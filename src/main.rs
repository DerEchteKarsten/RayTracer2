mod context;
use ash::ext::buffer_device_address;
use ash::khr::{
    acceleration_structure, deferred_host_operations, get_memory_requirements2,
    ray_tracing_pipeline, shader_float_controls, shader_non_semantic_info, swapchain,
};
use context::*;
mod camera;
use camera::*;
mod gltf;
mod pipelines;

use memoffset::offset_of;

mod model;
use gltf::Vertex;
use log::debug;
use model::*;

use anyhow::Result;
use ash::vk::{
    self, AabbPositionsKHR, AccelerationStructureTypeKHR, BufferUsageFlags, GeometryTypeKHR,
    Packed24_8, TransformMatrixKHR, KHR_SPIRV_1_4_NAME,
};
use gpu_allocator::MemoryLocation;

use ash::Device;

use glam::{vec3, Mat4, Vec3, Vec4};
use std::default::{self, Default};
use std::ffi::{CStr, CString};
use std::os::unix::thread;
use std::slice::from_ref;
use std::time::{Duration, Instant};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowAttributes;

use raw_window_handle::{HasRawWindowHandle, RawDisplayHandle, RawWindowHandle};

use std::mem::{size_of, transmute};

use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};


fn main() {
    let model_thread =
        std::thread::spawn(|| gltf::load_file("./src/models/sponza_scene.glb").unwrap());
    let image_thread = std::thread::spawn(|| image::open("./src/models/skybox.exr").unwrap());

    let device_extensions: [&CStr; 11] = [
        swapchain::NAME,
        ray_tracing_pipeline::NAME,
        acceleration_structure::NAME,
        ash::ext::descriptor_indexing::NAME,
        ash::ext::scalar_block_layout::NAME,
        get_memory_requirements2::NAME,
        buffer_device_address::NAME,
        deferred_host_operations::NAME,
        KHR_SPIRV_1_4_NAME,
        shader_float_controls::NAME,
        shader_non_semantic_info::NAME,
    ];

    let device_features = DeviceFeatures {
        ray_tracing_pipeline: true,
        acceleration_structure: true,
        runtime_descriptor_array: true,
        buffer_device_address: true,
        dynamic_rendering: true,
        synchronization2: true,
        attomics: true,
    };
    let event_loop = EventLoop::new().unwrap();
    let window = event_loop
        .create_window(WindowAttributes::default().with_inner_size(PhysicalSize {
            width: 1920,
            height: 1080,
        }))
        .unwrap();

    let window_size = window.inner_size();
    let mut ctx = Renderer::new(
        &window,
        &window,
        &device_features,
        device_extensions,
        window_size.width,
        window_size.height,
    )
    .unwrap();

    let mut camera = Camera::new(
        vec3(0.0, 0.0, 10.0),
        vec3(0.0, 0.0, -1.0),
        40.0,
        window_size.width as f32 / window_size.height as f32,
        0.1,
        1000.0,
    );

    let sampler_info = vk::SamplerCreateInfo {
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
        address_mode_u: vk::SamplerAddressMode::MIRRORED_REPEAT,
        address_mode_v: vk::SamplerAddressMode::MIRRORED_REPEAT,
        address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
        max_anisotropy: 1.0,
        border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
        compare_op: vk::CompareOp::NEVER,
        ..Default::default()
    };

    let mut controles = Controls {
        ..Default::default()
    };

    log::trace!("Starting Image Load");
    let image = image_thread.join().unwrap();
    let sky_box = Image::new_from_data(&mut ctx, image, vk::Format::R32G32B32A32_SFLOAT).unwrap();
    let sky_box_sampler = unsafe { ctx.device.create_sampler(&sampler_info, None) }.unwrap();
    log::trace!("Starting Model");
    let model = model_thread.join().unwrap();
    let model = Model::from_gltf(&mut ctx, model).unwrap();
    log::trace!("Model Loaded");

    let mut uniform_data = UniformData {
        proj_inverse: camera.projection_matrix().inverse(),
        view_inverse: camera.view_matrix().inverse(),
    };

    let uniform_buffer = ctx
        .create_buffer(
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            size_of::<UniformData>() as u64,
            Some("Uniform Buffer"),
        )
        .unwrap();
    uniform_buffer
        .copy_data_to_buffer(std::slice::from_ref(&uniform_data))
        .unwrap();

    let storage_images = {
        (0..ctx.swapchain.images.len())
            .map(|_| {
                let image = Image::new_2d(
                    &ctx.device,
                    &mut ctx.allocator,
                    vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                    MemoryLocation::GpuOnly,
                    vk::Format::R32G32B32A32_SFLOAT,
                    window_size.width,
                    window_size.height,
                )
                .unwrap();
                let view = create_image_view(&ctx.device, &image, vk::Format::R32G32B32A32_SFLOAT);
                Renderer::transition_image_layout_to_general(
                    &ctx.device,
                    &ctx.command_pool,
                    &image,
                    &ctx.graphics_queue,
                )
                .unwrap();
                (view, image)
            })
            .collect::<Vec<(vk::ImageView, Image)>>()
    };
    let post_proccesing_pipeline =
        create_post_proccesing_pipelien(&mut ctx, &storage_images).unwrap();

    let present_frame_buffers = {
        (0..ctx.swapchain.images.len())
            .map(|i| unsafe {
                ctx.device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .attachments(&[ctx.swapchain.images[i].view])
                            .attachment_count(1)
                            .height(window_size.height)
                            .width(window_size.width)
                            .render_pass(post_proccesing_pipeline.render_pass),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<vk::Framebuffer>>()
    };

    let model_mat = Mat4::from_scale(vec3(0.1, 0.1, 0.1));
    let tlas = {
        #[rustfmt::skip]
        let transform_matrix = vk::TransformMatrixKHR { matrix: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]};

        let instaces = &[model.instance(model_mat)];

        let instance_buffer = ctx
            .create_gpu_only_buffer_from_data(
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                instaces,
                Some("Instance Buffer"),
            )
            .unwrap();
        let instance_buffer_addr = instance_buffer.get_device_address(&ctx.device);

        let as_struct_geo = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: instance_buffer_addr,
                    }),
            });

        let as_ranges = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .first_vertex(0)
            .primitive_count(instaces.len() as _)
            .primitive_offset(0)
            .transform_offset(0);

        create_acceleration_structure(
            &mut ctx,
            AccelerationStructureTypeKHR::TOP_LEVEL,
            &[as_struct_geo],
            &[as_ranges],
            &[1],
        )
        .unwrap()
    };

    let shaders_create_info = [
        RayTracingShaderCreateInfo {
            source: &[(
                &include_bytes!("./shaders/raygen.rgen.spv")[..],
                vk::ShaderStageFlags::RAYGEN_KHR,
            )],
            group: RayTracingShaderGroup::RayGen,
        },
        RayTracingShaderCreateInfo {
            source: &[(
                &include_bytes!("./shaders/raymiss.rmiss.spv")[..],
                vk::ShaderStageFlags::MISS_KHR,
            )],
            group: RayTracingShaderGroup::Miss,
        },
        RayTracingShaderCreateInfo {
            source: &[
                (
                    &include_bytes!("./shaders/rayhit.rchit.spv")[..],
                    vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                ),
            ],
            group: RayTracingShaderGroup::Hit,
        },
    ];
    let pipeline = create_ray_tracing_pipeline(&ctx, &model, &shaders_create_info).unwrap();
    let shader_binding_table = ctx.create_shader_binding_table(&pipeline).unwrap();

    let temporal_reservoir_buffers = [
        ctx.create_buffer(
            BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            (window_size.width * window_size.height * 100) as u64,
            Some("Buffer"),
        )
        .unwrap(),
        ctx.create_buffer(
            BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            (window_size.width * window_size.height * 100) as u64,
            Some("Buffer"),
        )
        .unwrap(),
    ];

    let (static_set, dynamic_set, dynamic_set2) = {
        create_raytracing_descriptor_sets(
            &mut ctx,
            &pipeline,
            &tlas,
            &uniform_buffer,
            &model,
            &storage_images,
            &mut vec![WriteDescriptorSet {
                binding: 8,
                kind: WriteDescriptorSetKind::CombinedImageSampler {
                    view: sky_box.view,
                    sampler: sky_box_sampler,
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
            }],
            temporal_reservoir_buffers,
        )
        .unwrap()
    };
    let mut moved = false;
    let mut frame: u32 = 0;
    let mut now = Instant::now();
    let mut last_time = Instant::now();
    let mut frame_time = Duration::new(0, 0);
    #[allow(clippy::collapsible_match, clippy::single_match)]
    event_loop
        .run(move |event, control_flow| {
            control_flow.set_control_flow(ControlFlow::Poll);
            controles = controles.handle_event(&event, &window);
            match event {
                Event::NewEvents(_) => {
                    controles = controles.reset();
                    now = Instant::now();
                    frame_time = now - last_time;
                    last_time = now;
                    println!("{:?}", frame_time);
                }
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => control_flow.exit(),
                    WindowEvent::KeyboardInput { event, .. } => {
                        match (event.physical_key, event.state) {
                            (PhysicalKey::Code(KeyCode::Escape), ElementState::Released) => {
                                control_flow.exit()
                            }
                            _ => (),
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        // std::thread::sleep(Duration::from_secs(2));
                        frame += 1;
                        let new_cam = camera.update(&controles, frame_time);
                        moved = new_cam != camera;
                        if moved {
                            camera = new_cam;
                        }

                        uniform_data.proj_inverse = camera.projection_matrix().inverse();
                        uniform_data.view_inverse = camera.view_matrix().inverse();
                        uniform_buffer
                            .copy_data_to_buffer(std::slice::from_ref(&uniform_data))
                            .unwrap();

                        camera = new_cam;

                        ctx.render(|ctx, i| {
                            let cmd = &ctx.cmd_buffs[i as usize];

                            unsafe {
                                let frame_c = std::slice::from_raw_parts(
                                    &frame as *const u32 as *const u8,
                                    size_of::<u32>(),
                                );
                                let moved_c =
                                    &[if moved == true { 1 as u8 } else { 0 as u8 }, 0, 0, 0];

                                ctx.device.cmd_push_constants(
                                    *cmd,
                                    pipeline.layout,
                                    vk::ShaderStageFlags::RAYGEN_KHR,
                                    0,
                                    &frame_c,
                                );
                                ctx.device.cmd_push_constants(
                                    *cmd,
                                    pipeline.layout,
                                    vk::ShaderStageFlags::RAYGEN_KHR,
                                    size_of::<u32>() as u32,
                                    moved_c,
                                );

                                ctx.device.cmd_bind_descriptor_sets(
                                    *cmd,
                                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                                    pipeline.layout,
                                    0,
                                    &[
                                        static_set,
                                        dynamic_set[i as usize],
                                        dynamic_set[ctx.last_swapchain_image_index as usize],
                                        dynamic_set2[(frame % 2) as usize],
                                        dynamic_set2[1 - (frame % 2) as usize],
                                    ],
                                    &[],
                                );

                                ctx.device.cmd_bind_pipeline(
                                    *cmd,
                                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                                    pipeline.handle,
                                );

                                let call_region = vk::StridedDeviceAddressRegionKHR::default();

                                ctx.ray_tracing.pipeline_fn.cmd_trace_rays(
                                    *cmd,
                                    &shader_binding_table.raygen_region,
                                    &shader_binding_table.miss_region,
                                    &shader_binding_table.hit_region,
                                    &call_region,
                                    window_size.width,
                                    window_size.height,
                                    1,
                                );

                                let begin_info = vk::RenderPassBeginInfo::default()
                                    .render_pass(post_proccesing_pipeline.render_pass)
                                    .framebuffer(present_frame_buffers[i as usize])
                                    .render_area(vk::Rect2D {
                                        extent: vk::Extent2D {
                                            width: window_size.width,
                                            height: window_size.height,
                                        },
                                        offset: vk::Offset2D { x: 0, y: 0 },
                                    })
                                    .clear_values(&[vk::ClearValue {
                                        color: vk::ClearColorValue {
                                            float32: [0.0, 0.0, 0.0, 1.0],
                                        },
                                    }]);

                                let view_port = vk::Viewport::default()
                                    .height(window_size.height as f32)
                                    .width(window_size.width as f32)
                                    .max_depth(1.0)
                                    .min_depth(0.0)
                                    .x(0 as f32)
                                    .y(0 as f32);
                                ctx.device.cmd_set_viewport(*cmd, 0, &[view_port]);

                                let scissor = vk::Rect2D::default()
                                    .extent(vk::Extent2D {
                                        height: window_size.height,
                                        width: window_size.width,
                                    })
                                    .offset(vk::Offset2D { x: 0, y: 0 });
                                ctx.device.cmd_set_scissor(*cmd, 0, &[scissor]);

                                ctx.device.cmd_begin_render_pass(
                                    *cmd,
                                    &begin_info,
                                    vk::SubpassContents::INLINE,
                                );

                                ctx.device.cmd_bind_pipeline(
                                    *cmd,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    post_proccesing_pipeline.pipeline,
                                );

                                ctx.device.cmd_bind_descriptor_sets(
                                    *cmd,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    post_proccesing_pipeline.layout,
                                    0,
                                    &[post_proccesing_pipeline.descriptors[i as usize]],
                                    &[],
                                );
                                ctx.device.cmd_draw(*cmd, 6, 1, 0, 0);

                                ctx.device.cmd_end_render_pass(*cmd);
                            }
                        })
                        .unwrap();
                        window.request_redraw();
                    }
                    _ => (),
                },
                Event::LoopExiting => {
                    ctx.destroy();
                }
                _ => (),
            }
        })
        .unwrap();
}
