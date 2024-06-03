mod context;
use ash::extensions::ext;
use context::*;
mod camera;
use camera::*;
mod gltf;
// mod pipeline;

use memoffset::offset_of;

mod model;
use gltf::Vertex;
use log::debug;
use model::*;

use anyhow::Result;
use ash::extensions::khr::{self};
use ash::vk::{
    self, AabbPositionsKHR, AccelerationStructureTypeKHR, GeometryTypeKHR,
    KhrShaderNonSemanticInfoFn, Packed24_8, TransformMatrixKHR,
};
use gpu_allocator::MemoryLocation;

use ash::Device;

use glam::{vec3, Mat4, Vec3, Vec4};
use std::default::Default;
use std::ffi::{CStr, CString};
use std::os::unix::thread;
use std::slice::from_ref;
use std::time::{Duration, Instant};

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

use std::mem::{size_of, transmute};

use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

struct ComputePipeline {
    pub handel: vk::Pipeline,
    pub descriptors: Box<[Box<[vk::DescriptorSet]>]>,
    pub layout: vk::PipelineLayout,
    pub descriptor_layouts: Box<[vk::DescriptorSetLayout]>,
}

struct GBuffer {
    pub frame_buffers: Vec<vk::Framebuffer>,
    pub g_buffers: Vec<(vk::ImageView, Image)>,
    pub depth_buffers: Vec<(vk::ImageView, Image)>,
}

fn main() {
    let device_extensions = [
        khr::Swapchain::name(),
        khr::RayTracingPipeline::name(),
        khr::AccelerationStructure::name(),
        vk::ExtDescriptorIndexingFn::name(),
        vk::ExtScalarBlockLayoutFn::name(),
        vk::KhrGetMemoryRequirements2Fn::name(),
        khr::BufferDeviceAddress::name(),
        khr::DeferredHostOperations::name(),
        vk::KhrSpirv14Fn::name(),
        vk::KhrShaderFloatControlsFn::name(),
        vk::KhrBufferDeviceAddressFn::name(),
        KhrShaderNonSemanticInfoFn::name(),
    ];

    let device_extensions = device_extensions
        .into_iter()
        .map(|e| e.to_str().unwrap())
        .collect::<Vec<_>>();
    let device_extensions = device_extensions.as_slice();

    let device_features = DeviceFeatures {
        ray_tracing_pipeline: true,
        acceleration_structure: true,
        runtime_descriptor_array: true,
        buffer_device_address: true,
        dynamic_rendering: true,
        synchronization2: true,
    };
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("TEst")
        .with_inner_size(PhysicalSize {
            width: 1920,
            height: 1080,
        })
        .build(&event_loop)
        .unwrap();
    let window_size = window.inner_size();
    let mut ctx = Context::new(
        window.raw_window_handle(),
        window.raw_display_handle(),
        "Test",
        device_extensions,
        device_features,
        window_size,
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

    let mut uniform_data = UniformData {
        proj_inverse: camera.projection_matrix().inverse(),
        view_inverse: camera.view_matrix().inverse(),
    };
    let mut g_uniform_data = GUniformData {
        proj: camera.projection_matrix(),
        view: camera.view_matrix(),
        model: Mat4::IDENTITY,
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
                Context::transition_image_layout_to_general(
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

    let model_mat = Mat4::from_scale(vec3(0.1, 0.1, 0.1));
    let tlas = {
        #[rustfmt::skip]
        let transform_matrix = vk::TransformMatrixKHR { matrix: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]};

        let instaces = &[];

        let instance_buffer = ctx
            .create_gpu_only_buffer_from_data(
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                instaces,
                Some("Instance Buffer"),
            )
            .unwrap();
        let instance_buffer_addr = instance_buffer.get_device_address(&ctx.device);

        let as_struct_geo = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: instance_buffer_addr,
                    })
                    .build(),
            })
            .build();

        let as_ranges = vk::AccelerationStructureBuildRangeInfoKHR::builder()
            .first_vertex(0)
            .primitive_count(0)
            .primitive_offset(0)
            .transform_offset(0)
            .build();

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
                // (
                //     &include_bytes!("./shaders/anyhit.rahit.spv")[..],
                //     vk::ShaderStageFlags::ANY_HIT_KHR,
                // ),
                // (
                //     &include_bytes!("./shaders/rayint.rint.spv")[..],
                //     vk::ShaderStageFlags::INTERSECTION_KHR,
                // ),
            ],
            group: RayTracingShaderGroup::Hit,
        },
    ];
    let pipeline = create_ray_tracing_pipeline(&ctx, &shaders_create_info).unwrap();
    let shader_binding_table = ctx.create_shader_binding_table(&pipeline).unwrap();

    let (descriptor_pool, static_set, dynamic_set) = {
        create_raytracing_descriptor_sets(
            &mut ctx,
            &pipeline,
            &tlas,
            &uniform_buffer,
            &storage_images,
            &mut vec![],
        )
        .unwrap()
    };

    let present_frame_buffers = {
        (0..ctx.swapchain.images.len())
            .map(|i| unsafe {
                ctx.device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::builder()
                            .attachments(&[ctx.swapchain.images[i].1])
                            .attachment_count(1)
                            .height(window_size.height)
                            .width(window_size.width)
                            .layers(1)
                            .render_pass(post_proccesing_pipeline.render_pass)
                            .build(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<vk::Framebuffer>>()
    };

    let mut moved = false;
    let mut frame: u32 = 0;
    let mut now = Instant::now();
    let mut last_time = Instant::now();
    let mut frame_time = Duration::new(0, 0);
    #[allow(clippy::collapsible_match, clippy::single_match)]
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        controles = controles.handle_event(&event, &window);
        match event {
            Event::NewEvents(_) => {
                controles = controles.reset();
                now = Instant::now();
                frame_time = now - last_time;
                last_time = now;
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state,
                            ..
                        },
                    ..
                } => match (keycode, state) {
                    (VirtualKeyCode::Escape, ElementState::Released) => {
                        *control_flow = ControlFlow::Exit
                    }
                    _ => (),
                },
                _ => (),
            },
            Event::MainEventsCleared => {
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
                        let moved_c = &[if moved == true { 1 as u8 } else { 0 as u8 }, 0, 0, 0];

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

                        let begin_info = vk::RenderPassBeginInfo::builder()
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

                        let view_port = vk::Viewport::builder()
                            .height(window_size.height as f32)
                            .width(window_size.width as f32)
                            .max_depth(1.0)
                            .min_depth(0.0)
                            .x(0 as f32)
                            .y(0 as f32)
                            .build();
                        ctx.device.cmd_set_viewport(*cmd, 0, &[view_port]);

                        let scissor = vk::Rect2D::builder()
                            .extent(vk::Extent2D {
                                height: window_size.height,
                                width: window_size.width,
                            })
                            .offset(vk::Offset2D { x: 0, y: 0 })
                            .build();
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
            }
            Event::LoopDestroyed => {
                ctx.destroy();
            }
            _ => (),
        }
    });
}

struct PostProccesingPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptors: Vec<vk::DescriptorSet>,
    pub render_pass: vk::RenderPass,
}

fn create_post_proccesing_pipelien(
    ctx: &mut Context,
    storage_images: &Vec<(vk::ImageView, Image)>,
) -> Result<PostProccesingPipeline> {
    let attachments = [vk::AttachmentDescription::builder()
        .samples(vk::SampleCountFlags::TYPE_1)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .format(ctx.swapchain.format)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .build()];

    let subpasses = [vk::SubpassDescription::builder()
        .color_attachments(&[vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()])
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .build()];

    let dependencys = [vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .build()];

    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .dependencies(&dependencys)
        .subpasses(&subpasses);

    let render_pass = unsafe {
        ctx.device
            .create_render_pass(&render_pass_create_info, None)?
    };

    let descriptor_bindings = [vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build()];

    let descriptor_layout = ctx.create_descriptor_set_layout(&descriptor_bindings, &[])?;

    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&[descriptor_layout])
        .push_constant_ranges(&[])
        .build();
    let layout = unsafe { ctx.device.create_pipeline_layout(&layout_info, None) }?;

    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
        .blend_enable(false)
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .build()];

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .attachments(&color_blend_attachments)
        .logic_op(vk::LogicOp::COPY)
        .logic_op_enable(false)
        .blend_constants([0.0, 0.0, 0.0, 0.0])
        .build();

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_test_enable(false)
        .depth_write_enable(false);

    let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .cull_mode(vk::CullModeFlags::BACK)
        .depth_clamp_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(false)
        .rasterizer_discard_enable(false);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let entry_point_name: CString = CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::builder()
            .module(ctx.create_shader_module("./src/shaders/post_processing.frag.spv".to_string()))
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&entry_point_name)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .module(ctx.create_shader_module("./src/shaders/post_processing.vert.spv".to_string()))
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(&entry_point_name)
            .build(),
    ];

    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .scissor_count(1)
        .viewport_count(1);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .primitive_restart_enable(false)
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_attribute_descriptions(&[])
        .vertex_binding_descriptions(&[])
        .build();

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
        .color_blend_state(&color_blend_state)
        .depth_stencil_state(&depth_stencil_state)
        .dynamic_state(&dynamic_state)
        .input_assembly_state(&input_assembly_state)
        .vertex_input_state(&vertex_input_state)
        .layout(layout)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .render_pass(render_pass)
        .stages(&stages)
        .viewport_state(&viewport_state)
        .subpass(0)
        .build();
    let pipeline = unsafe {
        ctx.device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
            .unwrap()
    }[0];

    let pool_sizes = [vk::DescriptorPoolSize::builder()
        .descriptor_count(ctx.swapchain.images.len() as u32)
        .ty(vk::DescriptorType::STORAGE_IMAGE)
        .build()];

    let descriptor_pool =
        ctx.create_descriptor_pool((ctx.swapchain.images.len() as u32) * 2, &pool_sizes)?;
    let descriptors = allocate_descriptor_sets(
        &ctx.device,
        &descriptor_pool,
        &descriptor_layout,
        ctx.swapchain.images.len() as u32,
    )?;
    for i in 0..ctx.swapchain.images.len() {
        let write = WriteDescriptorSet {
            binding: 0,
            kind: WriteDescriptorSetKind::StorageImage {
                view: storage_images[i].0,
                layout: vk::ImageLayout::GENERAL,
            },
        };

        update_descriptor_sets(ctx, &descriptors[i], &[write]);
    }

    Ok(PostProccesingPipeline {
        pipeline,
        layout,
        render_pass,
        descriptors,
    })
}

fn create_raytracing_descriptor_sets(
    context: &mut Context,
    pipeline: &RayTracingPipeline,
    top_as: &AccelerationStructure,
    ubo_buffer: &Buffer,
    storage_images: &Vec<(vk::ImageView, Image)>,
) -> Result<(
    vk::DescriptorPool,
    vk::DescriptorSet,
    Vec<vk::DescriptorSet>,
)> {
    let size = context.swapchain.images.len() as u32;

    let mut pool_sizes = vec![
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1 + (size * 2))
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .build(),
    ];

    let pool = context.create_descriptor_pool((size * 2) + 1, &pool_sizes)?;

    let static_set =
        allocate_descriptor_set(&context.device, &pool, &pipeline.descriptor_set_layout)?;
    let dynamic_sets =
        allocate_descriptor_sets(&context.device, &pool, &pipeline.dynamic_layout, size)?;

    let sampler_create_info: vk::SamplerCreateInfo = vk::SamplerCreateInfo {
        mag_filter: vk::Filter::NEAREST,
        min_filter: vk::Filter::NEAREST,
        mipmap_mode: vk::SamplerMipmapMode::NEAREST,
        address_mode_u: vk::SamplerAddressMode::MIRRORED_REPEAT,
        address_mode_v: vk::SamplerAddressMode::MIRRORED_REPEAT,
        address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
        max_anisotropy: 1.0,
        border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
        compare_op: vk::CompareOp::NEVER,
        ..Default::default()
    };

    let sampler = unsafe { context.device.create_sampler(&sampler_create_info, None)? };

    for i in 0..context.swapchain.images.len() {
        let write = WriteDescriptorSet {
            binding: 2,
            kind: WriteDescriptorSetKind::StorageImage {
                view: storage_images[i].0.clone(),
                layout: vk::ImageLayout::GENERAL,
            },
        };

        update_descriptor_sets(context, &dynamic_sets[i], &[write]);
    }

    let w = [
        WriteDescriptorSet {
            binding: 0,
            kind: WriteDescriptorSetKind::AccelerationStructure {
                acceleration_structure: top_as.handle,
            },
        },
        WriteDescriptorSet {
            binding: 2,
            kind: WriteDescriptorSetKind::UniformBuffer {
                buffer: ubo_buffer.inner,
            },
        },
    ];

    update_descriptor_sets(context, &static_set, w.as_slice());

    for (i, (image_index, sampler_index)) in model.textures.iter().enumerate() {
        let view = &model.views[*image_index];
        let sampler = &model.samplers[*sampler_index];
        let img_info = vk::DescriptorImageInfo::builder()
            .image_view(*view)
            .sampler(*sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        unsafe {
            context.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_array_element(i as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_binding(7)
                    .dst_set(static_set.clone())
                    .image_info(from_ref(&img_info))
                    .build()],
                &[],
            )
        };
    }

    Ok((pool, static_set, dynamic_sets))
}

fn create_acceleration_structure(
    ctx: &mut Context,
    level: vk::AccelerationStructureTypeKHR,
    as_geometry: &[vk::AccelerationStructureGeometryKHR],
    as_ranges: &[vk::AccelerationStructureBuildRangeInfoKHR],
    max_primitive_counts: &[u32],
) -> Result<AccelerationStructure> {
    let build_geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .ty(level)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(as_geometry);

    let build_size = unsafe {
        ctx.ray_tracing
            .acceleration_structure_fn
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_geo_info,
                max_primitive_counts,
            )
    };

    let buffer = ctx.create_buffer(
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        MemoryLocation::GpuOnly,
        build_size.acceleration_structure_size,
        Some("Acceleration Structure Buffer"),
    )?;

    let create_info = vk::AccelerationStructureCreateInfoKHR::builder()
        .buffer(buffer.inner)
        .size(build_size.acceleration_structure_size)
        .ty(level);
    let handle = unsafe {
        ctx.ray_tracing
            .acceleration_structure_fn
            .create_acceleration_structure(&create_info, None)?
    };

    let scratch_buffer = ctx.create_aligned_buffer(
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        MemoryLocation::GpuOnly,
        build_size.build_scratch_size,
        Some("Acceleration Structure Scratch Buffer"),
        ctx.ray_tracing
            .acceleration_structure_properties
            .min_acceleration_structure_scratch_offset_alignment
            .into(),
    )?;

    let scratch_buffer_address = scratch_buffer.get_device_address(&ctx.device);

    let build_geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .ty(level)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(as_geometry)
        .dst_acceleration_structure(handle)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer_address,
        });

    ctx.execute_one_time_commands(|cmd_buffer| {
        unsafe {
            ctx.ray_tracing
                .acceleration_structure_fn
                .cmd_build_acceleration_structures(
                    *cmd_buffer,
                    from_ref(&build_geo_info),
                    from_ref(&as_ranges),
                )
        };
    })?;

    let address_info =
        vk::AccelerationStructureDeviceAddressInfoKHR::builder().acceleration_structure(handle);
    let address = unsafe {
        ctx.ray_tracing
            .acceleration_structure_fn
            .get_acceleration_structure_device_address(&address_info)
    };

    Ok(AccelerationStructure {
        buffer,
        handle,
        device_address: address,
    })
}

#[derive(Debug, Clone)]
pub struct RayTracingShaderCreateInfo<'a> {
    pub source: &'a [(&'a [u8], vk::ShaderStageFlags)],
    pub group: RayTracingShaderGroup,
}

#[derive(Debug, Clone, Copy)]
pub enum RayTracingShaderGroup {
    RayGen,
    Miss,
    Hit,
}

pub fn read_shader_from_bytes(bytes: &[u8]) -> Result<Vec<u32>> {
    let mut cursor = std::io::Cursor::new(bytes);
    Ok(ash::util::read_spv(&mut cursor)?)
}

fn module_from_bytes(device: &Device, source: &[u8]) -> Result<vk::ShaderModule> {
    let source = read_shader_from_bytes(source)?;

    let create_info = vk::ShaderModuleCreateInfo::builder().code(&source);
    let res = unsafe { device.create_shader_module(&create_info, None) }?;
    Ok(res)
}

fn create_ray_tracing_pipeline(
    ctx: &Context,
    shaders_create_info: &[RayTracingShaderCreateInfo],
) -> Result<RayTracingPipeline> {
    let static_layout_bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(
                vk::ShaderStageFlags::INTERSECTION_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )
            .build(),
    ];

    let dynamic_layout_bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
            .build(),
    ];

    let static_dsl = ctx.create_descriptor_set_layout(&static_layout_bindings, &[])?;
    let dynamic_dsl = ctx.create_descriptor_set_layout(&dynamic_layout_bindings, &[])?;
    let old_image_dsl = ctx.create_descriptor_set_layout(&dynamic_layout_bindings, &[])?;

    let dsls = [static_dsl, dynamic_dsl, old_image_dsl];

    let push_constants = &[vk::PushConstantRange::builder()
        .offset(0)
        .size((size_of::<u32>() * 2) as u32)
        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
        .build()];

    let pipe_layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&dsls)
        .push_constant_ranges(push_constants);
    let pipeline_layout = unsafe { ctx.device.create_pipeline_layout(&pipe_layout_info, None)? };

    let mut shader_group_info = RayTracingShaderGroupInfo {
        group_count: shaders_create_info.len() as u32,
        ..Default::default()
    };

    let mut modules = vec![];
    let mut stages = vec![];
    let mut groups = vec![];

    let entry_point_name: CString = CString::new("main").unwrap();

    for shader in shaders_create_info.iter() {
        let mut this_modules = vec![];
        let mut this_stages = vec![];

        shader.source.into_iter().for_each(|s| {
            let module = module_from_bytes(&ctx.device, s.0).unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::builder()
                .stage(s.1)
                .module(module)
                .name(&entry_point_name)
                .build();
            this_modules.push(module);
            this_stages.push(stage);
        });

        match shader.group {
            RayTracingShaderGroup::RayGen => shader_group_info.raygen_shader_count += 1,
            RayTracingShaderGroup::Miss => shader_group_info.miss_shader_count += 1,
            RayTracingShaderGroup::Hit => shader_group_info.hit_shader_count += 1,
        };

        let shader_index = stages.len();

        let mut group = vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR);
        group = match shader.group {
            RayTracingShaderGroup::RayGen | RayTracingShaderGroup::Miss => {
                group.general_shader(shader_index as _)
            }
            RayTracingShaderGroup::Hit => {
                group = group
                    .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                    .closest_hit_shader(shader_index as _);
                if shader.source.len() >= 2 {
                    group = group
                        .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                        .any_hit_shader((shader_index as u32) + 1);
                }
                if shader.source.len() >= 3 {
                    group = group
                        .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                        .any_hit_shader((shader_index as u32) + 1)
                        .intersection_shader((shader_index as u32) + 2);
                }

                group
            }
        };

        modules.append(&mut this_modules);
        stages.append(&mut this_stages);
        groups.push(group.build());
    }

    let pipe_info = vk::RayTracingPipelineCreateInfoKHR::builder()
        .layout(pipeline_layout)
        .stages(&stages)
        .groups(&groups)
        .max_pipeline_ray_recursion_depth(1);

    let inner = unsafe {
        ctx.ray_tracing.pipeline_fn.create_ray_tracing_pipelines(
            vk::DeferredOperationKHR::null(),
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipe_info),
            None,
        )?[0]
    };

    Ok(RayTracingPipeline {
        handle: inner,
        descriptor_set_layout: static_dsl,
        dynamic_layout: dynamic_dsl,
        layout: pipeline_layout,
        shader_group_info,
    })
}