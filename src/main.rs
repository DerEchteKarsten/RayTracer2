mod context;
use ash::extensions::ext;
use context::*;
mod camera;
use camera::*;
mod gltf;
// mod pipeline;

mod model;
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

use glam::{vec3, Mat4, Vec3};
use std::default::Default;
use std::ffi::CString;
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
    pub descriptor: vk::DescriptorSet,
    pub layout: vk::PipelineLayout,
    pub descriptor_layout: vk::DescriptorSetLayout,
}

fn main() {
    let model_thread = std::thread::spawn(|| gltf::load_file("./src/models/box.glb").unwrap());
    let image_thread = std::thread::spawn(|| image::open("./src/models/skybox.png").unwrap());

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
            width: 2560,
            height: 1440,
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

    let postprocessing_binding = [
        WriteDescriptorSet {
            binding: 0,
            kind: WriteDescriptorSetKind::StorageImage {
                view: ctx.storage_image.1,
                layout: vk::ImageLayout::GENERAL,
            },
        },
        WriteDescriptorSet {
            binding: 1,
            kind: WriteDescriptorSetKind::StorageImage {
                view: ctx.post_processing_image.1,
                layout: vk::ImageLayout::GENERAL,
            },
        },
    ];

    let spacial_reuse_buffer = ctx
        .create_gpu_only_buffer_from_data_with_size(
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &[()],
            (200 * window_size.width * window_size.height).into(),
            "Spacial Reuse".into(),
        )
        .unwrap();
    let temporal_reuse_buffer = ctx
        .create_gpu_only_buffer_from_data_with_size(
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &[()],
            (200 * window_size.width * window_size.height).into(),
            "Temporal Reuse".into(),
        )
        .unwrap();

    let initial_sample_buffer = ctx
        .create_gpu_only_buffer_from_data_with_size(
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &[()],
            (200 * window_size.width * window_size.height).into(),
            "Initial Samples".into(),
        )
        .unwrap();

    let reuse_bindings = [
        WriteDescriptorSet {
            binding: 0,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: spacial_reuse_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 1,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: temporal_reuse_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 2,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: initial_sample_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 3,
            kind: WriteDescriptorSetKind::StorageImage {
                layout: vk::ImageLayout::GENERAL,
                view: ctx.storage_image.1,
            },
        },
    ];

    let postprocessing_pass = create_compute_pipeline(
        &mut ctx,
        &postprocessing_binding,
        &[],
        include_bytes!("./shaders/post_processing.comp.spv"),
    );
    let reuse_pass = create_compute_pipeline(
        &mut ctx,
        &reuse_bindings,
        &[
            vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: 8,
            },
        ],
        include_bytes!("./shaders/reuse.comp.spv"),
    );

    log::trace!("Starting Image Load");
    let image = image_thread.join().unwrap();
    let sky_box = Image::new_from_data(&mut ctx, image, vk::Format::R8G8B8A8_SRGB).unwrap();
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

    let tlas = {
        #[rustfmt::skip]
        let transform_matrix = vk::TransformMatrixKHR { matrix: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]};

        let instaces = &[model.instance(Mat4::IDENTITY)];

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
            .primitive_count(instaces.len() as _)
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
    let pipeline = create_ray_tracing_pipeline(&ctx, &model, &shaders_create_info).unwrap();
    let shader_binding_table = ctx.create_shader_binding_table(&pipeline).unwrap();

    let (descriptor_pool, static_set) = create_raytracing_descriptor_sets(
        &mut ctx,
        &pipeline,
        &tlas,
        &uniform_buffer,
        &model,
        &mut vec![
            WriteDescriptorSet {
                binding: 8,
                kind: WriteDescriptorSetKind::CombinedImageSampler {
                    view: sky_box.1,
                    sampler: sky_box_sampler,
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
            },
            WriteDescriptorSet {
                binding: 9,
                kind: WriteDescriptorSetKind::StorageBuffer {
                    buffer: initial_sample_buffer.inner,
                },
            },
        ],
    )
    .unwrap();

    let mut moved = false;
    let mut frame = 0;
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
                debug!("{:?}", frame_time);
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
                    frame = 0;
                }
                camera = new_cam;

                uniform_data.proj_inverse = camera.projection_matrix().inverse();
                uniform_data.view_inverse = camera.view_matrix().inverse();

                uniform_buffer
                    .copy_data_to_buffer(std::slice::from_ref(&uniform_data))
                    .unwrap();
                //println!("{}", delta_time.as_micros());
                ctx.frame = ctx
                    .render(|i| {
                        record_command_buffers(
                            &ctx,
                            i,
                            &pipeline,
                            &shader_binding_table,
                            &postprocessing_pass,
                            &reuse_pass,
                            window_size,
                            &static_set,
                            &frame,
                            &moved,
                        );
                    })
                    .unwrap();
            }
            Event::LoopDestroyed => {
                unsafe { ctx.device.destroy_descriptor_pool(descriptor_pool, None) };
                ctx.destroy();
            }
            _ => (),
        }
    });
}

fn create_raytracing_descriptor_sets(
    context: &mut Context,
    pipeline: &RayTracingPipeline,
    top_as: &AccelerationStructure,
    ubo_buffer: &Buffer,
    model: &Model,
    wirtes: &mut Vec<WriteDescriptorSet>,
) -> Result<(vk::DescriptorPool, vk::DescriptorSet)> {
    let size = context.swapchain.images.len() as u32;

    let mut pool_sizes = vec![
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(4)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(model.images.len() as _)
            .build(),
    ];

    for w in wirtes.iter() {
        pool_sizes.push(match w.kind {
            WriteDescriptorSetKind::CombinedImageSampler {
                layout: _,
                sampler: _,
                view: _,
            } => vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .build(),
            WriteDescriptorSetKind::StorageBuffer { buffer: _ } => {
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .build()
            }
            WriteDescriptorSetKind::AccelerationStructure {
                acceleration_structure: _,
            } => vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .build(),
            WriteDescriptorSetKind::StorageImage { view: _, layout: _ } => {
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .build()
            }
            WriteDescriptorSetKind::UniformBuffer { buffer: _ } => {
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .build()
            }
        })
    }

    let pool = context.create_descriptor_pool((size * 2) + 1, &pool_sizes)?;

    let static_set =
        allocate_descriptor_set(&context.device, &pool, &pipeline.descriptor_set_layout)?;
    let w: Vec<_> = [
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
        WriteDescriptorSet {
            binding: 4,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: model.vertex_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 5,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: model.index_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 6,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: model.geometry_info_buffer.inner,
            },
        },
    ]
    .iter()
    .chain(wirtes.iter())
    .map(|x| x.clone())
    .collect();

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

    Ok((pool, static_set))
}

fn record_command_buffers(
    ctx: &Context,
    index: u32,
    pipeline: &RayTracingPipeline,
    shader_binding_table: &ShaderBindingTable,
    postprocessing: &ComputePipeline,
    temporal_reuse: &ComputePipeline,
    window_size: PhysicalSize<u32>,
    static_set: &vk::DescriptorSet,
    frame: &u32,
    moved: &bool,
) {
    let swapchain_image = &ctx.swapchain.images[index as usize].0;
    let cmd = &ctx.cmd_buffs[index as usize];

    unsafe {
        let frame_c =
            std::slice::from_raw_parts(frame as *const u32 as *const u8, size_of::<u32>());
        let moved_c = &[if *moved == true { 1 as u8 } else { 0 as u8 }, 0, 0, 0];

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

        ctx.device.cmd_bind_pipeline(
            *cmd,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline.handle,
        );

        ctx.device.cmd_bind_descriptor_sets(
            *cmd,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline.layout,
            0,
            &[*static_set],
            &[],
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

        ctx.device
            .cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::COMPUTE, temporal_reuse.handel);

        ctx.device.cmd_bind_descriptor_sets(
            *cmd,
            vk::PipelineBindPoint::COMPUTE,
            temporal_reuse.layout,
            0,
            &[temporal_reuse.descriptor],
            &[],
        );

        ctx.device.cmd_push_constants(
            *cmd,
            temporal_reuse.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &frame_c,
        );

        ctx.device
            .cmd_dispatch(*cmd, window_size.width, window_size.height, 1);

        ctx.device
            .cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::COMPUTE, postprocessing.handel);

        ctx.device.cmd_bind_descriptor_sets(
            *cmd,
            vk::PipelineBindPoint::COMPUTE,
            postprocessing.layout,
            0,
            &[postprocessing.descriptor],
            &[],
        );

        ctx.device
            .cmd_dispatch(*cmd, window_size.width, window_size.height, 1);

        ctx.pipeline_image_barriers(
            &cmd,
            &[
                ImageBarrier {
                    image: &swapchain_image,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_access_mask: vk::AccessFlags2::NONE,
                    dst_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                    src_stage_mask: vk::PipelineStageFlags2::NONE,
                    dst_stage_mask: vk::PipelineStageFlags2::TRANSFER,
                },
                ImageBarrier {
                    image: &ctx.post_processing_image.0,
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    src_access_mask: vk::AccessFlags2::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags2::TRANSFER_READ,
                    src_stage_mask: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    dst_stage_mask: vk::PipelineStageFlags2::TRANSFER,
                },
            ],
        );

        ctx.copy_image(
            &cmd,
            &ctx.post_processing_image.0,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        ctx.pipeline_image_barriers(
            &cmd,
            &[
                ImageBarrier {
                    image: &swapchain_image,
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
                    dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                },
                ImageBarrier {
                    image: &ctx.post_processing_image.0,
                    old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    new_layout: vk::ImageLayout::GENERAL,
                    src_access_mask: vk::AccessFlags2::TRANSFER_READ,
                    dst_access_mask: vk::AccessFlags2::NONE,
                    src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
                    dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                },
            ],
        );

        ctx.pipeline_image_barriers(
            cmd,
            &[ImageBarrier {
                image: &swapchain_image,
                old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            }],
        );
    }
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
    model: &Model,
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
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(
                vk::ShaderStageFlags::INTERSECTION_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(5)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(7)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(model.images.len() as _)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(8)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::MISS_KHR)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(9)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
            .build(),
    ];

    // let dynamic_layout_bindings = [vk::DescriptorSetLayoutBinding::builder()
    //     .binding(1)
    //     .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
    //     .descriptor_count(1)
    //     .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
    //     .build(),
    // ];

    let static_dsl = ctx.create_descriptor_set_layout(&static_layout_bindings, &[])?;
    // let dynamic_dsl = ctx.create_descriptor_set_layout(&dynamic_layout_bindings, &[])?;
    let dsls = [static_dsl];

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

    let entry_point_name = CString::new("main").unwrap();

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
        layout: pipeline_layout,
        shader_group_info,
    })
}

fn create_compute_pipeline(
    ctx: &mut Context,
    writes: &[WriteDescriptorSet],
    push_constant_ranges: &[vk::PushConstantRange],
    shader: &[u8],
) -> ComputePipeline {
    let mut bindings = Vec::with_capacity(writes.len());
    for w in writes.iter() {
        bindings.push(match w.kind {
            WriteDescriptorSetKind::CombinedImageSampler {
                layout: _,
                sampler: _,
                view: _,
            } => vk::DescriptorSetLayoutBinding {
                binding: w.binding,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                ..Default::default()
            },
            WriteDescriptorSetKind::StorageBuffer { buffer: _ } => vk::DescriptorSetLayoutBinding {
                binding: w.binding,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                ..Default::default()
            },
            WriteDescriptorSetKind::AccelerationStructure {
                acceleration_structure: _,
            } => vk::DescriptorSetLayoutBinding {
                binding: w.binding,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                ..Default::default()
            },
            WriteDescriptorSetKind::StorageImage { view: _, layout: _ } => {
                vk::DescriptorSetLayoutBinding {
                    binding: w.binding,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    ..Default::default()
                }
            }
            WriteDescriptorSetKind::UniformBuffer { buffer: _ } => vk::DescriptorSetLayoutBinding {
                binding: w.binding,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                ..Default::default()
            },
        })
    }

    let set_layout = create_descriptor_set_layout(&ctx.device, &bindings, &[]).unwrap();

    let layout = unsafe {
        ctx.device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[set_layout])
                    .push_constant_ranges(push_constant_ranges)
                    .build(),
                None,
            )
            .unwrap()
    };
    let entry_point_name = CString::new("main").unwrap();
    let compute_module = module_from_bytes(&ctx.device, shader).unwrap();
    let compute_stage = vk::PipelineShaderStageCreateInfo::builder()
        .module(compute_module)
        .name(&entry_point_name)
        .stage(vk::ShaderStageFlags::COMPUTE)
        .build();
    let create_info = vk::ComputePipelineCreateInfo::builder()
        .stage(compute_stage)
        .layout(layout)
        .build();
    let handel = unsafe {
        ctx.device
            .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .unwrap()
    }[0];

    let mut pool_sizes = Vec::with_capacity(writes.len());
    for w in writes.iter() {
        pool_sizes.push(match w.kind {
            WriteDescriptorSetKind::CombinedImageSampler {
                layout: _,
                sampler: _,
                view: _,
            } => vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .build(),
            WriteDescriptorSetKind::StorageBuffer { buffer: _ } => {
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .build()
            }
            WriteDescriptorSetKind::AccelerationStructure {
                acceleration_structure: _,
            } => vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .build(),
            WriteDescriptorSetKind::StorageImage { view: _, layout: _ } => {
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .build()
            }
            WriteDescriptorSetKind::UniformBuffer { buffer: _ } => {
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .build()
            }
        })
    }

    let pool = ctx.create_descriptor_pool(1, &pool_sizes).unwrap();
    let descriptor = allocate_descriptor_set(&ctx.device, &pool, &set_layout).unwrap();
    for w in writes.iter() {
        update_descriptor_sets(ctx, &descriptor, &[w.clone()]);
    }
    // update_descriptor_sets(ctx, &descriptor, writes.clone());
    // let view = ctx.storage_image.1.clone();

    // update_descriptor_sets(ctx, &descriptor, &[
    //     WriteDescriptorSet {
    //         binding: 0,
    //         kind: WriteDescriptorSetKind::StorageImage { view: view, layout: vk::ImageLayout::GENERAL }
    //     }
    // ]);
    // let view = ctx.post_processing_image.1.clone();
    // update_descriptor_sets(ctx, &descriptor, &[
    //     WriteDescriptorSet {
    //         binding: 1,
    //         kind: WriteDescriptorSetKind::StorageImage { view: view, layout: vk::ImageLayout::GENERAL }
    //     }
    // ]);
    ComputePipeline {
        descriptor,
        handel,
        layout,
        descriptor_layout: set_layout,
    }
}
