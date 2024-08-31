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
mod shader_params;
mod pipelines;
mod ui;

use memoffset::offset_of;

mod model;
use gltf::Vertex;
use log::debug;
use model::*;

use anyhow::Result;
use ash::vk::{
    self, AabbPositionsKHR, AccelerationStructureTypeKHR, BufferUsageFlags, GeometryTypeKHR, ImageLayout, Packed24_8, TransformMatrixKHR, KHR_SPIRV_1_4_NAME
};
use gpu_allocator::MemoryLocation;

use ash::Device;

use glam::{vec3, Mat4, Vec3, Vec4};
use pipelines::{create_render_recources, CalculateReservoirBufferParameters};
use shader_params::{GConst, RTXDI_ReservoirBufferParameters, RTXDI_RuntimeParameters, ReSTIRGI_BufferIndices, ReSTIRGI_FinalShadingParameters, ReSTIRGI_Parameters, ReSTIRGI_SpatialResamplingParameters, ReSTIRGI_TemporalResamplingParameters};
use simple_logger::SimpleLogger;
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

const NEIGHBOR_OFFSET_COUNT: u32 = 8192;
const RTXDI_RESERVOIR_BLOCK_SIZE: u32 = 16;

fn main() {
    let model_thread =
        std::thread::spawn(|| gltf::load_file("./src/models/box.glb").unwrap());
    let image_thread = std::thread::spawn(|| image::open("./src/models/skybox2.exr").unwrap());

    SimpleLogger::new().init().unwrap();

    let device_extensions: [&CStr; 10] = [
        swapchain::NAME,
        ray_tracing_pipeline::NAME,
        acceleration_structure::NAME,
        ash::ext::descriptor_indexing::NAME,
        ash::ext::scalar_block_layout::NAME,
        get_memory_requirements2::NAME,
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
        vec3(0.0, 0.0, 1.0),
        65.0,
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
    ctx.transition_image_layout(&sky_box.image, ImageLayout::SHADER_READ_ONLY_OPTIMAL).unwrap();
    log::trace!("Starting Model");
    let model = model_thread.join().unwrap();
    let model = Model::from_gltf(&mut ctx, model).unwrap();
    log::trace!("Model Loaded");


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

        ctx.create_acceleration_structure(
            AccelerationStructureTypeKHR::TOP_LEVEL,
            &[as_struct_geo],
            &[as_ranges],
            &[1],
        )
        .unwrap()
    };
    
    let renderer = create_render_recources(&mut ctx, &model, &tlas, sky_box.view).unwrap();

    let view = camera.planar_view_constants();

    let mut g_const = GConst {
        view,
        prevView: view,
        restirGI: ReSTIRGI_Parameters {
            bufferIndices: ReSTIRGI_BufferIndices { 
                secondarySurfaceReSTIRDIOutputBufferIndex: 0, //Reservoir to Resample
                temporalResamplingInputBufferIndex: 1,  //Resampeling input buffer
                temporalResamplingOutputBufferIndex: 0,
                spatialResamplingInputBufferIndex: 0, 
                spatialResamplingOutputBufferIndex: 2, //Resampeld reservoir output
                finalShadingInputBufferIndex: 2, //Final Shade input
                pad1: 0, 
                pad2: 0 
            },
            finalShadingParams: ReSTIRGI_FinalShadingParameters {
                enableFinalMIS: 0,
                enableFinalVisibility: 1,
                pad1: 0,
                pad2: 0,
            },
            reservoirBufferParams: CalculateReservoirBufferParameters(1920, 1080),
            spatialResamplingParams: ReSTIRGI_SpatialResamplingParameters {
                numSpatialSamples: 0,
                spatialBiasCorrectionMode: 3,
                spatialDepthThreshold: 0.2,
                spatialNormalThreshold: 0.1,
                spatialSamplingRadius: 0.0,

                pad1: 0,
                pad2: 0,
                pad3: 0,
            },
            temporalResamplingParams: ReSTIRGI_TemporalResamplingParameters { 
                boilingFilterStrength: 0.0, 
                depthThreshold: 0.2, 
                normalThreshold: 0.1, 
                enableBoilingFilter: 0, 
                enableFallbackSampling: 1, 
                enablePermutationSampling: 0, 
                maxHistoryLength: 20, 
                maxReservoirAge: 50, 
                temporalBiasCorrectionMode: 3, 
                uniformRandomNumber: rand::random(), 
                pad2: 0,
                pad3: 0, 
            }
        },
        runtimeParams: RTXDI_RuntimeParameters { 
            neighborOffsetMask: NEIGHBOR_OFFSET_COUNT - 1, 
            activeCheckerboardField: 0, 
            pad1: 0, 
            pad2: 0, 
        },
    };
    renderer.uniform_buffer.copy_data_to_aligned_buffer(std::slice::from_ref(&g_const), 16).unwrap();

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
                        camera = new_cam;
                        
                        let temp = g_const.view;
                        g_const.view = camera.planar_view_constants();
                        g_const.prevView = temp;   

                        renderer.uniform_buffer.copy_data_to_buffer(std::slice::from_ref(&g_const)).unwrap();
                        
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
                                    renderer.raytracing_pipeline.layout,
                                    vk::ShaderStageFlags::RAYGEN_KHR,
                                    0,
                                    &frame_c,
                                );
                                ctx.device.cmd_push_constants(
                                    *cmd,
                                    renderer.raytracing_pipeline.layout,
                                    vk::ShaderStageFlags::RAYGEN_KHR,
                                    size_of::<u32>() as u32,
                                    moved_c,
                                );

                                ctx.device.cmd_bind_descriptor_sets(
                                    *cmd,
                                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                                    renderer.raytracing_pipeline.layout,
                                    0,
                                    &[
                                        renderer.raytracing_pipeline.descriptor_set0,
                                        renderer.raytracing_pipeline.descriptor_set1[(frame % 2) as usize],
                                        renderer.raytracing_pipeline.descriptor_set2[1 - (frame % 2) as usize],
                                    ],
                                    &[],
                                );

                                ctx.device.cmd_bind_pipeline(
                                    *cmd,
                                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                                    renderer.raytracing_pipeline.handle,
                                );

                                let call_region = vk::StridedDeviceAddressRegionKHR::default();

                                ctx.ray_tracing.pipeline_fn.cmd_trace_rays(
                                    *cmd,
                                    &renderer.shader_binding_table.raygen_region,
                                    &renderer.shader_binding_table.miss_region,
                                    &renderer.shader_binding_table.hit_region,
                                    &call_region,
                                    window_size.width,
                                    window_size.height,
                                    1,
                                );

                                let memory_barriers = [vk::MemoryBarrier2 {
                                    src_access_mask: vk::AccessFlags2::empty(),
                                    src_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                                    dst_access_mask: vk::AccessFlags2::empty(),
                                    dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                                    ..Default::default()
                                }];

                                let dependency_info = vk::DependencyInfo::default()
                                    .memory_barriers(&memory_barriers);
                                ctx.device.cmd_pipeline_barrier2(*cmd, &dependency_info);

                                let handle_size = ctx.ray_tracing.pipeline_properties.shader_group_handle_size;
                                let handle_alignment = ctx
                                    .ray_tracing
                                    .pipeline_properties
                                    .shader_group_handle_alignment;
                                let aligned_handle_size = alinged_size(handle_size, handle_alignment);

                                let mut raygen_region2 = renderer.shader_binding_table.raygen_region;
                                raygen_region2.device_address += aligned_handle_size as u64;


                                ctx.ray_tracing.pipeline_fn.cmd_trace_rays(
                                    *cmd,
                                    &raygen_region2,
                                    &renderer.shader_binding_table.miss_region,
                                    &renderer.shader_binding_table.hit_region,
                                    &call_region,
                                    window_size.width,
                                    window_size.height,
                                    1,
                                );
                                let memory_barriers = [vk::MemoryBarrier2 {
                                    src_access_mask: vk::AccessFlags2::empty(),
                                    src_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                                    dst_access_mask: vk::AccessFlags2::empty(),
                                    dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                                    ..Default::default()
                                }];


                                let dependency_info = vk::DependencyInfo::default()
                                    .memory_barriers(&memory_barriers);
                                ctx.device.cmd_pipeline_barrier2(*cmd, &dependency_info);

                                let begin_info = vk::RenderPassBeginInfo::default()
                                    .render_pass(renderer.post_proccesing_pipeline.render_pass)
                                    .framebuffer(renderer.post_proccesing_pipeline.frame_buffers[i as usize])
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
                                    renderer.post_proccesing_pipeline.pipeline,
                                );

                                ctx.device.cmd_bind_descriptor_sets(
                                    *cmd,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    renderer.post_proccesing_pipeline.layout,
                                    0,
                                    &[
                                        renderer.post_proccesing_pipeline.static_descriptor,
                                        renderer.post_proccesing_pipeline.dynamic_descriptors[(frame % 2) as usize],
                                    ],
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
