#![feature(f16)]

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
mod shader_params;
mod ui;

use memoffset::offset_of;

mod model;
use gltf::Vertex;
use log::{debug, info};
use model::*;

use anyhow::Result;
use ash::vk::{
    self, AabbPositionsKHR, AccelerationStructureTypeKHR, AccessFlags, AccessFlags2,
    BufferUsageFlags, DependencyFlags, GeometryTypeKHR, ImageAspectFlags, ImageLayout,
    ImageUsageFlags, Packed24_8, PipelineBindPoint, PipelineStageFlags, PipelineStageFlags2,
    TransformMatrixKHR, KHR_SPIRV_1_4_NAME,
};
use gpu_allocator::MemoryLocation;

use ash::Device;

use glam::{uvec2, vec3, IVec2, Mat4, UVec2, Vec2, Vec3, Vec4};
use pipelines::{create_render_recources, CalculateReservoirBufferParameters};
use shader_params::{
    GConst, RTXDI_EnvironmentLightBufferParameters, RTXDI_LightBufferParameters,
    RTXDI_LightBufferRegion, RTXDI_RISBufferSegmentParameters, RTXDI_ReservoirBufferParameters,
    RTXDI_RuntimeParameters, ReSTIRDI_BufferIndices, ReSTIRDI_InitialSamplingParameters,
    ReSTIRDI_Parameters, ReSTIRDI_ShadingParameters, ReSTIRDI_SpatialResamplingParameters,
    ReSTIRDI_TemporalResamplingParameters, ReSTIRGI_BufferIndices, ReSTIRGI_FinalShadingParameters,
    ReSTIRGI_Parameters, ReSTIRGI_SpatialResamplingParameters,
    ReSTIRGI_TemporalResamplingParameters,
};
use simple_logger::SimpleLogger;
use std::default::{self, Default};
use std::ffi::{CStr, CString};
use std::os::raw::c_void;
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

#[repr(C)]
#[derive(Clone, Copy)]
struct MipLevelPushConstants {
    source_size: UVec2,
    num_dest_mip_levels: u32,
    source_mip_level: u32,
}

const NEIGHBOR_OFFSET_COUNT: u32 = 8192;
const RTXDI_RESERVOIR_BLOCK_SIZE: u32 = 16;
const WINDOW_SIZE: IVec2 = IVec2::new(1920, 1080);
const RESAMPLING: bool = false;
const MAX_LOCAL_LIGHTS: u32 = 1000;

fn main() {
    let model_thread = std::thread::spawn(|| gltf::load_file("./src/models/box.glb").unwrap());
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
            width: WINDOW_SIZE.x,
            height: WINDOW_SIZE.y,
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
    ctx.transition_image_layout(&sky_box.image, ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .unwrap();
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

    let reservoirBufferParams =
        CalculateReservoirBufferParameters(WINDOW_SIZE.x as u32, WINDOW_SIZE.y as u32);

    let renderer = create_render_recources(
        &mut ctx,
        &model,
        &tlas,
        sky_box.view,
        uvec2(sky_box.image.extent.width, sky_box.image.extent.height),
        reservoirBufferParams.reservoirArrayPitch as u64 * 2 * 32,
    )
    .unwrap();
    let view = camera.planar_view_constants();

    let mut g_const = GConst {
        view,
        prevView: view,
        restirGI: ReSTIRGI_Parameters {
            bufferIndices: ReSTIRGI_BufferIndices {
                secondarySurfaceReSTIRDIOutputBufferIndex: 0,
                temporalResamplingInputBufferIndex: 1,
                temporalResamplingOutputBufferIndex: 0,
                spatialResamplingInputBufferIndex: 0,
                spatialResamplingOutputBufferIndex: 1,
                finalShadingInputBufferIndex: 0,
                pad1: 0,
                pad2: 0,
            },
            finalShadingParams: ReSTIRGI_FinalShadingParameters {
                enableFinalMIS: 0,
                enableFinalVisibility: 0,
                pad1: 0,
                pad2: 0,
            },
            reservoirBufferParams,
            spatialResamplingParams: ReSTIRGI_SpatialResamplingParameters {
                spatialDepthThreshold: 0.1,
                spatialNormalThreshold: 0.6,

                numSpatialSamples: 8,
                spatialBiasCorrectionMode: 2,
                spatialSamplingRadius: 32.0,

                pad1: 0,
                pad2: 0,
                pad3: 0,
            },
            temporalResamplingParams: ReSTIRGI_TemporalResamplingParameters {
                boilingFilterStrength: 0.0,
                depthThreshold: 0.1,
                normalThreshold: 0.3,
                enableBoilingFilter: 0,
                enableFallbackSampling: 1,
                enablePermutationSampling: 0,
                maxHistoryLength: 20,
                maxReservoirAge: 50,
                temporalBiasCorrectionMode: 2,
                uniformRandomNumber: rand::random(),
                pad2: 0,
                pad3: 0,
            },
        },
        runtimeParams: RTXDI_RuntimeParameters {
            neighborOffsetMask: NEIGHBOR_OFFSET_COUNT - 1,
            activeCheckerboardField: 0,
            pad1: 0,
            pad2: 0,
        },
        environmentPdfTextureSize: uvec2(sky_box.image.extent.width, sky_box.image.extent.height),
        localLightPdfTextureSize: uvec2(
            renderer.local_lights_pdf.extent.width,
            renderer.local_lights_pdf.extent.height,
        ),
        environmentLightRISBufferSegmentParams: RTXDI_RISBufferSegmentParameters {
            bufferOffset: 1024 * 128,
            tileSize: 1024,
            tileCount: 128,
            pad1: 0,
        },
        localLightsRISBufferSegmentParams: RTXDI_RISBufferSegmentParameters {
            bufferOffset: 0,
            tileCount: 128,
            tileSize: 1024,
            pad1: 0,
        },
        restirDI: ReSTIRDI_Parameters {
            reservoirBufferParams,
            bufferIndices: ReSTIRDI_BufferIndices {
                shadingInputBufferIndex: 1,
                temporalResamplingInputBufferIndex: 1,
                temporalResamplingOutputBufferIndex: 0,
                spatialResamplingInputBufferIndex: 0,
                spatialResamplingOutputBufferIndex: 1,
                initialSamplingOutputBufferIndex: 1,
                pad1: 0,
                pad2: 0,
            },
            initialSamplingParams: ReSTIRDI_InitialSamplingParameters {
                numPrimaryLocalLightSamples: 1,
                numPrimaryInfiniteLightSamples: 1,
                numPrimaryEnvironmentSamples: 1,
                numPrimaryBrdfSamples: 1,
                brdfCutoff: 0.3,
                enableInitialVisibility: 0,
                environmentMapImportanceSampling: 1,
                localLightSamplingMode: 1,
            },
            temporalResamplingParams: ReSTIRDI_TemporalResamplingParameters {
                temporalDepthThreshold: 0.1,
                temporalNormalThreshold: 0.3,
                maxHistoryLength: 100,
                temporalBiasCorrection: 2,
                enablePermutationSampling: 0,
                permutationSamplingThreshold: 0.0,
                enableBoilingFilter: 0,
                boilingFilterStrength: 0.0,
                discardInvisibleSamples: 1,
                uniformRandomNumber: rand::random(),
                pad2: 0,
                pad3: 0,
            },
            spatialResamplingParams: ReSTIRDI_SpatialResamplingParameters {
                spatialDepthThreshold: 0.1,
                spatialNormalThreshold: 0.3,
                spatialBiasCorrection: 2,
                numSpatialSamples: 3,
                numDisocclusionBoostSamples: 2,
                spatialSamplingRadius: 32.0,
                neighborOffsetMask: NEIGHBOR_OFFSET_COUNT - 1,
                discountNaiveSamples: 0,
            },
            shadingParams: ReSTIRDI_ShadingParameters {
                enableFinalVisibility: 0,
                reuseFinalVisibility: 1,
                finalVisibilityMaxAge: 10,
                finalVisibilityMaxDistance: 1000.0,
                enableDenoiserInputPacking: 1,
                pad1: 0,
                pad2: 0,
                pad3: 0,
            },
        },
        lightBufferParams: RTXDI_LightBufferParameters {
            localLightBufferRegion: RTXDI_LightBufferRegion {
                firstLightIndex: 0,
                numLights: 100,
                pad1: 0,
                pad2: 0,
            },
            infiniteLightBufferRegion: RTXDI_LightBufferRegion {
                firstLightIndex: 100,
                numLights: 100,
                pad1: 0,
                pad2: 0,
            },
            environmentLightParams: RTXDI_EnvironmentLightBufferParameters {
                lightPresent: 1,
                lightIndex: 200,
                pad1: 0,
                pad2: 0,
            },
        },
    };

    let gi_reservoir_buffer_barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(AccessFlags::SHADER_WRITE | AccessFlags::SHADER_READ)
        .dst_access_mask(AccessFlags::SHADER_WRITE | AccessFlags::SHADER_READ)
        .src_queue_family_index(ctx.graphics_queue_family.index)
        .dst_queue_family_index(ctx.graphics_queue_family.index)
        .size(renderer.reservoirs.size)
        .buffer(renderer.reservoirs.inner)
        .offset(0);
    let regir_memory_barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(AccessFlags::SHADER_WRITE | AccessFlags::SHADER_READ)
        .dst_access_mask(AccessFlags::SHADER_WRITE | AccessFlags::SHADER_READ)
        .src_queue_family_index(ctx.graphics_queue_family.index)
        .dst_queue_family_index(ctx.graphics_queue_family.index)
        .size(renderer.ris_buffer.size)
        .buffer(renderer.ris_buffer.inner)
        .offset(0);

    let mut image_barriers = [vk::ImageMemoryBarrier::default()
        .src_access_mask(AccessFlags::SHADER_WRITE | AccessFlags::SHADER_READ)
        .dst_access_mask(AccessFlags::SHADER_WRITE | AccessFlags::SHADER_READ)
        .src_queue_family_index(ctx.graphics_queue_family.index)
        .dst_queue_family_index(ctx.graphics_queue_family.index)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: ImageAspectFlags::COLOR,
            base_array_layer: 0,
            base_mip_level: 0,
            layer_count: 1,
            level_count: 1,
        })
        .old_layout(ImageLayout::GENERAL)
        .new_layout(ImageLayout::GENERAL); 11];

    for i in 0..2 {
        image_barriers[0 + 5 * i] =
            image_barriers[0 + 5 * i].image(renderer.g_buffer[i].diffuse_albedo.image.inner);
        image_barriers[1 + 5 * i] =
            image_barriers[1 + 5 * i].image(renderer.g_buffer[i].normal.image.inner);
        image_barriers[2 + 5 * i] =
            image_barriers[2 + 5 * i].image(renderer.g_buffer[i].geo_normals.image.inner);
        image_barriers[3 + 5 * i] =
            image_barriers[3 + 5 * i].image(renderer.g_buffer[i].depth.image.inner);
        image_barriers[4 + 5 * i] =
            image_barriers[4 + 5 * i].image(renderer.g_buffer[i].motion_vectors.image.inner);
        image_barriers[5 + 5 * i] =
            image_barriers[5 + 5 * i].image(renderer.g_buffer[i].specular_rough.image.inner);
    }

    renderer
        .uniform_buffer
        .copy_data_to_aligned_buffer(std::slice::from_ref(&g_const), 16)
        .unwrap();

    let mut moved = false;
    let mut frame: u64 = 0;
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
                        frame += 1;
                        let new_cam = camera.update(&controles, frame_time);
                        moved = new_cam != camera;
                        camera = new_cam;

                        g_const.prevView = g_const.view;
                        g_const.view = camera.planar_view_constants();

                        renderer
                            .uniform_buffer
                            .copy_data_to_aligned_buffer(std::slice::from_ref(&g_const), 16)
                            .unwrap();

                        ctx.render(|ctx, i| {
                            let cmd = &ctx.cmd_buffs[i as usize];
                            // println!("{}", renderer.environment_pdf.mip_levels);

                            unsafe {
                                if frame == 1 {
                                    renderer
                                        .environment_presampeling_pipeline
                                        .execute_presampeling(&ctx, cmd, &[regir_memory_barrier]);
                                    renderer.environment_mip_pipeline.execute_mip_generation(
                                        &ctx,
                                        cmd,
                                        renderer.environment_pdf.extent.width,
                                        renderer.environment_pdf.extent.height,
                                        renderer.environment_pdf.mip_levels,
                                    );
                                }

                                let frame_c = std::slice::from_raw_parts(
                                    &frame as *const u64 as *const u8,
                                    size_of::<u32>(),
                                );

                                renderer
                                    .local_light_presampeling_pipeline
                                    .execute_presampeling(ctx, cmd, &[regir_memory_barrier]);

                                renderer.local_lights_mip_pipeline.execute_mip_generation(
                                    ctx,
                                    cmd,
                                    renderer.local_lights_pdf.extent.width,
                                    renderer.local_lights_pdf.extent.height,
                                    renderer.local_lights_pdf.mip_levels,
                                );

                                renderer
                                    .raytracing_pipeline
                                    .execute(&ctx, cmd, frame, frame_c);

                                if RESAMPLING {
                                    ctx.device.cmd_pipeline_barrier(
                                        *cmd,
                                        PipelineStageFlags::COMPUTE_SHADER,
                                        PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                                        DependencyFlags::BY_REGION,
                                        &[],
                                        &[regir_memory_barrier, gi_reservoir_buffer_barrier],
                                        &image_barriers,
                                    );

                                    renderer
                                        .temporal_reuse_pipeline
                                        .execute(&ctx, cmd, frame, frame_c);

                                    ctx.device.cmd_pipeline_barrier(
                                        *cmd,
                                        PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                                        PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                                        DependencyFlags::BY_REGION,
                                        &[],
                                        &[gi_reservoir_buffer_barrier, regir_memory_barrier],
                                        &[],
                                    );

                                    renderer
                                        .spatial_reuse_pipeline
                                        .execute(&ctx, cmd, frame, frame_c);
                                }

                                ctx.device.cmd_pipeline_barrier(
                                    *cmd,
                                    PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                                    PipelineStageFlags::FRAGMENT_SHADER,
                                    DependencyFlags::BY_REGION,
                                    &[],
                                    &[gi_reservoir_buffer_barrier],
                                    &image_barriers,
                                );

                                renderer
                                    .post_proccesing_pipeline
                                    .execute(ctx, cmd, frame_c, i, frame);
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
