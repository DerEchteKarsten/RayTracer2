mod context;
use ash::khr::{
    acceleration_structure, deferred_host_operations, get_memory_requirements2,
    ray_tracing_pipeline, shader_float_controls, shader_non_semantic_info, swapchain,
    synchronization2,
};
use context::*;
mod camera;
use camera::*;
mod gltf;
mod light_passes;
mod mip_pass;
mod postprocess;
mod prepare_lights;
mod render_resources;
mod shader_params;

use imgui::{Condition, TreeNodeFlags};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use light_passes::{calculate_reservoir_buffer_parameters, LightPasses};

mod model;
use log;
use mip_pass::GenerateMipsPass;
use model::*;

use ash::vk::{self, AccelerationStructureTypeKHR, ImageLayout, Offset2D, KHR_SPIRV_1_4_NAME};

use glam::{uvec2, vec3, IVec2, Mat4, UVec4};
use postprocess::PostProcessPass;
use prepare_lights::PrepareLightsTasks;
use render_resources::RenderResources;
use shader_params::{
    GConst, RTXDI_EnvironmentLightBufferParameters, RTXDI_LightBufferParameters,
    RTXDI_LightBufferRegion, RTXDI_RISBufferSegmentParameters, RTXDI_RuntimeParameters,
    ReSTIRDI_BufferIndices, ReSTIRDI_InitialSamplingParameters, ReSTIRDI_Parameters,
    ReSTIRDI_ShadingParameters, ReSTIRDI_SpatialResamplingParameters,
    ReSTIRDI_TemporalResamplingParameters, ReSTIRGI_BufferIndices, ReSTIRGI_FinalShadingParameters,
    ReSTIRGI_Parameters, ReSTIRGI_SpatialResamplingParameters,
    ReSTIRGI_TemporalResamplingParameters,
};
use simple_logger::SimpleLogger;
use std::default::Default;
use std::ffi::CStr;
use std::thread::sleep;
use std::time::{Duration, Instant};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowAttributes;

use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

const NEIGHBOR_OFFSET_COUNT: u32 = 8192;
const RTXDI_RESERVOIR_BLOCK_SIZE: u32 = 16;
const WINDOW_SIZE: IVec2 = IVec2::new(1920, 1080);
const RESAMPLING: bool = false;

fn main() {
    let model_thread = std::thread::spawn(|| gltf::load_file("./src/models/box.glb").unwrap());
    let image_thread = std::thread::spawn(|| image::open("./src/models/skybox2.exr").unwrap());

    SimpleLogger::new().init().unwrap();

    let device_extensions: [&CStr; 11] = [
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
        synchronization2::NAME,
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
    #[allow(deprecated)]
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

    let mut sampler_info = vk::SamplerCreateInfo {
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        address_mode_u: vk::SamplerAddressMode::CLAMP_TO_BORDER,
        address_mode_v: vk::SamplerAddressMode::CLAMP_TO_BORDER,
        address_mode_w: vk::SamplerAddressMode::CLAMP_TO_BORDER,
        max_anisotropy: 1.0,
        border_color: vk::BorderColor::FLOAT_OPAQUE_BLACK,
        compare_op: vk::CompareOp::NEVER,
        unnormalized_coordinates: 1,
        ..Default::default()
    };

    let skybox_pdf_sampler = unsafe { ctx.device.create_sampler(&sampler_info, None).unwrap() };
    sampler_info.unnormalized_coordinates = 0;
    let skybox_sampler = unsafe { ctx.device.create_sampler(&sampler_info, None).unwrap() };
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

    let reservoir_buffer_params =
        calculate_reservoir_buffer_parameters(WINDOW_SIZE.x as u32, WINDOW_SIZE.y as u32);

    let resources = RenderResources::new(
        &mut ctx,
        &model,
        uvec2(sky_box.image.extent.width, sky_box.image.extent.height),
    );

    let light_passes = LightPasses::new(
        &mut ctx,
        &model,
        &tlas,
        &resources,
        &sky_box.view,
        &skybox_sampler,
    );
    let post_process = PostProcessPass::new(
        &mut ctx,
        &resources.g_buffers,
        &resources,
        &light_passes.uniform_buffer,
        &sky_box.view,
        &skybox_sampler,
    )
    .unwrap();

    let mip_environment = GenerateMipsPass::new(
        &mut ctx,
        &resources.environment_pdf_texture_mips,
        resources.environment_pdf_texture.image.mip_levels,
        Some((&sky_box.view, &skybox_pdf_sampler)),
    )
    .unwrap();

    let mip_lights = GenerateMipsPass::new(
        &mut ctx,
        &resources.local_light_pdf_texture_mips,
        resources.local_light_pdf_texture.image.mip_levels,
        None,
    )
    .unwrap();

    let view = camera.planar_view_constants();
    let mut g_const = GConst {
        view,
        prev_view: view,
        restir_gi: ReSTIRGI_Parameters {
            buffer_indices: ReSTIRGI_BufferIndices {
                secondary_surface_re_stirdioutput_buffer_index: 0,
                temporal_resampling_input_buffer_index: 1,
                temporal_resampling_output_buffer_index: 0,
                spatial_resampling_input_buffer_index: 0,
                spatial_resampling_output_buffer_index: 1,
                final_shading_input_buffer_index: if RESAMPLING { 1 } else { 0 },
                pad1: 0,
                pad2: 0,
            },
            final_shading_params: ReSTIRGI_FinalShadingParameters {
                enable_final_mis: 0,
                enable_final_visibility: 0,
                pad1: 0,
                pad2: 0,
            },
            reservoir_buffer_params,
            spatial_resampling_params: ReSTIRGI_SpatialResamplingParameters {
                spatial_depth_threshold: 0.1,
                spatial_normal_threshold: 0.3,

                num_spatial_samples: 1,
                spatial_bias_correction_mode: 2,
                spatial_sampling_radius: 3.0,

                pad1: 0,
                pad2: 0,
                pad3: 0,
            },
            temporal_resampling_params: ReSTIRGI_TemporalResamplingParameters {
                boiling_filter_strength: 0.0,
                depth_threshold: 0.1,
                normal_threshold: 0.3,
                enable_boiling_filter: 0,
                enable_fallback_sampling: 1,
                enable_permutation_sampling: 0,
                max_history_length: 20,
                max_reservoir_age: 50,
                temporal_bias_correction_mode: 2,
                uniform_random_number: rand::random(),
                pad2: 0,
                pad3: 0,
            },
        },
        runtime_params: RTXDI_RuntimeParameters {
            neighbor_offset_mask: NEIGHBOR_OFFSET_COUNT - 1,
            active_checkerboard_field: 0,
            pad1: 0,
            pad2: 0,
        },
        environment_pdf_texture_size: uvec2(
            resources.environment_pdf_texture.image.extent.width,
            resources.environment_pdf_texture.image.extent.height,
        ),
        local_light_pdf_texture_size: uvec2(
            resources.local_light_pdf_texture.image.extent.width,
            resources.local_light_pdf_texture.image.extent.height,
        ),
        environment_light_risbuffer_segment_params: RTXDI_RISBufferSegmentParameters {
            buffer_offset: 1024 * 128,
            tile_size: 1024,
            tile_count: 128,
            pad1: 0,
        },
        local_lights_risbuffer_segment_params: RTXDI_RISBufferSegmentParameters {
            buffer_offset: 0,
            tile_count: 128,
            tile_size: 1024,
            pad1: 0,
        },
        restir_di: ReSTIRDI_Parameters {
            reservoir_buffer_params,
            buffer_indices: ReSTIRDI_BufferIndices {
                shading_input_buffer_index: 1,
                temporal_resampling_input_buffer_index: 1,
                temporal_resampling_output_buffer_index: 0,
                spatial_resampling_input_buffer_index: 0,
                spatial_resampling_output_buffer_index: 1,
                initial_sampling_output_buffer_index: 1,
                pad1: 0,
                pad2: 0,
            },
            initial_sampling_params: ReSTIRDI_InitialSamplingParameters {
                num_primary_local_light_samples: 0,
                num_primary_infinite_light_samples: 0,
                num_primary_environment_samples: 0,
                num_primary_brdf_samples: 1,
                brdf_cutoff: 0.0,
                enable_initial_visibility: 1,
                environment_map_importance_sampling: 1,
                local_light_sampling_mode: 2,
            },
            temporal_resampling_params: ReSTIRDI_TemporalResamplingParameters {
                temporal_depth_threshold: 0.1,
                temporal_normal_threshold: 0.3,
                max_history_length: 100,
                temporal_bias_correction: 2,
                enable_permutation_sampling: 0,
                permutation_sampling_threshold: 0.0,
                enable_boiling_filter: 0,
                boiling_filter_strength: 0.0,
                discard_invisible_samples: 1,
                uniform_random_number: rand::random(),
                pad2: 0,
                pad3: 0,
            },
            spatial_resampling_params: ReSTIRDI_SpatialResamplingParameters {
                spatial_depth_threshold: 0.1,
                spatial_normal_threshold: 0.3,
                spatial_bias_correction: 2,
                num_spatial_samples: 3,
                num_disocclusion_boost_samples: 2,
                spatial_sampling_radius: 32.0,
                neighbor_offset_mask: NEIGHBOR_OFFSET_COUNT - 1,
                discount_naive_samples: 0,
            },
            shading_params: ReSTIRDI_ShadingParameters {
                enable_final_visibility: 0,
                reuse_final_visibility: 1,
                final_visibility_max_age: 10,
                final_visibility_max_distance: 1000.0,
                enable_denoiser_input_packing: 1,
                pad1: 0,
                pad2: 0,
                pad3: 0,
            },
        },
        light_buffer_params: RTXDI_LightBufferParameters {
            local_light_buffer_region: RTXDI_LightBufferRegion {
                first_light_index: 0,
                num_lights: model.lights,
                pad1: 0,
                pad2: 0,
            },
            infinite_light_buffer_region: RTXDI_LightBufferRegion {
                first_light_index: model.lights,
                num_lights: 0,
                pad1: 0,
                pad2: 0,
            },
            environment_light_params: RTXDI_EnvironmentLightBufferParameters {
                light_present: 1,
                light_index: model.lights,
                pad1: 0,
                pad2: 0,
            },
        },
        enable_accumulation: 0,
        enable_brdf_additive_blend: 0,
        enable_brdf_indirect: 0,
        enable_restir_di: 0,
        enable_restir_gi: 0,
        frame: 0,
        refrence_mode: 0,
        textures: 0,
        blend_factor: 0.1,
        enable_spatial_resampling: 1,
        enable_temporal_resampling: 1,
        environment: 1,
    };

    let prepare_lights = PrepareLightsTasks::new(
        &mut ctx,
        &model,
        &resources,
        (&sky_box.view, &skybox_sampler),
    )
    .unwrap();

    light_passes
        .uniform_buffer
        .copy_data_to_aligned_buffer(std::slice::from_ref(&g_const), 16)
        .unwrap();

    let mut frame: u64 = 0;
    let mut last_time = Instant::now();
    let mut frame_time = Duration::new(0, 0);
    let mut skybox_dirty = true;

    let mut imgui = imgui::Context::create();
    imgui.style_mut().anti_aliased_lines = true;
    imgui.style_mut().anti_aliased_fill = true;

    let attachments = [vk::AttachmentDescription::default()
        .initial_layout(ImageLayout::PRESENT_SRC_KHR)
        .final_layout(ImageLayout::PRESENT_SRC_KHR)
        .format(ctx.swapchain.format)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .samples(vk::SampleCountFlags::TYPE_1)];
    let attachment = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(ImageLayout::GENERAL)];
    let subpasses = [vk::SubpassDescription::default()
        .color_attachments(&attachment)
        .input_attachments(&[])];

    let render_pass_create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);

    let imgui_pass = unsafe {
        ctx.device
            .create_render_pass(&render_pass_create_info, None)
            .unwrap()
    };

    let mut imgui_frame_buffers = vec![];
    for image in ctx.swapchain.images.iter() {
        let attachment = [image.view];
        let frame_buffer_info = vk::FramebufferCreateInfo::default()
            .attachments(&attachment)
            .height(WINDOW_SIZE.y as u32)
            .width(WINDOW_SIZE.x as u32)
            .layers(1)
            .render_pass(imgui_pass);
        imgui_frame_buffers.push(unsafe {
            ctx.device
                .create_framebuffer(&frame_buffer_info, None)
                .unwrap()
        })
    }

    let mut imgui_renderer = imgui_rs_vulkan_renderer::Renderer::with_default_allocator(
        &ctx.instance,
        ctx.physical_device.handel,
        ctx.device.clone(),
        ctx.graphics_queue,
        ctx.command_pool,
        imgui_pass,
        &mut imgui,
        None,
    )
    .unwrap();

    imgui.fonts().build_alpha8_texture();

    let mut platform = WinitPlatform::new(&mut imgui);
    platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Default);
    
    let mut override_blend_factor = false;
    let mut accumulation_duration = Instant::now();
    #[allow(clippy::collapsible_match, clippy::single_match, deprecated)]
    event_loop
        .run(move |event, control_flow| {
            control_flow.set_control_flow(ControlFlow::Poll);
            controles = controles.handle_event(&event, &window);
            platform.handle_event(imgui.io_mut(), &window, &event);

            match event {
                Event::NewEvents(_) => {
                    controles = controles.reset();
                    let now = Instant::now();
                    frame_time = now - last_time;
                    last_time = Instant::now();
                    imgui.io_mut().update_delta_time(frame_time);
                }
                Event::AboutToWait => {
                    platform
                        .prepare_frame(imgui.io_mut(), &window)
                        .expect("Failed to prepare frame");
                    window.request_redraw();
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
                        // sleep(Duration::from_millis(30));

                        let image_index = ctx.await_next_frame().unwrap();
                        println!("next frame --------------------------------");

                        frame += 1;
                        let ui = imgui.frame();

                        ui.window("GConstEditor")
                            .size([300.0, WINDOW_SIZE.y as f32], Condition::FirstUseEver)
                            .position([0.0, 0.0], Condition::FirstUseEver)
                            .build(|| {
                                ui.text(format!(
                                    "FPS: {:.0}, {:?}",
                                    1000.0 / frame_time.as_millis() as f64,
                                    frame_time
                                ));
                                ui.text(format!(
                                    "Frame: {}, Swapchain Image: {}", frame, image_index
                                ));

                                ui.separator();
                                ui.checkbox_flags("Refrence Mode", &mut g_const.refrence_mode, 0x1);
                                ui.checkbox_flags("Accumulation", &mut g_const.enable_accumulation, 0x1);
                                ui.checkbox_flags("Textures", &mut g_const.textures, 0x1);
                                ui.checkbox_flags("RestirDI", &mut g_const.enable_restir_di, 0x1);
                                ui.checkbox_flags("RestirGI", &mut g_const.enable_restir_gi, 0x1);
                                
                                ui.separator();
                                ui.checkbox_flags("Environment", &mut g_const.environment, 0x1);

                                ui.separator();
                                ui.checkbox_flags("Spatial Resampling", &mut g_const.enable_spatial_resampling, 0x1);
                                ui.checkbox_flags("Temporal Resampling", &mut g_const.enable_temporal_resampling, 0x1);

                                ui.separator();
                                ui.checkbox("Override Blend Factor", &mut override_blend_factor);
                                ui.input_float("Blend Factor", &mut g_const.blend_factor).build();
                                
                                ui.separator();
                                ui.checkbox_flags("Brdf Additive Blend", &mut g_const.enable_brdf_additive_blend, 0x1);
                                ui.checkbox_flags("Brdf Indirect", &mut g_const.enable_brdf_indirect, 0x1);

                                ui.separator();
                                ui.checkbox_flags("Checker Board Field", &mut g_const.runtime_params.active_checkerboard_field, 0x1);
                                ui.input_int("Neighbor Offset Mask", unsafe { std::mem::transmute(&mut g_const.runtime_params.neighbor_offset_mask) }).build();

                                

                            });

                            if g_const.enable_accumulation == 0 {
                                accumulation_duration = Instant::now();
                            }

                            if !override_blend_factor {
                                g_const.blend_factor = 100.0 / Instant::now().duration_since(accumulation_duration).as_millis() as f32;
                            }

                        platform.prepare_render(ui, &window);
                        let draw_data = imgui.render();
                        

                        let new_cam = camera.update(&controles, frame_time);
                        camera = new_cam;

                        g_const.prev_view = g_const.view;
                        g_const.view = camera.planar_view_constants();
                        g_const.frame = frame as u32;

                        println!(
                            "{}: {:?}------------------------------------------",
                            frame, frame_time
                        );

                        if frame_time.as_millis() > 16 {
                            log::error!("Over Frame Budget!!!!");
                            // panic!()
                        }
                        if let Err(e) = light_passes.update_uniform(&g_const) {
                            log::error!("{}", e);
                        }

                        ctx.render(image_index, |ctx| {
                            let cmd = &ctx.cmd_buffs[image_index as usize];
                            if frame == 1 {
                                prepare_lights.execute(
                                    ctx,
                                    cmd,
                                    &resources.task_buffer,
                                    (
                                        &resources.geometry_instance_to_light_buffer,
                                        &resources.geometry_instance_to_light_buffer_staging,
                                    ),
                                    &model,
                                );
                                let skybox_changed = light_passes.execute_presampeling(
                                    ctx,
                                    cmd,
                                    frame,
                                    skybox_dirty,
                                );
                                if skybox_dirty {
                                    mip_environment.execute_mip_generation(
                                        ctx,
                                        cmd,
                                        resources.environment_pdf_texture.image.extent.width,
                                        resources.environment_pdf_texture.image.extent.height,
                                        resources.environment_pdf_texture.image.mip_levels,
                                    );
                                    skybox_dirty = skybox_changed;
                                }
                                mip_lights.execute_mip_generation(
                                    ctx,
                                    cmd,
                                    resources.local_light_pdf_texture.image.extent.width,
                                    resources.local_light_pdf_texture.image.extent.height,
                                    resources.local_light_pdf_texture.image.mip_levels,
                                );
                            }

                            light_passes.execute(ctx, cmd, frame, &g_const);
                            post_process.execute(ctx, cmd, frame, image_index, &resources);
                            unsafe {
                                let begin_info = vk::RenderPassBeginInfo::default()
                                    .framebuffer(imgui_frame_buffers[image_index as usize])
                                    .render_pass(imgui_pass)
                                    .render_area(
                                        vk::Rect2D::default()
                                            .extent(vk::Extent2D {
                                                width: WINDOW_SIZE.x as u32,
                                                height: WINDOW_SIZE.y as u32,
                                            })
                                            .offset(Offset2D { x: 0, y: 0 }),
                                    );
                                ctx.device.cmd_begin_render_pass(
                                    *cmd,
                                    &begin_info,
                                    vk::SubpassContents::INLINE,
                                );
                                imgui_renderer.cmd_draw(*cmd, draw_data).unwrap();
                                ctx.device.cmd_end_render_pass(*cmd);
                            }
                        })
                        .unwrap();
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
