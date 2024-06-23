use std::{ffi::CStr, mem::size_of};

use ash::{
    ext::buffer_device_address,
    khr::{
        get_memory_requirements2, shader_float_controls, shader_non_semantic_info, swapchain,
        workgroup_memory_explicit_layout,
    },
    vk::{self, DescriptorType, ShaderStageFlags, KHR_SPIRV_1_4_NAME},
};
use bevy::{prelude::*, winit::WinitWindows};
use gpu_allocator::MemoryLocation;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

use crate::{
    oct_tree::{*},
    pipelines::{
        create_fullscreen_quad_pipeline, create_post_proccesing_pipelien, create_storage_images,
        PostProccesingPipeline, RayTracingPipeline,
    },
    Buffer, Camera, CameraUniformData, DeviceFeatures, Image, ImageAndView, Renderer,
    WriteDescriptorSet, WriteDescriptorSetKind, WINDOW_SIZE,
};

pub const DEVICE_EXTENSIONS: [&'static CStr; 10] = [
    swapchain::NAME,
    ash::ext::descriptor_indexing::NAME,
    ash::ext::scalar_block_layout::NAME,
    get_memory_requirements2::NAME,
    ash::ext::buffer_device_address::NAME,
    KHR_SPIRV_1_4_NAME,
    shader_float_controls::NAME,
    buffer_device_address::NAME,
    shader_non_semantic_info::NAME,
    workgroup_memory_explicit_layout::NAME,
];

pub const FULL_SCREEN_SCISSOR: vk::Rect2D = vk::Rect2D {
    extent: vk::Extent2D {
        width: WINDOW_SIZE.0,
        height: WINDOW_SIZE.1,
    },
    offset: vk::Offset2D { x: 0, y: 0 },
};

pub const FULL_SCREEN_VIEW_PORT: vk::Viewport = vk::Viewport {
    x: 0.0,
    y: 0.0,
    width: WINDOW_SIZE.0 as f32,
    height: WINDOW_SIZE.1 as f32,
    min_depth: 0.0,
    max_depth: 1.0,
};

fn render(
    mut renderer: NonSendMut<Renderer>,
    mut uniform_data: ResMut<CameraUniformData>,
    data: Res<FrameData>,
    raytracing_pipeline: Res<RayTracingPipeline>,
    post_proccesing_pipeline: Res<PostProccesingPipeline>,
    camera: Res<Camera>,
    time: Res<Time>
) {
    log::info!("{:?}", time.delta());
    uniform_data.proj_inverse = camera.projection_matrix().inverse();
    uniform_data.view_inverse = camera.view_matrix().inverse();
    uniform_data.input.z = WINDOW_SIZE.0 as f32;
    uniform_data.input.w = WINDOW_SIZE.1 as f32;
    data.uniform_buffer
        .copy_data_to_buffer(std::slice::from_ref(&(*uniform_data)))
        .unwrap();

    renderer
        .render(|renderer, i| {
            let cmd = &renderer.cmd_buffs[i as usize];

            unsafe {
                // let frame_c = std::slice::from_raw_parts(
                //     &frame as *const u32 as *const u8,
                //     size_of::<u32>(),
                // );
                // let moved_c = &[if moved == true { 1 as u8 } else { 0 as u8 }, 0, 0, 0];

                let begin_info = vk::RenderPassBeginInfo::default()
                    .clear_values(&[
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        },
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    ])
                    .render_pass(raytracing_pipeline.render_pass)
                    .framebuffer(data.frame_buffers[i as usize])
                    .render_area(vk::Rect2D {
                        extent: vk::Extent2D {
                            width: WINDOW_SIZE.0,
                            height: WINDOW_SIZE.1,
                        },
                        offset: vk::Offset2D { x: 0, y: 0 },
                    });

                renderer
                    .device
                    .cmd_set_viewport(*cmd, 0, &[FULL_SCREEN_VIEW_PORT]);
                renderer
                    .device
                    .cmd_set_scissor(*cmd, 0, &[FULL_SCREEN_SCISSOR]);

                renderer.device.cmd_begin_render_pass(
                    *cmd,
                    &begin_info,
                    vk::SubpassContents::INLINE,
                );

                renderer.device.cmd_bind_pipeline(
                    *cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    raytracing_pipeline.pipeline,
                );

                renderer.device.cmd_bind_descriptor_sets(
                    *cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    raytracing_pipeline.layout,
                    0,
                    &[raytracing_pipeline.descriptor],
                    &[],
                );
                renderer.device.cmd_draw(*cmd, 6, 1, 0, 0);

                renderer.device.cmd_end_render_pass(*cmd);

                let begin_info = vk::RenderPassBeginInfo::default()
                    .render_pass(post_proccesing_pipeline.render_pass)
                    .framebuffer(data.present_frame_buffers[i as usize])
                    .render_area(vk::Rect2D {
                        extent: vk::Extent2D {
                            width: WINDOW_SIZE.0,
                            height: WINDOW_SIZE.1,
                        },
                        offset: vk::Offset2D { x: 0, y: 0 },
                    })
                    .clear_values(&[vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    }]);

                renderer
                    .device
                    .cmd_set_viewport(*cmd, 0, &[FULL_SCREEN_VIEW_PORT]);
                renderer
                    .device
                    .cmd_set_scissor(*cmd, 0, &[FULL_SCREEN_SCISSOR]);

                renderer.device.cmd_begin_render_pass(
                    *cmd,
                    &begin_info,
                    vk::SubpassContents::INLINE,
                );

                renderer.device.cmd_bind_pipeline(
                    *cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    post_proccesing_pipeline.pipeline,
                );

                renderer.device.cmd_bind_descriptor_sets(
                    *cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    post_proccesing_pipeline.layout,
                    0,
                    &[post_proccesing_pipeline.descriptors[i as usize]],
                    &[],
                );
                renderer.device.cmd_draw(*cmd, 6, 1, 0, 0);

                renderer.device.cmd_end_render_pass(*cmd);
            }
        })
        .unwrap();
}

#[derive(Resource)]
struct FrameData {
    pub frame_buffers: Vec<vk::Framebuffer>,
    pub color_buffers: Vec<ImageAndView>,
    pub depth_buffers: Vec<ImageAndView>,
    pub present_frame_buffers: Vec<vk::Framebuffer>,
    pub uniform_buffer: Buffer,
}


fn init(world: &mut World) {
    let device_features = world.get_resource::<DeviceFeatures>().unwrap();
    let windows = world.get_non_send_resource::<WinitWindows>().unwrap();
    let window = windows.windows.values().into_iter().last().unwrap();
    let mut renderer = Renderer::new(
        window.raw_window_handle().unwrap(),
        window.raw_display_handle().unwrap(),
        &device_features,
        WINDOW_SIZE.0,
        WINDOW_SIZE.1,
    )
    .unwrap();

    let image_thread = std::thread::spawn(|| image::open("./src/models/skybox.png").unwrap());
    let default_sampler: vk::SamplerCreateInfo = vk::SamplerCreateInfo {
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

    let uniform_buffer = renderer
        .create_buffer(
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            size_of::<CameraUniformData>() as u64,
            Some("Uniform Buffer"),
        )
        .unwrap();

    let oct_tree_data = Octant::load("./models/monu2.vox").unwrap().build();

    let oct_tree_buffer = renderer
        .create_gpu_only_buffer_from_data(
            vk::BufferUsageFlags::STORAGE_BUFFER,
            oct_tree_data.as_slice(),
            Some("OctTreeData"),
        )
        .unwrap();

    let image = image_thread.join().unwrap();
    let sky_box = Image::new_from_data(&mut renderer, image, vk::Format::R8G8B8A8_SRGB).unwrap();
    let sky_box_sampler =
        unsafe { renderer.device.create_sampler(&default_sampler, None) }.unwrap();

    let raytracing_pipeline = create_fullscreen_quad_pipeline(
        &mut renderer,
        &uniform_buffer,
        &oct_tree_buffer,
        &[vk::DescriptorSetLayoutBinding {
            binding: 2,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            stage_flags: ShaderStageFlags::FRAGMENT,
            ..Default::default()
        }],
        &[vk::DescriptorPoolSize {
            descriptor_count: 1,
            ty: DescriptorType::COMBINED_IMAGE_SAMPLER,
        }],
        &[WriteDescriptorSet {
            binding: 2,
            kind: WriteDescriptorSetKind::CombinedImageSampler {
                view: sky_box.view,
                sampler: sky_box_sampler,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        }],
    )
    .unwrap();

    let (frame_buffers, color_buffers, depth_buffers) =
        create_storage_images(&mut renderer, &raytracing_pipeline).unwrap();

    let post_proccesing_pipeline =
        create_post_proccesing_pipelien(&mut renderer, &color_buffers).unwrap();

    let present_frame_buffers = {
        (0..renderer.swapchain.images.len())
            .map(|i| unsafe {
                renderer
                    .device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .attachments(&[renderer.swapchain.images[i].view])
                            .attachment_count(1)
                            .height(WINDOW_SIZE.1)
                            .width(WINDOW_SIZE.0)
                            .layers(1)
                            .render_pass(post_proccesing_pipeline.render_pass),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<vk::Framebuffer>>()
    };
    world.insert_non_send_resource(renderer);
    world.insert_resource(FrameData {
        color_buffers,
        depth_buffers,
        frame_buffers,
        present_frame_buffers,
        uniform_buffer,
    });
    world.insert_resource(raytracing_pipeline);
    world.insert_resource(post_proccesing_pipeline);
}

pub fn RenderPlugin(app: &mut App) {
    app.add_systems(Startup, init)
    .add_systems(Update, render);
}
