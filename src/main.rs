mod renderer;
use ash::khr::{
    buffer_device_address, get_memory_requirements2, shader_float_controls,
    shader_non_semantic_info, swapchain, workgroup_memory_explicit_layout,
};
use bevy_a11y::{AccessibilityPlugin, AccessibilityRequested};
use bevy_app::App;
use bevy_input::InputPlugin;
use bevy_time::{Time, TimePlugin};
use bevy_winit::{WinitPlugin, WinitSettings, WinitWindows};
use pipelines::{
    create_fullscreen_quad_pipeline, create_post_proccesing_pipelien, create_storage_images,
    PostProccesingPipeline, RayTracingPipeline,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use renderer::*;
mod camera;
use camera::*;
mod pipelines;

use anyhow::Result;
use ash::ext;
use ash::vk::{self, DescriptorType, ShaderStageFlags, SwapchainKHR, KHR_SPIRV_1_4_NAME};
use dot_vox::Voxel;
use gpu_allocator::MemoryLocation;

use glam::{vec3, BVec3, UVec3, Vec3, Vec4};
use std::default::Default;
use std::ffi::{CStr, CString};
use std::rc::Rc;
use std::time::{Duration, Instant};

use std::mem::size_of;

const DEVICE_EXTENSIONS: [&'static CStr; 10] = [
    swapchain::NAME,
    ext::descriptor_indexing::NAME,
    ext::scalar_block_layout::NAME,
    get_memory_requirements2::NAME,
    ext::buffer_device_address::NAME,
    KHR_SPIRV_1_4_NAME,
    shader_float_controls::NAME,
    buffer_device_address::NAME,
    shader_non_semantic_info::NAME,
    workgroup_memory_explicit_layout::NAME,
];

const APP_NAME: &'static str = "Test";
const WINDOW_SIZE: (u32, u32) = (2560, 1400);

const FULL_SCREEN_SCISSOR: vk::Rect2D = vk::Rect2D {
    extent: vk::Extent2D {
        width: WINDOW_SIZE.0,
        height: WINDOW_SIZE.1,
    },
    offset: vk::Offset2D { x: 0, y: 0 },
};

const FULL_SCREEN_VIEW_PORT: vk::Viewport = vk::Viewport {
    x: 0.0,
    y: 0.0,
    width: WINDOW_SIZE.0 as f32,
    height: WINDOW_SIZE.1 as f32,
    min_depth: 0.0,
    max_depth: 1.0,
};

use bevy_app::prelude::*;
use bevy_core::prelude::*;
use bevy_ecs::prelude::*;
use bevy_window::{prelude::*, RequestRedraw, WindowFocused, WindowResolution};

fn main() {
    App::new()
        .insert_resource(AccessibilityRequested::default())
        .insert_resource(DeviceFeatures {
            ray_tracing_pipeline: false,
            acceleration_structure: false,
            runtime_descriptor_array: true,
            buffer_device_address: true,
            dynamic_rendering: true,
            synchronization2: true,
        })
        .insert_resource(Camera::new(
            vec3(1.5, 1.25, -5.0),
            vec3(0.0, 0.0, 1.0),
            40.0,
            WINDOW_SIZE.0 as f32 / WINDOW_SIZE.1 as f32,
            0.1,
            1000.0,
        ))
        .init_resource::<Controls>()
        .init_resource::<CameraUniformData>()
        .add_plugins((
            AccessibilityPlugin,
            InputPlugin,
            WindowPlugin {
                close_when_requested: true,
                exit_condition: bevy_window::ExitCondition::OnPrimaryClosed,
                primary_window: Some(Window {
                    resolution: WindowResolution::new(WINDOW_SIZE.0 as f32, WINDOW_SIZE.1 as f32),
                    present_mode: bevy_window::PresentMode::Fifo,
                    ..Default::default()
                }),
            },
            WinitPlugin {
                run_on_any_thread: true
            },
            CameraPlugin,
            TimePlugin
        ))
        .add_systems(Startup, init)
        .add_systems(Update, render)
        .run();
}

fn render(
    mut renderer: NonSendMut<Renderer>,
    mut uniform_data: ResMut<CameraUniformData>,
    data: Res<FrameData>,
    raytracing_pipeline: Res<RayTracingPipeline>,
    post_proccesing_pipeline: Res<PostProccesingPipeline>,
    camera: Res<Camera>,
    controles: Res<Controls>,
    time: Res<Time>
) {
    log::info!("{:?}", time.delta());
    uniform_data.proj_inverse = camera.projection_matrix().inverse();
    uniform_data.view_inverse = camera.view_matrix().inverse();
    uniform_data.input.x = if controles.left_mouse { 1.0 } else { 0.0 };
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
    let DEFAULT_SAMPLER: vk::SamplerCreateInfo = vk::SamplerCreateInfo {
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

    let oct_tree_data = load_model("./models/Helena.vox").unwrap();

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
        unsafe { renderer.device.create_sampler(&DEFAULT_SAMPLER, None) }.unwrap();

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
                view: sky_box.1,
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

// let col = (((index % 8) as f32 / 8.0 as f32) * 255.0) as u32;

#[derive(Debug, Default)]
struct Octant {
    children: Option<Box<[Octant; 8]>>,
    color: Option<u32>,
}

impl Clone for Octant {
    fn clone(&self) -> Self {
        Octant {
            children: match self.children.as_ref() {
                None => None,
                Some(children) => Some(children.clone()),
            },
            color: self.color,
        }
    }
}

fn count(tree: &Octant, depth: u32, total: &mut u32) {
    if let Some(color) = tree.color {
        *total += 1;
        println!("Color: {}", color);
    }
    if depth == 0 {
        return;
    }
    if let Some(children) = tree.children.as_ref() {
        for i in 0..8 {
            count(&children[i as usize], depth - 1, total);
        }
    }
    return;
}

fn filter<T>(tree: &Octant, candidates: &mut Vec<Octant>, predicate: &T)
where
    T: Fn(&Octant) -> bool,
{
    if predicate(tree) {
        candidates.push(tree.clone());
    }
    if let Some(children) = tree.children.as_ref() {
        for i in 0..8 {
            filter(&children[i as usize], candidates, predicate);
        }
    }
    return;
}

fn append_voxel(tree: &mut Octant, color: u32, level_dim: u32, level_pos: UVec3) {
    if level_dim <= 1 {
        if tree.color.is_some() {
            println!("Override")
        } else {
            tree.color = Some(color);
        }
        return;
    }

    let level_dim = level_dim >> 1;
    let cmp = level_pos.cmpge(UVec3::new(level_dim, level_dim, level_dim));
    let child_slot_index = cmp.x as u32 | (cmp.y as u32) << 1 | (cmp.z as u32) << 2;
    let new_pos = level_pos
        - (UVec3::new(
            cmp.x as u32 * level_dim,
            cmp.y as u32 * level_dim,
            cmp.z as u32 * level_dim,
        ));

    match tree.children.as_mut() {
        None => {
            let mut children: [Octant; 8] = Default::default();

            append_voxel(
                &mut children[child_slot_index as usize],
                color,
                level_dim,
                new_pos,
            );
            tree.children = Some(Box::new(children));
        }
        Some(children) => {
            append_voxel(
                &mut children[child_slot_index as usize],
                color,
                level_dim,
                new_pos,
            );
        }
    }
}

fn load_model(path: &str) -> Result<Vec<u32>> {
    let mut tree = Vec::new();
    let mut octree = Octant::default();
    let file = dot_vox::load(path).unwrap();
    let models = file.models;
    for (n, m) in models.iter().enumerate() {
        let mut size = m.size.x;
        if size < m.size.y {
            size = m.size.y
        }
        if size < m.size.z {
            size = m.size.z
        }
        size = 2_u32.pow(((size as f32).log2()).ceil() as u32);
        for v in m.voxels.iter() {
            let color = file.palette[v.i as usize];
            let u32_color = ((color.b as u32) << 16) | ((color.g as u32) << 8) | (color.r as u32);
            let pos = UVec3::new(size - v.x as u32, v.z as u32, v.y as u32);
            append_voxel(&mut octree, u32_color, size, pos);
        }
    }

    let mut candidates = vec![&octree];
    let mut new_candidates = vec![];
    for _ in 0..10 {
        let i = tree.len() as u32 + candidates.len() as u32 * 8;
        for c in &candidates {
            for child in c.children.as_ref().unwrap().iter() {
                let node = if child.children.is_some() {
                    let candidate = new_candidates.len() as u32;
                    new_candidates.push(child);
                    0x8000_0000 + i + candidate * 8
                } else {
                    if let Some(color) = child.color {
                        0xC000_0000 | color
                    } else {
                        0x0000_0000
                    }
                };
                tree.push(node);
            }
        }

        candidates = new_candidates;
        new_candidates = vec![]
    }
    return Ok(tree);
}

fn build_oct_tree(depth: u32) -> Vec<u32> {
    let mut tree = Vec::new();
    let mut none_empty = 1;
    for j in 0..depth {
        let count = none_empty;
        none_empty = 0;
        for _ in 0..count * 8 {
            let i = tree.len() as u32 + 1;
            let node = if rand::random::<f32>() < 1.0 / ((depth - j + 1) as f32) {
                0xa000_0000
            } else if j == depth - 1 {
                none_empty += 1;
                // let col = (((index % 8) as f32 / 8.0 as f32) * 255.0) as u32;
                let col = rand::random::<u32>() % 2_u32.pow(24);
                0xC000_0000 + col
            } else {
                none_empty += 1;
                0x8000_0000 + 8 * i
            };
            tree.push(node);
        }
    }
    return tree;
}
