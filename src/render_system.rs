use std::{
    f32::consts::PI,
    ffi::CStr,
    mem::{size_of, size_of_val},
    time::Duration,
};

use ash::{
    ext::{buffer_device_address, shader_atomic_float, shader_atomic_float2},
    khr::{
        get_memory_requirements2, shader_float_controls, shader_non_semantic_info, swapchain,
        workgroup_memory_explicit_layout,
    },
    vk::{self, DescriptorType, ShaderStageFlags, KHR_SPIRV_1_4_NAME},
};
use bevy::{
    app::{App, Startup, Update},
    prelude::{NonSendMut, Res, ResMut, Resource, World},
    time::Time,
    winit::WinitWindows,
};
use glam::*;
use gpu_allocator::MemoryLocation;
use log::info;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

use crate::{
    copy_buffer,
    oct_tree::*,
    pipelines::{
        create_frame_buffers, create_main_render_pass, create_raytracing_pipeline, MainPass,
    },
    Buffer, Camera, CameraUniformData, DeviceFeatures, Image, ImageAndView, Renderer,
    WriteDescriptorSet, WriteDescriptorSetKind, WINDOW_SIZE,
};

pub const DEVICE_EXTENSIONS: [&'static CStr; 10] = [
    swapchain::NAME,
    ash::ext::descriptor_indexing::NAME,
    ash::ext::scalar_block_layout::NAME,
    get_memory_requirements2::NAME,
    KHR_SPIRV_1_4_NAME,
    shader_float_controls::NAME,
    shader_non_semantic_info::NAME,
    workgroup_memory_explicit_layout::NAME,
    shader_atomic_float::NAME,
    shader_atomic_float2::NAME,
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
    uniform_data: Res<CameraUniformData>,
    mut gizzmo_buffer: ResMut<GizzmoBuffer>,
    mut data: ResMut<FrameData>,
    main_pass: Res<MainPass>,
    time: Res<Time>,
) {
    data.uniform_buffer
        .copy_data_to_buffer(std::slice::from_ref(&(*uniform_data)))
        .unwrap();

    gizzmo_buffer.update(&mut renderer);

    // std::thread::sleep(Duration::from_millis(100));
    data.frame += 1;
    renderer
        .render(|renderer, i| {
            let cmd = &renderer.cmd_buffs[i as usize];
            let current_frame = renderer.frame;
            unsafe {
                let frame_c = std::slice::from_raw_parts(
                    &data.frame as *const u32 as *const u8,
                    size_of::<u32>(),
                );
                
                renderer.device.cmd_fill_buffer(*cmd, main_pass.hash_map_buffer.inner, 0, main_pass.hash_map_buffer.size, 0);

                renderer.device.cmd_pipeline_barrier(
                    *cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::DependencyFlags::DEVICE_GROUP,
                    &[],
                    &[vk::BufferMemoryBarrier::default()
                        .buffer(main_pass.hash_map_buffer.inner)
                        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                        .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                        .offset(0)
                        .size(main_pass.hash_map_buffer.size)],
                    &[],
                );
                let begin_info = vk::RenderPassBeginInfo::default()
                    .clear_values(&[
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                uint32: [0, 0, 0, 0],
                            },
                        },
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        },
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        },
                    ])
                    .render_pass(main_pass.render_pass)
                    .framebuffer(main_pass.frame_buffers[i as usize])
                    .render_area(vk::Rect2D {
                        extent: vk::Extent2D {
                            width: WINDOW_SIZE.0,
                            height: WINDOW_SIZE.1,
                        },
                        offset: vk::Offset2D { x: 0, y: 0 },
                    });

                renderer
                    .device
                    .cmd_set_viewport(*cmd, 0, &[vk::Viewport {
                        width: WINDOW_SIZE.0 as f32 / 8.0,
                        height: WINDOW_SIZE.1 as f32 / 8.0,
                        x: 0.0,
                        y: 0.0,
                        max_depth: 1.0,
                        min_depth: 0.0,
                    }]);
                renderer
                    .device
                    .cmd_set_scissor(*cmd, 0, &[vk::Rect2D {
                        extent: vk::Extent2D { width: WINDOW_SIZE.0 /8, height: WINDOW_SIZE.0 /8 },
                        offset: vk::Offset2D {
                            x: 0,
                            y: 0
                        }
                    }]);

                renderer.device.cmd_begin_render_pass(
                    *cmd,
                    &begin_info,
                    vk::SubpassContents::INLINE,
                );
                renderer.device.cmd_bind_pipeline(
                    *cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    main_pass.beam_trace.pipeline,
                );

                renderer.device.cmd_bind_descriptor_sets(
                    *cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    main_pass.beam_trace.layout,
                    0,
                    &[main_pass.beam_trace.descriptors[0]],
                    &[],
                );

                renderer.device.cmd_push_constants(
                    *cmd,
                    main_pass.beam_trace.layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    &(1.0 - data.oct_tree_level as f32).exp2().to_ne_bytes(),
                );

                renderer.device.cmd_draw(*cmd, 6, 1, 0, 0);

                renderer
                    .device
                    .cmd_next_subpass(*cmd, vk::SubpassContents::INLINE);

                renderer
                    .device
                    .cmd_set_viewport(*cmd, 0, &[FULL_SCREEN_VIEW_PORT]);
                renderer
                    .device
                    .cmd_set_scissor(*cmd, 0, &[FULL_SCREEN_SCISSOR]);
                renderer.device.cmd_bind_pipeline(
                    *cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    main_pass.ray_tracing.pipeline,
                );

                renderer.device.cmd_bind_descriptor_sets(
                    *cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    main_pass.ray_tracing.layout,
                    0,
                    &[main_pass.ray_tracing.descriptors[0], main_pass.ray_tracing.descriptors2[i as usize]],
                    &[],
                );

                renderer.device.cmd_push_constants(
                    *cmd,
                    main_pass.ray_tracing.layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    frame_c,
                );

                renderer.device.cmd_draw(*cmd, 6, 1, 0, 0);

                renderer
                    .device
                    .cmd_next_subpass(*cmd, vk::SubpassContents::INLINE);

                renderer.device.cmd_bind_pipeline(
                    *cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    main_pass.post_proccesing.pipeline,
                );

                renderer.device.cmd_bind_descriptor_sets(
                    *cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    main_pass.post_proccesing.layout,
                    0,
                    &[
                        main_pass.post_proccesing.descriptors[0],
                        main_pass.post_proccesing.descriptors2[i as usize],
                    ],
                    &[],
                );
                renderer.device.cmd_push_constants(
                    *cmd,
                    main_pass.post_proccesing.layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    frame_c,
                );

                renderer.device.cmd_draw(*cmd, 6, 1, 0, 0);

                renderer.device.cmd_end_render_pass(*cmd);
            }
        })
        .unwrap();
}

#[derive(Resource)]
struct FrameData {
    pub oct_tree_level: u32,
    pub uniform_buffer: Buffer,
    pub frame: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Gizzmo {
    pub color: glam::Vec4,
    pub position: glam::Vec3,
    pub radius: f32,
}

#[derive(Resource)]
pub struct GizzmoBuffer {
    pub staging: Buffer,
    pub buffer: Buffer,
    pub data: [Gizzmo; 10],
    pub dirty: bool,
}

impl GizzmoBuffer {
    pub fn sphere(&mut self, i: usize, color: glam::Vec3, pos: glam::Vec3, radius: f32) {
        self.data[i as usize].color = glam::vec4(color.x, color.y, color.z, 1.0);
        self.data[i as usize].position = pos;
        self.data[i as usize].radius = radius;
        self.dirty = true;
    }

    pub fn update(&mut self, renderer: &mut Renderer) {
        self.staging.copy_data_to_buffer(&self.data).unwrap();
        renderer
            .execute_one_time_commands(|cmd_buffer| {
                copy_buffer(&renderer.device, cmd_buffer, &self.staging, &self.buffer);
            })
            .unwrap();
        // info!("{:?}", self.data);
        self.dirty = false;
    }
}

const ligth_rotation: f32 = PI / 1.5;
const light_hight: f32 = PI / 4.0;

fn polar_form(theta: f32, thi: f32) -> glam::Vec3 {
    return vec3(
        f32::sin(theta) * f32::cos(thi),
        f32::sin(theta) * f32::sin(thi),
        f32::cos(theta),
    );
}

fn angle_axis3x3(angle: f32, axis: glam::Vec3) -> glam::Mat3 {
    let s = f32::sin(angle);
    let c = f32::cos(angle);

    let t = 1.0 - c;
    let x = axis.x;
    let y = axis.y;
    let z = axis.z;

    return glam::Mat3::from_cols(
        vec3(t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        vec3(t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        vec3(t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    );
}

fn get_cone_sample(direction: glam::Vec3, angle: f32) -> glam::Vec3 {
    let cos_angle = f32::cos(angle);

    // Generate points on the spherical cap around the north pole [1].
    // [1] See https://math.stackexchange.com/a/205589/81266
    let z = rand::random::<f32>() * (1.0 - cos_angle) + cos_angle;
    let phi = rand::random::<f32>() * 2.0 * PI;

    let x = f32::sqrt(1.0 - z * z) * f32::cos(phi);
    let y = f32::sqrt(1.0 - z * z) * f32::sin(phi);
    let north = vec3(0., 0., 1.);

    // Find the rotation axis `u` and rotation angle `rot` [1]
    let axis = Vec3::normalize(Vec3::cross(north, Vec3::normalize(direction)));
    let angle = f32::acos(Vec3::dot(Vec3::normalize(direction), north));
    // Convert rotation axis and angle to 3x3 rotation matrix [2]
    let r = Affine3A::from_axis_angle(axis, angle);

    return r.matrix3 * vec3(x, y, z);
}

fn init(world: &mut World) {
    let device_features = world.get_resource::<DeviceFeatures>().unwrap();
    let windows = world.get_non_send_resource::<WinitWindows>().unwrap();
    let window = windows.windows.values().into_iter().last().unwrap();
    let mut renderer = Renderer::new(
        &window,
        &window,
        &device_features,
        WINDOW_SIZE.0,
        WINDOW_SIZE.1,
    )
    .unwrap();
    let game_world = world.get_resource::<GameWorld>().unwrap();

    let image_thread = std::thread::spawn(|| image::open("./models/skybox.exr").unwrap());
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
    
    let oct_tree_buffer = {
        let oct_tree_data = &game_world.build_tree;

        renderer
            .create_gpu_only_buffer_from_data(
                vk::BufferUsageFlags::STORAGE_BUFFER,
                oct_tree_data.as_slice(),
                Some("OctTreeData"),
            )
            .unwrap()
    };
    let gizzmo_buffer = {
        let data = [Gizzmo {
            position: glam::Vec3::ZERO,
            radius: -1.0,
            color: vec4(1.0, 0.0, 0.0, 1.0),
        }; 10];
        let size = size_of_val(&data);
        let gizzmo_staging_buffer = renderer
            .create_buffer(
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                size as u64,
                Some("Gizzmo Staging Buffer"),
            )
            .unwrap();
        gizzmo_staging_buffer.copy_data_to_buffer(&data).unwrap();

        let buffer = renderer
            .create_buffer(
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                size as u64,
                Some("Gizzmo Buffer"),
            )
            .unwrap();

        renderer
            .execute_one_time_commands(|cmd_buffer| {
                copy_buffer(
                    &renderer.device,
                    cmd_buffer,
                    &gizzmo_staging_buffer,
                    &buffer,
                );
            })
            .unwrap();
        GizzmoBuffer {
            buffer,
            data,
            staging: gizzmo_staging_buffer,
            dirty: false,
        }
    };

    let light_dir = polar_form(ligth_rotation, light_hight);
    let mut light_data = vec![];

    for _ in 0..20 {
        let dir = get_cone_sample(light_dir, 0.1);
        light_data.push(dir.xyzx());
    }

    let lights_buffer = renderer
        .create_gpu_only_buffer_from_data(
            vk::BufferUsageFlags::STORAGE_BUFFER,
            light_data.as_slice(),
            Some("lights_buffer"),
        )
        .unwrap();

    let image = image_thread.join().unwrap();
    let sky_box =
        Image::new_from_data(&mut renderer, image, vk::Format::R32G32B32A32_SFLOAT).unwrap();
    let sky_box_sampler =
        unsafe { renderer.device.create_sampler(&default_sampler, None) }.unwrap();

    let main_pass = create_main_render_pass(
        &mut renderer,
        &oct_tree_buffer,
        &uniform_buffer,
        &sky_box,
        &sky_box_sampler,
        &[
            vk::DescriptorSetLayoutBinding {
                binding: 4,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 5,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ],
        &[vk::DescriptorPoolSize {
            descriptor_count: 2,
            ty: DescriptorType::STORAGE_BUFFER,
        }],
        &[
            WriteDescriptorSet {
                binding: 4,
                kind: WriteDescriptorSetKind::StorageBuffer {
                    buffer: gizzmo_buffer.buffer.inner,
                },
            },
            WriteDescriptorSet {
                binding: 5,
                kind: WriteDescriptorSetKind::StorageBuffer {
                    buffer: lights_buffer.inner,
                },
            },
        ],
    )
    .unwrap();
    let oct_tree_level = game_world.tree_level.clone();
    world.insert_resource(gizzmo_buffer);
    world.insert_non_send_resource(renderer);
    world.insert_resource(FrameData {
        oct_tree_level,
        uniform_buffer,
        frame: 0,
    });
    world.insert_resource(main_pass);
}

pub fn RenderPlugin(app: &mut App) {
    app.add_systems(Startup, init).add_systems(Update, render);
}