use std::{
    f32::consts::PI,
    ffi::CStr,
    mem::{size_of, size_of_val},
    time::Duration,
};

use ash::{
    ext::buffer_device_address,
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
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

use crate::{
    copy_buffer,
    oct_tree::*,
    pipelines::{
        create_fullscreen_quad_pipeline, create_post_proccesing_pipelien, create_storage_images,
        PostProccesingPipeline, RayTracingPipeline,
    },
    Buffer, Camera, CameraUniformData, DeviceFeatures, Image, ImageAndView, Renderer,
    WriteDescriptorSet, WriteDescriptorSetKind, WINDOW_SIZE,
};

pub const DEVICE_EXTENSIONS: [&'static CStr; 8] = [
    swapchain::NAME,
    ash::ext::descriptor_indexing::NAME,
    ash::ext::scalar_block_layout::NAME,
    get_memory_requirements2::NAME,
    KHR_SPIRV_1_4_NAME,
    shader_float_controls::NAME,
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
    uniform_data: Res<CameraUniformData>,
    mut gizzmo_buffer: ResMut<GizzmoBuffer>,
    mut data: ResMut<FrameData>,
    raytracing_pipeline: Res<RayTracingPipeline>,
    post_proccesing_pipeline: Res<PostProccesingPipeline>,
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

            unsafe {
                let frame_c = std::slice::from_raw_parts(
                    &data.frame as *const u32 as *const u8,
                    size_of::<u32>(),
                );

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

                renderer.device.cmd_push_constants(
                    *cmd,
                    raytracing_pipeline.layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    frame_c,
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
        window.raw_window_handle().unwrap(),
        window.raw_display_handle().unwrap(),
        &device_features,
        WINDOW_SIZE.0,
        WINDOW_SIZE.1,
    )
    .unwrap();
    let game_world = world.get_resource::<GameWorld>().unwrap();

    let image_thread = std::thread::spawn(|| image::open("./src/models/skybox.exr").unwrap());
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

    let oct_tree_data = &game_world.build_tree;

    let oct_tree_buffer = renderer
        .create_gpu_only_buffer_from_data(
            vk::BufferUsageFlags::STORAGE_BUFFER,
            oct_tree_data.as_slice(),
            Some("OctTreeData"),
        )
        .unwrap();

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

    let raytracing_pipeline = create_fullscreen_quad_pipeline(
        &mut renderer,
        &uniform_buffer,
        &oct_tree_buffer,
        &[
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage_flags: ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 3,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 4,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ],
        &[
            vk::DescriptorPoolSize {
                descriptor_count: 1,
                ty: DescriptorType::COMBINED_IMAGE_SAMPLER,
            },
            vk::DescriptorPoolSize {
                descriptor_count: 2,
                ty: DescriptorType::STORAGE_BUFFER,
            },
        ],
        &[
            WriteDescriptorSet {
                binding: 2,
                kind: WriteDescriptorSetKind::CombinedImageSampler {
                    view: sky_box.view,
                    sampler: sky_box_sampler,
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
            },
            WriteDescriptorSet {
                binding: 3,
                kind: WriteDescriptorSetKind::StorageBuffer {
                    buffer: gizzmo_buffer.buffer.inner,
                },
            },
            WriteDescriptorSet {
                binding: 4,
                kind: WriteDescriptorSetKind::StorageBuffer {
                    buffer: lights_buffer.inner,
                },
            },
        ],
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
    world.insert_resource(gizzmo_buffer);
    world.insert_non_send_resource(renderer);
    world.insert_resource(FrameData {
        color_buffers,
        depth_buffers,
        frame_buffers,
        present_frame_buffers,
        uniform_buffer,
        frame: 0,
    });
    world.insert_resource(raytracing_pipeline);
    world.insert_resource(post_proccesing_pipeline);
}

pub fn RenderPlugin(app: &mut App) {
    app.add_systems(Startup, init).add_systems(Update, render);
}
