use std::{ffi::CStr, mem::size_of};

use ash::{
    ext::{shader_atomic_float, shader_atomic_float2},
    khr::{
        get_memory_requirements2, shader_float_controls, shader_non_semantic_info, swapchain,
        workgroup_memory_explicit_layout,
    },
    vk::{self, KHR_SPIRV_1_4_NAME},
};
use bevy::{prelude::*, winit::WinitWindows};
use g_buffer_pass::GBufferPass;
use gpu_allocator::MemoryLocation;
use oct_tree::GameWorld;
use post_processing_pass::PostProcessingPass;
use render_resources::RenderResources;
use renderer::{Buffer, DeviceFeatures, Image, ImageBarrier, Renderer};

use crate::{CameraUniformData, WINDOW_SIZE};

pub mod g_buffer_pass;
pub mod lighting_pass;
pub mod oct_tree;
pub mod post_processing_pass;
pub mod render_resources;

pub mod renderer;

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

#[derive(Resource)]
struct RenderPasses {
    g_buffer_pass: GBufferPass,
    post_processing_pass: PostProcessingPass,
}

fn render(
    mut renderer: NonSendMut<Renderer>,
    uniform_data: Res<CameraUniformData>,
    mut data: ResMut<FrameData>,
    main_pass: Res<RenderPasses>,
) {
    data.uniform_buffer
        .copy_data_to_buffer(std::slice::from_ref(&(*uniform_data)))
        .unwrap();

    // std::thread::sleep(Duration::from_millis(100));
    data.frame += 1;
    renderer
        .render(|renderer, i| {
            let cmd = &renderer.cmd_buffs[i as usize];

            main_pass.g_buffer_pass.execute(&renderer, cmd, data.frame);
            main_pass
                .post_processing_pass
                .execute(&renderer, cmd, data.frame, i);

            renderer.pipeline_image_barriers(
                cmd,
                &[ImageBarrier {
                    src_access_mask: vk::AccessFlags2::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags2::MEMORY_READ,
                    src_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    dst_stage_mask: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    image: &renderer.swapchain.images[i as usize].image,
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                }],
            );
        })
        .unwrap();
}

#[derive(Resource)]
struct FrameData {
    pub oct_tree_level: u32,
    pub uniform_buffer: Buffer,
    pub frame: u64,
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

    let image = image_thread.join().unwrap();
    let sky_box =
        Image::new_from_data(&mut renderer, image, vk::Format::R32G32B32A32_SFLOAT).unwrap();
    let sky_box_sampler =
        unsafe { renderer.device.create_sampler(&default_sampler, None) }.unwrap();

    let resources = RenderResources::new(&mut renderer);

    let render_passes = RenderPasses {
        g_buffer_pass: GBufferPass::new(
            &mut renderer,
            &resources,
            &oct_tree_buffer,
            &uniform_buffer,
        )
        .unwrap(),
        post_processing_pass: PostProcessingPass::new(
            &mut renderer,
            &resources,
            &uniform_buffer,
            &sky_box.view,
            &sky_box_sampler,
        )
        .unwrap(),
    };

    let oct_tree_level = game_world.tree_level.clone();
    world.insert_resource(resources);
    world.insert_non_send_resource(renderer);
    world.insert_resource(FrameData {
        oct_tree_level,
        uniform_buffer,
        frame: 0,
    });
    world.insert_resource(render_passes);
}

pub fn RenderPlugin(app: &mut App) {
    app.add_systems(Startup, init).add_systems(Update, render);
}
