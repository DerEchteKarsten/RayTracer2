mod context;
use context::*;
mod camera;
use camera::*;

use anyhow::Result;
use ash::extensions::khr::{self};
use ash::vk::{
    self,
    KhrShaderNonSemanticInfoFn,
};
use dot_vox::Voxel;
use gpu_allocator::MemoryLocation;


use glam::{vec3, BVec3, UVec3, Vec3, Vec4};
use std::default::Default;
use std::ffi::{CStr, CString};
use std::rc::Rc;
use std::time::{Duration, Instant};

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

use std::mem::size_of;

use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const DEVICE_EXTENSIONS: [&'static CStr; 10] = [
    khr::Swapchain::name(),
    vk::ExtDescriptorIndexingFn::name(),
    vk::ExtScalarBlockLayoutFn::name(),
    vk::KhrGetMemoryRequirements2Fn::name(),
    khr::BufferDeviceAddress::name(),
    vk::KhrSpirv14Fn::name(),
    vk::KhrShaderFloatControlsFn::name(),
    vk::KhrBufferDeviceAddressFn::name(),
    KhrShaderNonSemanticInfoFn::name(),
    vk::KhrWorkgroupMemoryExplicitLayoutFn::name()
];

const APP_NAME: &'static str = "Test";
const WINDOW_SIZE: PhysicalSize<u32> = PhysicalSize {
    width: 2560,
    height: 1400,
};

const FULL_SCREEN_SCISSOR: vk::Rect2D = vk::Rect2D{
        extent: vk::Extent2D { width: WINDOW_SIZE.width, height: WINDOW_SIZE.height },
        offset: vk::Offset2D {x:0,y:0}
    };

const FULL_SCREEN_VIEW_PORT: vk::Viewport = vk::Viewport {
    x: 0.0,
    y: 0.0,
    width: WINDOW_SIZE.width as f32,
    height: WINDOW_SIZE.height as f32,
    min_depth: 0.0,
    max_depth: 1.0
};

fn main() {
  
    let device_features: DeviceFeatures = DeviceFeatures {
        ray_tracing_pipeline: false,
        acceleration_structure: false,
        runtime_descriptor_array: true,
        buffer_device_address: true,
        dynamic_rendering: true,
        synchronization2: true,
    };
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(APP_NAME)
        .with_inner_size(WINDOW_SIZE)
        .build(&event_loop)
        .unwrap();
    let mut ctx = Context::new(
        window.raw_window_handle(),
        window.raw_display_handle(),
        device_features,
        WINDOW_SIZE,
    )
    .unwrap();

    let mut camera = Camera::new(
        vec3(1.5, 1.25, -5.0),
        vec3(0.0, 0.0, 1.0),
        40.0,
        WINDOW_SIZE.width as f32 / WINDOW_SIZE.height as f32,
        0.1,
        1000.0,
    );

    let mut controles = Controls {
        ..Default::default()
    };

    let mut uniform_data = UniformData {
        proj_inverse: camera.projection_matrix().inverse(),
        view_inverse: camera.view_matrix().inverse(),
        input: Vec4::new(0.0, 0.0, 0.0, 0.0),
    };

    let uniform_buffer = ctx
        .create_buffer(
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            size_of::<UniformData>() as u64,
            Some("Uniform Buffer"),
        )
        .unwrap();
    
    let oct_tree_data = load_model("./models/monu2.vox").unwrap();
    //build_oct_tree(3);

    let oct_tree_buffer = ctx.create_gpu_only_buffer_from_data(vk::BufferUsageFlags::STORAGE_BUFFER, oct_tree_data.as_slice(), Some("OctTreeData")).unwrap();

    let raytracing_pipeline = create_fullscreen_quad_pipeline(&mut ctx, &uniform_buffer, &oct_tree_buffer).unwrap();
    
    let storage_images = create_storage_images(&mut ctx, &raytracing_pipeline).unwrap();

    let post_proccesing_pipeline =
        create_post_proccesing_pipelien(&mut ctx, &storage_images.color_buffers).unwrap();

    let present_frame_buffers = {
        (0..ctx.swapchain.images.len())
            .map(|i| unsafe {
                ctx.device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::builder()
                            .attachments(&[ctx.swapchain.images[i].1])
                            .attachment_count(1)
                            .height(WINDOW_SIZE.height)
                            .width(WINDOW_SIZE.width)
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
    // let mut frame: u32 = 0;
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
                // frame += 1;
                let new_cam = camera.update(&controles, frame_time);
                moved = new_cam != camera;
                if moved {
                    camera = new_cam;
                }

                uniform_data.proj_inverse = camera.projection_matrix().inverse();
                uniform_data.view_inverse = camera.view_matrix().inverse();
                uniform_data.input.x = if controles.left_mouse {1.0} else {0.0};
                uniform_data.input.z = WINDOW_SIZE.width as f32;
                uniform_data.input.w = WINDOW_SIZE.height as f32;
                uniform_buffer
                    .copy_data_to_buffer(std::slice::from_ref(&uniform_data))
                    .unwrap();
                camera = new_cam;

                ctx.render(|ctx, i| {
                    let cmd = &ctx.cmd_buffs[i as usize];

                    unsafe {
                        // let frame_c = std::slice::from_raw_parts(
                        //     &frame as *const u32 as *const u8,
                        //     size_of::<u32>(),
                        // );
                        // let moved_c = &[if moved == true { 1 as u8 } else { 0 as u8 }, 0, 0, 0];

                        let begin_info = vk::RenderPassBeginInfo::builder()
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
                            .framebuffer(storage_images.frame_buffers[i as usize])
                            .render_area(vk::Rect2D {
                                extent: vk::Extent2D {
                                    width: WINDOW_SIZE.width,
                                    height: WINDOW_SIZE.height,
                                },
                                offset: vk::Offset2D { x: 0, y: 0 },
                            });

                        ctx.device.cmd_set_viewport(*cmd, 0, &[FULL_SCREEN_VIEW_PORT]);
                        ctx.device.cmd_set_scissor(*cmd, 0, &[FULL_SCREEN_SCISSOR]);

                        ctx.device.cmd_begin_render_pass(
                            *cmd,
                            &begin_info,
                            vk::SubpassContents::INLINE,
                        );

                        ctx.device.cmd_bind_pipeline(
                            *cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            raytracing_pipeline.pipeline,
                        );

                        ctx.device.cmd_bind_descriptor_sets(
                            *cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            raytracing_pipeline.layout,
                            0,
                            &[raytracing_pipeline.descriptor],
                            &[],
                        );
                        ctx.device.cmd_draw(*cmd, 6, 1, 0, 0);

                        ctx.device.cmd_end_render_pass(*cmd);

                        let begin_info = vk::RenderPassBeginInfo::builder()
                            .render_pass(post_proccesing_pipeline.render_pass)
                            .framebuffer(present_frame_buffers[i as usize])
                            .render_area(vk::Rect2D {
                                extent: vk::Extent2D {
                                    width: WINDOW_SIZE.width,
                                    height: WINDOW_SIZE.height,
                                },
                                offset: vk::Offset2D { x: 0, y: 0 },
                            })
                            .clear_values(&[vk::ClearValue {
                                color: vk::ClearColorValue {
                                    float32: [0.0, 0.0, 0.0, 1.0],
                                },
                            }]);

                        ctx.device.cmd_set_viewport(*cmd, 0, &[FULL_SCREEN_VIEW_PORT]);
                        ctx.device.cmd_set_scissor(*cmd, 0, &[FULL_SCREEN_SCISSOR]);

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

// let col = (((index % 8) as f32 / 8.0 as f32) * 255.0) as u32;

#[derive(Debug, Default)]
struct Octant {
    children: Option<Box<[Octant;8]>>,
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
        *total+=1;
        println!("Color: {}", color);
    }
    if depth == 0 {
        return;
    }
    if let Some(children) = tree.children.as_ref() {
        for i in 0..8 {
            count(&children[i as usize], depth-1, total);
        }
    }
    return;
}


fn filter<T>(tree: &Octant, candidates: &mut Vec<Octant>, predicate: &T) 
where
    T: Fn(&Octant) -> bool
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

fn append_voxel(tree: &mut Octant, color: u32, level_dim: u32, level_pos: UVec3){
    if level_dim <= 1 {
        if tree.color.is_some() {
            println!("Override")
        }else {
            tree.color = Some(color);
        }
        return;
    }
  
    let level_dim = level_dim >> 1;
    let cmp = level_pos.cmpge(UVec3::new(level_dim, level_dim, level_dim));
    let child_slot_index = cmp.x as u32 | (cmp.y as u32) << 1 | (cmp.z as u32) << 2;
    let new_pos = level_pos - (UVec3::new(cmp.x as u32 * level_dim, cmp.y as u32 * level_dim, cmp.z as u32 * level_dim));

    match tree.children.as_mut() {
        None => {
            let mut children: [Octant; 8] = Default::default();

            append_voxel(&mut children[child_slot_index as usize], color, level_dim, new_pos); 
            tree.children = Some(Box::new(children));
        }
        Some(children) => {
            append_voxel(&mut children[child_slot_index as usize], color, level_dim, new_pos); 
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
            let pos = UVec3::new(v.x as u32, v.z as u32, v.y as u32);
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
                    let candidate = new_candidates.len() as u32 ;
                    new_candidates.push(child);
                    0x8000_0000 + i + candidate *8 
                }else {
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

fn build_oct_tree(depth: u32) -> Vec<u32>{
    let mut tree = Vec::new();
    let mut none_empty = 1;
    for j in 0..depth {
        let count = none_empty;
        none_empty = 0;
        for _ in 0..count*8 {
            let i = tree.len() as u32 + 1;
            let node = if rand::random::<f32>() < 1.0/((depth-j+1) as f32) {
                0xa000_0000
            }else if j == depth-1 {
                none_empty+=1;
                // let col = (((index % 8) as f32 / 8.0 as f32) * 255.0) as u32;
                let col = rand::random::<u32>() % 2_u32.pow(24);
                0xC000_0000 + col
            }else {
                none_empty+=1;
                0x8000_0000 + 8*i 
            };
            tree.push(node);
        }
    }
    return tree;
}

struct StorageImages {
    frame_buffers: Vec<vk::Framebuffer>,
    color_buffers: Vec<(vk::ImageView, Image)>,
    depth_buffers: Vec<(vk::ImageView, Image)>,
}

fn create_storage_images<'a>(
    ctx: &mut Context,
    g_buffer_pipeline: &RasterPipeline,
) -> Result<StorageImages> {
    let size = ctx.swapchain.images.len();
    let color_buffers = (0..size)
        .map(|_| {
            let image = Image::new_2d(
                &ctx.device,
                &mut ctx.allocator,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                MemoryLocation::GpuOnly,
                vk::Format::R32G32B32A32_SFLOAT,
                WINDOW_SIZE.width,
                WINDOW_SIZE.height,
            )
            .unwrap();
            let image_view = ctx.create_image_view(&image).unwrap();
            (image_view, image)
        })
        .collect::<Vec<(vk::ImageView, Image)>>();
    let depth_buffers = (0..size)
        .map(|_| {
            let image = Image::new_2d(
                &ctx.device,
                &mut ctx.allocator,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                MemoryLocation::GpuOnly,
                vk::Format::R32_SFLOAT,
                WINDOW_SIZE.width,
                WINDOW_SIZE.height,
            )
            .unwrap();
            let image_view = ctx.create_image_view(&image).unwrap();
            (image_view, image)
        })
        .collect::<Vec<(vk::ImageView, Image)>>();

    let frame_buffers = (0..size)
        .map(|i| {
            let attachments = [
                color_buffers[i].0,
                depth_buffers[i].0,
            ];
            let create_info = vk::FramebufferCreateInfo::builder()
                .attachment_count(2)
                .attachments(&attachments)
                .height(WINDOW_SIZE.height)
                .width(WINDOW_SIZE.width)
                .layers(1)
                .render_pass(g_buffer_pipeline.render_pass);
            unsafe { ctx.device.create_framebuffer(&create_info, None) }.unwrap()
        })
        .collect::<Vec<vk::Framebuffer>>();

    Ok(StorageImages {
        frame_buffers,
        color_buffers,
        depth_buffers,
    })
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

struct RasterPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_layout: vk::DescriptorSetLayout,
    pub render_pass: vk::RenderPass,
    pub descriptor: vk::DescriptorSet,
}

fn create_fullscreen_quad_pipeline(
    ctx: &mut Context,
    uniform_buffer: &Buffer,
    oct_tree_buffer: &Buffer,
) -> Result<RasterPipeline> {
    let attachments = [
        vk::AttachmentDescription::builder()
            .samples(vk::SampleCountFlags::TYPE_1)
            .final_layout(vk::ImageLayout::GENERAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .build(),
            vk::AttachmentDescription::builder()
            .samples(vk::SampleCountFlags::TYPE_1)
            .final_layout(vk::ImageLayout::GENERAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .format(vk::Format::R32_SFLOAT)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .build(),
    ];

    let subpasses = [vk::SubpassDescription::builder()
        .color_attachments(&[
            vk::AttachmentReference::builder()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .build(),
            vk::AttachmentReference::builder()
                .attachment(1)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .build(),
        ])
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .build()];

    let dependencys = [vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .build()];

    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .dependencies(&dependencys)
        .subpasses(&subpasses);

    let render_pass = unsafe {
        ctx.device
            .create_render_pass(&render_pass_create_info, None)?
    };

    let descriptor_bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX)
            .build(),
    ];

    let descriptor_layout = ctx.create_descriptor_set_layout(&descriptor_bindings, &[])?;

    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&[descriptor_layout])
        .build();
    let layout = unsafe { ctx.device.create_pipeline_layout(&layout_info, None) }?;

    let color_blend_attachments = [
        vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(false)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build(),
        vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(false)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build(),
    ];

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

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .primitive_restart_enable(false)
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_attribute_descriptions(&[])
        .vertex_binding_descriptions(&[])
        .build();

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
            .module(ctx.create_shader_module("./src/shaders/default.frag.spv".to_string()))
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&entry_point_name)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .module(ctx.create_shader_module("./src/shaders/default.vert.spv".to_string()))
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(&entry_point_name)
            .build(),
    ];

    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .scissor_count(1)
        .viewport_count(1);

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

    let pool_sizes = [
        vk::DescriptorPoolSize::builder()
            .descriptor_count(1)
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .build(),
        vk::DescriptorPoolSize::builder()
            .descriptor_count(2)
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .build(),
    ];

    let descriptor_pool = ctx.create_descriptor_pool(1, &pool_sizes)?;
    let descriptor = allocate_descriptor_set(&ctx.device, &descriptor_pool, &descriptor_layout)?;

    let writes = vec![
        WriteDescriptorSet {
            binding: 0,
            kind: WriteDescriptorSetKind::UniformBuffer {
                buffer: uniform_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 1,
            kind: WriteDescriptorSetKind::StorageBuffer { buffer: oct_tree_buffer.inner }
        }
    ];
    update_descriptor_sets(ctx, &descriptor, &writes);

    Ok(RasterPipeline {
        pipeline,
        render_pass,
        descriptor,
        descriptor_layout,
        layout,
    })
}
