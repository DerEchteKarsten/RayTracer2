use anyhow::{Ok, Result};
use ash::vk::{self};
use bevy::prelude::*;
use gpu_allocator::MemoryLocation;

use std::default::Default;
use std::ffi::CString;

use crate::{
    allocate_descriptor_set, allocate_descriptor_sets, create_descriptor_pool,
    create_descriptor_set_layout, update_descriptor_sets, Buffer, Image, ImageAndView, Renderer,
    WriteDescriptorSet, WriteDescriptorSetKind, WINDOW_SIZE,
};

pub fn create_compute_pipeline(
    renderer: &mut Renderer,
    hash_map_buffers: &Vec<Buffer>,
) -> Result<Pipeline> {
    let bindings = [vk::DescriptorSetLayoutBinding {
        descriptor_count: 1,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        binding: 0,
        ..Default::default()
    }];

    let descriptor_set_layout = renderer.create_descriptor_set_layout(&bindings, &[])?;
    let descriptor_set_layout2 = renderer.create_descriptor_set_layout(&bindings, &[])?;

    let set_layouts = [descriptor_set_layout, descriptor_set_layout2];
    let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    let layout = unsafe { renderer.device.create_pipeline_layout(&layout_info, None) }?;

    let create_info = vk::ComputePipelineCreateInfo::default()
        .layout(layout)
        .stage(
            renderer
                .create_shader_stage(
                    vk::ShaderStageFlags::COMPUTE,
                    "./src/shaders/temp_reuse.comp.spv".to_owned(),
                )
                .0,
        );

    let pipeline = unsafe {
        renderer
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
    }
    .unwrap();

    let pool_sizes = [vk::DescriptorPoolSize::default()
        .descriptor_count(2 * renderer.swapchain.images.len() as u32)
        .ty(vk::DescriptorType::STORAGE_BUFFER)];
    let descriptor_pool =
        renderer.create_descriptor_pool(renderer.swapchain.images.len() as u32 * 2, &pool_sizes)?;
    let descriptors = allocate_descriptor_sets(
        &renderer.device,
        &descriptor_pool,
        &descriptor_set_layout,
        renderer.swapchain.images.len() as u32,
    )?;
    let descriptors2 = allocate_descriptor_sets(
        &renderer.device,
        &descriptor_pool,
        &descriptor_set_layout2,
        renderer.swapchain.images.len() as u32,
    )?;

    for i in 0..renderer.swapchain.images.len() {
        let writes = [WriteDescriptorSet {
            binding: 0,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: hash_map_buffers[i].inner,
            },
        }];

        update_descriptor_sets(renderer, &descriptors[i], &writes);
        update_descriptor_sets(renderer, &descriptors2[i], &writes);
    }

    Ok(Pipeline {
        descriptors2,
        descriptors,
        layout,
        pipeline: *pipeline.first().unwrap(),
    })
}

pub fn create_frame_buffers<'a>(
    ctx: &mut Renderer,
    render_pass: &vk::RenderPass,
) -> Result<(Vec<vk::Framebuffer>, Vec<ImageAndView>)> {
    let size = ctx.swapchain.images.len();
    let voxel_index_buffers = (0..size)
        .map(|_| {
            let image = Image::new_2d(
                &ctx.device,
                &mut ctx.allocator,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                MemoryLocation::GpuOnly,
                vk::Format::R32_UINT,
                WINDOW_SIZE.0,
                WINDOW_SIZE.1,
            )
            .unwrap();
            let image_view = ctx.create_image_view(&image).unwrap();
            ImageAndView {
                image,
                view: image_view,
            }
        })
        .collect::<Vec<ImageAndView>>();
    let frame_buffers = (0..size)
        .map(|i| {
            let attachments = [voxel_index_buffers[i].view, ctx.swapchain.images[i].view];
            let create_info = vk::FramebufferCreateInfo::default()
                .attachment_count(2)
                .attachments(&attachments)
                .height(WINDOW_SIZE.1)
                .width(WINDOW_SIZE.0)
                .layers(1)
                .render_pass(*render_pass);
            unsafe { ctx.device.create_framebuffer(&create_info, None) }.unwrap()
        })
        .collect::<Vec<vk::Framebuffer>>();

    Ok((frame_buffers, voxel_index_buffers))
}

pub struct Pipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptors: Vec<vk::DescriptorSet>,
    pub descriptors2: Vec<vk::DescriptorSet>,
}

#[derive(Resource)]
pub struct MainPass {
    pub ray_tracing: Pipeline,
    pub post_proccesing: Pipeline,
    pub temporal_reuse: Pipeline,
    pub render_pass: vk::RenderPass,
    pub voxel_index_buffers: Vec<ImageAndView>,
    pub frame_buffers: Vec<vk::Framebuffer>,
    pub hash_map_buffers: Vec<Buffer>,
}

pub fn create_post_proccesing_pipeline(
    ctx: &mut Renderer,
    render_pass: &vk::RenderPass,
    storage_images: &Vec<ImageAndView>,
    hash_map_buffers: &Vec<Buffer>,
    oct_tree_buffer: &Buffer,
    uniform_buffer: &Buffer,
    sky_box: &ImageAndView,
    sky_box_sampler: &vk::Sampler,
) -> Result<Pipeline> {
    let static_descriptor_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];

    let descriptor_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];

    let descriptor_layout = ctx.create_descriptor_set_layout(&descriptor_bindings, &[])?;
    let static_descriptor_layout =
        ctx.create_descriptor_set_layout(&static_descriptor_bindings, &[])?;

    let descriptor_layouts = [static_descriptor_layout, descriptor_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&descriptor_layouts)
        .push_constant_ranges(&[]);
    let layout = unsafe { ctx.device.create_pipeline_layout(&layout_info, None) }?;

    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )];

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&color_blend_attachments)
        .logic_op(vk::LogicOp::COPY)
        .logic_op_enable(false)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_test_enable(false)
        .depth_write_enable(false);

    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .cull_mode(vk::CullModeFlags::BACK)
        .depth_clamp_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(false)
        .rasterizer_discard_enable(false);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let entry_point_name: CString = CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .module(ctx.create_shader_module("./src/shaders/post_processing.frag.spv".to_string()))
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&entry_point_name),
        vk::PipelineShaderStageCreateInfo::default()
            .module(ctx.create_shader_module("./src/shaders/post_processing.vert.spv".to_string()))
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(&entry_point_name),
    ];

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .scissor_count(1)
        .viewport_count(1);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .primitive_restart_enable(false)
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&[])
        .vertex_binding_descriptions(&[]);

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
        .color_blend_state(&color_blend_state)
        .depth_stencil_state(&depth_stencil_state)
        .dynamic_state(&dynamic_state)
        .input_assembly_state(&input_assembly_state)
        .vertex_input_state(&vertex_input_state)
        .layout(layout)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .render_pass(*render_pass)
        .stages(&stages)
        .viewport_state(&viewport_state)
        .subpass(1);
    let pipeline = unsafe {
        ctx.device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
            .unwrap()
    }[0];

    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .descriptor_count(1)
            .ty(vk::DescriptorType::UNIFORM_BUFFER),
        vk::DescriptorPoolSize::default()
            .descriptor_count(ctx.swapchain.images.len() as u32)
            .ty(vk::DescriptorType::STORAGE_IMAGE),
        vk::DescriptorPoolSize::default()
            .descriptor_count(ctx.swapchain.images.len() as u32 + 1)
            .ty(vk::DescriptorType::STORAGE_BUFFER),
        vk::DescriptorPoolSize::default()
            .descriptor_count(1)
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
    ];

    let descriptor_pool =
        ctx.create_descriptor_pool((ctx.swapchain.images.len() as u32) + 1, &pool_sizes)?;
    let descriptors = allocate_descriptor_sets(
        &ctx.device,
        &descriptor_pool,
        &descriptor_layout,
        ctx.swapchain.images.len() as u32,
    )?;
    let static_descriptor =
        allocate_descriptor_set(&ctx.device, &descriptor_pool, &static_descriptor_layout)?;
    for i in 0..ctx.swapchain.images.len() {
        let writes = [
            WriteDescriptorSet {
                binding: 0,
                kind: WriteDescriptorSetKind::StorageImage {
                    view: storage_images[i].view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
            WriteDescriptorSet {
                binding: 1,
                kind: WriteDescriptorSetKind::StorageBuffer {
                    buffer: hash_map_buffers[i].inner,
                },
            },
        ];

        update_descriptor_sets(ctx, &descriptors[i], &writes);
    }
    let write = [
        WriteDescriptorSet {
            binding: 0,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: oct_tree_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 1,
            kind: WriteDescriptorSetKind::UniformBuffer {
                buffer: uniform_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 2,
            kind: WriteDescriptorSetKind::CombinedImageSampler {
                view: sky_box.view,
                sampler: *sky_box_sampler,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        },
    ];
    update_descriptor_sets(ctx, &static_descriptor, &write);

    Ok(Pipeline {
        descriptors2: descriptors,
        pipeline,
        layout,
        descriptors: vec![static_descriptor],
    })
}

pub fn create_raytracing_pipeline(
    renderer: &mut Renderer,
    render_pass: &vk::RenderPass,
    hash_map_buffers: &Vec<Buffer>,
    oct_tree_buffer: &Buffer,
    uniform_buffer: &Buffer,
    sky_box: &ImageAndView,
    sky_box_sampler: &vk::Sampler,
    bindings: &[vk::DescriptorSetLayoutBinding],
    sizes: &[vk::DescriptorPoolSize],
    own_writes: &[WriteDescriptorSet],
) -> Result<Pipeline> {
    let mut descriptor_bindings = vec![
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let dynamic_bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

    descriptor_bindings.extend_from_slice(bindings);

    let descriptor_layout = renderer.create_descriptor_set_layout(&descriptor_bindings, &[])?;
    let dynamic_layout = renderer.create_descriptor_set_layout(&dynamic_bindings, &[])?;

    let set_layouts = [descriptor_layout, dynamic_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&[vk::PushConstantRange {
            offset: 0,
            size: 4,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
        }]);
    let layout = unsafe { renderer.device.create_pipeline_layout(&layout_info, None) }?;

    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )];

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&color_blend_attachments)
        .logic_op(vk::LogicOp::COPY)
        .logic_op_enable(false)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_test_enable(false)
        .depth_write_enable(false);

    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .primitive_restart_enable(false)
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&[])
        .vertex_binding_descriptions(&[]);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .cull_mode(vk::CullModeFlags::BACK)
        .depth_clamp_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(false)
        .rasterizer_discard_enable(false);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let entry_point_name: CString = CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .module(renderer.create_shader_module("./src/shaders/default.frag.spv".to_string()))
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&entry_point_name),
        vk::PipelineShaderStageCreateInfo::default()
            .module(renderer.create_shader_module("./src/shaders/default.vert.spv".to_string()))
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(&entry_point_name),
    ];

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .scissor_count(1)
        .viewport_count(1);

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
        .color_blend_state(&color_blend_state)
        .depth_stencil_state(&depth_stencil_state)
        .dynamic_state(&dynamic_state)
        .input_assembly_state(&input_assembly_state)
        .vertex_input_state(&vertex_input_state)
        .layout(layout)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .render_pass(*render_pass)
        .stages(&stages)
        .viewport_state(&viewport_state)
        .subpass(0);
    let pipeline = unsafe {
        renderer
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
            .unwrap()
    }[0];

    let mut static_sizes = vec![
        vk::DescriptorPoolSize::default()
            .descriptor_count(renderer.swapchain.images.len() as u32 + 1)
            .ty(vk::DescriptorType::STORAGE_BUFFER),
        vk::DescriptorPoolSize::default()
            .descriptor_count(1)
            .ty(vk::DescriptorType::UNIFORM_BUFFER),
        vk::DescriptorPoolSize::default()
            .descriptor_count(1)
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
    ];
    static_sizes.extend_from_slice(&sizes);

    let descriptor_pool = renderer
        .create_descriptor_pool(renderer.swapchain.images.len() as u32 + 1, &static_sizes)?;
    let descriptors =
        allocate_descriptor_set(&renderer.device, &descriptor_pool, &descriptor_layout)?;
    let old_descriptors = allocate_descriptor_sets(
        &renderer.device,
        &descriptor_pool,
        &dynamic_layout,
        renderer.swapchain.images.len() as u32,
    )?;

    let writes = [
        WriteDescriptorSet {
            binding: 0,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: oct_tree_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 1,
            kind: WriteDescriptorSetKind::UniformBuffer {
                buffer: uniform_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 2,
            kind: WriteDescriptorSetKind::CombinedImageSampler {
                view: sky_box.view,
                sampler: *sky_box_sampler,
                layout: vk::ImageLayout::GENERAL,
            },
        },
    ];

    update_descriptor_sets(renderer, &descriptors, &own_writes);
    update_descriptor_sets(renderer, &descriptors, &writes);
    for i in 0..renderer.swapchain.images.len() {
        let writes = [WriteDescriptorSet {
            binding: 0,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: hash_map_buffers[i].inner,
            },
        }];
        update_descriptor_sets(renderer, &old_descriptors[i], &writes);
    }

    Ok(Pipeline {
        descriptors2: old_descriptors,
        pipeline,
        descriptors: vec![descriptors],
        layout,
    })
}

pub fn create_main_render_pass(
    renderer: &mut Renderer,
    oct_tree_buffer: &Buffer,
    unifrom_buffer: &Buffer,
    sky_box: &ImageAndView,
    sky_box_sampler: &vk::Sampler,
    bindings: &[vk::DescriptorSetLayoutBinding],
    sizes: &[vk::DescriptorPoolSize],
    own_writes: &[WriteDescriptorSet],
) -> Result<MainPass> {
    let attachments = [
        vk::AttachmentDescription::default()
            .samples(vk::SampleCountFlags::TYPE_1)
            .final_layout(vk::ImageLayout::GENERAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .format(vk::Format::R32_UINT)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE),
        vk::AttachmentDescription::default()
            .samples(vk::SampleCountFlags::TYPE_1)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .format(renderer.swapchain.format)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE),
    ];

    let attachments1 = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let attachments2 = [vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

    let input_attachments = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::GENERAL)];

    let subpasses = [
        vk::SubpassDescription::default()
            .color_attachments(&attachments1)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS),
        vk::SubpassDescription::default()
            .input_attachments(&input_attachments)
            .color_attachments(&attachments2)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS),
    ];

    let dependencys = [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(1)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
    ];

    let render_pass_create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .dependencies(&dependencys)
        .subpasses(&subpasses);

    let render_pass = unsafe {
        renderer
            .device
            .create_render_pass(&render_pass_create_info, None)?
    };

    let (frame_buffers, voxel_index_buffers) = create_frame_buffers(renderer, &render_pass)?;

    let mut hash_map_buffers = vec![];

    for _ in 0..renderer.swapchain.images.len() {
        hash_map_buffers.push(renderer.create_buffer(
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            100000 * 1000,
            Some("hashmap_buffer"),
        )?);
    }

    let ray_tracing_pipeline = create_raytracing_pipeline(
        renderer,
        &render_pass,
        &hash_map_buffers,
        oct_tree_buffer,
        unifrom_buffer,
        sky_box,
        sky_box_sampler,
        bindings,
        sizes,
        own_writes,
    )?;
    let post_proccesing_pipeline = create_post_proccesing_pipeline(
        renderer,
        &render_pass,
        &voxel_index_buffers,
        &hash_map_buffers,
        oct_tree_buffer,
        unifrom_buffer,
        sky_box,
        sky_box_sampler,
    )?;
    let temporal_reuse_pipeline = create_compute_pipeline(renderer, &hash_map_buffers)?;
    Ok(MainPass {
        hash_map_buffers,
        ray_tracing: ray_tracing_pipeline,
        temporal_reuse: temporal_reuse_pipeline,
        post_proccesing: post_proccesing_pipeline,
        render_pass,
        frame_buffers,
        voxel_index_buffers,
    })
}
