use anyhow::Result;
use ash::vk::{self};
use bevy::prelude::*;
use gpu_allocator::MemoryLocation;

use std::default::Default;
use std::ffi::CString;

use crate::{
    allocate_descriptor_set, allocate_descriptor_sets, update_descriptor_sets, Buffer, Image,
    ImageAndView, Renderer, WriteDescriptorSet, WriteDescriptorSetKind, WINDOW_SIZE,
};

pub fn create_storage_images<'a>(
    ctx: &mut Renderer,
    g_buffer_pipeline: &RayTracingPipeline,
) -> Result<(Vec<vk::Framebuffer>, Vec<ImageAndView>, Vec<ImageAndView>)> {
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
            let attachments = [color_buffers[i].view, depth_buffers[i].view];
            let create_info = vk::FramebufferCreateInfo::default()
                .attachment_count(2)
                .attachments(&attachments)
                .height(WINDOW_SIZE.1)
                .width(WINDOW_SIZE.0)
                .layers(1)
                .render_pass(g_buffer_pipeline.render_pass);
            unsafe { ctx.device.create_framebuffer(&create_info, None) }.unwrap()
        })
        .collect::<Vec<vk::Framebuffer>>();

    Ok((frame_buffers, color_buffers, depth_buffers))
}

#[derive(Resource)]
pub struct PostProccesingPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptors: Vec<vk::DescriptorSet>,
    pub render_pass: vk::RenderPass,
}

pub fn create_post_proccesing_pipelien(
    ctx: &mut Renderer,
    storage_images: &Vec<ImageAndView>,
) -> Result<PostProccesingPipeline> {
    let attachments = [vk::AttachmentDescription::default()
        .samples(vk::SampleCountFlags::TYPE_1)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .format(ctx.swapchain.format)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)];
    let color_attachments = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpasses = [vk::SubpassDescription::default()
        .color_attachments(&color_attachments)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)];

    let dependencys = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

    let render_pass_create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .dependencies(&dependencys)
        .subpasses(&subpasses);

    let render_pass = unsafe {
        ctx.device
            .create_render_pass(&render_pass_create_info, None)?
    };

    let descriptor_bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

    let descriptor_layout = ctx.create_descriptor_set_layout(&descriptor_bindings, &[])?;
    let descriptor_layouts = [descriptor_layout];
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
        .render_pass(render_pass)
        .stages(&stages)
        .viewport_state(&viewport_state)
        .subpass(0);
    let pipeline = unsafe {
        ctx.device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
            .unwrap()
    }[0];

    let pool_sizes = [vk::DescriptorPoolSize::default()
        .descriptor_count(ctx.swapchain.images.len() as u32)
        .ty(vk::DescriptorType::STORAGE_IMAGE)];

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
                view: storage_images[i].view,
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

#[derive(Resource)]
pub struct RayTracingPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub _descriptor_layout: vk::DescriptorSetLayout,
    pub render_pass: vk::RenderPass,
    pub descriptor: vk::DescriptorSet,
}

pub fn create_fullscreen_quad_pipeline(
    ctx: &mut Renderer,
    uniform_buffer: &Buffer,
    oct_tree_buffer: &Buffer,
    bindings: &[vk::DescriptorSetLayoutBinding],
    sizes: &[vk::DescriptorPoolSize],
    own_writes: &[WriteDescriptorSet],
) -> Result<RayTracingPipeline> {
    let attachments = [
        vk::AttachmentDescription::default()
            .samples(vk::SampleCountFlags::TYPE_1)
            .final_layout(vk::ImageLayout::GENERAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE),
        vk::AttachmentDescription::default()
            .samples(vk::SampleCountFlags::TYPE_1)
            .final_layout(vk::ImageLayout::GENERAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .format(vk::Format::R32_SFLOAT)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE),
    ];
    let color_attachments = [
        vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
    ];
    let subpasses = [vk::SubpassDescription::default()
        .color_attachments(&color_attachments)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)];

    let dependencys = [vk::SubpassDependency::default()
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
        )];

    let render_pass_create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .dependencies(&dependencys)
        .subpasses(&subpasses);

    let render_pass = unsafe {
        ctx.device
            .create_render_pass(&render_pass_create_info, None)?
    };

    let mut descriptor_bindings = vec![
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];

    descriptor_bindings.extend_from_slice(bindings);

    let descriptor_layout = ctx.create_descriptor_set_layout(&descriptor_bindings, &[])?;
    let set_layouts = [descriptor_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&[vk::PushConstantRange {
            offset: 0,
            size: 4,
            stage_flags: vk::ShaderStageFlags::FRAGMENT
        }]);
    let layout = unsafe { ctx.device.create_pipeline_layout(&layout_info, None) }?;

    let color_blend_attachments = [
        vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            ),
        vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            ),
    ];

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
            .module(ctx.create_shader_module("./src/shaders/default.frag.spv".to_string()))
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&entry_point_name),
        vk::PipelineShaderStageCreateInfo::default()
            .module(ctx.create_shader_module("./src/shaders/default.vert.spv".to_string()))
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
        .render_pass(render_pass)
        .stages(&stages)
        .viewport_state(&viewport_state)
        .subpass(0);
    let pipeline = unsafe {
        ctx.device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
            .unwrap()
    }[0];

    let mut pool_sizes = vec![
        vk::DescriptorPoolSize::default()
            .descriptor_count(1)
            .ty(vk::DescriptorType::UNIFORM_BUFFER),
        vk::DescriptorPoolSize::default()
            .descriptor_count(2)
            .ty(vk::DescriptorType::STORAGE_BUFFER),
    ];
    pool_sizes.extend_from_slice(sizes);
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
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: oct_tree_buffer.inner,
            },
        },
    ];
    update_descriptor_sets(ctx, &descriptor, &writes);
    update_descriptor_sets(ctx, &descriptor, &own_writes);

    Ok(RayTracingPipeline {
        pipeline,
        render_pass,
        descriptor,
        _descriptor_layout: descriptor_layout,
        layout,
    })
}
