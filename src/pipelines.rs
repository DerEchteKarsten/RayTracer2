use anyhow::{Ok, Result};
use ash::vk::{self, ImageLayout};
use gpu_allocator::MemoryLocation;

use std::{array::from_ref, default::Default};
use std::ffi::CString;

use crate::{
    allocate_descriptor_set, allocate_descriptor_sets, create_descriptor_pool, create_descriptor_set_layout, module_from_bytes, update_descriptor_sets, Buffer, Image, ImageAndView, Model, RayTracingShaderGroupInfo, Renderer, WriteDescriptorSet, WriteDescriptorSetKind, FRAMES_IN_FLIGHT
};

struct RendererGraph {
    pub raytracing_pipeline: RayTracingPipeline,
    pub post_proccesing_pipeline: PostProccesingPipeline,
}
struct PostProccesingPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptors: Vec<vk::DescriptorSet>,
    pub render_pass: vk::RenderPass,
}

pub struct RayTracingPipeline {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub dynamic_layout: vk::DescriptorSetLayout,
    pub dynamic_layout2: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
    pub handle: vk::Pipeline,
    pub shader_group_info: RayTracingShaderGroupInfo,
}


fn create_post_proccesing_pipelien(
    ctx: &mut Renderer,
    storage_images: &Vec<(vk::ImageView, Image)>,
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

    let binding = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpasses = [vk::SubpassDescription::default()
        .color_attachments(&binding)
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

    let binding = [descriptor_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&binding)
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

#[derive(Debug, Clone)]
pub struct RayTracingShaderCreateInfo<'a> {
    pub source: &'a [(&'a [u8], vk::ShaderStageFlags)],
    pub group: RayTracingShaderGroup,
}

#[derive(Debug, Clone, Copy)]
pub enum RayTracingShaderGroup {
    RayGen,
    Miss,
    Hit,
}

fn create_ray_tracing_pipeline(
    ctx: &Renderer,
    model: &Model,
    shaders_create_info: &[RayTracingShaderCreateInfo],
) -> Result<RayTracingPipeline> {
    let static_layout_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(
                vk::ShaderStageFlags::INTERSECTION_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            ),
        vk::DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(5)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(7)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(model.images.len() as _)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(8)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::MISS_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(9)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
    ];

    let dynamic_layout_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
    ];

    let dynamic_layout_bindings2 = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)];

    let static_dsl = ctx.create_descriptor_set_layout(&static_layout_bindings, &[])?;
    let dynamic_dsl = ctx.create_descriptor_set_layout(&dynamic_layout_bindings, &[])?;
    let old_image_dsl = ctx.create_descriptor_set_layout(&dynamic_layout_bindings, &[])?;
    let dynamic_dsl2 = ctx.create_descriptor_set_layout(&dynamic_layout_bindings2, &[])?;
    let dynamic_dsl22 = ctx.create_descriptor_set_layout(&dynamic_layout_bindings2, &[])?;

    let dsls = [
        static_dsl,
        dynamic_dsl,
        old_image_dsl,
        dynamic_dsl2,
        dynamic_dsl22,
    ];

    let push_constants = &[vk::PushConstantRange::default()
        .offset(0)
        .size((size_of::<u32>() * 2) as u32)
        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)];

    let pipe_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&dsls)
        .push_constant_ranges(push_constants);
    let pipeline_layout = unsafe { ctx.device.create_pipeline_layout(&pipe_layout_info, None)? };

    let mut shader_group_info = RayTracingShaderGroupInfo {
        group_count: shaders_create_info.len() as u32,
        ..Default::default()
    };

    let mut modules = vec![];
    let mut stages = vec![];
    let mut groups = vec![];

    let entry_point_name: CString = CString::new("main").unwrap();

    for shader in shaders_create_info.iter() {
        let mut this_modules = vec![];
        let mut this_stages = vec![];

        shader.source.into_iter().for_each(|s| {
            let module = module_from_bytes(&ctx.device, s.0).unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(s.1)
                .module(module)
                .name(&entry_point_name);
            this_modules.push(module);
            this_stages.push(stage);
        });

        match shader.group {
            RayTracingShaderGroup::RayGen => shader_group_info.raygen_shader_count += 1,
            RayTracingShaderGroup::Miss => shader_group_info.miss_shader_count += 1,
            RayTracingShaderGroup::Hit => shader_group_info.hit_shader_count += 1,
        };

        let shader_index = stages.len();

        let mut group = vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR);
        group = match shader.group {
            RayTracingShaderGroup::RayGen | RayTracingShaderGroup::Miss => {
                group.general_shader(shader_index as _)
            }
            RayTracingShaderGroup::Hit => {
                group = group
                    .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                    .closest_hit_shader(shader_index as _);
                if shader.source.len() >= 2 {
                    group = group
                        .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                        .any_hit_shader((shader_index as u32) + 1);
                }
                if shader.source.len() >= 3 {
                    group = group
                        .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                        .any_hit_shader((shader_index as u32) + 1)
                        .intersection_shader((shader_index as u32) + 2);
                }

                group
            }
        };

        modules.append(&mut this_modules);
        stages.append(&mut this_stages);
        groups.push(group);
    }

    let pipe_info = vk::RayTracingPipelineCreateInfoKHR::default()
        .layout(pipeline_layout)
        .stages(&stages)
        .groups(&groups)
        .max_pipeline_ray_recursion_depth(1);

    let inner = unsafe {
        ctx.ray_tracing
            .pipeline_fn
            .create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipe_info),
                None,
            )
            .unwrap()[0]
    };

    let size = ctx.swapchain.images.len() as u32;

    let mut pool_sizes = vec![
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1 + (size * 2)),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(6),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count((model.images.len() as u32) + size),
    ];

    let pool = ctx.create_descriptor_pool((size * 2) + 3, &pool_sizes)?;

    let static_set =
        allocate_descriptor_set(&ctx.device, &pool, &static_dsl)?;
    let dynamic_sets =
        allocate_descriptor_sets(&ctx.device, &pool, &dynamic_dsl, size)?;
    let dynamic_sets2 =
        allocate_descriptor_sets(&ctx.device, &pool, &dynamic_dsl2, 2)?;

    let sampler_create_info: vk::SamplerCreateInfo = vk::SamplerCreateInfo {
        mag_filter: vk::Filter::NEAREST,
        min_filter: vk::Filter::NEAREST,
        mipmap_mode: vk::SamplerMipmapMode::NEAREST,
        address_mode_u: vk::SamplerAddressMode::MIRRORED_REPEAT,
        address_mode_v: vk::SamplerAddressMode::MIRRORED_REPEAT,
        address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
        max_anisotropy: 1.0,
        border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
        compare_op: vk::CompareOp::NEVER,
        ..Default::default()
    };

    let sampler = unsafe { ctx.device.create_sampler(&sampler_create_info, None)? };

    for i in 0..ctx.swapchain.images.len() {
        let write = WriteDescriptorSet {
            binding: 2,
            kind: WriteDescriptorSetKind::StorageImage {
                view: storage_images[i].0.clone(),
                layout: vk::ImageLayout::GENERAL,
            },
        };

        update_descriptor_sets(&mut ctx, &dynamic_sets[i], &[write]);
    }

    let w = vec![
        WriteDescriptorSet {
            binding: 0,
            kind: WriteDescriptorSetKind::AccelerationStructure {
                acceleration_structure: top_as.handle,
            },
        },
        WriteDescriptorSet {
            binding: 2,
            kind: WriteDescriptorSetKind::UniformBuffer {
                buffer: ubo_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 4,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: model.vertex_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 5,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: model.index_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 6,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: model.geometry_info_buffer.inner,
            },
        },
    ];

    update_descriptor_sets(&mut ctx, &static_set, w.as_slice());
    for i in 0..temporal_buffers.len() {
        let write = WriteDescriptorSet {
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: temporal_buffers[i].inner.clone(),
            },
            binding: 0,
        };
        update_descriptor_sets(&mut ctx, &dynamic_sets2[i], &[write]);
    }

    for (i, (image_index, sampler_index)) in model.textures.iter().enumerate() {
        let view = &model.views[*image_index];
        let sampler = &model.samplers[*sampler_index];
        let img_info = vk::DescriptorImageInfo::default()
            .image_view(*view)
            .sampler(*sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        unsafe {
            ctx.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_array_element(i as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_binding(7)
                    .dst_set(static_set.clone())
                    .image_info(from_ref(&img_info))],
                &[],
            )
        };
    }

    Ok(RayTracingPipeline {
        handle: inner,
        descriptor_set_layout: static_dsl,
        dynamic_layout: dynamic_dsl,
        layout: pipeline_layout,
        shader_group_info,
        dynamic_layout2: dynamic_dsl2,
    })
}