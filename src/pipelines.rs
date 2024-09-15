use anyhow::{Ok, Result};
use ash::vk::{self, BufferUsageFlags, ImageLayout, PipelineCache, ShaderStageFlags};
use glam::Vec2;
use gpu_allocator::MemoryLocation;
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng};
use ndarray::{arr3, prelude::*};

use std::ffi::CString;
use std::mem::size_of;
use std::{array::from_ref, default::Default};

use crate::shader_params::{GConst, RTXDI_ReservoirBufferParameters};
use crate::{
    allocate_descriptor_set, allocate_descriptor_sets, create_descriptor_pool,
    create_descriptor_set_layout, module_from_bytes, update_descriptor_sets, AccelerationStructure,
    Buffer, Image, ImageAndView, Model, RayTracingShaderGroupInfo, Renderer, ShaderBindingTable,
    WriteDescriptorSet, WriteDescriptorSetKind, FRAMES_IN_FLIGHT, NEIGHBOR_OFFSET_COUNT,
    RTXDI_RESERVOIR_BLOCK_SIZE, WINDOW_SIZE,
};

pub struct GBuffer {
    pub depth: ImageAndView,
    pub normal: ImageAndView,
    pub geo_normals: ImageAndView,
    pub diffuse_albedo: ImageAndView,
    pub specular_rough: ImageAndView,
    pub motion_vectors: ImageAndView,
}

pub struct RendererResources {
    pub raytracing_pipeline: RayTracingPipeline,
    pub post_proccesing_pipeline: PostProccesingPipeline,
    pub g_buffer: Vec<GBuffer>,
    pub reservoirs: Buffer,
    pub neighbors: Buffer,
    pub shader_binding_table: ShaderBindingTable,
    pub uniform_buffer: Buffer,
    pub inference_pipeline: InferencePipeline,
    pub input_buffers: Buffer,
    pub output_buffers: Buffer,
    pub weights_buffer: Buffer,
    pub exspected_outputs: Array2<f32>,
}

pub struct PostProccesingPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub dynamic_descriptors: Vec<vk::DescriptorSet>,
    pub static_descriptor: vk::DescriptorSet,

    pub render_pass: vk::RenderPass,
    pub frame_buffers: Vec<vk::Framebuffer>,
}

pub struct RayTracingPipeline {
    pub shader_group_info: RayTracingShaderGroupInfo,
    pub layout: vk::PipelineLayout,
    pub handle: vk::Pipeline,
    pub descriptor_set0: vk::DescriptorSet,
    pub descriptor_set1: Vec<vk::DescriptorSet>,
    pub descriptor_set2: Vec<vk::DescriptorSet>,
}

pub struct InferencePipeline {
    pub layout: vk::PipelineLayout,
    pub handle: vk::Pipeline,
    pub descriptor_set0: vk::DescriptorSet,
}

pub fn create_inference_pipeline(
    ctx: &mut Renderer,
    input_buffer: &Buffer,
    output_buffer: &Buffer,
    weights_buffer: &Buffer,
) -> Result<InferencePipeline> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];

    let descriptor0_layout = ctx.create_descriptor_set_layout(&bindings, &[])?;

    let descriptor_layouts = [descriptor0_layout];

    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(&[])
        .set_layouts(&descriptor_layouts);

    let layout = unsafe { ctx.device.create_pipeline_layout(&layout_info, None)? };

    let entry_point_name: CString = CString::new("main").unwrap();
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .layout(layout)
        .stage(
            vk::PipelineShaderStageCreateInfo::default()
                .module(ctx.create_shader_module("./src/shaders/training.comp.spv".to_string()))
                .stage(vk::ShaderStageFlags::COMPUTE)
                .name(&entry_point_name),
        );

    let handle = unsafe {
        ctx.device
            .create_compute_pipelines(PipelineCache::null(), &[pipeline_info], None)
            .unwrap()[0]
    };

    let pool = ctx
        .create_descriptor_pool(
            1,
            &[vk::DescriptorPoolSize::default()
                .descriptor_count(3)
                .ty(vk::DescriptorType::STORAGE_BUFFER)],
        )
        .unwrap();

    let descriptor_set0 = allocate_descriptor_set(&ctx.device, &pool, &descriptor0_layout).unwrap();

    let writes = [
        WriteDescriptorSet {
            binding: 0,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: input_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 1,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: output_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 2,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: weights_buffer.inner,
            },
        },
    ];

    update_descriptor_sets(ctx, &descriptor_set0, &writes);

    Ok(InferencePipeline {
        descriptor_set0,
        handle,
        layout,
    })
}

fn create_post_proccesing_pipelien(
    ctx: &mut Renderer,
    g_buffer: &Vec<GBuffer>,
    skybox_sampler: &vk::Sampler,
    skybox_view: &vk::ImageView,
    uniform_buffer: &Buffer,
    reservoirs: &Buffer,
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

    let static_bindings = [
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
    let dynamic_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(5)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];

    let static_layout = ctx.create_descriptor_set_layout(&static_bindings, &[])?;
    let dynamic_layout = ctx.create_descriptor_set_layout(&dynamic_bindings, &[])?;

    let binding = [static_layout, dynamic_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&binding)
        .push_constant_ranges(&[vk::PushConstantRange {
            offset: 0,
            size: 4,
            stage_flags: ShaderStageFlags::FRAGMENT,
        }]);
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

    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .descriptor_count(6 * 2)
            .ty(vk::DescriptorType::STORAGE_IMAGE),
        vk::DescriptorPoolSize::default()
            .descriptor_count(1)
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        vk::DescriptorPoolSize::default()
            .descriptor_count(1)
            .ty(vk::DescriptorType::UNIFORM_BUFFER),
        vk::DescriptorPoolSize::default()
            .descriptor_count(1)
            .ty(vk::DescriptorType::STORAGE_BUFFER),
    ];

    let descriptor_pool = ctx.create_descriptor_pool(3, &pool_sizes)?;
    let dynamic_descriptors =
        allocate_descriptor_sets(&ctx.device, &descriptor_pool, &dynamic_layout, 2)?;

    let static_descriptor = allocate_descriptor_set(&ctx.device, &descriptor_pool, &static_layout)?;

    let writes = [
        WriteDescriptorSet {
            binding: 0,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: reservoirs.inner,
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
                sampler: *skybox_sampler,
                view: *skybox_view,
                layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        },
    ];

    update_descriptor_sets(ctx, &static_descriptor, &writes);

    for i in 0..2 {
        let writes = [
            WriteDescriptorSet {
                binding: 0,
                kind: WriteDescriptorSetKind::StorageImage {
                    view: g_buffer[i].depth.view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
            WriteDescriptorSet {
                binding: 1,
                kind: WriteDescriptorSetKind::StorageImage {
                    view: g_buffer[i].normal.view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
            WriteDescriptorSet {
                binding: 2,
                kind: WriteDescriptorSetKind::StorageImage {
                    view: g_buffer[i].geo_normals.view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
            WriteDescriptorSet {
                binding: 3,
                kind: WriteDescriptorSetKind::StorageImage {
                    view: g_buffer[i].diffuse_albedo.view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
            WriteDescriptorSet {
                binding: 4,
                kind: WriteDescriptorSetKind::StorageImage {
                    view: g_buffer[i].specular_rough.view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
            WriteDescriptorSet {
                binding: 5,
                kind: WriteDescriptorSetKind::StorageImage {
                    view: g_buffer[i].motion_vectors.view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
        ];

        update_descriptor_sets(ctx, &dynamic_descriptors[i], &writes);
    }

    let mut frame_buffers = vec![];
    for i in 0..ctx.swapchain.images.len() {
        let attachments = [ctx.swapchain.images[i].view];
        let frame_buffer_create_info = vk::FramebufferCreateInfo::default()
            .attachments(&attachments)
            .layers(1)
            .width(WINDOW_SIZE.x as u32)
            .height(WINDOW_SIZE.y as u32)
            .render_pass(render_pass)
            .attachment_count(1);
        let frame_buffer = unsafe {
            ctx.device
                .create_framebuffer(&frame_buffer_create_info, None)
        }?;
        frame_buffers.push(frame_buffer);
    }

    Ok(PostProccesingPipeline {
        pipeline,
        layout,
        render_pass,
        dynamic_descriptors,
        frame_buffers,
        static_descriptor,
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
    ctx: &mut Renderer,
    model: &Model,
    top_as: &AccelerationStructure,
    skybox_view: &vk::ImageView,
    skybox_sampler: &vk::Sampler,
    reservoirs: &Buffer,
    uniform_buffer: &Buffer,
    neighbors: &Buffer,
    g_buffer: &Vec<GBuffer>,
    shaders_create_info: &[RayTracingShaderCreateInfo],
) -> Result<RayTracingPipeline> {
    let static_layout_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::INTERSECTION_KHR | vk::ShaderStageFlags::RAYGEN_KHR),
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
    ];

    let dynamic_layout_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(5)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
    ];

    let static_dsl = ctx.create_descriptor_set_layout(&static_layout_bindings, &[])?;
    let dynamic_dsl = ctx.create_descriptor_set_layout(&dynamic_layout_bindings, &[])?;
    let old_image_dsl = ctx.create_descriptor_set_layout(&dynamic_layout_bindings, &[])?;

    let dsls = [static_dsl, dynamic_dsl, old_image_dsl];

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

    let mut pool_sizes = vec![
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(6 * 4),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(5),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count((model.images.len() as u32) + 1),
    ];

    let pool = ctx.create_descriptor_pool(1 + 2 + 2, &pool_sizes)?;

    let static_set = allocate_descriptor_set(&ctx.device, &pool, &static_dsl)?;
    let dynamic_sets = allocate_descriptor_sets(&ctx.device, &pool, &dynamic_dsl, 2)?;
    let dynamic_sets2 = allocate_descriptor_sets(&ctx.device, &pool, &old_image_dsl, 2)?;

    let writes = [
        WriteDescriptorSet {
            binding: 0,
            kind: WriteDescriptorSetKind::AccelerationStructure {
                acceleration_structure: top_as.handle,
            },
        },
        WriteDescriptorSet {
            binding: 2,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: neighbors.inner,
            },
        },
        WriteDescriptorSet {
            binding: 3,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: reservoirs.inner,
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
        WriteDescriptorSet {
            binding: 8,
            kind: WriteDescriptorSetKind::CombinedImageSampler {
                view: *skybox_view,
                sampler: *skybox_sampler,
                layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        },
    ];

    update_descriptor_sets(ctx, &static_set, &writes);

    let write = [WriteDescriptorSet {
        binding: 1,
        kind: WriteDescriptorSetKind::UniformBuffer {
            buffer: uniform_buffer.inner,
        },
    }];
    update_descriptor_sets(ctx, &static_set, &write);

    for i in 0..2 {
        let writes = [
            WriteDescriptorSet {
                binding: 0,
                kind: WriteDescriptorSetKind::StorageImage {
                    view: g_buffer[i].depth.view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
            WriteDescriptorSet {
                binding: 1,
                kind: WriteDescriptorSetKind::StorageImage {
                    view: g_buffer[i].normal.view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
            WriteDescriptorSet {
                binding: 2,
                kind: WriteDescriptorSetKind::StorageImage {
                    view: g_buffer[i].geo_normals.view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
            WriteDescriptorSet {
                binding: 3,
                kind: WriteDescriptorSetKind::StorageImage {
                    view: g_buffer[i].diffuse_albedo.view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
            WriteDescriptorSet {
                binding: 4,
                kind: WriteDescriptorSetKind::StorageImage {
                    view: g_buffer[i].specular_rough.view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
            WriteDescriptorSet {
                binding: 5,
                kind: WriteDescriptorSetKind::StorageImage {
                    view: g_buffer[i].motion_vectors.view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
        ];
        update_descriptor_sets(ctx, &dynamic_sets[i], &writes);
        update_descriptor_sets(ctx, &dynamic_sets2[i], &writes);
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
        layout: pipeline_layout,
        descriptor_set0: static_set,
        descriptor_set1: dynamic_sets,
        descriptor_set2: dynamic_sets2,
        shader_group_info,
    })
}

fn fill_neighbor_offset_buffer(neighbor_offset_count: u32) -> Vec<u8> {
    let mut buffer = vec![];

    let R = 250;
    let phi2 = 1.0 / 1.3247179572447;
    let mut u = 0.5;
    let mut v = 0.5;
    while buffer.len() < (neighbor_offset_count * 2) as usize {
        u += phi2;
        v += phi2 * phi2;
        if u >= 1.0 {
            u -= 1.0;
        }
        if v >= 1.0 {
            v -= 1.0;
        }

        let rSq = (u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5);
        if rSq > 0.25 {
            continue;
        }

        buffer.push(((u - 0.5) * R as f32) as u8);
        buffer.push(((v - 0.5) * R as f32) as u8);
    }

    buffer
}

pub fn create_render_recources(
    ctx: &mut Renderer,
    model: &Model,
    top_as: &AccelerationStructure,
    skybox_view: vk::ImageView,
    reservoir_buffer_size: u64,
) -> Result<RendererResources> {
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

    let skybox_sampler = unsafe {
        ctx.device
            .create_sampler(&sampler_create_info, None)
            .unwrap()
    };

    let width = WINDOW_SIZE.x as u32;
    let height = WINDOW_SIZE.y as u32;

    let mut g_buffer = vec![];
    for i in 0..2 {
        let depth_image = ctx
            .create_image(
                vk::ImageUsageFlags::STORAGE,
                MemoryLocation::GpuOnly,
                vk::Format::R32_SFLOAT,
                width,
                height,
            )
            .unwrap();
        Renderer::transition_image_layout_to_general(
            &ctx.device,
            &ctx.command_pool,
            &depth_image,
            &ctx.graphics_queue,
        )
        .unwrap();
        let depth_image_view = ctx.create_image_view(&depth_image).unwrap();

        let normal_image = ctx
            .create_image(
                vk::ImageUsageFlags::STORAGE,
                MemoryLocation::GpuOnly,
                vk::Format::R32_UINT,
                width,
                height,
            )
            .unwrap();
        Renderer::transition_image_layout_to_general(
            &ctx.device,
            &ctx.command_pool,
            &normal_image,
            &ctx.graphics_queue,
        )
        .unwrap();
        let normal_image_view = ctx.create_image_view(&normal_image).unwrap();

        let geo_normal_image = ctx
            .create_image(
                vk::ImageUsageFlags::STORAGE,
                MemoryLocation::GpuOnly,
                vk::Format::R32_UINT,
                width,
                height,
            )
            .unwrap();
        Renderer::transition_image_layout_to_general(
            &ctx.device,
            &ctx.command_pool,
            &geo_normal_image,
            &ctx.graphics_queue,
        )
        .unwrap();
        let geo_normal_image_view = ctx.create_image_view(&geo_normal_image).unwrap();

        let diffuse_albedo_image = ctx
            .create_image(
                vk::ImageUsageFlags::STORAGE,
                MemoryLocation::GpuOnly,
                vk::Format::R32_UINT,
                width,
                height,
            )
            .unwrap();
        Renderer::transition_image_layout_to_general(
            &ctx.device,
            &ctx.command_pool,
            &diffuse_albedo_image,
            &ctx.graphics_queue,
        )
        .unwrap();
        let diffuse_albedo_image_view = ctx.create_image_view(&diffuse_albedo_image).unwrap();

        let specular_rough_image = ctx
            .create_image(
                vk::ImageUsageFlags::STORAGE,
                MemoryLocation::GpuOnly,
                vk::Format::R32_UINT,
                width,
                height,
            )
            .unwrap();
        Renderer::transition_image_layout_to_general(
            &ctx.device,
            &ctx.command_pool,
            &specular_rough_image,
            &ctx.graphics_queue,
        )
        .unwrap();
        let specular_rough_image_view = ctx.create_image_view(&specular_rough_image)?;

        let motion_vectors = ctx
            .create_image(
                vk::ImageUsageFlags::STORAGE,
                MemoryLocation::GpuOnly,
                vk::Format::R32G32B32A32_SFLOAT,
                width,
                height,
            )
            .unwrap();
        Renderer::transition_image_layout_to_general(
            &ctx.device,
            &ctx.command_pool,
            &motion_vectors,
            &ctx.graphics_queue,
        )
        .unwrap();
        let motion_vectors_image_view = ctx.create_image_view(&motion_vectors).unwrap();

        let g_buffe = GBuffer {
            depth: ImageAndView {
                image: depth_image,
                view: depth_image_view,
            },
            diffuse_albedo: ImageAndView {
                image: diffuse_albedo_image,
                view: diffuse_albedo_image_view,
            },
            geo_normals: ImageAndView {
                image: geo_normal_image,
                view: geo_normal_image_view,
            },
            motion_vectors: ImageAndView {
                image: motion_vectors,
                view: motion_vectors_image_view,
            },
            normal: ImageAndView {
                image: normal_image,
                view: normal_image_view,
            },
            specular_rough: ImageAndView {
                image: specular_rough_image,
                view: specular_rough_image_view,
            },
        };
        g_buffer.push(g_buffe);
    }

    let uniform_buffer = ctx
        .create_buffer(
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            size_of::<GConst>() as u64,
            None,
        )
        .unwrap();
    let reservoirs = ctx
        .create_buffer(
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            reservoir_buffer_size,
            None,
        )
        .unwrap();
    let neighbor_offsets = fill_neighbor_offset_buffer(NEIGHBOR_OFFSET_COUNT);
    let neighbors = ctx
        .create_gpu_only_buffer_from_data(
            vk::BufferUsageFlags::STORAGE_BUFFER,
            neighbor_offsets.as_slice(),
            None,
        )
        .unwrap();

    let post_proccesing_pipeline = create_post_proccesing_pipelien(
        ctx,
        &g_buffer,
        &skybox_sampler,
        &skybox_view,
        &uniform_buffer,
        &reservoirs,
    )
    .unwrap();

    let shaders_create_info = [
        RayTracingShaderCreateInfo {
            source: &[(
                &include_bytes!("./shaders/raygen.rgen.spv")[..],
                vk::ShaderStageFlags::RAYGEN_KHR,
            )],
            group: RayTracingShaderGroup::RayGen,
        },
        RayTracingShaderCreateInfo {
            source: &[(
                &include_bytes!("./shaders/spatial_resampling.rgen.spv")[..],
                vk::ShaderStageFlags::RAYGEN_KHR,
            )],
            group: RayTracingShaderGroup::RayGen,
        },
        RayTracingShaderCreateInfo {
            source: &[(
                &include_bytes!("./shaders/temporal_resampling.rgen.spv")[..],
                vk::ShaderStageFlags::RAYGEN_KHR,
            )],
            group: RayTracingShaderGroup::RayGen,
        },
        RayTracingShaderCreateInfo {
            source: &[(
                &include_bytes!("./shaders/raymiss.rmiss.spv")[..],
                vk::ShaderStageFlags::MISS_KHR,
            )],
            group: RayTracingShaderGroup::Miss,
        },
        RayTracingShaderCreateInfo {
            source: &[(
                &include_bytes!("./shaders/rayhit.rchit.spv")[..],
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )],
            group: RayTracingShaderGroup::Hit,
        },
    ];

    let raytracing_pipeline = create_ray_tracing_pipeline(
        ctx,
        model,
        top_as,
        &skybox_view,
        &skybox_sampler,
        &reservoirs,
        &uniform_buffer,
        &neighbors,
        &g_buffer,
        &shaders_create_info,
    )
    .unwrap();

    let shader_binding_table = ctx
        .create_shader_binding_table(
            &raytracing_pipeline.handle,
            &raytracing_pipeline.shader_group_info,
        )
        .unwrap();

    // let input_buffers = ctx.create_buffer(BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, 64*64*4, None).unwrap();
    // let output_buffers = ctx.create_buffer(BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, 64*3*4, None).unwrap();
    // let weights_buffer = ctx.create_buffer(BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, 64*64, None).unwrap();

    let inputs = Array2::from_shape_fn((64, 64), |(x, y)| {
        (0.5 - rand::random::<f32>()) * 2.0  
    });

    let weights = Array3::from_shape_fn((7, 64, 64), |(i, x, y)| {
        (0.5 - rand::random::<f32>()) * 2.0  
    });

    let mut outputs = Array2::<f32>::zeros(Dim([64, 64]));
    let mut final_outputs = Array2::<f32>::zeros(Dim([64, 3]));
    let mut last_outputs = inputs.to_owned().clone();
    for i in 0..7 as usize {
        if i == 6 {
            let current_weights = weights.index_axis(Axis(0), i);
            let current_weights = current_weights.slice(s![0..3, ..]);
            for x in 0..64 as usize {
                for y in 0..3 as usize {
                    let mut dotp = 0.0;
                    for other in 0..64 as usize {
                        dotp += current_weights[[y,other]] * last_outputs[[x, other]];
                    }
                    final_outputs[[x,y]] = f32::max(dotp, 0.0);
                }
            }
        }else {
            let current_weights = weights.index_axis(Axis(0), i);
            for x in 0..64 as usize {
                for y in 0..64 as usize {
                    let mut dopt = 0.0;
                    for other in 0..64 as usize {
                        dopt += current_weights[[y, other]] * last_outputs[[x, other]];
                    }
                    outputs[[x,y]] = f32::max(dopt, 0.0);
                }
            }
            last_outputs = outputs.clone();
        }
    }
    println!("{:?}", final_outputs);

    let input_buffer =
        ctx.create_gpu_only_buffer_from_data(BufferUsageFlags::STORAGE_BUFFER, inputs.flatten().as_slice().unwrap(), None)?;
    let output_buffer = ctx.create_gpu_only_buffer_from_data(
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_SRC,
        &[0.0 as f32; 64 * 3],
        None,
    )?;
    let weights_buffer = ctx.create_gpu_only_buffer_from_data(
        BufferUsageFlags::STORAGE_BUFFER,
        weights.flatten().as_slice().unwrap(),
        None,
    )?;

    let inference_pipeline =
        create_inference_pipeline(ctx, &input_buffer, &output_buffer, &weights_buffer).unwrap();

    Ok(RendererResources {
        g_buffer,
        neighbors,
        post_proccesing_pipeline,
        raytracing_pipeline,
        reservoirs,
        shader_binding_table,
        uniform_buffer,
        inference_pipeline,
        input_buffers: input_buffer,
        output_buffers: output_buffer,
        weights_buffer,
        exspected_outputs: final_outputs,
    })
}

fn gen_rand_arr<const SIZE: usize>(rng: &mut ThreadRng) -> [f16; SIZE] {
    let mut arr = [0.0; SIZE];
    for x in &mut arr {
        *x = rng.gen::<f32>() as f16;
    }
    arr
}

pub fn CalculateReservoirBufferParameters(
    renderWidth: u32,
    renderHeight: u32,
) -> RTXDI_ReservoirBufferParameters {
    let renderWidthBlocks = (renderWidth + RTXDI_RESERVOIR_BLOCK_SIZE - 1) / RTXDI_RESERVOIR_BLOCK_SIZE;
    let renderHeightBlocks = (renderHeight + RTXDI_RESERVOIR_BLOCK_SIZE - 1) / RTXDI_RESERVOIR_BLOCK_SIZE;
    let mut params = RTXDI_ReservoirBufferParameters::default();
    params.reservoirBlockRowPitch = renderWidthBlocks * (RTXDI_RESERVOIR_BLOCK_SIZE * RTXDI_RESERVOIR_BLOCK_SIZE);
    params.reservoirArrayPitch = params.reservoirBlockRowPitch * renderHeightBlocks;
    return params;
}