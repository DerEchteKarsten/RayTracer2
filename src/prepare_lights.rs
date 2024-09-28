use std::ffi::CString;

use anyhow::Result;
use ash::vk::{self, BufferUsageFlags, PipelineCache, Sampler, ShaderStageFlags};
use gpu_allocator::MemoryLocation;

use crate::{allocate_descriptor_set, update_descriptor_sets, Buffer, ImageAndView, Model, Renderer, WriteDescriptorSet, WriteDescriptorSetKind};

struct PrepareLightsTask {
    task_buffer: Buffer,
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor: vk::DescriptorSet,
}

impl PrepareLightsTask {
    fn new(ctx: &mut Renderer, model: &Model, lights_buffer: &Buffer, local_lights_pdf: &ImageAndView, environment: (&vk::ImageView, &Sampler)) -> Result<Self> {
        let bindings = vec![
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
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
    
        let descriptor0_layout = ctx.create_descriptor_set_layout(&bindings, &[])?;
        let descriptor_layouts = [descriptor0_layout];
    
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&[vk::PushConstantRange {
                offset: 0,
                size: size_of::<u32>() as u32,
                stage_flags: ShaderStageFlags::COMPUTE,
            }])
            .set_layouts(&descriptor_layouts);
    
        let layout = unsafe { ctx.device.create_pipeline_layout(&layout_info, None)? };
    
        let entry_point_name: CString = CString::new("main").unwrap();
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .layout(layout)
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .module(ctx.create_shader_module("./src/shaders/prepare_lights.comp.spv"))
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
                &[
                    vk::DescriptorPoolSize::default()
                        .descriptor_count(1)
                        .ty(vk::DescriptorType::STORAGE_IMAGE),
                    vk::DescriptorPoolSize::default()
                        .descriptor_count(5)
                        .ty(vk::DescriptorType::STORAGE_BUFFER),
                    vk::DescriptorPoolSize::default()
                        .descriptor_count(1)
                        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                ],
            )
            .unwrap();
    
        let descriptor_set0 = allocate_descriptor_set(&ctx.device, &pool, &descriptor0_layout).unwrap();
    
        let task_buffer = ctx
            .create_buffer(
                BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                16 * 500,
                None,
            )
            .unwrap();

        let writes = [
            WriteDescriptorSet {
                binding: 0,
                kind: WriteDescriptorSetKind::StorageBuffer {
                    buffer: model.geometry_info_buffer.inner,
                },
            },
            WriteDescriptorSet {
                binding: 1,
                kind: WriteDescriptorSetKind::StorageBuffer {
                    buffer: model.vertex_buffer.inner,
                },
            },
            WriteDescriptorSet {
                binding: 2,
                kind: WriteDescriptorSetKind::StorageBuffer {
                    buffer: model.index_buffer.inner,
                },
            },
            WriteDescriptorSet {
                binding: 3,
                kind: WriteDescriptorSetKind::StorageBuffer {
                    buffer: lights_buffer.inner,
                },
            },
            WriteDescriptorSet {
                binding: 4,
                kind: WriteDescriptorSetKind::StorageImage {
                    view: local_lights_pdf.view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
            WriteDescriptorSet {
                binding: 5,
                kind: WriteDescriptorSetKind::CombinedImageSampler {
                    view: *environment.0,
                    sampler: *environment.1,
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
            },
            WriteDescriptorSet {
                binding: 6,
                kind: WriteDescriptorSetKind::StorageBuffer {
                    buffer: task_buffer.inner,
                },
            },
        ];
        update_descriptor_sets(ctx, &descriptor_set0, &writes);
    
        Ok(Self {
            task_buffer,
            descriptor: descriptor_set0,
            handle,
            layout,
        })
    }
}
