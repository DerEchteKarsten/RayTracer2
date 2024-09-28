use std::ffi::CString;

use crate::{
    allocate_descriptor_set, render_recources::RenderResources, update_descriptor_sets, Buffer,
    Model, Renderer, WriteDescriptorSet, WriteDescriptorSetKind,
};
use anyhow::Result;
use ash::vk::{self, AccessFlags, PipelineCache, PipelineStageFlags, Sampler, ShaderStageFlags};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct PrepareLightsTask {
    geometry_index: u32,
    light_buffer_offset: u32,
    triangle_count: u32,
    pad: u32,
}

pub struct PrepareLightsTasks {
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor: vk::DescriptorSet,
}

impl PrepareLightsTasks {
    pub fn new(
        ctx: &mut Renderer,
        model: &Model,
        resources: &RenderResources,
        environment: (&vk::ImageView, &Sampler),
    ) -> Result<Self> {
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

        let descriptor0_layout = ctx.create_descriptor_set_layout(&bindings)?;
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

        let descriptor_set0 =
            allocate_descriptor_set(&ctx.device, &pool, &descriptor0_layout).unwrap();

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
                    buffer: resources.light_buffer.inner,
                },
            },
            WriteDescriptorSet {
                binding: 4,
                kind: WriteDescriptorSetKind::StorageImage {
                    view: resources.local_light_pdf_texture.view,
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
                    buffer: resources.task_buffer.inner,
                },
            },
        ];
        update_descriptor_sets(ctx, &descriptor_set0, &writes);

        Ok(Self {
            descriptor: descriptor_set0,
            handle,
            layout,
        })
    }

    pub fn execute(
        &self,
        ctx: &Renderer,
        cmd: &vk::CommandBuffer,
        light_tasks_buffer: &Buffer,
        geometry_to_light_buffer_and_staging: (&Buffer, &Buffer),
        model: &Model,
    ) {
        unsafe {
            let mut geometry_to_light = vec![0xffffffffu32; model.geometry_infos.len()];
            let mut light_tasks = vec![];
            let mut light_buffer_offset = 0;
            for (geometry_index, geometry) in model.geometry_infos.iter().enumerate() {
                if geometry.emission[0] == 0.0
                    && geometry.emission[1] == 0.0
                    && geometry.emission[2] == 0.0
                {
                    continue;
                }
                geometry_to_light[geometry_index] = light_buffer_offset;
                let triangle_count = model.index_counts[geometry_index] / 3;
                light_tasks.push(PrepareLightsTask {
                    geometry_index: geometry_index as u32,
                    light_buffer_offset,
                    triangle_count,
                    pad: 0,
                });
                light_buffer_offset += triangle_count;
            }

            light_tasks_buffer
                .copy_data_to_buffer(light_tasks.as_slice())
                .unwrap();
            geometry_to_light_buffer_and_staging
                .1
                .copy_data_to_buffer(geometry_to_light.as_slice())
                .unwrap();

            ctx.device.cmd_copy_buffer(
                *cmd,
                geometry_to_light_buffer_and_staging.1.inner,
                geometry_to_light_buffer_and_staging.0.inner,
                &[vk::BufferCopy::default()
                    .dst_offset(0)
                    .src_offset(0)
                    .size(geometry_to_light_buffer_and_staging.1.size)],
            );
            ctx.memory_barrier(
                cmd,
                PipelineStageFlags::TRANSFER,
                PipelineStageFlags::COMPUTE_SHADER,
                AccessFlags::TRANSFER_WRITE,
                AccessFlags::SHADER_READ,
            );
            ctx.device
                .cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::COMPUTE, self.handle);

            let num_tasks = light_tasks.len() as u32;
            let num_tasks =
                std::slice::from_raw_parts(&num_tasks as *const u32 as *const _, size_of::<u32>());

            ctx.device.cmd_push_constants(
                *cmd,
                self.layout,
                ShaderStageFlags::COMPUTE,
                0,
                num_tasks,
            );

            ctx.device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.descriptor],
                &[],
            );
            ctx.device.cmd_dispatch(
                *cmd,
                f32::ceil(light_buffer_offset as f32 / 256.0) as u32,
                1,
                1,
            );
        }
    }
}
