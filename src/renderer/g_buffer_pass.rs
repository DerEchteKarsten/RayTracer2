use std::default;

use anyhow::Result;
use ash::vk::{self, DescriptorSet, Pipeline, PipelineCache, PipelineLayout};

use crate::WINDOW_SIZE;

use super::{
    render_resources::RenderResources,
    renderer::{
        calculate_pool_sizes, Buffer, CalculatePoolSizesDesc, Renderer, WriteDescriptorSet,
    },
};

pub(super) struct GBufferPass {
    pub(super) layout: PipelineLayout,
    pub(super) beam_coarse_tracing: Pipeline,
    pub(super) beam_fine_tracing: Pipeline,
    pub(super) static_descriptors: vk::DescriptorSet,
    pub(super) dynamic_descriptors: Vec<vk::DescriptorSet>,
}

impl GBufferPass {
    pub(super) fn new(
        ctx: &Renderer,
        resources: &RenderResources,
        oct_tree_buffer: &Buffer,
        uniform_buffer: &Buffer,
    ) -> Result<Self> {
        let dynamic_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let static_bindings = [
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
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dyn_layout = ctx.create_descriptor_set_layout(&dynamic_bindings).unwrap();
        let static_layout = ctx.create_descriptor_set_layout(&static_bindings).unwrap();

        let binding = [static_layout, dyn_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&binding);
        let layout = unsafe {
            ctx.device
                .create_pipeline_layout(&layout_info, None)
                .unwrap()
        };

        let descriptor_pool = ctx
            .create_descriptor_pool(&[
                CalculatePoolSizesDesc {
                    bindings: &dynamic_bindings,
                    num_sets: 2,
                },
                CalculatePoolSizesDesc {
                    bindings: &static_bindings,
                    num_sets: 1,
                },
            ])
            .unwrap();

        let dynamic_descriptors = ctx
            .allocate_descriptor_sets(&descriptor_pool, &dyn_layout, 2)
            .unwrap();
        let static_descriptors = ctx
            .allocate_descriptor_sets(&descriptor_pool, &static_layout, 1)
            .unwrap()[0];

        for i in 0..2 {
            let writes = [WriteDescriptorSet {
                binding: 0,
                kind: super::renderer::WriteDescriptorSetKind::StorageImage {
                    view: resources.g_buffer[i].view,
                    layout: vk::ImageLayout::GENERAL,
                },
            }];
            ctx.update_descriptor_sets(&dynamic_descriptors[i as usize], &writes);
        }

        let writes = [
            WriteDescriptorSet {
                binding: 0,
                kind: super::renderer::WriteDescriptorSetKind::StorageBuffer {
                    buffer: oct_tree_buffer.inner,
                },
            },
            WriteDescriptorSet {
                binding: 1,
                kind: super::renderer::WriteDescriptorSetKind::StorageBuffer {
                    buffer: resources.hash_map_buffer.inner,
                },
            },
            WriteDescriptorSet {
                binding: 2,
                kind: super::renderer::WriteDescriptorSetKind::UniformBuffer {
                    buffer: uniform_buffer.inner,
                },
            },
        ];
        ctx.update_descriptor_sets(&static_descriptors, &writes);

        Ok(Self {
            beam_coarse_tracing: ctx
                .create_compute_pipeline("./src/shaders/bin/beam_coarse.comp.spv", layout),
            beam_fine_tracing: ctx
                .create_compute_pipeline("./src/shaders/bin/beam_fine.comp.spv", layout),
            dynamic_descriptors,
            static_descriptors,
            layout,
        })
    }

    pub(super) fn execute(&self, ctx: &Renderer, cmd: &vk::CommandBuffer, frame: u64) {
        unsafe {
            ctx.device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[
                    self.static_descriptors,
                    self.dynamic_descriptors[(frame % 2) as usize],
                ],
                &[],
            );

            ctx.memory_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::MEMORY_WRITE,
            );

            ctx.device.cmd_bind_pipeline(
                *cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.beam_coarse_tracing,
            );
            ctx.device
                .cmd_dispatch(*cmd, WINDOW_SIZE.0 / 16, WINDOW_SIZE.1 / 16, 1);

            ctx.memory_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::MEMORY_WRITE,
            );

            ctx.device.cmd_bind_pipeline(
                *cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.beam_fine_tracing,
            );
            ctx.device
                .cmd_dispatch(*cmd, WINDOW_SIZE.0 / 8, WINDOW_SIZE.1 / 8, 1);
        }
    }
}
