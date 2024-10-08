use anyhow::Result;
use ash::vk::{self, Pipeline, PipelineLayout};

use crate::WINDOW_SIZE;

use super::{
    render_resources::RenderResources,
    renderer::{Buffer, CalculatePoolSizesDesc, ImageBarrier, Renderer, WriteDescriptorSet},
};

pub(super) struct PostProcessingPass {
    pub(super) layout: PipelineLayout,
    pub(super) pipeline: Pipeline,
    pub(super) static_descriptors: vk::DescriptorSet,
    pub(super) dynamic_descriptors: Vec<vk::DescriptorSet>,
    pub(super) swapchain_image_descriptors: Vec<vk::DescriptorSet>,
}

impl PostProcessingPass {
    pub(super) fn new(
        ctx: &Renderer,
        resources: &RenderResources,
        uniform_buffer: &Buffer,
        skybox: &vk::ImageView,
        skybox_sampler: &vk::Sampler,
    ) -> Result<Self> {
        let images = ctx.swapchain.images.len() as u32;
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
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dyn_layout = ctx.create_descriptor_set_layout(&dynamic_bindings).unwrap();
        let static_layout = ctx.create_descriptor_set_layout(&static_bindings).unwrap();

        let binding = [static_layout, dyn_layout, dyn_layout];
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
                    num_sets: 2 + images,
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
        let swapchain_image_descriptors = ctx
            .allocate_descriptor_sets(&descriptor_pool, &dyn_layout, images)
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

        for i in 0..images {
            let writes = [WriteDescriptorSet {
                binding: 0,
                kind: super::renderer::WriteDescriptorSetKind::StorageImage {
                    view: ctx.swapchain.images[i as usize].view,
                    layout: vk::ImageLayout::GENERAL,
                },
            }];
            ctx.update_descriptor_sets(&swapchain_image_descriptors[i as usize], &writes);
        }

        let writes = [
            WriteDescriptorSet {
                binding: 0,
                kind: super::renderer::WriteDescriptorSetKind::StorageBuffer {
                    buffer: resources.hash_map_buffer.inner,
                },
            },
            WriteDescriptorSet {
                binding: 1,
                kind: super::renderer::WriteDescriptorSetKind::CombinedImageSampler {
                    view: *skybox,
                    sampler: *skybox_sampler,
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
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
            pipeline: ctx
                .create_compute_pipeline("./src/shaders/bin/post_processing.comp.spv", layout),
            dynamic_descriptors,
            static_descriptors,
            swapchain_image_descriptors,
            layout,
        })
    }

    pub(super) fn execute(&self, ctx: &Renderer, cmd: &vk::CommandBuffer, frame: u64, i: u32) {
        unsafe {
            ctx.device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[
                    self.static_descriptors,
                    self.dynamic_descriptors[(frame % 2) as usize],
                    self.swapchain_image_descriptors[i as usize],
                ],
                &[],
            );

            ctx.device
                .cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);

            ctx.device.cmd_pipeline_barrier(
                *cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_queue_family_index(ctx.graphics_queue_family.index)
                    .dst_queue_family_index(ctx.graphics_queue_family.index)
                    .image(ctx.swapchain.images[i as usize].image.inner)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags::MEMORY_READ)
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)],
            );

            ctx.device
                .cmd_dispatch(*cmd, WINDOW_SIZE.0 / 8, WINDOW_SIZE.1 / 8, 1);
        }
    }
}
