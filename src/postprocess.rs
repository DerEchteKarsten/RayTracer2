use anyhow::Result;
use ash::vk::{self, PipelineCache, ShaderStageFlags};

use crate::{
    allocate_descriptor_sets, calculate_pool_sizes, render_recources::GBuffer,
    update_descriptor_sets, ImageBarrier, Renderer, WriteDescriptorSet, WINDOW_SIZE,
};

pub struct PostProcessPass {
    handel: vk::Pipeline,
    layout: vk::PipelineLayout,
    source_descriptors: Vec<vk::DescriptorSet>,
    target_descriptors: Vec<vk::DescriptorSet>,
}

impl PostProcessPass {
    pub fn new(ctx: &mut Renderer, g_buffers: &[GBuffer; 2]) -> Result<Self> {
        let source_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE), // Albedo
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE), // Specular
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE), // Emissiv
        ];

        let target_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE), // Output
        ];

        let source_layout = ctx.create_descriptor_set_layout(&source_bindings)?;
        let target_layout = ctx.create_descriptor_set_layout(&target_bindings)?;

        let layouts = [source_layout, target_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&[])
            .set_layouts(&layouts);
        let layout = unsafe { ctx.device.create_pipeline_layout(&layout_info, None)? };
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .layout(layout)
            .stage(
                ctx.create_shader_stage(
                    ShaderStageFlags::COMPUTE,
                    "./src/shaders/post_processing.comp.spv",
                )
                .0,
            );
        let handel = unsafe {
            ctx.device
                .create_compute_pipelines(PipelineCache::null(), &[pipeline_info], None)
        }
        .unwrap();

        let pool = ctx.create_descriptor_pool(
            2 + ctx.swapchain.images.len() as u32,
            &calculate_pool_sizes(&[&source_bindings, &target_bindings]),
        )?;

        let source_descriptors = allocate_descriptor_sets(&ctx.device, &pool, &source_layout, 2)?;
        let target_descriptors = allocate_descriptor_sets(
            &ctx.device,
            &pool,
            &target_layout,
            ctx.swapchain.images.len() as u32,
        )?;

        for i in 0..2 {
            let writes = [
                WriteDescriptorSet {
                    binding: 0,
                    kind: crate::WriteDescriptorSetKind::StorageImage {
                        view: g_buffers[i].diffuse_albedo.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                },
                WriteDescriptorSet {
                    binding: 1,
                    kind: crate::WriteDescriptorSetKind::StorageImage {
                        view: g_buffers[i].specular_rough.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                },
                WriteDescriptorSet {
                    binding: 2,
                    kind: crate::WriteDescriptorSetKind::StorageImage {
                        view: g_buffers[i].emissive.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                },
            ];
            update_descriptor_sets(ctx, &source_descriptors[i], &writes);
        }

        for i in 0..ctx.swapchain.images.len() {
            let writes = [WriteDescriptorSet {
                binding: 0,
                kind: crate::WriteDescriptorSetKind::StorageImage {
                    view: ctx.swapchain.images[i].view,
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
            }];
            update_descriptor_sets(ctx, &target_descriptors[i], &writes);
        }

        Ok(Self {
            handel: handel[0],
            layout,
            source_descriptors,
            target_descriptors,
        })
    }

    pub fn execute(&self, ctx: &Renderer, cmd: &vk::CommandBuffer, frame: u64, i: u32) {
        unsafe {
            ctx.pipeline_image_barriers(
                cmd,
                &[ImageBarrier {
                    image: &ctx.swapchain.images[i as usize].image,
                    old_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_access_mask: vk::AccessFlags2::NONE,
                    dst_access_mask: vk::AccessFlags2::SHADER_WRITE,
                    src_stage_mask: vk::PipelineStageFlags2::NONE,
                    dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                }],
            );
            ctx.device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[
                    self.source_descriptors[(frame % 2) as usize],
                    self.target_descriptors[i as usize],
                ],
                &[],
            );
            ctx.device
                .cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::COMPUTE, self.handel);
            ctx.device
                .cmd_dispatch(*cmd, WINDOW_SIZE.x as u32 / 8, WINDOW_SIZE.y as u32 / 8, 1);
            ctx.pipeline_image_barriers(
                cmd,
                &[ImageBarrier {
                    image: &ctx.swapchain.images[i as usize].image,
                    old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    src_access_mask: vk::AccessFlags2::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags2::NONE,
                    src_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    dst_stage_mask: vk::PipelineStageFlags2::NONE,
                }],
            );
        }
    }
}
