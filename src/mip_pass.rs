use std::ffi::CString;

use crate::context::*;
use anyhow::Result;
use ash::vk::{self, PipelineCache, ShaderStageFlags};
use glam::UVec2;

#[repr(C)]
#[derive(Clone, Copy)]
struct MipLevelPushConstants {
    source_size: UVec2,
    num_dest_mip_levels: u32,
    source_mip_level: u32,
}

pub struct GenerateMipsPass {
    handle: vk::Pipeline,
    descriptor: vk::DescriptorSet,
    layout: vk::PipelineLayout,
}

impl GenerateMipsPass {
    pub fn new(
        ctx: &mut Renderer,
        input_images: &[vk::ImageView],
        mips: u32,
        environment: Option<(&vk::ImageView, &vk::Sampler)>,
    ) -> Result<Self> {
        let mut bindings = vec![vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(32)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        if environment.is_some() {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            )
        }
        let descriptor0_layout = ctx.create_descriptor_set_layout(&bindings)?;
        let descriptor_layouts = [descriptor0_layout];

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&[vk::PushConstantRange {
                offset: 0,
                size: size_of::<MipLevelPushConstants>() as u32,
                stage_flags: ShaderStageFlags::COMPUTE,
            }])
            .set_layouts(&descriptor_layouts);

        let layout = unsafe { ctx.device.create_pipeline_layout(&layout_info, None)? };

        let path = if environment.is_some() {
            "./src/shaders/bin/env_mip_levels.spv"
        } else {
            "./src/shaders/bin/mip_levels.spv"
        };

        let entry_point_name: CString = CString::new("main").unwrap();
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .layout(layout)
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .module(ctx.create_shader_module(path).unwrap())
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .name(&entry_point_name),
            );

        let handle = unsafe {
            ctx.device
                .create_compute_pipelines(PipelineCache::null(), &[pipeline_info], None)
                .unwrap()[0]
        };

        let pool = ctx
            .create_descriptor_pool(&[CalculatePoolSizesDesc {
                bindings: bindings.as_slice(),
                num_sets: 1,
            }])
            .unwrap();

        let descriptor_set0 = ctx
            .allocate_descriptor_sets(&pool, &descriptor0_layout, 1)
            .unwrap()[0];

        for i in 0..mips {
            let image_info = [vk::DescriptorImageInfo::default()
                .image_view(input_images[i as usize])
                .image_layout(vk::ImageLayout::GENERAL)];
            let write = vk::WriteDescriptorSet::default()
                .descriptor_count(32)
                .dst_binding(0)
                .dst_array_element(i)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .dst_set(descriptor_set0)
                .image_info(&image_info);
            unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
        }

        if let Some(environment) = environment {
            let write = WriteDescriptorSet {
                binding: 1,
                kind: WriteDescriptorSetKind::CombinedImageSampler {
                    view: *environment.0,
                    sampler: *environment.1,
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
            };
            ctx.update_descriptor_sets(&descriptor_set0, &[write]);
        }

        Ok(Self {
            descriptor: descriptor_set0,
            handle,
            layout,
        })
    }

    pub fn execute_mip_generation(
        &self,
        ctx: &Renderer,
        cmd: &vk::CommandBuffer,
        source_width: u32,
        source_height: u32,
        mip_levels: u32,
    ) {
        unsafe {
            ctx.memory_barrier(
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::AccessFlags2::SHADER_WRITE,
            );
            ctx.device
                .cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::COMPUTE, self.handle);
            ctx.device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.descriptor],
                &[],
            );

            let mip_levels_per_pass = 5;
            let mut width = source_width;
            let mut height = source_height;

            for mip in (0..mip_levels).step_by(mip_levels_per_pass) {
                let constants = MipLevelPushConstants {
                    num_dest_mip_levels: mip_levels,
                    source_mip_level: mip,
                    source_size: UVec2 {
                        x: source_width,
                        y: source_height,
                    },
                };
                let constants = std::slice::from_raw_parts(
                    &constants as *const _ as *const u8,
                    size_of::<MipLevelPushConstants>(),
                );

                ctx.device.cmd_push_constants(
                    *cmd,
                    self.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &constants,
                );

                ctx.device.cmd_dispatch(
                    *cmd,
                    f32::ceil(width as f32 / 32.0) as u32,
                    f32::ceil(height as f32 / 32.0) as u32,
                    1,
                );

                width = u32::max(1u32, width >> mip_levels_per_pass);
                height = u32::max(1u32, height >> mip_levels_per_pass);
            }
        }
    }
}
