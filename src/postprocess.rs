use anyhow::Result;
use ash::vk::{self, ImageLayout, PipelineCache, ShaderStageFlags};

use crate::{
    calculate_pool_sizes,
    render_resources::{GBuffer, RenderResources},
    Buffer, CalculatePoolSizesDesc, ImageBarrier, Renderer, WriteDescriptorSet, WINDOW_SIZE,
};

pub struct PostProcessPass {
    handel: vk::Pipeline,
    layout: vk::PipelineLayout,
    source_descriptors: Vec<vk::DescriptorSet>,
    target_descriptors: Vec<vk::DescriptorSet>,
    static_descriptor: vk::DescriptorSet,
}

impl PostProcessPass {
    pub fn new(
        ctx: &mut Renderer,
        g_buffers: &[GBuffer; 2],
        resources: &RenderResources,
        uniform_buffer: &Buffer,
        skybox: &vk::ImageView,
        skybox_sampler: &vk::Sampler,
    ) -> Result<Self> {
        let source_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_count(1)
                .binding(0)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE), // Depth
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_count(1)
                .binding(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE), // Albedo
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_count(1)
                .binding(2)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE), // Specular
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_count(1)
                .binding(3)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE), // Emissiv
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_count(1)
                .binding(4)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE), // Motion Vectors
        ];

        let static_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_count(1)
                .binding(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE), // Diffuse
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_count(1)
                .binding(2)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE), // Specular
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_count(1)
                .binding(3)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER), // Uniform
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_count(1)
                .binding(4)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER), // Skybox
        ];

        let target_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE), // Output
        ];

        let source_layout = ctx.create_descriptor_set_layout(&source_bindings)?;
        let target_layout = ctx.create_descriptor_set_layout(&target_bindings)?;
        let static_layout = ctx.create_descriptor_set_layout(&static_bindings)?;

        let layouts = [source_layout, static_layout, target_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&[])
            .set_layouts(&layouts);
        let layout = unsafe { ctx.device.create_pipeline_layout(&layout_info, None)? };
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .layout(layout)
            .stage(
                ctx.create_shader_stage(
                    ShaderStageFlags::COMPUTE,
                    "./src/shaders/bin/post_processing.spv",
                )
                .unwrap()
                .0,
            );
        let handel = unsafe {
            ctx.device
                .create_compute_pipelines(PipelineCache::null(), &[pipeline_info], None)
        }
        .unwrap();

        let pool = ctx.create_descriptor_pool(&[
            CalculatePoolSizesDesc {
                bindings: &source_bindings,
                num_sets: 2,
            },
            CalculatePoolSizesDesc {
                bindings: &static_bindings,
                num_sets: 1,
            },
            CalculatePoolSizesDesc {
                bindings: &target_bindings,
                num_sets: ctx.swapchain.images.len() as u32,
            },
        ])?;

        let source_descriptors = ctx.allocate_descriptor_sets(&pool, &source_layout, 2)?;
        let target_descriptors =
            ctx.allocate_descriptor_sets(&pool, &target_layout, ctx.swapchain.images.len() as u32)?;
        let static_descriptor = ctx.allocate_descriptor_sets(&pool, &static_layout, 1)?[0];

        for i in 0..2 {
            let writes = [
                WriteDescriptorSet {
                    binding: 0,
                    kind: crate::WriteDescriptorSetKind::StorageImage {
                        view: g_buffers[i].depth.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                },
                WriteDescriptorSet {
                    binding: 1,
                    kind: crate::WriteDescriptorSetKind::StorageImage {
                        view: g_buffers[i].diffuse_albedo.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                },
                WriteDescriptorSet {
                    binding: 2,
                    kind: crate::WriteDescriptorSetKind::StorageImage {
                        view: g_buffers[i].specular_rough.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                },
                WriteDescriptorSet {
                    binding: 3,
                    kind: crate::WriteDescriptorSetKind::StorageImage {
                        view: g_buffers[i].emissive.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                },
                WriteDescriptorSet {
                    binding: 4,
                    kind: crate::WriteDescriptorSetKind::StorageImage {
                        view: resources.motion_vectors.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                },
            ];
            ctx.update_descriptor_sets(&source_descriptors[i], &writes);
        }

        let writes = [
            WriteDescriptorSet {
                binding: 1,
                kind: crate::WriteDescriptorSetKind::StorageImage {
                    view: resources.diffuse_lighting.view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
            WriteDescriptorSet {
                binding: 2,
                kind: crate::WriteDescriptorSetKind::StorageImage {
                    view: resources.specular_lighting.view,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
            WriteDescriptorSet {
                binding: 3,
                kind: crate::WriteDescriptorSetKind::UniformBuffer {
                    buffer: uniform_buffer.inner,
                },
            },
            WriteDescriptorSet {
                binding: 4,
                kind: crate::WriteDescriptorSetKind::CombinedImageSampler {
                    view: *skybox,
                    sampler: *skybox_sampler,
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
            },
        ];

        ctx.update_descriptor_sets(&static_descriptor, &writes);

        for i in 0..ctx.swapchain.images.len() {
            let writes = [WriteDescriptorSet {
                binding: 0,
                kind: crate::WriteDescriptorSetKind::StorageImage {
                    view: ctx.swapchain.images[i].view,
                    layout: vk::ImageLayout::GENERAL,
                },
            }];
            ctx.update_descriptor_sets(&target_descriptors[i], &writes);
        }

        Ok(Self {
            handel: handel[0],
            layout,
            source_descriptors,
            target_descriptors,
            static_descriptor,
        })
    }

    pub fn execute(
        &self,
        ctx: &Renderer,
        cmd: &vk::CommandBuffer,
        frame: u64,
        i: u32,
        resources: &RenderResources,
    ) {
        unsafe {
            ctx.pipeline_image_barriers(
                cmd,
                &[
                    ImageBarrier {
                        image: &ctx.swapchain.images[i as usize].image,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::GENERAL,
                        src_access_mask: vk::AccessFlags2::NONE,
                        dst_access_mask: vk::AccessFlags2::SHADER_WRITE,
                        src_stage_mask: vk::PipelineStageFlags2::NONE,
                        dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    },
                    ImageBarrier {
                        image: &resources.diffuse_lighting.image,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::UNDEFINED,
                        src_access_mask: vk::AccessFlags2::MEMORY_READ
                            | vk::AccessFlags2::MEMORY_WRITE,
                        dst_access_mask: vk::AccessFlags2::MEMORY_READ
                            | vk::AccessFlags2::MEMORY_WRITE,
                        src_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                        dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                    },
                    ImageBarrier {
                        image: &resources.specular_lighting.image,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::UNDEFINED,
                        src_access_mask: vk::AccessFlags2::MEMORY_READ
                            | vk::AccessFlags2::MEMORY_WRITE,
                        dst_access_mask: vk::AccessFlags2::MEMORY_READ
                            | vk::AccessFlags2::MEMORY_WRITE,
                        src_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                        dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                    },
                ],
            );
            ctx.memory_barrier(
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            );

            ctx.device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[
                    self.source_descriptors[(frame % 2) as usize],
                    self.static_descriptor,
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
                    old_layout: vk::ImageLayout::GENERAL,
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
