use anyhow::{Ok, Result};
use ash::vk::{
    self, AccessFlags, BorderColor, BufferUsageFlags, CompareOp, ImageLayout, ImageUsageFlags,
    ImageView, PipelineCache, SamplerAddressMode, SamplerMipmapMode, ShaderStageFlags,
};
use glam::{vec3, UVec2, Vec2};
use gpu_allocator::MemoryLocation;
use ndarray::{arr3, prelude::*};
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng};

use std::ffi::CString;
use std::mem::size_of;
use std::{array::from_ref, default::Default};

use crate::shader_params::{GConst, RTXDI_ReservoirBufferParameters};
use crate::{
    allocate_descriptor_set, allocate_descriptor_sets, create_descriptor_pool,
    create_descriptor_set_layout, module_from_bytes, update_descriptor_sets, AccelerationStructure,
    Buffer, Image, ImageAndView, MipLevelPushConstants, Model, RayTracingShaderCreateInfo,
    RayTracingShaderGroup, RayTracingShaderGroupInfo, Renderer, ShaderBindingTable,
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
    pub di_reservoirs: Buffer,
    pub lights_buffer: Buffer,
    pub ris_lights_buffer: Buffer,
    pub ris_buffer: Buffer,
    pub neighbors: Buffer,
    pub uniform_buffer: Buffer,
    pub spatial_reuse_pipeline: RayTracingPipeline,
    pub temporal_reuse_pipeline: RayTracingPipeline,
    pub environment_presampeling_pipeline: ComputePipeline,
    pub local_light_presampeling_pipeline: ComputePipeline,
    pub local_lights_pdf: Image,
    pub environment_pdf: Image,
    pub environment_mip_pipeline: ComputePipeline,
    pub local_lights_mip_pipeline: ComputePipeline,
    pub prepare_lights_pipeline: ComputePipeline,
    pub light_tasks_buffer: Buffer,
    pub geometry_to_light_buffer_and_staging: (Buffer, Buffer),
}

pub struct PostProccesingPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub dynamic_descriptors: Vec<vk::DescriptorSet>,
    pub static_descriptor: vk::DescriptorSet,

    pub render_pass: vk::RenderPass,
    pub frame_buffers: Vec<vk::Framebuffer>,
}

impl PostProccesingPipeline {
    pub fn execute(
        &self,
        ctx: &Renderer,
        cmd: &vk::CommandBuffer,
        frame_c: &[u8],
        i: u32,
        frame: u64,
    ) {
        unsafe {
            let begin_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.frame_buffers[i as usize])
                .render_area(vk::Rect2D {
                    extent: vk::Extent2D {
                        width: WINDOW_SIZE.x as u32,
                        height: WINDOW_SIZE.y as u32,
                    },
                    offset: vk::Offset2D { x: 0, y: 0 },
                })
                .clear_values(&[vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }]);

            let view_port = vk::Viewport::default()
                .height(WINDOW_SIZE.y as f32)
                .width(WINDOW_SIZE.x as f32)
                .max_depth(1.0)
                .min_depth(0.0)
                .x(0 as f32)
                .y(0 as f32);
            ctx.device.cmd_set_viewport(*cmd, 0, &[view_port]);

            let scissor = vk::Rect2D::default()
                .extent(vk::Extent2D {
                    width: WINDOW_SIZE.x as u32,
                    height: WINDOW_SIZE.y as u32,
                })
                .offset(vk::Offset2D { x: 0, y: 0 });
            ctx.device.cmd_set_scissor(*cmd, 0, &[scissor]);

            ctx.device
                .cmd_begin_render_pass(*cmd, &begin_info, vk::SubpassContents::INLINE);

            ctx.device
                .cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            ctx.device.cmd_push_constants(
                *cmd,
                self.layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                &frame_c,
            );

            ctx.device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[
                    self.static_descriptor,
                    self.dynamic_descriptors[(frame % 2) as usize],
                ],
                &[],
            );
            ctx.device.cmd_draw(*cmd, 6, 1, 0, 0);
            ctx.device.cmd_end_render_pass(*cmd);
        }
    }
}

pub struct RayTracingPipeline {
    pub shader_binding_table: ShaderBindingTable,
    pub shader_group_info: RayTracingShaderGroupInfo,
    pub layout: vk::PipelineLayout,
    pub handle: vk::Pipeline,
    pub descriptor_set0: vk::DescriptorSet,
    pub descriptor_set1: Vec<vk::DescriptorSet>,
    pub descriptor_set2: Vec<vk::DescriptorSet>,
}

impl RayTracingPipeline {
    pub fn execute(&self, ctx: &Renderer, cmd: &vk::CommandBuffer, frame: u64, frame_c: &[u8]) {
        unsafe {
            ctx.device.cmd_push_constants(
                *cmd,
                self.layout,
                vk::ShaderStageFlags::RAYGEN_KHR,
                0,
                frame_c,
            );

            ctx.device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.layout,
                0,
                &[
                    self.descriptor_set0,
                    self.descriptor_set1[(frame % 2) as usize],
                    self.descriptor_set2[1 - (frame % 2) as usize],
                ],
                &[],
            );

            ctx.device
                .cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::RAY_TRACING_KHR, self.handle);

            let call_region = vk::StridedDeviceAddressRegionKHR::default();

            ctx.ray_tracing.pipeline_fn.cmd_trace_rays(
                *cmd,
                &self.shader_binding_table.raygen_region,
                &self.shader_binding_table.miss_region,
                &self.shader_binding_table.hit_region,
                &call_region,
                WINDOW_SIZE.x as u32,
                WINDOW_SIZE.y as u32,
                1,
            );
        }
    }
}

#[derive(Clone, Copy)]
pub struct ComputePipeline {
    pub descriptor: vk::DescriptorSet,
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl ComputePipeline {
    pub fn execute_presampeling(
        self,
        ctx: &Renderer,
        cmd: &vk::CommandBuffer,
        barriers: &[vk::BufferMemoryBarrier],
    ) {
        unsafe {
            ctx.device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.descriptor],
                &[],
            );
            ctx.device
                .cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::COMPUTE, self.handle);

            ctx.device.cmd_dispatch(*cmd, 1024 / 256, 128, 1);

            ctx.device.cmd_pipeline_barrier(
                *cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                barriers,
                &[],
            );
        }
    }
    pub fn execute_mip_generation(
        self,
        ctx: &Renderer,
        cmd: &vk::CommandBuffer,
        source_width: u32,
        source_height: u32,
        mip_levels: u32,
    ) {
        unsafe {
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

                ctx.device
                    .cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::COMPUTE, self.handle);
                ctx.device.cmd_push_constants(
                    *cmd,
                    self.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &constants,
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
                    f32::ceil(width as f32 / 32.0) as u32,
                    f32::ceil(height as f32 / 32.0) as u32,
                    1,
                );

                width = u32::max(1u32, width >> mip_levels_per_pass);
                height = u32::max(1u32, height >> mip_levels_per_pass);
            }
            ctx.device.cmd_pipeline_barrier(
                *cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[],
            );
        }
    }
    pub fn execute_light_preperation(
        self,
        ctx: &Renderer,
        light_tasks_buffer: &Buffer,
        geometry_to_light_buffer_and_staging: &(Buffer, Buffer),
        model: &Model,
        cmd: &vk::CommandBuffer,
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
            ctx.device.cmd_pipeline_barrier(
                *cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[vk::BufferMemoryBarrier::default()
                    .buffer(geometry_to_light_buffer_and_staging.0.inner)
                    .offset(0)
                    .size(geometry_to_light_buffer_and_staging.0.size)
                    .src_access_mask(AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ)
                    .src_queue_family_index(ctx.graphics_queue_family.index)
                    .dst_queue_family_index(ctx.graphics_queue_family.index)],
                &[],
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct PrepareLightsTask {
    geometry_index: u32,
    light_buffer_offset: u32,
    triangle_count: u32,
    pad: u32,
}

pub fn create_mip_pipeline(
    ctx: &mut Renderer,
    input_images: &[vk::ImageView],
    mips: u32,
    environment: Option<&vk::ImageView>,
) -> Result<ComputePipeline> {
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
    let descriptor0_layout = ctx.create_descriptor_set_layout(&bindings, &[])?;
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
        "./src/shaders/env_mip_levels.comp.spv"
    } else {
        "./src/shaders/mip_levels.comp.spv"
    };

    let entry_point_name: CString = CString::new("main").unwrap();
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .layout(layout)
        .stage(
            vk::PipelineShaderStageCreateInfo::default()
                .module(ctx.create_shader_module(path))
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
                    .descriptor_count(32)
                    .ty(vk::DescriptorType::STORAGE_IMAGE),
                vk::DescriptorPoolSize::default()
                    .descriptor_count(1)
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            ],
        )
        .unwrap();

    let descriptor_set0 = allocate_descriptor_set(&ctx.device, &pool, &descriptor0_layout).unwrap();

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
        let sampler_create_info = vk::SamplerCreateInfo::default()
            .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
            .unnormalized_coordinates(true);
        let sampler = unsafe { ctx.device.create_sampler(&sampler_create_info, None)? };
        let write = WriteDescriptorSet {
            binding: 1,
            kind: WriteDescriptorSetKind::CombinedImageSampler {
                view: *environment,
                sampler,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        };
        update_descriptor_sets(ctx, &descriptor_set0, &[write]);
    }

    Ok(ComputePipeline {
        descriptor: descriptor_set0,
        handle,
        layout,
    })
}

pub fn create_lights_pipeline(
    ctx: &mut Renderer,
    local_lights_pdf: &vk::ImageView,
    environment: (&vk::ImageView, &vk::Sampler),
    lights_buffer: &Buffer,
    light_tasks_buffer: &Buffer,
    model: &Model,
) -> Result<ComputePipeline> {
    let mut bindings = vec![
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
                view: *local_lights_pdf,
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
                buffer: light_tasks_buffer.inner,
            },
        },
    ];
    update_descriptor_sets(ctx, &descriptor_set0, &writes);

    Ok(ComputePipeline {
        descriptor: descriptor_set0,
        handle,
        layout,
    })
}

pub fn create_presampling_pipeline(
    ctx: &mut Renderer,
    ris_lights_buffer: &Buffer,
    di_reservoirs: &Buffer,
    lights_buffer: &Buffer,
    ris_buffer: &Buffer,
    input_image: (&vk::ImageView, &vk::Sampler),
    output_image: (&vk::ImageView, &vk::Sampler),
    uniform_buffer: &Buffer,
    path: &str,
) -> Result<ComputePipeline> {
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
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(5)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(6)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
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
                .module(ctx.create_shader_module(path))
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
                    .descriptor_count(4)
                    .ty(vk::DescriptorType::STORAGE_BUFFER),
                vk::DescriptorPoolSize::default()
                    .descriptor_count(2)
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                vk::DescriptorPoolSize::default()
                    .descriptor_count(1)
                    .ty(vk::DescriptorType::UNIFORM_BUFFER),
            ],
        )
        .unwrap();

    let descriptor_set0 = allocate_descriptor_set(&ctx.device, &pool, &descriptor0_layout).unwrap();

    let writes = [
        WriteDescriptorSet {
            binding: 0,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: di_reservoirs.inner,
            },
        },
        WriteDescriptorSet {
            binding: 1,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: lights_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 2,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: ris_lights_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 3,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: ris_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 4,
            kind: WriteDescriptorSetKind::CombinedImageSampler {
                view: input_image.0.clone(),
                sampler: input_image.1.clone(),
                layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        },
        WriteDescriptorSet {
            binding: 5,
            kind: WriteDescriptorSetKind::CombinedImageSampler {
                view: output_image.0.clone(),
                sampler: output_image.1.clone(),
                layout: ImageLayout::GENERAL,
            },
        },
    ];

    update_descriptor_sets(ctx, &descriptor_set0, &writes);

    let writes = [WriteDescriptorSet {
        binding: 6,
        kind: WriteDescriptorSetKind::UniformBuffer {
            buffer: uniform_buffer.inner,
        },
    }];

    update_descriptor_sets(ctx, &descriptor_set0, &writes);

    Ok(ComputePipeline {
        descriptor: descriptor_set0,
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
    di_reservoirs: &Buffer,
    lights_buffer: &Buffer,
    ris_buffer: &Buffer,
    ris_lights_buffer: &Buffer,
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
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(5)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(6)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
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
            .module(ctx.create_shader_module("./src/shaders/post_processing.frag.spv"))
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&entry_point_name),
        vk::PipelineShaderStageCreateInfo::default()
            .module(ctx.create_shader_module("./src/shaders/post_processing.vert.spv"))
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
            .descriptor_count(5)
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
            binding: 2,
            kind: WriteDescriptorSetKind::CombinedImageSampler {
                sampler: *skybox_sampler,
                view: *skybox_view,
                layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        },
        WriteDescriptorSet {
            binding: 3,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: di_reservoirs.inner,
            },
        },
        WriteDescriptorSet {
            binding: 4,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: lights_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 5,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: ris_lights_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 6,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: ris_buffer.inner,
            },
        },
    ];

    update_descriptor_sets(ctx, &static_descriptor, &writes);

    let write = [WriteDescriptorSet {
        binding: 1,
        kind: WriteDescriptorSetKind::UniformBuffer {
            buffer: uniform_buffer.inner,
        },
    }];
    update_descriptor_sets(ctx, &static_descriptor, &write);

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
    environment_pdf: (&vk::ImageView, &vk::Sampler),
    local_lights_pdf: (&vk::ImageView, &vk::Sampler),
    di_reservoirs: &Buffer,
    ris_lights_buffer: &Buffer,
    lights_buffer: &Buffer,
    ris_buffer: &Buffer,
    geom_to_lights: &Buffer,
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
            .stage_flags(vk::ShaderStageFlags::MISS_KHR | vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(9)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(10)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(11)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(12)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(13)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(14)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(15)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
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
        .size(size_of::<u32>() as u32)
        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)];

    let pipe_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&dsls)
        .push_constant_ranges(push_constants);
    let pipeline_layout = unsafe { ctx.device.create_pipeline_layout(&pipe_layout_info, None)? };

    let (inner, shader_group_info) =
        ctx.create_raytracing_pipeline(pipeline_layout, &shaders_create_info)?;

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
            .descriptor_count(10),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count((model.images.len() as u32) + 3),
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
        WriteDescriptorSet {
            binding: 9,
            kind: WriteDescriptorSetKind::CombinedImageSampler {
                view: *environment_pdf.0,
                sampler: *environment_pdf.1,
                layout: ImageLayout::GENERAL,
            },
        },
        WriteDescriptorSet {
            binding: 10,
            kind: WriteDescriptorSetKind::CombinedImageSampler {
                view: *local_lights_pdf.0,
                sampler: *local_lights_pdf.1,
                layout: ImageLayout::GENERAL,
            },
        },
        WriteDescriptorSet {
            binding: 11,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: di_reservoirs.inner,
            },
        },
        WriteDescriptorSet {
            binding: 12,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: lights_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 13,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: ris_lights_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 14,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: ris_buffer.inner,
            },
        },
        WriteDescriptorSet {
            binding: 15,
            kind: WriteDescriptorSetKind::StorageBuffer {
                buffer: geom_to_lights.inner,
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

    let shader_binding_table = ctx
        .create_shader_binding_table(&inner, &shader_group_info)
        .unwrap();

    Ok(RayTracingPipeline {
        shader_binding_table,
        handle: inner,
        layout: pipeline_layout,
        descriptor_set0: static_set,
        descriptor_set1: dynamic_sets,
        descriptor_set2: dynamic_sets2,
        shader_group_info,
    })
}

fn create_reuse_pipeline(
    ctx: &mut Renderer<'_>,
    top_as: &AccelerationStructure,
    reservoirs: &Buffer,
    uniform_buffer: &Buffer,
    neighbors: &Buffer,
    g_buffer: &[GBuffer],
    reuse_shaders: &[RayTracingShaderCreateInfo],
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
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
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
        .size(size_of::<u32>() as u32)
        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)];

    let pipe_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&dsls)
        .push_constant_ranges(push_constants);
    let pipeline_layout = unsafe { ctx.device.create_pipeline_layout(&pipe_layout_info, None)? };

    let (inner, shader_group_info) =
        ctx.create_raytracing_pipeline(pipeline_layout, &reuse_shaders)?;

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
    let shader_binding_table = ctx
        .create_shader_binding_table(&inner, &shader_group_info)
        .unwrap();

    Ok(RayTracingPipeline {
        shader_binding_table,
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
    skybox_size: UVec2,
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

    let di_reservoirs = ctx
        .create_buffer(
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            reservoir_buffer_size,
            None,
        )
        .unwrap();

    let lights_buffer = ctx
        .create_buffer(
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            48 * model.lights as u64,
            None,
        )
        .unwrap();
    let ris_lights_buffer = ctx
        .create_buffer(
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            size_of::<u32>() as u64 * 8 * WINDOW_SIZE.x as u64 * WINDOW_SIZE.y as u64,
            None,
        )
        .unwrap();

    let ris_buffer = ctx
        .create_buffer(
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            size_of::<u32>() as u64 * 2 * WINDOW_SIZE.x as u64 * WINDOW_SIZE.y as u64,
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
        &di_reservoirs,
        &lights_buffer,
        &ris_buffer,
        &ris_lights_buffer,
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

    let spatial_reuse_shaders = [
        RayTracingShaderCreateInfo {
            source: &[(
                &include_bytes!("./shaders/spatial_resampling.rgen.spv")[..],
                vk::ShaderStageFlags::RAYGEN_KHR,
            )],
            group: RayTracingShaderGroup::RayGen,
        },
        RayTracingShaderCreateInfo {
            source: &[(
                &include_bytes!("./shaders/visibility.rmiss.spv")[..],
                vk::ShaderStageFlags::MISS_KHR,
            )],
            group: RayTracingShaderGroup::Miss,
        },
        RayTracingShaderCreateInfo {
            source: &[(
                &include_bytes!("./shaders/visibility.rchit.spv")[..],
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )],
            group: RayTracingShaderGroup::Hit,
        },
    ];

    let temporal_reuse_shaders = [
        RayTracingShaderCreateInfo {
            source: &[(
                &include_bytes!("./shaders/temporal_resampling.rgen.spv")[..],
                vk::ShaderStageFlags::RAYGEN_KHR,
            )],
            group: RayTracingShaderGroup::RayGen,
        },
        RayTracingShaderCreateInfo {
            source: &[(
                &include_bytes!("./shaders/visibility.rmiss.spv")[..],
                vk::ShaderStageFlags::MISS_KHR,
            )],
            group: RayTracingShaderGroup::Miss,
        },
        RayTracingShaderCreateInfo {
            source: &[(
                &include_bytes!("./shaders/visibility.rchit.spv")[..],
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )],
            group: RayTracingShaderGroup::Hit,
        },
    ];

    let environment_pdf = ctx
        .create_mipimage(
            ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED,
            MemoryLocation::GpuOnly,
            vk::Format::R16_SFLOAT,
            skybox_size.x,
            skybox_size.y,
            ((u32::max(skybox_size.x, skybox_size.y) as f32)
                .log(2.0)
                .ceil() as u32)
                + 1,
        )
        .unwrap();

    let environenvironment_pdf_views = ctx.create_image_views(&environment_pdf).unwrap();
    let environenvironment_pdf_view = ctx.create_image_view(&environment_pdf).unwrap();

    let (width, height, mips) = ComputePdfTextureSize(model.lights);

    let local_lights_pdf = ctx
        .create_mipimage(
            ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED,
            MemoryLocation::GpuOnly,
            vk::Format::R32_SFLOAT,
            width,
            height,
            mips,
        )
        .unwrap();

    let local_lights_pdf_views = ctx.create_image_views(&local_lights_pdf).unwrap();
    let local_lights_pdf_view = ctx.create_image_view(&local_lights_pdf).unwrap();

    ctx.transition_image_layout(&local_lights_pdf, ImageLayout::GENERAL)
        .unwrap();
    ctx.transition_image_layout(&environment_pdf, ImageLayout::GENERAL)
        .unwrap();

    let mut sampler_create_info: vk::SamplerCreateInfo = vk::SamplerCreateInfo {
        mag_filter: vk::Filter::NEAREST,
        min_filter: vk::Filter::NEAREST,
        mipmap_mode: vk::SamplerMipmapMode::NEAREST,
        address_mode_u: vk::SamplerAddressMode::CLAMP_TO_BORDER,
        address_mode_v: vk::SamplerAddressMode::CLAMP_TO_BORDER,
        address_mode_w: vk::SamplerAddressMode::CLAMP_TO_BORDER,
        max_anisotropy: 1.0,
        border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
        min_lod: 0.0,
        max_lod: environment_pdf.mip_levels as f32,
        ..Default::default()
    };

    let skybox_pdf_sampler = unsafe {
        ctx.device
            .create_sampler(&sampler_create_info, None)
            .unwrap()
    };

    sampler_create_info.max_lod = local_lights_pdf.mip_levels as f32;

    let local_lights_pdf_sampler = unsafe {
        ctx.device
            .create_sampler(&sampler_create_info, None)
            .unwrap()
    };

    let geometry_to_light_buffer = ctx
        .create_buffer(
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            model.geometry_infos.len() as u64,
            None,
        )
        .unwrap();
    let geometry_to_light_staging_buffer = ctx
        .create_buffer(
            BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::CpuToGpu,
            geometry_to_light_buffer.size,
            None,
        )
        .unwrap();
    let geometry_to_light_buffer_and_staging =
        (geometry_to_light_buffer, geometry_to_light_staging_buffer);
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
        (&environenvironment_pdf_view, &skybox_pdf_sampler),
        (&local_lights_pdf_view, &local_lights_pdf_sampler),
        &di_reservoirs,
        &ris_lights_buffer,
        &lights_buffer,
        &ris_buffer,
        &geometry_to_light_buffer_and_staging.0,
    )
    .unwrap();

    let environment_presampeling_pipeline = create_presampling_pipeline(
        ctx,
        &ris_lights_buffer,
        &di_reservoirs,
        &lights_buffer,
        &ris_buffer,
        (&skybox_view, &skybox_sampler),
        (&environenvironment_pdf_view, &skybox_pdf_sampler),
        &uniform_buffer,
        "./src/shaders/presample_environment.comp.spv",
    )
    .unwrap();

    let local_light_presampeling_pipeline = create_presampling_pipeline(
        ctx,
        &ris_lights_buffer,
        &di_reservoirs,
        &lights_buffer,
        &ris_buffer,
        (&skybox_view, &skybox_sampler),
        (&local_lights_pdf_view, &local_lights_pdf_sampler),
        &uniform_buffer,
        "./src/shaders/presample_locallights.comp.spv",
    )
    .unwrap();

    let spatial_reuse_pipeline = create_reuse_pipeline(
        ctx,
        top_as,
        &reservoirs,
        &uniform_buffer,
        &neighbors,
        &g_buffer,
        &spatial_reuse_shaders,
    )
    .unwrap();

    let temporal_reuse_pipeline = create_reuse_pipeline(
        ctx,
        top_as,
        &reservoirs,
        &uniform_buffer,
        &neighbors,
        &g_buffer,
        &temporal_reuse_shaders,
    )
    .unwrap();
    let light_tasks_buffer = ctx
        .create_buffer(
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            16 * 500,
            None,
        )
        .unwrap();

    let environment_mip_pipeline = create_mip_pipeline(
        ctx,
        &environenvironment_pdf_views,
        environment_pdf.mip_levels,
        Some(&skybox_view),
    )
    .unwrap();
    let local_lights_mip_pipeline = create_mip_pipeline(
        ctx,
        &local_lights_pdf_views,
        local_lights_pdf.mip_levels,
        None,
    )
    .unwrap();
    let prepare_lights_pipeline = create_lights_pipeline(
        ctx,
        &local_lights_pdf_view,
        (&skybox_view, &skybox_sampler),
        &lights_buffer,
        &light_tasks_buffer,
        model,
    )
    .unwrap();

    Ok(RendererResources {
        g_buffer,
        neighbors,
        post_proccesing_pipeline,
        raytracing_pipeline,
        reservoirs,
        uniform_buffer,
        spatial_reuse_pipeline,
        temporal_reuse_pipeline,
        di_reservoirs,
        lights_buffer,
        ris_lights_buffer,
        ris_buffer,
        local_lights_pdf,
        environment_presampeling_pipeline,
        local_light_presampeling_pipeline,
        environment_pdf,
        environment_mip_pipeline,
        local_lights_mip_pipeline,
        prepare_lights_pipeline,
        light_tasks_buffer,
        geometry_to_light_buffer_and_staging,
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
    let renderWidthBlocks =
        (renderWidth + RTXDI_RESERVOIR_BLOCK_SIZE - 1) / RTXDI_RESERVOIR_BLOCK_SIZE;
    let renderHeightBlocks =
        (renderHeight + RTXDI_RESERVOIR_BLOCK_SIZE - 1) / RTXDI_RESERVOIR_BLOCK_SIZE;
    let mut params = RTXDI_ReservoirBufferParameters::default();
    params.reservoirBlockRowPitch =
        renderWidthBlocks * (RTXDI_RESERVOIR_BLOCK_SIZE * RTXDI_RESERVOIR_BLOCK_SIZE);
    params.reservoirArrayPitch = params.reservoirBlockRowPitch * renderHeightBlocks;
    return params;
}

fn ComputePdfTextureSize(maxItems: u32) -> (u32, u32, u32) {
    // Compute the size of a power-of-2 rectangle that fits all items, 1 item per pixel
    let mut textureWidth = f64::max(1.0, f64::ceil(f64::sqrt(maxItems as f64)));
    textureWidth = f64::exp2(f64::ceil(f64::log2(textureWidth)));
    let mut textureHeight = f64::max(1.0, f64::ceil(maxItems as f64 / textureWidth));
    textureHeight = f64::exp2(f64::ceil(f64::log2(textureHeight)));
    let textureMips = f64::max(1.0, f64::log2(f64::max(textureWidth, textureHeight)) + 1.0);

    (
        textureWidth as u32,
        textureHeight as u32,
        textureMips as u32,
    )
}
