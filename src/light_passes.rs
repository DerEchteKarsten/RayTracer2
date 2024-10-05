use std::{ffi::CString, sync::LazyLock};

use anyhow::Result;
use ash::vk::{
    self, AccessFlags, DescriptorType, ImageView, PipelineBindPoint, PipelineCache,
    PipelineStageFlags, ShaderStageFlags,
};
use gpu_allocator::MemoryLocation;

use crate::{
    context::*,
    render_resources::RenderResources,
    shader_params::{GConst, RTXDI_ReservoirBufferParameters},
    Model, RTXDI_RESERVOIR_BLOCK_SIZE, WINDOW_SIZE,
};

struct RayTracingPass {
    shader_binding_table: ShaderBindingTable,
    handle: vk::Pipeline,
}

static RAYMISS_BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {std::fs::read("./src/shaders/bin/raymiss.spv").unwrap()});
static RAYHIT_BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {std::fs::read("./src/shaders/bin/rayhit.spv").unwrap()});
impl RayTracingPass {
    fn new(
        ctx: &mut Renderer,
        layout: vk::PipelineLayout,
        path: &str,
    ) -> Result<Self> {
        let raygen_bytes = std::fs::read(path)?;

        let shaders_create_info = [
            RayTracingShaderCreateInfo {
                source: &[(raygen_bytes.as_slice(), vk::ShaderStageFlags::RAYGEN_KHR)],
                group: RayTracingShaderGroup::RayGen,
            },
            RayTracingShaderCreateInfo {
                source: &[(RAYMISS_BYTES.as_slice(), vk::ShaderStageFlags::MISS_KHR)],
                group: RayTracingShaderGroup::Miss,
            },
            RayTracingShaderCreateInfo {
                source: &[(
                    RAYHIT_BYTES.as_slice(),
                    vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                )],
                group: RayTracingShaderGroup::Hit,
            },
        ];

        let (handle, shader_group_info) =
            ctx.create_raytracing_pipeline(layout, &shaders_create_info)?;
        let shader_binding_table = ctx.create_shader_binding_table(&handle, &shader_group_info)?;
        Ok(Self {
            shader_binding_table,
            handle,
        })
    }
    fn execute(&self, ctx: &Renderer, cmd: &vk::CommandBuffer) {
        unsafe {
            ctx.device
                .cmd_bind_pipeline(*cmd, PipelineBindPoint::RAY_TRACING_KHR, self.handle);
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

struct ComputePass {
    handle: vk::Pipeline,
}

impl ComputePass {
    fn new(ctx: &mut Renderer, layout: vk::PipelineLayout, path: &str) -> Result<Self> {
        let entry_point_name: CString = CString::new("main").unwrap();
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .layout(layout)
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .module(ctx.create_shader_module(path)?)
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .name(&entry_point_name),
            );
        let handel = unsafe {
            ctx.device
                .create_compute_pipelines(PipelineCache::null(), &[pipeline_info], None)
        }
        .unwrap();

        Ok(Self { handle: handel[0] })
    }
    fn execute(&self, ctx: &Renderer, cmd: &vk::CommandBuffer, x: u32, y: u32, z: u32) {
        unsafe {
            ctx.device
                .cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::COMPUTE, self.handle);
            ctx.device.cmd_dispatch(*cmd, x, y, z);
        }
    }
}

pub struct LightPasses {
    presample_lights_pass: ComputePass,
    presample_environment_map_pass: ComputePass,

    g_buffer_pass: RayTracingPass,

    di_fused_resampling_pass: RayTracingPass,

    brdf_ray_tracing_pass: RayTracingPass,
    shade_secondary_surfaces_pass: RayTracingPass,
    gi_temporal_resampling_pass: RayTracingPass,
    gi_spatial_resampling_pass: RayTracingPass,
    gi_final_shading_pass: RayTracingPass,

    layout: vk::PipelineLayout,
    static_set: vk::DescriptorSet,
    current_set: vk::DescriptorSet,
    prev_set: vk::DescriptorSet,

    pub uniform_buffer: Buffer,
}

impl LightPasses {
    fn get_dynamic_descriptor_bindings<'a>() -> Vec<vk::DescriptorSetLayoutBinding<'a>> {
        let mut bindings = vec![
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), //Depth
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), //Normals
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // GeoNormals
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // Diffuse Albedo
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // Specular Rough
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // Emission
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // Prev Depth
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // Prev Normal
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // Prev GeoNormals
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // Prev Diffuse Albedo
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // Prev Specular Rough
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // Prev Emission
        ];
        bindings = bindings
            .iter_mut()
            .enumerate()
            .map(|(i, b)| {
                b.binding(i as u32)
                    .stage_flags(ShaderStageFlags::ALL)
                    .descriptor_count(1)
            })
            .collect();
        bindings
    }

    fn get_static_descriptor_bindings<'a>(
        num_textures: u32,
    ) -> Vec<vk::DescriptorSetLayoutBinding<'a>> {
        let mut bindings = vec![
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), //Motion Vectors
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::ACCELERATION_STRUCTURE_KHR), // Acceleration Structure
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::UNIFORM_BUFFER), // Unifrom Buffer
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // Geometrie Infos
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // Vertex Buffer
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // Index Buffer
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER), // Skybox
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // Light Data Buffer
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // Neighbour Offsets Buffer
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER), // Environment Pdf
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER), // Local Lights Pdf
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // Geom To Lights
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // DI Reservoirs
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // Diffuse Lighting
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // Specular Lighting
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // Temporal Sample Positions
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // GI Reservoirs
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // RIS Buffer
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // RIS Light Data
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // Secondary GBuffer
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER) // Textures
                .descriptor_count(num_textures),
        ];
        bindings = bindings
            .iter_mut()
            .enumerate()
            .map(|(i, b)| {
                b.binding(i as u32)
                    .stage_flags(ShaderStageFlags::ALL)
                    .descriptor_count(1)
            })
            .collect();
        bindings
    }

    pub fn new(
        ctx: &mut Renderer,
        model: &Model,
        top_as: &AccelerationStructure,
        resources: &RenderResources,
        skybox_view: &ImageView,
        skybox_sampler: &vk::Sampler,
    ) -> Self {
        unsafe {
            let static_bindings = Self::get_static_descriptor_bindings(model.textures.len() as u32);
            let static_set_layout = ctx.create_descriptor_set_layout(&static_bindings).unwrap();

            let dynamic_bindings = Self::get_dynamic_descriptor_bindings();
            let dynamic_set_layout = ctx.create_descriptor_set_layout(&dynamic_bindings).unwrap();

            let set_layouts = &[static_set_layout, dynamic_set_layout, dynamic_set_layout];

            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .push_constant_ranges(&[vk::PushConstantRange {
                    offset: 0,
                    size: 4,
                    stage_flags: ShaderStageFlags::ALL,
                }])
                .set_layouts(set_layouts);

            let layout = ctx
                .device
                .create_pipeline_layout(&layout_info, None)
                .unwrap();

            let pool_sizes = &calculate_pool_sizes(&[
                CalculatePoolSizesDesc {
                    bindings: static_bindings.as_slice(),
                    num_sets: 1,
                },
                CalculatePoolSizesDesc {
                    bindings: dynamic_bindings.as_slice(),
                    num_sets: 2,
                },
            ]);

            let descriptor_pool = ctx.create_descriptor_pool(3, pool_sizes).unwrap();

            let static_set =
                allocate_descriptor_set(&ctx.device, &descriptor_pool, &static_set_layout).unwrap();
            let dynamic_sets =
                allocate_descriptor_sets(&ctx.device, &descriptor_pool, &dynamic_set_layout, 2)
                    .unwrap();

            let uniform_buffer = ctx
                .create_aligned_buffer(
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryLocation::CpuToGpu,
                    size_of::<GConst>() as u64,
                    None,
                    16,
                )
                .unwrap();

            let static_writes = vec![
                WriteDescriptorSetKind::StorageImage {
                    view: resources.motion_vectors.view,
                    layout: vk::ImageLayout::GENERAL,
                },
                WriteDescriptorSetKind::AccelerationStructure {
                    acceleration_structure: top_as.handle,
                },
                WriteDescriptorSetKind::UniformBuffer {
                    buffer: uniform_buffer.inner,
                },
                WriteDescriptorSetKind::StorageBuffer {
                    buffer: model.geometry_info_buffer.inner,
                },
                WriteDescriptorSetKind::StorageBuffer {
                    buffer: model.vertex_buffer.inner,
                },
                WriteDescriptorSetKind::StorageBuffer {
                    buffer: model.index_buffer.inner,
                },
                WriteDescriptorSetKind::CombinedImageSampler {
                    view: *skybox_view,
                    sampler: *skybox_sampler,
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
                WriteDescriptorSetKind::StorageBuffer {
                    buffer: resources.light_buffer.inner,
                },
                WriteDescriptorSetKind::StorageBuffer {
                    buffer: resources.neighbor_offsets_buffer.inner,
                },
                WriteDescriptorSetKind::CombinedImageSampler {
                    view: resources.environment_pdf_texture.view,
                    sampler: resources.environment_pdf_sampler,
                    layout: vk::ImageLayout::GENERAL,
                },
                WriteDescriptorSetKind::CombinedImageSampler {
                    view: resources.local_light_pdf_texture.view,
                    sampler: resources.local_light_pdf_sampler,
                    layout: vk::ImageLayout::GENERAL,
                },
                WriteDescriptorSetKind::StorageBuffer {
                    buffer: resources.geometry_instance_to_light_buffer.inner,
                },
                WriteDescriptorSetKind::StorageBuffer {
                    buffer: resources.di_reservoir_buffer.inner,
                },
                WriteDescriptorSetKind::StorageImage {
                    view: resources.diffuse_lighting.view,
                    layout: vk::ImageLayout::GENERAL,
                },
                WriteDescriptorSetKind::StorageImage {
                    view: resources.specular_lighting.view,
                    layout: vk::ImageLayout::GENERAL,
                },
                WriteDescriptorSetKind::StorageImage {
                    view: resources.temporal_sample_position.view,
                    layout: vk::ImageLayout::GENERAL,
                },
                WriteDescriptorSetKind::StorageBuffer {
                    buffer: resources.gi_reservoir_buffer.inner,
                },
                WriteDescriptorSetKind::StorageBuffer {
                    buffer: resources.ris_buffer.inner,
                },
                WriteDescriptorSetKind::StorageBuffer {
                    buffer: resources.ris_light_data_buffer.inner,
                },
                WriteDescriptorSetKind::StorageBuffer {
                    buffer: resources.secondary_gbuffer.inner,
                },
            ];

            let static_writes = static_writes
                .into_iter()
                .enumerate()
                .map(|(i, b)| WriteDescriptorSet {
                    binding: i as u32,
                    kind: b,
                })
                .collect::<Vec<WriteDescriptorSet>>();
            update_descriptor_sets(ctx, &static_set, static_writes.as_slice());

            for i in 0..2 as usize {
                let dynamic_writes = vec![
                    WriteDescriptorSetKind::StorageImage {
                        view: resources.g_buffers[i].depth.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                    WriteDescriptorSetKind::StorageImage {
                        view: resources.g_buffers[i].normal.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                    WriteDescriptorSetKind::StorageImage {
                        view: resources.g_buffers[i].geo_normals.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                    WriteDescriptorSetKind::StorageImage {
                        view: resources.g_buffers[i].diffuse_albedo.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                    WriteDescriptorSetKind::StorageImage {
                        view: resources.g_buffers[i].specular_rough.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                    WriteDescriptorSetKind::StorageImage {
                        view: resources.g_buffers[i].emissive.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                    WriteDescriptorSetKind::StorageImage {
                        view: resources.g_buffers[1 - i].depth.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                    WriteDescriptorSetKind::StorageImage {
                        view: resources.g_buffers[1 - i].normal.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                    WriteDescriptorSetKind::StorageImage {
                        view: resources.g_buffers[1 - i].geo_normals.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                    WriteDescriptorSetKind::StorageImage {
                        view: resources.g_buffers[1 - i].diffuse_albedo.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                    WriteDescriptorSetKind::StorageImage {
                        view: resources.g_buffers[1 - i].specular_rough.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                    WriteDescriptorSetKind::StorageImage {
                        view: resources.g_buffers[1 - i].emissive.view,
                        layout: vk::ImageLayout::GENERAL,
                    },
                ];

                let dynamic_writes = dynamic_writes
                    .into_iter()
                    .enumerate()
                    .map(|(i, b)| WriteDescriptorSet {
                        binding: i as u32,
                        kind: b,
                    })
                    .collect::<Vec<WriteDescriptorSet>>();

                update_descriptor_sets(ctx, &dynamic_sets[i], dynamic_writes.as_slice());
            }

            for (i, (image_index, sampler_index)) in model.textures.iter().enumerate() {
                let view = &model.views[*image_index];
                let sampler = &model.samplers[*sampler_index];
                let img_info = vk::DescriptorImageInfo::default()
                    .image_view(*view)
                    .sampler(*sampler)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

                ctx.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_array_element(i as u32)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .dst_binding(20)
                        .dst_set(static_set.clone())
                        .image_info(&[img_info])],
                    &[],
                );
            }

            Self {
                presample_lights_pass: ComputePass::new(
                    ctx,
                    layout,
                    "./src/shaders/bin/presample_locallights.spv",
                )
                .unwrap(),
                presample_environment_map_pass: ComputePass::new(
                    ctx,
                    layout,
                    "./src/shaders/bin/presample_environment.spv",
                )
                .unwrap(),
                g_buffer_pass: RayTracingPass::new(
                    ctx,
                    layout,
                    "./src/shaders/bin/g_buffer.spv",
                )
                .unwrap(),
                di_fused_resampling_pass: RayTracingPass::new(
                    ctx,
                    layout,
                    "./src/shaders/bin/di_fused_resampling.spv",
                )
                .unwrap(),
                brdf_ray_tracing_pass: RayTracingPass::new(
                    ctx,
                    layout,
                    "./src/shaders/bin/brdf_rays.spv",
                )
                .unwrap(),
                shade_secondary_surfaces_pass: RayTracingPass::new(
                    ctx,
                    layout,
                    "./src/shaders/bin/shade_secondary_surfaces.spv",
                )
                .unwrap(),
                gi_temporal_resampling_pass: RayTracingPass::new(
                    ctx,
                    layout,
                    "./src/shaders/bin/temporal_resampling.spv",
                )
                .unwrap(),
                gi_spatial_resampling_pass: RayTracingPass::new(
                    ctx,
                    layout,
                    "./src/shaders/bin/spatial_resampling.spv",
                )
                .unwrap(),
                gi_final_shading_pass: RayTracingPass::new(
                    ctx,
                    layout,
                    "./src/shaders/bin/gi_final_shading.spv",
                )
                .unwrap(),
                layout,
                static_set,
                current_set: dynamic_sets[0],
                prev_set: dynamic_sets[1],
                uniform_buffer,
            }
        }
    }

    pub fn execute_presampeling(
        &self,
        ctx: &Renderer,
        cmd: &vk::CommandBuffer,
        frame: u64,
        skybox_changed: bool,
    ) -> bool {
        unsafe {
            ctx.device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[
                    self.static_set,
                    if frame % 2 == 0 {
                        self.current_set
                    } else {
                        self.prev_set
                    },
                    if frame % 2 == 0 {
                        self.prev_set
                    } else {
                        self.current_set
                    },
                ],
                &[],
            );
            self.presample_lights_pass
                .execute(ctx, cmd, 1024 / 256, 128, 1);
            if skybox_changed {
                self.presample_environment_map_pass
                    .execute(ctx, cmd, 1024 / 256, 128, 1);
                return false;
            } else {
                return true;
            }
        }
    }

    pub fn execute(&self, ctx: &Renderer, cmd: &vk::CommandBuffer, frame: u64) {
        unsafe {
            ctx.memory_barrier(
                cmd,
                PipelineStageFlags::ALL_COMMANDS,
                PipelineStageFlags::ALL_COMMANDS,
                AccessFlags::MEMORY_WRITE | AccessFlags::MEMORY_READ,
                AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
            );

            ctx.device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.layout,
                0,
                &[
                    self.static_set,
                    if frame % 2 == 0 {
                        self.current_set
                    } else {
                        self.prev_set
                    },
                    if frame % 2 == 0 {
                        self.prev_set
                    } else {
                        self.current_set
                    },
                ],
                &[],
            );

            let frame_c =
                std::slice::from_raw_parts(&frame as *const u64 as *const u8, size_of::<u32>());

            ctx.device.cmd_push_constants(
                *cmd,
                self.layout,
                vk::ShaderStageFlags::ALL,
                0,
                frame_c,
            );

            self.g_buffer_pass.execute(ctx, cmd);

            ctx.memory_barrier(
                cmd,
                PipelineStageFlags::ALL_COMMANDS,
                PipelineStageFlags::ALL_COMMANDS,
                AccessFlags::MEMORY_WRITE | AccessFlags::MEMORY_READ,
                AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
            );

            // self.di_fused_resampling_pass.execute(ctx, cmd);
            self.brdf_ray_tracing_pass.execute(ctx, cmd);

            ctx.memory_barrier(
                cmd,
                PipelineStageFlags::ALL_COMMANDS,
                PipelineStageFlags::ALL_COMMANDS,
                AccessFlags::MEMORY_WRITE | AccessFlags::MEMORY_READ,
                AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
            );

            self.shade_secondary_surfaces_pass.execute(ctx, cmd);

            // ctx.memory_barrier(
            //     cmd,
            //     PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            //     PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            //     AccessFlags::SHADER_WRITE,
            //     AccessFlags::SHADER_READ,
            // );

            // self.gi_temporal_resampling_pass.execute(ctx, cmd);
            // self.gi_spatial_resampling_pass.execute(ctx, cmd);

            ctx.memory_barrier(
                cmd,
                PipelineStageFlags::ALL_COMMANDS,
                PipelineStageFlags::ALL_COMMANDS,
                AccessFlags::MEMORY_WRITE | AccessFlags::MEMORY_READ,
                AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
            );
            
            self.gi_final_shading_pass.execute(ctx, cmd);
        }
    }

    pub fn update_uniform(&self, g_const: &GConst) -> Result<()> {
        self.uniform_buffer
            .copy_data_to_aligned_buffer(std::slice::from_ref(g_const), 16)
    }
}

pub fn fill_neighbor_offset_buffer(neighbor_offset_count: u32) -> Vec<u8> {
    let mut buffer = vec![];

    let r = 250;
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

        let r_sq = (u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5);
        if r_sq > 0.25 {
            continue;
        }

        buffer.push(((u - 0.5) * r as f32) as u8);
        buffer.push(((v - 0.5) * r as f32) as u8);
    }

    buffer
}

pub fn compute_pdf_texture_size(max_items: u32) -> (u32, u32, u32) {
    // Compute the size of a power-of-2 rectangle that fits all items, 1 item per pixel
    let mut texture_width = f64::max(1.0, f64::ceil(f64::sqrt(max_items as f64)));
    texture_width = f64::exp2(f64::ceil(f64::log2(texture_width)));
    let mut texture_height = f64::max(1.0, f64::ceil(max_items as f64 / texture_width));
    texture_height = f64::exp2(f64::ceil(f64::log2(texture_height)));
    let texture_mips = f64::max(
        1.0,
        f64::log2(f64::max(texture_width, texture_height)) + 1.0,
    );

    (
        texture_width as u32,
        texture_height as u32,
        texture_mips as u32,
    )
}

pub fn calculate_reservoir_buffer_parameters(
    render_width: u32,
    render_height: u32,
) -> RTXDI_ReservoirBufferParameters {
    let render_width_blocks =
        (render_width + RTXDI_RESERVOIR_BLOCK_SIZE - 1) / RTXDI_RESERVOIR_BLOCK_SIZE;
    let render_height_blocks =
        (render_height + RTXDI_RESERVOIR_BLOCK_SIZE - 1) / RTXDI_RESERVOIR_BLOCK_SIZE;
    let mut params = RTXDI_ReservoirBufferParameters::default();
    params.reservoir_block_row_pitch =
        render_width_blocks * (RTXDI_RESERVOIR_BLOCK_SIZE * RTXDI_RESERVOIR_BLOCK_SIZE);
    params.reservoir_array_pitch = params.reservoir_block_row_pitch * render_height_blocks;
    return params;
}
