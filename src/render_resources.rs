use std::cell::LazyCell;

use ash::vk::{self, BufferUsageFlags, ImageLayout, ImageUsageFlags, ImageView};
use glam::UVec2;
use gpu_allocator::MemoryLocation;

use crate::{
    light_passes::{
        calculate_reservoir_buffer_parameters, compute_pdf_texture_size,
        fill_neighbor_offset_buffer,
    },
    Buffer, ImageAndView, ImageBarrier, Model, Renderer, NEIGHBOR_OFFSET_COUNT, WINDOW_SIZE,
};

pub struct RenderResources {
    pub task_buffer: Buffer,
    pub light_buffer: Buffer,
    pub geometry_instance_to_light_buffer: Buffer,
    pub geometry_instance_to_light_buffer_staging: Buffer,
    pub ris_buffer: Buffer,
    pub ris_light_data_buffer: Buffer,
    pub neighbor_offsets_buffer: Buffer,
    pub di_reservoir_buffer: Buffer,
    pub environment_pdf_texture: ImageAndView,
    pub environment_pdf_texture_mips: Vec<ImageView>,
    pub environment_pdf_sampler: vk::Sampler,
    pub local_light_pdf_texture: ImageAndView,
    pub local_light_pdf_texture_mips: Vec<ImageView>,
    pub local_light_pdf_sampler: vk::Sampler,
    pub gi_reservoir_buffer: Buffer,
    pub motion_vectors: ImageAndView,
    pub g_buffers: [GBuffer; 2],
    pub diffuse_lighting: ImageAndView,
    pub specular_lighting: ImageAndView,
    pub secondary_gbuffer: Buffer,
}

#[derive(Default)]
pub struct GBuffer {
    pub depth: ImageAndView,
    pub normal: ImageAndView,
    pub geo_normals: ImageAndView,
    pub diffuse_albedo: ImageAndView,
    pub specular_rough: ImageAndView,
    pub emissive: ImageAndView,
}

impl GBuffer {
    fn new(ctx: &mut Renderer) -> Self {
        let depth = ctx
            .create_storage_image(
                WINDOW_SIZE.x as u32,
                WINDOW_SIZE.y as u32,
                vk::Format::R32_SFLOAT,
            )
            .unwrap();
        let normal = ctx
            .create_storage_image(
                WINDOW_SIZE.x as u32,
                WINDOW_SIZE.y as u32,
                vk::Format::R32_UINT,
            )
            .unwrap();
        let geo_normals = ctx
            .create_storage_image(
                WINDOW_SIZE.x as u32,
                WINDOW_SIZE.y as u32,
                vk::Format::R32_UINT,
            )
            .unwrap();
        let diffuse_albedo = ctx
            .create_storage_image(
                WINDOW_SIZE.x as u32,
                WINDOW_SIZE.y as u32,
                vk::Format::R32_UINT,
            )
            .unwrap();
        let specular_rough = ctx
            .create_storage_image(
                WINDOW_SIZE.x as u32,
                WINDOW_SIZE.y as u32,
                vk::Format::R32_UINT,
            )
            .unwrap();
        let emissive = ctx
            .create_storage_image(
                WINDOW_SIZE.x as u32,
                WINDOW_SIZE.y as u32,
                vk::Format::R16G16B16A16_SFLOAT,
            )
            .unwrap();

        Self {
            depth,
            diffuse_albedo,
            geo_normals,
            normal,
            specular_rough,
            emissive,
        }
    }

    pub fn barriers(
        &self,
        src: vk::PipelineStageFlags2,
        dst: vk::PipelineStageFlags2,
    ) -> [vk::ImageMemoryBarrier2; 6] {
        let binding = [
            &self.depth,
            &self.diffuse_albedo,
            &self.geo_normals,
            &self.normal,
            &self.specular_rough,
            &self.emissive,
        ];
        let mut barriers = [vk::ImageMemoryBarrier2::default(); 6];
        binding.iter().enumerate().for_each(|(i, image)| {
            barriers[i] = image.image.barrier(
                vk::AccessFlags2::MEMORY_WRITE,
                vk::AccessFlags2::MEMORY_READ,
                src,
                dst,
            )
        });
        barriers
    }
}

impl RenderResources {
    pub fn new(ctx: &mut Renderer, model: &Model, skybox_size: UVec2) -> Self {
        let reservoir_buffer_params =
            calculate_reservoir_buffer_parameters(WINDOW_SIZE.x as u32, WINDOW_SIZE.y as u32);
        let reservoir_buffer_size = reservoir_buffer_params.reservoir_array_pitch as u64 * 2 * 32;

        let task_buffer = ctx
            .create_buffer(
                BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::CpuToGpu,
                16 * 500,
                None,
            )
            .unwrap();
        let light_buffer = ctx
            .create_buffer(
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuOnly,
                48 * model.lights as u64,
                None,
            )
            .unwrap();
        let geometry_instance_to_light_buffer = ctx
            .create_buffer(
                BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                model.geometry_infos.len() as u64 * 4,
                None,
            )
            .unwrap();
        let geometry_instance_to_light_buffer_staging = ctx
            .create_buffer(
                BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                model.geometry_infos.len() as u64 * 4,
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
        let ris_light_data_buffer = ctx
            .create_buffer(
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuOnly,
                model.lights as u64 * 36,
                None,
            )
            .unwrap();
        let neighbor_offsets = fill_neighbor_offset_buffer(NEIGHBOR_OFFSET_COUNT);
        let neighbor_offsets_buffer = ctx
            .create_gpu_only_buffer_from_data(
                vk::BufferUsageFlags::STORAGE_BUFFER,
                neighbor_offsets.as_slice(),
                None,
            )
            .unwrap();
        let di_reservoir_buffer = ctx
            .create_buffer(
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuOnly,
                reservoir_buffer_params.reservoir_array_pitch as u64 * 2 * 24,
                None,
            )
            .unwrap();
        let gi_reservoir_buffer = ctx
            .create_buffer(
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuOnly,
                reservoir_buffer_size,
                None,
            )
            .unwrap();

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

        let environment_pdf_texture_mips = ctx.create_image_views(&environment_pdf).unwrap();
        let environenvironment_pdf_view = ctx.create_image_view(&environment_pdf).unwrap();

        let (width, height, mips) = compute_pdf_texture_size(model.lights);

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

        let local_light_pdf_texture_mips = ctx.create_image_views(&local_lights_pdf).unwrap();
        let local_light_pdf_image_view = ctx.create_image_view(&local_lights_pdf).unwrap();

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

        let environment_pdf_sampler = unsafe {
            ctx.device
                .create_sampler(&sampler_create_info, None)
                .unwrap()
        };

        sampler_create_info.max_lod = local_lights_pdf.mip_levels as f32;

        let local_light_pdf_sampler = unsafe {
            ctx.device
                .create_sampler(&sampler_create_info, None)
                .unwrap()
        };

        let environment_pdf_texture = ImageAndView {
            image: environment_pdf,
            view: environenvironment_pdf_view,
        };
        let local_light_pdf_texture = ImageAndView {
            image: local_lights_pdf,
            view: local_light_pdf_image_view,
        };
        let mut g_buffers: [GBuffer; 2] = Default::default();
        for i in 0..2 {
            g_buffers[i] = GBuffer::new(ctx);
        }

        let motion_vectors = ctx
            .create_storage_image(
                WINDOW_SIZE.x as u32,
                WINDOW_SIZE.y as u32,
                vk::Format::R32G32B32A32_SFLOAT,
            )
            .unwrap();
        let diffuse_lighting = ctx
            .create_storage_image(
                WINDOW_SIZE.x as u32,
                WINDOW_SIZE.y as u32,
                vk::Format::R16G16B16A16_SFLOAT,
            )
            .unwrap();
        let specular_lighting = ctx
            .create_storage_image(
                WINDOW_SIZE.x as u32,
                WINDOW_SIZE.y as u32,
                vk::Format::R16G16B16A16_SFLOAT,
            )
            .unwrap();

        let secondary_gbuffer = ctx
            .create_buffer(
                BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuOnly,
                (12 + 4 + 8 + 4 + 4 + 12 + 4)
                    * reservoir_buffer_params.reservoir_array_pitch as u64,
                None,
            )
            .unwrap();

        Self {
            motion_vectors,
            g_buffers,
            task_buffer,
            light_buffer,
            geometry_instance_to_light_buffer,
            geometry_instance_to_light_buffer_staging,
            ris_buffer,
            ris_light_data_buffer,
            neighbor_offsets_buffer,
            di_reservoir_buffer,
            environment_pdf_texture,
            local_light_pdf_texture,
            environment_pdf_texture_mips,
            local_light_pdf_texture_mips,
            gi_reservoir_buffer,
            environment_pdf_sampler,
            local_light_pdf_sampler,
            diffuse_lighting,
            specular_lighting,
            secondary_gbuffer,
        }
    }
}
