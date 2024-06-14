use std::{
    borrow::BorrowMut,
    ffi::{c_char, CStr, CString},
    mem::{align_of, size_of, size_of_val},
    num,
    ops::Sub,
    os::raw::c_void,
    rc::Rc,
    time::{Duration, Instant},
};

use anyhow::Result;
use ash::{
    extensions::{ext::DebugUtils, khr},
    vk::{self, FramebufferMixedSamplesCombinationNV, ImageView},
    Device, Entry, Instance,
};
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc},
    AllocatorDebugSettings, MemoryLocation,
};
use image::{DynamicImage, GenericImageView};
use log::debug;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use simple_logger::SimpleLogger;
use std::slice::from_ref;
use winit::dpi::PhysicalSize;

use crate::{module_from_bytes, rendergraph::Renderer, RayTracingShaderCreateInfo, RayTracingShaderGroup};

pub const FRAMES_IN_FLIGHT: u32 = 2;

impl Renderer {
    pub fn create_aligned_buffer(
        &mut self,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        size: vk::DeviceSize,
        name: Option<&str>,
        alignment: u64,
    ) -> Result<Buffer> {
        Buffer::new(
            &self.device,
            &mut self.allocator,
            usage,
            memory_location,
            size,
            name,
            Some(alignment),
        )
    }

    pub fn create_buffer(
        &mut self,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        size: vk::DeviceSize,
        name: Option<&str>,
    ) -> Result<Buffer> {
        Buffer::new(
            &self.device,
            &mut self.allocator,
            usage,
            memory_location,
            size,
            name,
            None,
        )
    }


    pub fn copy_buffer_to_image(
        &self,
        cmd: &vk::CommandBuffer,
        src: &Buffer,
        dst: &Image,
        layout: vk::ImageLayout,
    ) {
        let region = vk::BufferImageCopy::builder()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(dst.extent);

        unsafe {
            self.device.cmd_copy_buffer_to_image(
                *cmd,
                src.inner,
                dst.inner,
                layout,
                std::slice::from_ref(&region),
            );
        };
    }

    pub fn get_buffer_address(&self, buffer: &Buffer) -> u64 {
        buffer.get_device_address(&self.device)
    }

    pub fn create_gpu_only_buffer_from_data_with_size<T: Copy>(
        &mut self,
        usage: vk::BufferUsageFlags,
        data: &[T],
        size: u64,
        name: Option<&str>,
    ) -> Result<Buffer> {
        let staging_buffer = self.create_buffer(
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            size,
            Some(format!("Staging Buffer for: {}", name.unwrap_or("Buffer")).as_str()),
        )?;
        staging_buffer.copy_data_to_buffer(data)?;

        let buffer = self.create_buffer(
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            size,
            name,
        )?;

        self.execute_one_time_commands(|cmd_buffer| {
            copy_buffer(&self.device, cmd_buffer, &staging_buffer, &buffer);
        })?;

        Ok(buffer)
    }

    pub fn create_gpu_only_buffer_from_data<T: Copy>(
        &mut self,
        device: &Device,
        usage: vk::BufferUsageFlags,
        data: &[T],
        name: Option<&str>,
    ) -> Result<Buffer> {
        let size = size_of_val(data) as _;
        self.create_gpu_only_buffer_from_data_with_size(usage, data, size, name)
    }

    pub fn execute_one_time_commands<R, F: FnOnce(&vk::CommandBuffer) -> R>(
        &self,
        executor: F,
    ) -> Result<R> {
        let command_buffer = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_pool(self.command_pool)
                    .build(),
            )?
        }[0];

        begin_command_buffer(
            &command_buffer,
            &self.device,
            Some(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let executor_result = executor(&command_buffer);

        unsafe { self.device.end_command_buffer(command_buffer)? };

        let fence = self.create_fence(None)?;
        queue_submit(
            &self.device,
            &self.graphics_queue.handel,
            &command_buffer,
            None,
            None,
            &fence,
        )?;
        fence.wait(&self.device, None)?;

        free_command_buffer(&self.command_pool, &self.device, command_buffer)?;

        Ok(executor_result)
    }

    pub fn create_fence(&self, flags: Option<vk::FenceCreateFlags>) -> Result<Fence> {
        Fence::new(&self.device, flags)
    }

    pub fn create_image(
        &mut self,
        usage: vk::ImageUsageFlags,
        memory_location: MemoryLocation,
        format: vk::Format,
        width: u32,
        height: u32,
    ) -> Result<Image> {
        Image::new_2d(
            &self.device,
            &mut self.allocator,
            usage,
            memory_location,
            format,
            width,
            height,
        )
    }

    pub fn create_image_view(&self, image: &Image) -> Result<vk::ImageView> {
        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image.inner)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(image.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let res = unsafe { self.device.create_image_view(&view_info, None)? };

        Ok(res)
    }

    pub fn create_descriptor_set_layout(
        &self,
        bindings: &[vk::DescriptorSetLayoutBinding],
        flags: &[vk::DescriptorBindingFlags],
    ) -> Result<vk::DescriptorSetLayout> {
        create_descriptor_set_layout(&self.device, bindings, flags)
    }

    pub fn create_descriptor_pool(
        &self,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> Result<vk::DescriptorPool> {
        create_descriptor_pool(&self.device, max_sets, pool_sizes)
    }

    pub fn create_shader_module(&self, code_path: &str) -> vk::ShaderModule {
        let mut code = std::fs::File::open(code_path).unwrap();
        let decoded_code = ash::util::read_spv(&mut code).unwrap();
        let create_info = vk::ShaderModuleCreateInfo::builder().code(&decoded_code);

        unsafe { self.device.create_shader_module(&create_info, None) }.unwrap()
    }

    pub fn create_shader_stage(
        &self,
        stage: vk::ShaderStageFlags,
        path: &str,
    ) -> (vk::PipelineShaderStageCreateInfo, vk::ShaderModule) {
        let module = self.create_shader_module(path);
        (
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(stage)
                .module(module)
                .name(CStr::from_bytes_with_nul("main\0".as_bytes()).unwrap())
                .build(),
            module,
        )
    }

    pub fn pipeline_image_barriers(&self, cmd: &vk::CommandBuffer, barriers: &[ImageBarrier]) {
        let barriers = barriers
            .iter()
            .map(|b| {
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(b.src_stage_mask)
                    .src_access_mask(b.src_access_mask)
                    .old_layout(b.old_layout)
                    .dst_stage_mask(b.dst_stage_mask)
                    .dst_access_mask(b.dst_access_mask)
                    .new_layout(b.new_layout)
                    .image(b.image.inner)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: b.image.mip_levls,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .build()
            })
            .collect::<Vec<_>>();

        let dependency_info = vk::DependencyInfo::builder().image_memory_barriers(&barriers);

        unsafe { self.device.cmd_pipeline_barrier2(*cmd, &dependency_info) };
    }

    pub fn copy_image(
        &self,
        cmd: &vk::CommandBuffer,
        src_image: &Image,
        src_layout: vk::ImageLayout,
        dst_image: &Image,
        dst_layout: vk::ImageLayout,
    ) {
        let region = vk::ImageCopy::builder()
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                mip_level: 0,
                layer_count: 1,
            })
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                mip_level: 0,
                layer_count: 1,
            })
            .extent(vk::Extent3D {
                width: src_image.extent.width,
                height: src_image.extent.height,
                depth: 1,
            });

        unsafe {
            self.device.cmd_copy_image(
                *cmd,
                src_image.inner,
                src_layout,
                dst_image.inner,
                dst_layout,
                std::slice::from_ref(&region),
            )
        };
    }
}

pub struct Target {
    pub ash_surface_ext: khr::Surface,
    pub vk_surface: vk::SurfaceKHR,
    pub swapchain: Swapchain,
    pub frames_in_flight: Vec<Frame>,
    pub frame_in_flight: usize,
}

impl Target {
    pub fn current_frame<'a>(self) -> & 'a Frame {
        &self.frames_in_flight[self.frame_in_flight]
    }
}

pub fn transition_image_layout_to_general(
    device: &Device,
    command_pool: &vk::CommandPool,
    image: &Image,
    queue: &vk::Queue,
) -> Result<()> {
    transition_image_layout_to(
        device,
        command_pool,
        image,
        queue,
        vk::ImageLayout::GENERAL,
        vk::ImageAspectFlags::COLOR,
    )
}

pub fn transition_image_layout_to(
    device: &Device,
    command_pool: &vk::CommandPool,
    image: &Image,
    queue: &vk::Queue,
    layout: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
) -> Result<()> {
    let cmd = allocate_command_buffer(&device, &command_pool, vk::CommandBufferLevel::PRIMARY)?;

    let barrier = vk::ImageMemoryBarrier2::builder()
        .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
        .src_access_mask(vk::AccessFlags2::NONE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
        .dst_access_mask(vk::AccessFlags2::NONE)
        .new_layout(layout)
        .image(image.inner)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .build();
    unsafe {
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                .build(),
        )?
    };
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::builder()
                .image_memory_barriers(&[barrier])
                .build(),
        )
    };
    unsafe { device.end_command_buffer(cmd)? };

    let fence = Fence {
        handel: unsafe { device.create_fence(&vk::FenceCreateInfo::builder().build(), None)? },
    };
    queue_submit(&device, &queue, &cmd, None, None, &fence)?;
    fence.wait(&device, None)?;

    free_command_buffer(&command_pool, &device, cmd)?;
    Ok(())
}

pub fn copy_buffer_to_image(
    device: &Device,
    cmd: &vk::CommandBuffer,
    src: &Buffer,
    dst: &Image,
    layout: vk::ImageLayout,
) {
    let region = vk::BufferImageCopy::builder()
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_extent(dst.extent);

    unsafe {
        device.cmd_copy_buffer_to_image(
            *cmd,
            src.inner,
            dst.inner,
            layout,
            std::slice::from_ref(&region),
        );
    };
}

pub fn create_fence(device: &Device, flags: Option<vk::FenceCreateFlags>) -> Result<Fence> {
    Fence::new(&device, flags)
}

pub fn create_image_view(device: &Device, image: &Image) -> Result<vk::ImageView> {
    let view_info = vk::ImageViewCreateInfo::builder()
        .image(image.inner)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(image.format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let res = unsafe { device.create_image_view(&view_info, None)? };

    Ok(res)
}

pub fn create_shader_module(device: &Device, code_path: String) -> vk::ShaderModule {
    let mut code = std::fs::File::open(code_path).unwrap();
    let decoded_code = ash::util::read_spv(&mut code).unwrap();
    let create_info = vk::ShaderModuleCreateInfo::builder().code(&decoded_code);

    unsafe { device.create_shader_module(&create_info, None) }.unwrap()
}

pub fn create_compute_pipeline(renderer: &Renderer, layout: &vk::PipelineLayout, code_path: &str) -> Result<vk::Pipeline> {
    let entry_point_name: CString = CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(renderer.create_shader_module(code_path))
        .name(&entry_point_name)
        .build();
    
    let create_info = vk::ComputePipelineCreateInfo::builder()
        .layout(*layout)
        .stage(stage)
        .build();
    let pipelines = unsafe { renderer.device.create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None) }.unwrap();
    Ok(pipelines[0])
}

pub fn create_shader_stage(
    device: &Device,
    stage: vk::ShaderStageFlags,
    path: String,
) -> (vk::PipelineShaderStageCreateInfo, vk::ShaderModule) {
    let module = create_shader_module(device, path);
    (
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(stage)
            .module(module)
            .name(CStr::from_bytes_with_nul("main\0".as_bytes()).unwrap())
            .build(),
        module,
    )
}

pub fn pipeline_image_barriers(device: &Device, cmd: &vk::CommandBuffer, barriers: &[ImageBarrier]) {
    let barriers = barriers
        .iter()
        .map(|b| {
            vk::ImageMemoryBarrier2::builder()
                .src_stage_mask(b.src_stage_mask)
                .src_access_mask(b.src_access_mask)
                .old_layout(b.old_layout)
                .dst_stage_mask(b.dst_stage_mask)
                .dst_access_mask(b.dst_access_mask)
                .new_layout(b.new_layout)
                .image(b.image.inner)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: b.image.mip_levls,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build()
        })
        .collect::<Vec<_>>();

    let dependency_info = vk::DependencyInfo::builder().image_memory_barriers(&barriers);

    unsafe { device.cmd_pipeline_barrier2(*cmd, &dependency_info) };
}

pub fn copy_image(
    device: &Device,
    cmd: &vk::CommandBuffer,
    src_image: &Image,
    src_layout: vk::ImageLayout,
    dst_image: &Image,
    dst_layout: vk::ImageLayout,
) {
    let region = vk::ImageCopy::builder()
        .src_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_array_layer: 0,
            mip_level: 0,
            layer_count: 1,
        })
        .dst_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_array_layer: 0,
            mip_level: 0,
            layer_count: 1,
        })
        .extent(vk::Extent3D {
            width: src_image.extent.width,
            height: src_image.extent.height,
            depth: 1,
        });

    unsafe {
        device.cmd_copy_image(
            *cmd,
            src_image.inner,
            src_layout,
            dst_image.inner,
            dst_layout,
            std::slice::from_ref(&region),
        )
    };
}

#[derive(Clone, Copy, Debug)]
pub struct ImageBarrier<'a> {
    pub image: &'a Image,
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
    pub src_access_mask: vk::AccessFlags2,
    pub dst_access_mask: vk::AccessFlags2,
    pub src_stage_mask: vk::PipelineStageFlags2,
    pub dst_stage_mask: vk::PipelineStageFlags2,
}

pub fn create_descriptor_set_layout(
    device: &Device,
    bindings: &[vk::DescriptorSetLayoutBinding],
    flags: &[vk::DescriptorBindingFlags],
) -> Result<vk::DescriptorSetLayout> {
    let mut dsl_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

    let mut info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(flags);
    dsl_info = dsl_info.push_next(&mut info);

    let res = unsafe { device.create_descriptor_set_layout(&dsl_info, None)? };

    Ok(res)
}

pub fn create_descriptor_pool(
    device: &Device,
    max_sets: u32,
    pool_sizes: &[vk::DescriptorPoolSize],
) -> Result<vk::DescriptorPool> {
    let pool_info = vk::DescriptorPoolCreateInfo::builder()
        .max_sets(max_sets)
        .pool_sizes(pool_sizes);
    let res = unsafe { device.create_descriptor_pool(&pool_info, None)? };

    Ok(res)
}

pub fn allocate_descriptor_sets(
    device: &Device,
    pool: &vk::DescriptorPool,
    layout: &vk::DescriptorSetLayout,
    count: u32,
) -> Result<Vec<vk::DescriptorSet>> {
    let layouts = (0..count).map(|_| *layout).collect::<Vec<_>>();

    let sets_alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(*pool)
        .set_layouts(&layouts);
    let res = unsafe { device.allocate_descriptor_sets(&sets_alloc_info)? };
    Ok(res)
}

pub fn allocate_descriptor_set(
    device: &Device,
    pool: &vk::DescriptorPool,
    layout: &vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet> {
    Ok(allocate_descriptor_sets(device, pool, layout, 1)?
        .into_iter()
        .next()
        .unwrap())
}

#[derive(Debug)]
pub struct Image {
    pub inner: vk::Image,
    pub allocation: Option<Allocation>,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levls: u32,
}

impl Image {
    pub(crate) fn from_swapchain_image(
        swapchain_image: vk::Image,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Self {
        let extent = vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        };

        Self {
            inner: swapchain_image,
            allocation: None,
            format,
            extent,
            mip_levls: 1,
        }
    }

    pub(crate) fn new_2d(
        device: &Device,
        allocator: &mut Allocator,
        usage: vk::ImageUsageFlags,
        memory_location: MemoryLocation,
        format: vk::Format,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };

        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let inner = unsafe { device.create_image(&image_info, None)? };
        let requirements = unsafe { device.get_image_memory_requirements(inner) };

        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: "image",
            requirements,
            location: memory_location,
            linear: true,
        })?;

        unsafe { device.bind_image_memory(inner, allocation.memory(), allocation.offset())? };

        Ok(Self {
            inner,
            allocation: Some(allocation),
            format,
            extent,
            mip_levls: 1,
        })
    }

    pub fn new_from_data(
        ctx: &mut Renderer,
        image: DynamicImage,
        format: vk::Format,
    ) -> Result<(Self, vk::ImageView)> {
        let (width, height) = image.dimensions();
        let image_extent = vk::Extent2D { width, height };
        let image_buffer = if format != vk::Format::R8G8B8A8_SRGB {
            let image_data = image.to_rgba32f();
            let image_data_raw = image_data.as_raw();

            let image_buffer = ctx.create_buffer(
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                (size_of::<f32>() * image_data.len()) as u64,
                Some("image buffer"),
            )?;
            image_buffer.copy_data_to_buffer(image_data_raw.as_slice())?;
            image_buffer
        } else {
            let image_data = image.to_rgba8();
            let image_data_raw = image_data.as_raw();

            let image_buffer = ctx.create_buffer(
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                (size_of::<u8>() * image_data.len()) as u64,
                Some("image buffer"),
            )?;
            image_buffer.copy_data_to_buffer(image_data_raw.as_slice())?;
            image_buffer
        };

        log::debug!("TODO: Mip Levels");

        let texture_image = Self::new_2d(
            &ctx.device,
            &mut ctx.allocator,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            MemoryLocation::GpuOnly,
            format,
            width,
            height,
        )?;

        ctx.execute_one_time_commands(|cmd| {
            let texture_barrier = ImageBarrier {
                src_access_mask: vk::AccessFlags2::NONE,
                src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
                dst_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::TRANSFER_KHR,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                image: &texture_image,
            };
            ctx.pipeline_image_barriers(cmd, &[texture_barrier]);

            ctx.copy_buffer_to_image(
                cmd,
                &image_buffer,
                &texture_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            let texture_barrier_end = ImageBarrier {
                src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                src_stage_mask: vk::PipelineStageFlags2::TRANSFER_KHR,
                dst_access_mask: vk::AccessFlags2::SHADER_READ,
                dst_stage_mask: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image: &texture_image,
            };
            ctx.pipeline_image_barriers(cmd, &[texture_barrier_end]);
        })?;

        let image_view = create_image_view(&ctx.device, &texture_image)?;
        Ok((
            Self {
                allocation: texture_image.allocation,
                extent: vk::Extent3D {
                    width,
                    depth: 1,
                    height,
                },
                format: format,
                inner: texture_image.inner,
                mip_levls: 1,
            },
            image_view,
        ))
    }
}

pub fn copy_buffer(device: &Device, cmd: &vk::CommandBuffer, src_buffer: &Buffer, dst_buffer: &Buffer) {
    unsafe {
        let region = vk::BufferCopy::builder().size(src_buffer.size);
        device.cmd_copy_buffer(
            *cmd,
            src_buffer.inner,
            dst_buffer.inner,
            std::slice::from_ref(&region),
        )
    };
}

pub fn alinged_size_u64(size: u64, alignment: u64) -> u64 {
    (size + (alignment - 1)) & !(alignment - 1)
}

pub fn alinged_size(size: u32, alignment: u32) -> u32 {
    (size + (alignment - 1)) & !(alignment - 1)
}

#[derive(Debug)]
pub struct Queue {
    pub family: QueueFamily,
    pub handel: vk::Queue,
}

#[derive(Clone, Debug)]
pub struct QueueFamily {
    index: u32,
    handel: vk::QueueFamilyProperties,
    supports_present: bool,
}

impl QueueFamily {
    pub fn supports_compute(&self) -> bool {
        self.handel.queue_flags.contains(vk::QueueFlags::COMPUTE)
    }

    pub fn supports_graphics(&self) -> bool {
        self.handel.queue_flags.contains(vk::QueueFlags::GRAPHICS)
    }

    pub fn supports_present(&self) -> bool {
        self.supports_present
    }

    pub fn has_queues(&self) -> bool {
        self.handel.queue_count > 0
    }

    pub fn supports_timestamp_queries(&self) -> bool {
        self.handel.timestamp_valid_bits > 0
    }
}

#[derive(Clone, Debug, Default)]
pub struct DeviceFeatures {
    pub ray_tracing_pipeline: bool,
    pub acceleration_structure: bool,
    pub runtime_descriptor_array: bool,
    pub buffer_device_address: bool,
    pub dynamic_rendering: bool,
    pub synchronization2: bool,
}

impl DeviceFeatures {
    pub fn is_compatible_with(&self, requirements: &Self) -> bool {
        (!requirements.ray_tracing_pipeline || self.ray_tracing_pipeline)
            && (!requirements.acceleration_structure || self.acceleration_structure)
            && (!requirements.runtime_descriptor_array || self.runtime_descriptor_array)
            && (!requirements.buffer_device_address || self.buffer_device_address)
            && (!requirements.dynamic_rendering || self.dynamic_rendering)
            && (!requirements.synchronization2 || self.synchronization2)
    }
}

#[derive(Clone, Debug)]
pub struct PhysicalDevice {
    pub handel: vk::PhysicalDevice,
    pub name: String,
    pub device_type: vk::PhysicalDeviceType,
    pub limits: vk::PhysicalDeviceLimits,
    pub queue_families: Vec<QueueFamily>,
    pub supported_extensions: Vec<String>,
    pub supported_surface_formats: Vec<vk::SurfaceFormatKHR>,
    pub supported_present_modes: Vec<vk::PresentModeKHR>,
    pub supported_device_features: DeviceFeatures,
}

impl PhysicalDevice {
    pub fn new(
        instance: &Instance,
        ash_surface: &khr::Surface,
        vk_surface: &vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let props = unsafe { instance.get_physical_device_properties(physical_device) };

        let name = unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
                .to_str()?
                .to_owned()
        };

        let device_type = props.device_type;
        let limits = props.limits;

        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_families = queue_family_properties
            .into_iter()
            .enumerate()
            .map(|(index, p)| {
                let present_support = unsafe {
                    ash_surface
                        .get_physical_device_surface_support(
                            physical_device,
                            index as _,
                            *vk_surface,
                        )
                        .unwrap()
                };

                QueueFamily {
                    index: index as _,
                    handel: p,
                    supports_present: present_support,
                }
            })
            .collect::<Vec<_>>();

        let extension_properties =
            unsafe { instance.enumerate_device_extension_properties(physical_device)? };
        let supported_extensions = extension_properties
            .into_iter()
            .map(|p| {
                let name = unsafe { CStr::from_ptr(p.extension_name.as_ptr()) };
                name.to_str().unwrap().to_owned()
            })
            .collect();

        let supported_surface_formats = unsafe {
            ash_surface.get_physical_device_surface_formats(physical_device, *vk_surface)?
        };

        let supported_present_modes = unsafe {
            ash_surface.get_physical_device_surface_present_modes(physical_device, *vk_surface)?
        };

        let mut ray_tracing_feature = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
        let mut acceleration_struct_feature =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut features12 = vk::PhysicalDeviceVulkan12Features::builder()
            .runtime_descriptor_array(true)
            .buffer_device_address(true);
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut ray_tracing_feature)
            .push_next(&mut acceleration_struct_feature)
            .push_next(&mut features12)
            .push_next(&mut features13);
        unsafe { instance.get_physical_device_features2(physical_device, &mut features) };
        let supported_device_features = DeviceFeatures {
            ray_tracing_pipeline: ray_tracing_feature.ray_tracing_pipeline == vk::TRUE,
            acceleration_structure: acceleration_struct_feature.acceleration_structure == vk::TRUE,
            runtime_descriptor_array: features12.runtime_descriptor_array == vk::TRUE,
            buffer_device_address: features12.buffer_device_address == vk::TRUE,
            dynamic_rendering: features13.dynamic_rendering == vk::TRUE,
            synchronization2: features13.synchronization2 == vk::TRUE,
        };

        Ok(Self {
            handel: physical_device,
            name,
            device_type,
            limits,
            queue_families,
            supported_extensions,
            supported_surface_formats,
            supported_present_modes,
            supported_device_features,
        })
    }

    pub fn supports_extensions(&self, extensions: &[&str]) -> bool {
        let supported_extensions = self
            .supported_extensions
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        extensions.iter().all(|e| supported_extensions.contains(e))
    }
}



pub fn create_raster_pipeline(
    ctx: &mut Renderer,
    layout: &vk::PipelineLayout,
    render_pass: vk::RenderPass,
    vertex_stage: &str,
    fragment_stage: &str
) -> Result<vk::Pipeline> {
    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
        .blend_enable(false)
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .build()];

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .attachments(&color_blend_attachments)
        .logic_op(vk::LogicOp::COPY)
        .logic_op_enable(false)
        .blend_constants([0.0, 0.0, 0.0, 0.0])
        .build();

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_test_enable(false)
        .depth_write_enable(false);

    let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .cull_mode(vk::CullModeFlags::BACK)
        .depth_clamp_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(false)
        .rasterizer_discard_enable(false);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let entry_point_name: CString = CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::builder()
            .module(ctx.create_shader_module(fragment_stage))
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&entry_point_name)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .module(ctx.create_shader_module(vertex_stage))
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(&entry_point_name)
            .build(),
    ];

    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .scissor_count(1)
        .viewport_count(1);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .primitive_restart_enable(false)
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_attribute_descriptions(&[])
        .vertex_binding_descriptions(&[])
        .build();

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
        .color_blend_state(&color_blend_state)
        .depth_stencil_state(&depth_stencil_state)
        .dynamic_state(&dynamic_state)
        .input_assembly_state(&input_assembly_state)
        .vertex_input_state(&vertex_input_state)
        .layout(*layout)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .render_pass(render_pass)
        .stages(&stages)
        .viewport_state(&viewport_state)
        .subpass(0)
        .build();
    let pipeline = unsafe {
        ctx.device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
            .unwrap()
    }[0];

    Ok(pipeline)
}

pub fn create_ray_tracing_pipeline(
    ctx: &Renderer,
    shaders_create_info: &[RayTracingShaderCreateInfo],
    layout: vk::PipelineLayout,
) -> Result<(vk::Pipeline, RayTracingShaderGroupInfo)> {
    let mut shader_group_info = RayTracingShaderGroupInfo {
        group_count: shaders_create_info.len() as u32,
        ..Default::default()
    };

    let mut modules = vec![];
    let mut stages = vec![];
    let mut groups = vec![];

    let entry_point_name: CString = CString::new("main").unwrap();

    for shader in shaders_create_info.iter() {
        let mut this_modules = vec![];
        let mut this_stages = vec![];

        shader.source.into_iter().for_each(|s| {
            let module = module_from_bytes(&ctx.device, s.0).unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::builder()
                .stage(s.1)
                .module(module)
                .name(&entry_point_name)
                .build();
            this_modules.push(module);
            this_stages.push(stage);
        });

        match shader.group {
            RayTracingShaderGroup::RayGen => shader_group_info.raygen_shader_count += 1,
            RayTracingShaderGroup::Miss => shader_group_info.miss_shader_count += 1,
            RayTracingShaderGroup::Hit => shader_group_info.hit_shader_count += 1,
        };

        let shader_index = stages.len();

        let mut group = vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR);
        group = match shader.group {
            RayTracingShaderGroup::RayGen | RayTracingShaderGroup::Miss => {
                group.general_shader(shader_index as _)
            }
            RayTracingShaderGroup::Hit => {
                group = group
                    .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                    .closest_hit_shader(shader_index as _);
                if shader.source.len() >= 2 {
                    group = group
                        .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                        .any_hit_shader((shader_index as u32) + 1);
                }
                if shader.source.len() >= 3 {
                    group = group
                        .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                        .any_hit_shader((shader_index as u32) + 1)
                        .intersection_shader((shader_index as u32) + 2);
                }

                group
            }
        };

        modules.append(&mut this_modules);
        stages.append(&mut this_stages);
        groups.push(group.build());
    }

    let pipe_info = vk::RayTracingPipelineCreateInfoKHR::builder()
        .layout(layout)
        .stages(&stages)
        .groups(&groups)
        .max_pipeline_ray_recursion_depth(1);

    let inner = unsafe {
        ctx.ray_tracing_context.pipeline_fn.create_ray_tracing_pipelines(
            vk::DeferredOperationKHR::null(),
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipe_info),
            None,
        )?[0]
    };

    Ok((inner, shader_group_info))
}


pub struct ShaderBindingTable {
    pub _buffer: Buffer,
    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub miss_region: vk::StridedDeviceAddressRegionKHR,
    pub hit_region: vk::StridedDeviceAddressRegionKHR,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RayTracingShaderGroupInfo {
    pub group_count: u32,
    pub raygen_shader_count: u32,
    pub miss_shader_count: u32,
    pub hit_shader_count: u32,
}

impl ShaderBindingTable {
    pub fn new(ctx: &mut Renderer, pipeline: &vk::Pipeline, desc: RayTracingShaderGroupInfo) -> Result<Self> {
        let handle_size = ctx.ray_tracing_context.pipeline_properties.shader_group_handle_size;
        let handle_alignment = ctx
            .ray_tracing_context
            .pipeline_properties
            .shader_group_handle_alignment;
        let aligned_handle_size = alinged_size(handle_size, handle_alignment);
        let handle_pad = aligned_handle_size - handle_size;

        let group_alignment = ctx
            .ray_tracing_context
            .pipeline_properties
            .shader_group_base_alignment;

        let data_size = desc.group_count * handle_size;
        let handles = unsafe {
            ctx.ray_tracing_context
                .pipeline_fn
                .get_ray_tracing_shader_group_handles(
                    *pipeline,
                    0,
                    desc.group_count,
                    data_size as _,
                )?
        };

        let raygen_region_size = alinged_size(
            desc.raygen_shader_count * aligned_handle_size,
            group_alignment,
        );

        let miss_region_size = alinged_size(
            desc.miss_shader_count * aligned_handle_size,
            group_alignment,
        );
        let hit_region_size =
            alinged_size(desc.hit_shader_count * aligned_handle_size, group_alignment);

        let buffer_size = raygen_region_size + miss_region_size + hit_region_size;
        let mut stb_data = Vec::<u8>::with_capacity(buffer_size as _);
        let groups_shader_count = [
            desc.raygen_shader_count,
            desc.miss_shader_count,
            desc.hit_shader_count,
        ];

        let buffer_usage = vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let memory_location = MemoryLocation::CpuToGpu;

        let buffer = ctx.create_aligned_buffer(
            buffer_usage,
            memory_location,
            buffer_size as _,
            Some("Shader Binding Table"),
            ctx.ray_tracing_context
                .pipeline_properties
                .shader_group_base_alignment
                .into(),
        )?;

        let address = buffer.get_device_address(&ctx.device);

        let mut offset = 0;
        for group_shader_count in groups_shader_count {
            let group_size = group_shader_count * aligned_handle_size;
            let aligned_group_size = alinged_size(group_size, group_alignment);
            let group_pad = aligned_group_size - group_size;

            for _ in 0..group_shader_count {
                for _ in 0..handle_size as usize {
                    stb_data.push(handles[offset]);
                    offset += 1;
                }

                for _ in 0..handle_pad {
                    stb_data.push(0x0);
                }
            }

            for _ in 0..group_pad {
                stb_data.push(0x0);
            }
        }

        buffer.copy_data_to_buffer(&stb_data)?;

        let raygen_region = vk::StridedDeviceAddressRegionKHR::builder()
            .device_address(address)
            .size(raygen_region_size as _)
            .stride(raygen_region_size as _)
            .build();

        let miss_region = vk::StridedDeviceAddressRegionKHR::builder()
            .device_address(address + raygen_region.size)
            .size(miss_region_size as _)
            .stride(aligned_handle_size as _)
            .build();

        let hit_region = vk::StridedDeviceAddressRegionKHR::builder()
            .device_address(address + raygen_region.size + miss_region.size)
            .size(hit_region_size as _)
            .stride(aligned_handle_size as _)
            .build();

        Ok(Self {
            _buffer: buffer,
            raygen_region,
            miss_region,
            hit_region,
        })
    }
}

#[derive(Debug)]
pub struct AccelerationStructure {
    pub handle: vk::AccelerationStructureKHR,
    pub device_address: u64,
    pub buffer: Buffer,
}

pub struct Frame {
    pub fence: Fence,
    pub image_available: vk::Semaphore,
    pub render_finished: vk::Semaphore,
}

impl Frame {
    pub fn new(device: &Device) -> Result<Self> {
        let fence = Fence::new(device, Some(vk::FenceCreateFlags::SIGNALED))?;

        let image_available =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::builder().build(), None)? };
        let render_finished =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::builder().build(), None)? };

        Ok(Self {
            fence,
            image_available,
            render_finished,
        })
    }
}

pub struct Swapchain {
    pub ash_swapchain: khr::Swapchain,
    pub vk_swapchain: vk::SwapchainKHR,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub present_mode: vk::PresentModeKHR,
    pub images: Vec<(Image, vk::ImageView)>,
}

impl Swapchain {
    pub fn new(
        instance: &Instance,
        device: &Device,
        vk_surface: vk::SurfaceKHR,
        ash_surface: &khr::Surface,
        physical_device: &PhysicalDevice,
        width: u32,
        height: u32,
        present_queue_family: &QueueFamily,
        graphics_queue_family: &QueueFamily,
    ) -> Result<Self> {
        let format = {
            let formats = unsafe {
                ash_surface
                    .get_physical_device_surface_formats(physical_device.handel, vk_surface)?
            };
            if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
                vk::SurfaceFormatKHR {
                    format: vk::Format::B8G8R8A8_UNORM,
                    color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                }
            } else {
                *formats
                    .iter()
                    .find(|format| {
                        format.format == vk::Format::B8G8R8A8_UNORM
                            && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                    })
                    .unwrap_or(&formats[0])
            }
        };

        let present_mode = {
            let present_modes = unsafe {
                ash_surface
                    .get_physical_device_surface_present_modes(physical_device.handel, vk_surface)?
            };
            if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                vk::PresentModeKHR::IMMEDIATE
            } else {
                vk::PresentModeKHR::FIFO
            }
        };

        let capabilities = unsafe {
            ash_surface
                .get_physical_device_surface_capabilities(physical_device.handel, vk_surface)?
        };

        let extent = {
            if capabilities.current_extent.width != std::u32::MAX {
                capabilities.current_extent
            } else {
                let min = capabilities.min_image_extent;
                let max = capabilities.max_image_extent;
                let width = width.min(max.width).max(min.width);
                let height = height.min(max.height).max(min.height);
                vk::Extent2D { width, height }
            }
        };

        let image_count = capabilities.min_image_count + 1;

        let families_indices = [graphics_queue_family.index, present_queue_family.index];

        let create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::builder()
                .surface(vk_surface)
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                );

            builder = if graphics_queue_family.index != present_queue_family.index {
                builder
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families_indices)
            } else {
                builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            };

            builder
                .pre_transform(capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
        };

        let ash_swapchain = khr::Swapchain::new(&instance, &device);
        let vk_swapchain = unsafe { ash_swapchain.create_swapchain(&create_info, None).unwrap() };

        let images = unsafe { ash_swapchain.get_swapchain_images(vk_swapchain).unwrap() };

        let images = images
            .into_iter()
            .map(|i| Image::from_swapchain_image(i, format.format, extent))
            .collect::<Vec<_>>();

        let views = images
            .iter()
            .map(|i| create_image_view(device, &i).unwrap())
            .collect::<Vec<_>>();

        let images_and_views = images
            .into_iter()
            .zip(views)
            .collect::<Vec<(Image, vk::ImageView)>>();
        Ok(Self {
            ash_swapchain,
            vk_swapchain,
            extent,
            format: format.format,
            color_space: format.color_space,
            present_mode,
            images: images_and_views,
        })
    }
}


pub fn create_image_view_aspekt(
    device: &Device,
    image: &Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> vk::ImageView {
    let image_view_info = vk::ImageViewCreateInfo::builder()
        .image(image.inner)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        })
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(image.mip_levls)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        );
    unsafe { device.create_image_view(&image_view_info, None) }.unwrap()
}

pub fn create_depth_view(device: &Device, image: &Image, format: vk::Format) -> vk::ImageView {
    create_image_view_aspekt(device, image, format, vk::ImageAspectFlags::DEPTH)
}

pub fn new_command_pool(
    device: &Device,
    queue_family: &QueueFamily,
    flags: Option<vk::CommandPoolCreateFlags>,
) -> Result<vk::CommandPool> {
    let flags = flags.unwrap_or_else(vk::CommandPoolCreateFlags::empty);

    let command_pool_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_family.index)
        .flags(flags);
    let ret = unsafe { device.create_command_pool(&command_pool_info, None)? };
    Ok(ret)
}

pub fn allocate_command_buffers(
    pool: &vk::CommandPool,
    device: &Device,
    level: vk::CommandBufferLevel,
    count: u32,
) -> Result<Vec<vk::CommandBuffer>> {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(*pool)
        .level(level)
        .command_buffer_count(count);

    let buffers = unsafe { device.allocate_command_buffers(&allocate_info)? };
    let buffers = buffers.into_iter().collect();

    Ok(buffers)
}

pub fn allocate_command_buffer(
    device: &Device,
    pool: &vk::CommandPool,
    level: vk::CommandBufferLevel,
) -> Result<vk::CommandBuffer> {
    let buffers = allocate_command_buffers(pool, device, level, 1)?;
    let buffer = buffers.into_iter().next().unwrap();

    Ok(buffer)
}

pub fn free_command_buffers(pool: &vk::CommandPool, device: &Device, buffer: &[vk::CommandBuffer]) {
    unsafe { device.free_command_buffers(*pool, &buffer) };
}

pub fn free_command_buffer(
    pool: &vk::CommandPool,
    device: &Device,
    buffer: vk::CommandBuffer,
) -> Result<()> {
    let buffs = [buffer];
    unsafe { device.free_command_buffers(*pool, &buffs) };

    Ok(())
}

pub struct Fence {
    pub(crate) handel: vk::Fence,
}

impl Fence {
    pub(crate) fn new(device: &ash::Device, flags: Option<vk::FenceCreateFlags>) -> Result<Self> {
        let flags = flags.unwrap_or_else(vk::FenceCreateFlags::empty);

        let fence_info = vk::FenceCreateInfo::builder().flags(flags);
        let handel = unsafe { device.create_fence(&fence_info, None)? };

        Ok(Self { handel })
    }

    pub fn wait(&self, device: &Device, timeout: Option<u64>) -> Result<()> {
        let timeout = timeout.unwrap_or(std::u64::MAX);

        unsafe { device.wait_for_fences(from_ref(&self.handel), true, timeout)? };

        Ok(())
    }

    pub fn reset(&self, device: &Device) -> Result<()> {
        unsafe { device.reset_fences(&[self.handel])? };

        Ok(())
    }
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_fence(self.handel, None);
        }
    }
}

pub struct RayTracingContext {
    pub pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    pub pipeline_fn: khr::RayTracingPipeline,
    pub acceleration_structure_properties: vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
    pub acceleration_structure_fn: khr::AccelerationStructure,
}

impl RayTracingContext {
    pub(crate) fn new(instance: &Instance, pdevice: &vk::PhysicalDevice, device: &Device) -> Self {
        let pipeline_properties =
            unsafe { khr::RayTracingPipeline::get_properties(&instance, *pdevice) };
        let pipeline_fn = khr::RayTracingPipeline::new(&instance, &device);

        let acceleration_structure_properties =
            unsafe { khr::AccelerationStructure::get_properties(&instance, *pdevice) };
        let acceleration_structure_fn = khr::AccelerationStructure::new(&instance, &device);

        Self {
            pipeline_properties,
            pipeline_fn,
            acceleration_structure_properties,
            acceleration_structure_fn,
        }
    }
}

#[derive(Debug)]
pub struct Buffer {
    pub inner: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: vk::DeviceSize,
}

impl Buffer {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        size: vk::DeviceSize,
        name: Option<&str>,
        alignment: Option<u64>,
    ) -> Result<Self> {
        let create_info = vk::BufferCreateInfo::builder().size(size).usage(usage);
        let inner = unsafe { device.create_buffer(&create_info, None)? };
        let mut requirements = unsafe { device.get_buffer_memory_requirements(inner) };
        if let Some(a) = alignment {
            requirements.alignment = a;
        }

        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: name.unwrap_or("buffer"),
            requirements,
            location: memory_location,
            linear: true,
        })?;

        unsafe { device.bind_buffer_memory(inner, allocation.memory(), allocation.offset())? };

        Ok(Self {
            inner,
            allocation: Some(allocation),
            size,
        })
    }

    pub fn copy_data_to_buffer<T: Copy>(&self, data: &[T]) -> Result<()> {
        unsafe {
            let data_ptr = self
                .allocation
                .as_ref()
                .unwrap()
                .mapped_ptr()
                .unwrap()
                .as_ptr();
            let mut align =
                ash::util::Align::new(data_ptr, align_of::<T>() as _, size_of_val(data) as _);
            align.copy_from_slice(data);
        };

        Ok(())
    }

    pub fn get_device_address(&self, device: &Device) -> u64 {
        let addr_info = vk::BufferDeviceAddressInfo::builder().buffer(self.inner);
        unsafe { device.get_buffer_device_address(&addr_info) }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) -> Result<()> {
        unsafe { device.destroy_buffer(self.inner, None) };
        allocator.free(self.allocation.take().unwrap()).unwrap();
        Ok(())
    }
}

pub fn create_device(
    instance: &Instance,
    physical_device: &PhysicalDevice,
    queue_families: &[&QueueFamily; 3],
    required_extensions: &[&str],
) -> Result<Device> {
    let queue_priorities = [1.0f32];

    let queue_create_infos = {
        let mut indices = queue_families.iter().map(|f| f.index).collect::<Vec<_>>();
        indices.dedup();

        indices
            .iter()
            .map(|index| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(*index)
                    .queue_priorities(&queue_priorities)
                    .build()
            })
            .collect::<Vec<_>>()
    };

    let mut ray_tracing_feature = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder()
        .ray_tracing_pipeline(true)
        .build();
    let mut acceleration_struct_feature =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
            .acceleration_structure(true)
            .build();
    let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::builder()
        .runtime_descriptor_array(true)
        .buffer_device_address(true)
        .descriptor_indexing(true)
        .shader_sampled_image_array_non_uniform_indexing(true)
        .build();
    let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::builder()
        .dynamic_rendering(true)
        .synchronization2(true)
        .build();

    let mut features = vk::PhysicalDeviceFeatures2::builder()
        .features(vk::PhysicalDeviceFeatures::default())
        .push_next(&mut acceleration_struct_feature)
        .push_next(&mut ray_tracing_feature)
        .push_next(&mut vulkan_12_features)
        .push_next(&mut vulkan_13_features)
        .build();

    let device_extensions_as_ptr = required_extensions
        .into_iter()
        .map(|e| e.as_ptr() as *const i8)
        .collect::<Vec<_>>();

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(device_extensions_as_ptr.as_slice())
        .push_next(&mut features);

    let device = unsafe {
        instance
            .create_device(physical_device.handel, &device_create_info, None)
            .unwrap()
    };

    Ok(device)
}

pub fn enumerate_physical_devices(
    instance: &Instance,
    ash_surface: &khr::Surface,
    vk_surface: &vk::SurfaceKHR,
) -> Result<Vec<PhysicalDevice>> {
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };

    let mut physical_devices = physical_devices
        .into_iter()
        .map(|pd| PhysicalDevice::new(&instance, ash_surface, vk_surface, pd))
        .collect::<Result<Vec<PhysicalDevice>>>()?;

    physical_devices.sort_by_key(|pd| match pd.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        _ => 2,
    });
    Ok(physical_devices)
}

pub fn get_queue(device: &Device, queue_family: &QueueFamily, queue_index: u32) -> vk::Queue {
    unsafe { device.get_device_queue(queue_family.index, queue_index) }
}

pub fn select_suitable_physical_device(
    devices: &[PhysicalDevice],
    required_extensions: &[&str],
    required_device_features: &DeviceFeatures,
) -> Result<(PhysicalDevice, QueueFamily, QueueFamily, QueueFamily)> {
    let mut graphics = None;
    let mut present = None;
    let mut compute = None;

    let device = devices
        .iter()
        .find(|device| {
            for family in device.queue_families.iter().filter(|f| f.has_queues()) {
                if family.supports_graphics()
                    && family.supports_timestamp_queries()
                    && graphics.is_none()
                {
                    graphics = Some(family.clone());
                }

                if family.supports_compute() && compute.is_none() {
                    compute = Some(family.clone())
                }

                if family.supports_present() && present.is_none() {
                    present = Some(family.clone());
                }

                if graphics.is_some() && present.is_some() && compute.is_some() {
                    break;
                }
            }

            let extention_support = device.supports_extensions(required_extensions);

            graphics.is_some()
                && present.is_some()
                && extention_support
                && !device.supported_surface_formats.is_empty()
                && !device.supported_present_modes.is_empty()
                && device
                    .supported_device_features
                    .is_compatible_with(required_device_features)
        })
        .ok_or_else(|| anyhow::anyhow!("Could not find a suitable device"))?;

    Ok((device.clone(), graphics.unwrap(), present.unwrap(), compute.unwrap()))
}

fn begin_command_buffer(
    cmd: &vk::CommandBuffer,
    device: &Device,
    flags: Option<vk::CommandBufferUsageFlags>,
) -> Result<()> {
    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(flags.unwrap_or(vk::CommandBufferUsageFlags::empty()));
    unsafe { device.begin_command_buffer(*cmd, &begin_info)? };
    Ok(())
}

pub fn queue_submit(
    device: &Device,
    queue: &vk::Queue,
    command_buffer: &vk::CommandBuffer,
    wait_semaphore: Option<vk::SemaphoreSubmitInfo>,
    signal_semaphore: Option<vk::SemaphoreSubmitInfo>,
    fence: &Fence,
) -> Result<()> {
    let wait_semaphore_submit_info = wait_semaphore.map(|s| {
        vk::SemaphoreSubmitInfo::builder()
            .semaphore(s.semaphore)
            .stage_mask(s.stage_mask)
    });

    let signal_semaphore_submit_info = signal_semaphore.map(|s| {
        vk::SemaphoreSubmitInfo::builder()
            .semaphore(s.semaphore)
            .stage_mask(s.stage_mask)
    });

    let cmd_buffer_submit_info =
        vk::CommandBufferSubmitInfo::builder().command_buffer(*command_buffer);

    let submit_info = vk::SubmitInfo2::builder()
        .command_buffer_infos(std::slice::from_ref(&cmd_buffer_submit_info));

    let submit_info = match wait_semaphore_submit_info.as_ref() {
        Some(info) => submit_info.wait_semaphore_infos(std::slice::from_ref(info)),
        None => submit_info,
    };

    let submit_info = match signal_semaphore_submit_info.as_ref() {
        Some(info) => submit_info.signal_semaphore_infos(std::slice::from_ref(info)),
        None => submit_info,
    };

    unsafe { device.queue_submit2(*queue, std::slice::from_ref(&submit_info), fence.handel)? };

    Ok(())
}
