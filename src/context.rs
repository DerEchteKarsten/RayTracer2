use anyhow::Result;
use ash::{
    ext::debug_utils,
    khr::{self, acceleration_structure, ray_tracing_pipeline},
    vk::{self, ImageAspectFlags, ImageUsageFlags, ImageView},
    Device, Entry, Instance,
};

use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc},
    AllocationSizes, AllocatorDebugSettings, MemoryLocation,
};
use image::{DynamicImage, GenericImageView};
use log::debug;
use raw_window_handle::{
    HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle,
};
use std::{
    borrow::BorrowMut,
    ffi::{c_char, CStr, CString},
    mem::{align_of, size_of, size_of_val},
    os::raw::c_void,
    ptr,
    time::Instant,
};
// use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use simple_logger::SimpleLogger;
use std::slice::from_ref;

pub const FRAMES_IN_FLIGHT: u32 = 1;

#[derive(Debug, Clone, Copy, Default)]
pub struct RayTracingShaderGroupInfo {
    pub group_count: u32,
    pub raygen_shader_count: u32,
    pub miss_shader_count: u32,
    pub hit_shader_count: u32,
}
#[derive(Debug, Clone)]
pub struct RayTracingShaderCreateInfo<'a> {
    pub source: &'a [(&'a [u8], vk::ShaderStageFlags)],
    pub group: RayTracingShaderGroup,
}

#[derive(Debug, Clone, Copy)]
pub enum RayTracingShaderGroup {
    RayGen,
    Miss,
    Hit,
}

pub struct RayTracingContext<'a> {
    pub pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'a>,
    pub pipeline_fn: ray_tracing_pipeline::Device,
    pub acceleration_structure_properties: vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'a>,
    pub acceleration_structure_fn: acceleration_structure::Device,
}

impl<'a> RayTracingContext<'a> {
    pub(crate) fn new(instance: &Instance, pdevice: &vk::PhysicalDevice, device: &Device) -> Self {
        let (pipeline_properties, acceleration_structure_properties) = unsafe {
            let mut rt_pipeline_properties =
                vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            let mut acc_properties =
                vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
            let mut subgroups = vk::PhysicalDeviceSubgroupProperties::default();

            let mut physical_device_properties2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut rt_pipeline_properties)
                .push_next(&mut acc_properties)
                .push_next(&mut subgroups);
            instance.get_physical_device_properties2(*pdevice, &mut physical_device_properties2);
            debug!("{:?}", subgroups);
            (rt_pipeline_properties, acc_properties)
        };
        let pipeline_fn = khr::ray_tracing_pipeline::Device::new(&instance, &device);

        let acceleration_structure_fn =
            khr::acceleration_structure::Device::new(&instance, &device);

        Self {
            pipeline_properties,
            pipeline_fn,
            acceleration_structure_properties,
            acceleration_structure_fn,
        }
    }
}

#[derive()]
pub struct Renderer<'a> {
    pub ray_tracing: RayTracingContext<'a>,
    pub allocator: Allocator,
    pub command_pool: vk::CommandPool,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub device: ash::Device,
    pub present_queue_family: QueueFamily,
    pub graphics_queue_family: QueueFamily,
    pub physical_device: PhysicalDevice,
    pub ash_surface: khr::surface::Instance,
    pub vk_surface: vk::SurfaceKHR,
    pub instance: Instance,
    pub frames_in_flight: Vec<Frame>,
    pub swapchain: Swapchain,
    pub frame: u64,
    pub last_swapchain_image_index: u32,
    pub cmd_buffs: Vec<vk::CommandBuffer>,
    pub last_frame: u64,
    _entry: Entry,
}

pub struct ImageAndView {
    pub image: Image,
    pub view: ImageView,
}

impl<'a> Renderer<'a> {
    pub fn new(
        window_handle: &dyn HasRawWindowHandle,
        display_handle: &dyn HasRawDisplayHandle,
        required_device_features: &DeviceFeatures,
        device_extensions: [&CStr; 10],
        window_width: u32,
        window_height: u32,
    ) -> Result<Self> {
        let required_extensions = device_extensions
            .into_iter()
            .map(|e| e.to_str().unwrap())
            .collect::<Vec<_>>();
        let required_extensions = required_extensions.as_slice();

        // SimpleLogger::default().env().init().unwrap();
        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);

        let layer_names = unsafe {
            [CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            )]
        };

        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let mut instance_extensions =
            ash_window::enumerate_required_extensions(display_handle.raw_display_handle().unwrap())
                .unwrap()
                .to_vec();

        //#[cfg(debug_assertions)]
        instance_extensions.push(debug_utils::NAME.as_ptr());

        let mut validation_features = vk::ValidationFeaturesEXT::default()
            .enabled_validation_features(&[vk::ValidationFeatureEnableEXT::DEBUG_PRINTF]);

        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);

        //#[cfg(debug_assertions)]
        {
            instance_info = instance_info
                .enabled_layer_names(&layers_names_raw)
                .push_next(&mut validation_features);
        }

        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        //#[cfg(debug_assertions)]
        {
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                        | vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));
            let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
            let debug_call_back = unsafe {
                debug_utils_loader
                    .create_debug_utils_messenger(&debug_info, None)
                    .unwrap()
            };
        }
        let vk_surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                display_handle.raw_display_handle().unwrap(),
                window_handle.raw_window_handle().unwrap(),
                None,
            )
        }
        .unwrap();
        let ash_surface = khr::surface::Instance::new(&entry, &instance);

        let physical_devices = enumerate_physical_devices(&instance, &ash_surface, &vk_surface)?;
        let (physical_device, graphics_queue_family, present_queue_family) =
            select_suitable_physical_device(
                physical_devices.as_slice(),
                required_extensions,
                &required_device_features,
            )?;

        let queue_families = [&graphics_queue_family, &present_queue_family];

        let device = new_device(
            &instance,
            &physical_device,
            &queue_families,
            required_extensions,
        )?;

        let graphics_queue = get_queue(&device, &graphics_queue_family, 0);
        let present_queue = get_queue(&device, &present_queue_family, 0);

        let command_pool = new_command_pool(
            &device,
            &graphics_queue_family,
            Some(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ),
        )?;
        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: physical_device.handel,
            debug_settings: AllocatorDebugSettings {
                log_allocations: false,
                log_frees: false,
                log_leaks_on_shutdown: false,
                log_memory_information: false,
                ..Default::default()
            },
            buffer_device_address: true,
            allocation_sizes: AllocationSizes::new(64, 64),
        })?;

        let swapchain = Swapchain::new(
            &instance,
            &device,
            vk_surface,
            &ash_surface,
            &physical_device,
            window_width,
            window_height,
            &present_queue_family,
            &graphics_queue_family,
        )?;

        let mut frames_in_flight = Vec::with_capacity(FRAMES_IN_FLIGHT as usize);
        for _ in 0..FRAMES_IN_FLIGHT {
            frames_in_flight.push(Frame::new(&device)?);
        }
        let mut cmd_buffs = Vec::with_capacity(swapchain.images.len());

        for _ in &swapchain.images {
            cmd_buffs.push(allocate_command_buffer(
                &device,
                &command_pool,
                vk::CommandBufferLevel::PRIMARY,
            )?);
        }

        let ray_tracing = RayTracingContext::new(&instance, &physical_device.handel, &device);

        Ok(Self {
            ray_tracing,
            last_swapchain_image_index: 0,
            last_frame: 0,
            frame: 0,
            allocator,
            command_pool,
            present_queue,
            graphics_queue,
            device,
            present_queue_family,
            graphics_queue_family,
            physical_device,
            vk_surface,
            ash_surface,
            instance,
            frames_in_flight,
            swapchain,
            cmd_buffs,
            _entry: entry,
        })
    }

    pub fn read_shader_from_bytes(bytes: &[u8]) -> Result<Vec<u32>> {
        let mut cursor = std::io::Cursor::new(bytes);
        Ok(ash::util::read_spv(&mut cursor)?)
    }

    pub fn module_from_bytes(&self, source: &[u8]) -> Result<vk::ShaderModule> {
        let source = read_shader_from_bytes(source)?;

        let create_info = vk::ShaderModuleCreateInfo::default().code(&source);
        let res = unsafe { self.device.create_shader_module(&create_info, None) }?;
        Ok(res)
    }

    pub fn transition_image_layout_to_general(
        device: &Device,
        command_pool: &vk::CommandPool,
        image: &Image,
        queue: &vk::Queue,
    ) -> Result<()> {
        Self::transition_image_layout_to(
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

        let barrier = vk::ImageMemoryBarrier2::default()
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
                level_count: image.mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });
        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?
        };
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[barrier]),
            )
        };
        unsafe { device.end_command_buffer(cmd)? };

        let fence = Fence {
            handel: unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? },
        };
        queue_submit(&device, &queue, &cmd, None, None, &fence)?;
        fence.wait(&device, None)?;

        free_command_buffer(&command_pool, &device, cmd)?;
        Ok(())
    }

    pub fn copy_buffer_to_image(
        &self,
        cmd: &vk::CommandBuffer,
        src: &Buffer,
        dst: &Image,
        layout: vk::ImageLayout,
    ) {
        let region = vk::BufferImageCopy::default()
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
                &vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_pool(self.command_pool),
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
            &self.graphics_queue,
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
            1,
        )
    }

    pub fn create_mipimage(
        &mut self,
        usage: vk::ImageUsageFlags,
        memory_location: MemoryLocation,
        format: vk::Format,
        width: u32,
        height: u32,
        mips: u32,
    ) -> Result<Image> {
        Image::new_2d(
            &self.device,
            &mut self.allocator,
            usage,
            memory_location,
            format,
            width,
            height,
            mips,
        )
    }

    pub fn create_image_view(&self, image: &Image) -> Result<vk::ImageView> {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image.inner)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(image.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: image.mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        let res = unsafe { self.device.create_image_view(&view_info, None)? };

        Ok(res)
    }

    pub fn create_image_views(&self, image: &Image) -> Result<Vec<vk::ImageView>> {
        let mut image_views = vec![];
        for i in 0..image.mip_levels {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image.inner)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(image.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: i,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let res = unsafe { self.device.create_image_view(&view_info, None)? };
            image_views.push(res);
        }

        Ok(image_views)
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
        let create_info = vk::ShaderModuleCreateInfo::default().code(&decoded_code);

        unsafe { self.device.create_shader_module(&create_info, None) }.unwrap()
    }

    pub fn create_shader_stage(
        &self,
        stage: vk::ShaderStageFlags,
        path: &str,
    ) -> (vk::PipelineShaderStageCreateInfo, vk::ShaderModule) {
        let module = self.create_shader_module(path);
        (
            vk::PipelineShaderStageCreateInfo::default()
                .stage(stage)
                .module(module)
                .name(CStr::from_bytes_with_nul("main\0".as_bytes()).unwrap()),
            module,
        )
    }

    pub fn render<F>(&mut self, func: F) -> Result<()>
    where
        F: FnOnce(&mut Renderer, u32),
    {
        let last_time = Instant::now();
        let (image_index, frame_index) = {
            let frame_index: u64 = (self.frame + 1) % FRAMES_IN_FLIGHT as u64;
            let frame = &self.frames_in_flight[frame_index as usize];
            frame.fence.wait(&self.device, None)?;
            frame.fence.reset(&self.device)?;

            let image_index = match unsafe {
                self.swapchain.ash_swapchain.acquire_next_image(
                    self.swapchain.vk_swapchain,
                    u64::MAX,
                    frame.image_available,
                    vk::Fence::null(),
                )
            } {
                Err(err) => {
                    if err == vk::Result::ERROR_OUT_OF_DATE_KHR || err == vk::Result::SUBOPTIMAL_KHR
                    {
                        debug!("Suboptimal");
                    }
                    0
                }
                Ok(v) => v.0,
            };
            let cmd = &self.cmd_buffs[image_index as usize];
            begin_command_buffer(cmd, &self.device, None)?;
            (image_index, frame_index)
        };
        self.frame = frame_index;
        func(self, image_index);
        self.last_frame = frame_index;
        let frame = &self.frames_in_flight[frame_index as usize];
        let cmd = &self.cmd_buffs[image_index as usize];
        unsafe { self.device.end_command_buffer(*cmd) }.unwrap();
        let sms = [vk::SemaphoreSubmitInfo::default()
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .semaphore(frame.image_available)];
        let cmbs = [vk::CommandBufferSubmitInfo::default().command_buffer(*cmd)];
        let sss = [vk::SemaphoreSubmitInfo::default()
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .semaphore(frame.render_finished)];
        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&sms)
            .command_buffer_infos(&cmbs)
            .signal_semaphore_infos(&sss);
        frame.fence.reset(&self.device)?;

        unsafe {
            self.device
                .queue_submit2(self.graphics_queue, &[submit_info], frame.fence.handel)?;
        };
        // unsafe { self.device.queue_wait_idle(self.graphics_queue).unwrap() };
        // println!("{:?}", Instant::now().duration_since(last_time));

        let rf = &[frame.render_finished];
        let sc = &[self.swapchain.vk_swapchain];
        let image_indices = vec![image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(rf)
            .swapchains(sc)
            .image_indices(&image_indices);
        match unsafe {
            self.swapchain
                .ash_swapchain
                .queue_present(self.present_queue, &present_info)
        } {
            Err(err) => {
                if err == vk::Result::ERROR_OUT_OF_DATE_KHR || err == vk::Result::SUBOPTIMAL_KHR {
                    debug!("Suboptimal");
                }
            }
            Ok(_) => {}
        };

        Ok(())
    }

    pub fn pipeline_image_barriers(&self, cmd: &vk::CommandBuffer, barriers: &[ImageBarrier]) {
        let barriers = barriers
            .iter()
            .map(|b| {
                vk::ImageMemoryBarrier2::default()
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
                        level_count: b.image.mip_levels,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
            })
            .collect::<Vec<_>>();

        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);

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
        let region = vk::ImageCopy::default()
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

    pub fn destroy(&self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
            todo!();
        }
    }

    fn create_frame(&mut self) -> Result<Frame> {
        Frame::new(&self.device)
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

    pub fn get_buffer_address(&self, buffer: &Buffer) -> u64 {
        buffer.get_device_address(&self.device)
    }

    pub fn create_shader_binding_table(
        &mut self,
        pipeline: &vk::Pipeline,
        shader_group_infos: &RayTracingShaderGroupInfo,
    ) -> Result<ShaderBindingTable> {
        ShaderBindingTable::new(self, pipeline, shader_group_infos)
    }

    pub fn create_acceleration_structure(
        &mut self,
        level: vk::AccelerationStructureTypeKHR,
        as_geometry: &[vk::AccelerationStructureGeometryKHR],
        as_ranges: &[vk::AccelerationStructureBuildRangeInfoKHR],
        max_primitive_counts: &[u32],
    ) -> Result<AccelerationStructure> {
        let build_geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(level)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(as_geometry);

        let build_size = unsafe {
            let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
            self.ray_tracing
                .acceleration_structure_fn
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_geo_info,
                    max_primitive_counts,
                    &mut size_info,
                );
            size_info
        };

        let buffer = self.create_buffer(
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
            build_size.acceleration_structure_size,
            Some("Acceleration Structure Buffer"),
        )?;

        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(buffer.inner)
            .size(build_size.acceleration_structure_size)
            .ty(level);
        let handle = unsafe {
            self.ray_tracing
                .acceleration_structure_fn
                .create_acceleration_structure(&create_info, None)?
        };

        let scratch_buffer = self.create_aligned_buffer(
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
            build_size.build_scratch_size,
            Some("Acceleration Structure Scratch Buffer"),
            self.ray_tracing
                .acceleration_structure_properties
                .min_acceleration_structure_scratch_offset_alignment
                .into(),
        )?;

        let scratch_buffer_address = scratch_buffer.get_device_address(&self.device);

        let build_geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(level)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(as_geometry)
            .dst_acceleration_structure(handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_buffer_address,
            });

        self.execute_one_time_commands(|cmd_buffer| {
            unsafe {
                self.ray_tracing
                    .acceleration_structure_fn
                    .cmd_build_acceleration_structures(
                        *cmd_buffer,
                        from_ref(&build_geo_info),
                        from_ref(&as_ranges),
                    )
            };
        })?;

        let address_info =
            vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle);
        let address = unsafe {
            self.ray_tracing
                .acceleration_structure_fn
                .get_acceleration_structure_device_address(&address_info)
        };

        Ok(AccelerationStructure {
            buffer,
            handle,
            device_address: address,
        })
    }

    pub fn transition_image_layout(&self, image: &Image, layout: vk::ImageLayout) -> Result<()> {
        Renderer::transition_image_layout_to(
            &self.device,
            &self.command_pool,
            image,
            &self.graphics_queue,
            layout,
            ImageAspectFlags::COLOR,
        )
    }

    pub fn create_raytracing_pipeline(
        &self,
        pipeline_layout: vk::PipelineLayout,
        shaders_create_info: &[RayTracingShaderCreateInfo],
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
                let module = module_from_bytes(&self.device, s.0).unwrap();
                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(s.1)
                    .module(module)
                    .name(&entry_point_name);
                this_modules.push(module);
                this_stages.push(stage);
            });

            match shader.group {
                RayTracingShaderGroup::RayGen => shader_group_info.raygen_shader_count += 1,
                RayTracingShaderGroup::Miss => shader_group_info.miss_shader_count += 1,
                RayTracingShaderGroup::Hit => shader_group_info.hit_shader_count += 1,
            };

            let shader_index = stages.len();

            let mut group = vk::RayTracingShaderGroupCreateInfoKHR::default()
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
            groups.push(group);
        }

        let pipe_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .layout(pipeline_layout)
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(1);

        let pipeline = unsafe {
            self.ray_tracing.pipeline_fn.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipe_info),
                None,
            )
        }
        .unwrap();
        Ok((pipeline[0], shader_group_info))
    }
}

pub struct ShaderBindingTable {
    pub _buffer: Buffer,
    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub miss_region: vk::StridedDeviceAddressRegionKHR,
    pub hit_region: vk::StridedDeviceAddressRegionKHR,
}

impl ShaderBindingTable {
    pub fn new(
        ctx: &mut Renderer,
        pipeline: &vk::Pipeline,
        shaders: &RayTracingShaderGroupInfo,
    ) -> Result<Self> {
        let desc = shaders;

        let handle_size = ctx.ray_tracing.pipeline_properties.shader_group_handle_size;
        let handle_alignment = ctx
            .ray_tracing
            .pipeline_properties
            .shader_group_handle_alignment;
        let aligned_handle_size = alinged_size(handle_size, handle_alignment);
        let handle_pad = aligned_handle_size - handle_size;

        let group_alignment = ctx
            .ray_tracing
            .pipeline_properties
            .shader_group_base_alignment;

        let data_size = desc.group_count * handle_size;
        let handles = unsafe {
            ctx.ray_tracing
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
            ctx.ray_tracing
                .pipeline_properties
                .shader_group_base_alignment
                .into(),
        )?;

        let address = ctx.get_buffer_address(&buffer);

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

        let raygen_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(address)
            .size(raygen_region_size as _)
            .stride(raygen_region_size as _); //REMINDER

        let miss_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(address + raygen_region.size)
            .size(miss_region_size as _)
            .stride(aligned_handle_size as _);

        let hit_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(address + raygen_region.size + miss_region.size)
            .size(hit_region_size as _)
            .stride(aligned_handle_size as _);

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
    let mut dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);

    let mut info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(flags);
    dsl_info = dsl_info.push_next(&mut info);

    let res = unsafe { device.create_descriptor_set_layout(&dsl_info, None)? };

    Ok(res)
}

pub fn create_descriptor_pool(
    device: &Device,
    max_sets: u32,
    pool_sizes: &[vk::DescriptorPoolSize],
) -> Result<vk::DescriptorPool> {
    let pool_info = vk::DescriptorPoolCreateInfo::default()
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

    let sets_alloc_info = vk::DescriptorSetAllocateInfo::default()
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

pub fn update_descriptor_sets(
    renderer: &mut Renderer,
    set: &vk::DescriptorSet,
    writes: &[WriteDescriptorSet],
) {
    use WriteDescriptorSetKind::*;

    for w in writes {
        match w.kind {
            StorageImage { view, layout } => {
                let img_info = vk::DescriptorImageInfo::default()
                    .image_view(view.clone())
                    .image_layout(layout);
                unsafe {
                    renderer.device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::default()
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .dst_binding(w.binding)
                            .dst_set(set.clone())
                            .image_info(&[img_info])],
                        &[],
                    )
                }
            }
            AccelerationStructure {
                acceleration_structure,
            } => {
                let mut write_set_as = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                    .acceleration_structures(std::slice::from_ref(&acceleration_structure));

                let mut write = vk::WriteDescriptorSet::default()
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .dst_binding(w.binding)
                    .dst_set(set.clone())
                    .push_next(write_set_as.borrow_mut());
                write.descriptor_count = 1;
                unsafe { renderer.device.update_descriptor_sets(&[write], &[]) }
            }
            UniformBuffer { buffer } => {
                let buffer_info = vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .range(vk::WHOLE_SIZE);

                let buffer_infos = [buffer_info];
                let write = vk::WriteDescriptorSet::default()
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .dst_binding(w.binding)
                    .dst_set(set.clone())
                    .buffer_info(&buffer_infos);
                unsafe { renderer.device.update_descriptor_sets(&[write], &[]) }
            }
            CombinedImageSampler {
                view,
                sampler,
                layout,
            } => {
                let img_info = vk::DescriptorImageInfo::default()
                    .image_view(view.clone())
                    .sampler(sampler.clone())
                    .image_layout(layout);
                let img_infos = [img_info];
                let write = vk::WriteDescriptorSet::default()
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_binding(w.binding)
                    .dst_set(set.clone())
                    .image_info(&img_infos);
                unsafe { renderer.device.update_descriptor_sets(&[write], &[]) }
            }
            StorageBuffer { buffer } => {
                let buffer_info = vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .range(vk::WHOLE_SIZE);
                let buffer_infos = [buffer_info];
                let write = vk::WriteDescriptorSet::default()
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .dst_binding(w.binding)
                    .dst_set(set.clone())
                    .buffer_info(&buffer_infos);
                //TODO WTF is this hack
                unsafe { renderer.device.update_descriptor_sets(&[write], &[]) };
            }
        }
    }
}

#[derive(Clone)]
pub struct WriteDescriptorSet {
    pub binding: u32,
    pub kind: WriteDescriptorSetKind,
}

#[derive(Clone, Debug)]
pub enum WriteDescriptorSetKind {
    StorageImage {
        view: vk::ImageView,
        layout: vk::ImageLayout,
    },
    AccelerationStructure {
        acceleration_structure: vk::AccelerationStructureKHR,
    },
    UniformBuffer {
        buffer: vk::Buffer,
    },
    CombinedImageSampler {
        view: vk::ImageView,
        sampler: vk::Sampler,
        layout: vk::ImageLayout,
    },
    StorageBuffer {
        buffer: vk::Buffer,
    },
}

#[derive(Debug)]
pub struct Image {
    pub inner: vk::Image,
    pub allocation: Option<Allocation>,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
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
            mip_levels: 1,
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
        mips: u32,
    ) -> Result<Self> {
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(mips)
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
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe { device.bind_image_memory(inner, allocation.memory(), allocation.offset())? };

        Ok(Self {
            inner,
            allocation: Some(allocation),
            format,
            extent,
            mip_levels: mips,
        })
    }

    pub fn new_from_data(
        renderer: &mut Renderer,
        image: DynamicImage,
        format: vk::Format,
    ) -> Result<ImageAndView> {
        let (width, height) = image.dimensions();
        let image_buffer = if format != vk::Format::R8G8B8A8_SRGB {
            let image_data = image.to_rgba32f();
            let image_data_raw = image_data.as_raw();

            let image_buffer = renderer.create_buffer(
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

            let image_buffer = renderer.create_buffer(
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
            &renderer.device,
            &mut renderer.allocator,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            MemoryLocation::GpuOnly,
            format,
            width,
            height,
            1,
        )?;

        renderer.execute_one_time_commands(|cmd| {
            let texture_barrier = ImageBarrier {
                src_access_mask: vk::AccessFlags2::NONE,
                src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
                dst_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::TRANSFER_KHR,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                image: &texture_image,
            };
            renderer.pipeline_image_barriers(cmd, &[texture_barrier]);

            renderer.copy_buffer_to_image(
                cmd,
                &image_buffer,
                &texture_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            let texture_barrier_end = ImageBarrier {
                src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                src_stage_mask: vk::PipelineStageFlags2::TRANSFER_KHR,
                dst_access_mask: vk::AccessFlags2::SHADER_READ,
                dst_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image: &texture_image,
            };
            renderer.pipeline_image_barriers(cmd, &[texture_barrier_end]);
        })?;

        let image_view = create_image_view(&renderer.device, &texture_image, format);
        Ok(ImageAndView {
            image: Self {
                allocation: texture_image.allocation,
                extent: vk::Extent3D {
                    width,
                    depth: 1,
                    height,
                },
                format: format,
                inner: texture_image.inner,
                mip_levels: 1,
            },
            view: image_view,
        })
    }
}

pub fn copy_buffer(
    device: &Device,
    cmd: &vk::CommandBuffer,
    src_buffer: &Buffer,
    dst_buffer: &Buffer,
) {
    unsafe {
        let region = vk::BufferCopy::default().size(src_buffer.size);
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

#[derive(Clone, Debug)]
pub struct QueueFamily {
    pub index: u32,
    pub handel: vk::QueueFamilyProperties,
    pub supports_present: bool,
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

#[derive(Clone, Debug)]
pub struct DeviceFeatures {
    pub ray_tracing_pipeline: bool,
    pub acceleration_structure: bool,
    pub runtime_descriptor_array: bool,
    pub buffer_device_address: bool,
    pub dynamic_rendering: bool,
    pub synchronization2: bool,
    pub attomics: bool,
}

impl DeviceFeatures {
    pub fn is_compatible_with(&self, requirements: &Self) -> bool {
        (!requirements.ray_tracing_pipeline || self.ray_tracing_pipeline)
            && (!requirements.acceleration_structure || self.acceleration_structure)
            && (!requirements.runtime_descriptor_array || self.runtime_descriptor_array)
            && (!requirements.buffer_device_address || self.buffer_device_address)
            && (!requirements.dynamic_rendering || self.dynamic_rendering)
            && (!requirements.synchronization2 || self.synchronization2)
            && (!requirements.attomics || self.attomics)
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
    fn new(
        instance: &Instance,
        ash_surface: &khr::surface::Instance,
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
        let mut atomics = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
            .shader_buffer_float32_atomics(true)
            .shader_buffer_float64_atomic_add(true)
            .shader_buffer_float32_atomic_add(true);
        let mut atomics2 = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
            .shader_buffer_float32_atomics(true)
            .shader_buffer_float32_atomic_add(true);
        let features = vk::PhysicalDeviceFeatures::default().shader_int64(true);
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .runtime_descriptor_array(true)
            .buffer_device_address(true)
            .shader_buffer_int64_atomics(true);

        let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .features(features)
            .push_next(&mut atomics)
            .push_next(&mut features12)
            .push_next(&mut ray_tracing_feature)
            .push_next(&mut acceleration_struct_feature)
            .push_next(&mut features13)
            .push_next(&mut atomics2);
        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };

        let supported_device_features = DeviceFeatures {
            ray_tracing_pipeline: ray_tracing_feature.ray_tracing_pipeline == vk::TRUE,
            acceleration_structure: acceleration_struct_feature.acceleration_structure == vk::TRUE,
            runtime_descriptor_array: features12.runtime_descriptor_array == vk::TRUE,
            buffer_device_address: features12.buffer_device_address == vk::TRUE,
            dynamic_rendering: features13.dynamic_rendering == vk::TRUE,
            synchronization2: features13.synchronization2 == vk::TRUE,
            attomics: atomics.shader_buffer_float32_atomics == vk::TRUE,
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

pub struct Frame {
    pub fence: Fence,
    pub image_available: vk::Semaphore,
    pub render_finished: vk::Semaphore,
}

impl Frame {
    fn new(device: &Device) -> Result<Self> {
        let fence = Fence::new(device, Some(vk::FenceCreateFlags::SIGNALED))?;

        let image_available =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };
        let render_finished =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };

        Ok(Self {
            fence,
            image_available,
            render_finished,
        })
    }
}

pub struct Swapchain {
    pub ash_swapchain: khr::swapchain::Device,
    pub vk_swapchain: vk::SwapchainKHR,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub present_mode: vk::PresentModeKHR,
    pub images: Vec<ImageAndView>,
}

impl Swapchain {
    fn new(
        instance: &Instance,
        device: &Device,
        vk_surface: vk::SurfaceKHR,
        ash_surface: &khr::surface::Instance,
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
            let mut builder = vk::SwapchainCreateInfoKHR::default()
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

        let ash_swapchain = khr::swapchain::Device::new(&instance, &device);
        let vk_swapchain = unsafe { ash_swapchain.create_swapchain(&create_info, None).unwrap() };

        let images = unsafe { ash_swapchain.get_swapchain_images(vk_swapchain).unwrap() };

        let images = images
            .into_iter()
            .map(|i| Image::from_swapchain_image(i, format.format, extent))
            .collect::<Vec<_>>();

        let views = images
            .iter()
            .map(|i| create_image_view(device, &i, format.format))
            .collect::<Vec<_>>();

        let images_and_views = images
            .into_iter()
            .zip(views)
            .map(|e| ImageAndView {
                image: e.0,
                view: e.1,
            })
            .collect::<Vec<ImageAndView>>();
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

pub fn create_image_view(device: &Device, image: &Image, format: vk::Format) -> vk::ImageView {
    create_image_view_aspekt(device, image, format, vk::ImageAspectFlags::COLOR)
}

pub fn create_image_view_aspekt(
    device: &Device,
    image: &Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> vk::ImageView {
    let image_view_info = vk::ImageViewCreateInfo::default()
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
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(image.mip_levels)
                .base_array_layer(0)
                .layer_count(1),
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

    let command_pool_info = vk::CommandPoolCreateInfo::default()
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
    let allocate_info = vk::CommandBufferAllocateInfo::default()
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

        let fence_info = vk::FenceCreateInfo::default().flags(flags);
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
        let create_info = vk::BufferCreateInfo::default().size(size).usage(usage);
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
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe { device.bind_buffer_memory(inner, allocation.memory(), allocation.offset())? };

        Ok(Self {
            inner,
            allocation: Some(allocation),
            size,
        })
    }

    pub fn copy_data_to_buffer<T: Copy>(&self, data: &[T]) -> Result<()> {
        self.copy_data_to_aligned_buffer(data, align_of::<T>() as _)
    }

    pub fn copy_data_to_aligned_buffer<T: Copy>(&self, data: &[T], alignment: u32) -> Result<()> {
        unsafe {
            let data_ptr = self
                .allocation
                .as_ref()
                .unwrap()
                .mapped_ptr()
                .unwrap()
                .as_ptr();
            let mut align = ash::util::Align::new(data_ptr, alignment as _, size_of_val(data) as _);
            align.copy_from_slice(data);
        };

        Ok(())
    }

    pub fn get_device_address(&self, device: &Device) -> u64 {
        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(self.inner);
        unsafe { device.get_buffer_device_address(&addr_info) }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) -> Result<()> {
        unsafe { device.destroy_buffer(self.inner, None) };
        allocator.free(self.allocation.take().unwrap()).unwrap();
        Ok(())
    }
}

fn new_device(
    instance: &Instance,
    physical_device: &PhysicalDevice,
    queue_families: &[&QueueFamily; 2],
    required_extensions: &[&str],
) -> Result<Device> {
    let queue_priorities = [1.0f32];

    let queue_create_infos = {
        let mut indices = queue_families.iter().map(|f| f.index).collect::<Vec<_>>();
        indices.dedup();

        indices
            .iter()
            .map(|index| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*index)
                    .queue_priorities(&queue_priorities)
            })
            .collect::<Vec<_>>()
    };
    // let mut atomics = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
    //     .shader_buffer_float64_atomic_add(true);
    let mut ray_tracing_feature =
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
    let mut acceleration_struct_feature =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
    let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
        .runtime_descriptor_array(true)
        .buffer_device_address(true)
        .descriptor_indexing(true)
        .shader_sampled_image_array_non_uniform_indexing(true);
    let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .maintenance4(true)
        .synchronization2(true);

    let features = vk::PhysicalDeviceFeatures::default()
        .shader_int64(true)
        .fragment_stores_and_atomics(true);

    let mut features = vk::PhysicalDeviceFeatures2::default()
        .features(features)
        .push_next(&mut vulkan_12_features)
        .push_next(&mut vulkan_13_features)
        .push_next(&mut ray_tracing_feature)
        .push_next(&mut acceleration_struct_feature);
    // .push_next(&mut atomics);

    let device_extensions_as_ptr = required_extensions
        .into_iter()
        .map(|e| e.as_ptr() as *const i8)
        .collect::<Vec<_>>();

    let device_create_info = vk::DeviceCreateInfo::default()
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

fn enumerate_physical_devices(
    instance: &Instance,
    ash_surface: &khr::surface::Instance,
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

fn get_queue(device: &Device, queue_family: &QueueFamily, queue_index: u32) -> vk::Queue {
    unsafe { device.get_device_queue(queue_family.index, queue_index) }
}

fn select_suitable_physical_device(
    devices: &[PhysicalDevice],
    required_extensions: &[&str],
    required_device_features: &DeviceFeatures,
) -> Result<(PhysicalDevice, QueueFamily, QueueFamily)> {
    let mut graphics = None;
    let mut present = None;

    let device = devices
        .iter()
        .find(|device| {
            for family in device.queue_families.iter().filter(|f| f.has_queues()) {
                if family.supports_graphics()
                    && family.supports_compute()
                    && family.supports_timestamp_queries()
                    && graphics.is_none()
                {
                    graphics = Some(family.clone());
                }

                if family.supports_present() && present.is_none() {
                    present = Some(family.clone());
                }

                if graphics.is_some() && present.is_some() {
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

    Ok((device.clone(), graphics.unwrap(), present.unwrap()))
}

fn begin_command_buffer(
    cmd: &vk::CommandBuffer,
    device: &Device,
    flags: Option<vk::CommandBufferUsageFlags>,
) -> Result<()> {
    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(flags.unwrap_or(vk::CommandBufferUsageFlags::empty()));
    unsafe { device.begin_command_buffer(*cmd, &begin_info)? };
    Ok(())
}

//#[cfg(debug_assertions)]
unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Flag;
    if p_callback_data != ptr::null() && (*p_callback_data).p_message != ptr::null() {
        let message = CStr::from_ptr((*p_callback_data).p_message);
        match flag {
            Flag::VERBOSE => log::info!("{:?} - {:?}", typ, message),
            Flag::INFO => {
                let message = message.to_str().unwrap();
                let index = if message.contains("DEBUG-PRINTF") {
                    170
                } else {
                    0
                };
                log::info!("{:?} - {:?}", typ, message[index..].to_owned())
            }
            Flag::WARNING => log::warn!("{:?} - {:?}", typ, message),
            _ => log::error!("{:?} - {:?}", typ, message),
        }
    }
    vk::FALSE
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
        vk::SemaphoreSubmitInfo::default()
            .semaphore(s.semaphore)
            .stage_mask(s.stage_mask)
    });

    let signal_semaphore_submit_info = signal_semaphore.map(|s| {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(s.semaphore)
            .stage_mask(s.stage_mask)
    });

    let cmd_buffer_submit_info =
        vk::CommandBufferSubmitInfo::default().command_buffer(*command_buffer);

    let submit_info = vk::SubmitInfo2::default()
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

pub fn read_shader_from_bytes(bytes: &[u8]) -> Result<Vec<u32>> {
    let mut cursor = std::io::Cursor::new(bytes);
    Ok(ash::util::read_spv(&mut cursor)?)
}

pub fn module_from_bytes(device: &Device, source: &[u8]) -> Result<vk::ShaderModule> {
    let source = read_shader_from_bytes(source)?;

    let create_info = vk::ShaderModuleCreateInfo::default().code(&source);
    let res = unsafe { device.create_shader_module(&create_info, None) }?;
    Ok(res)
}
