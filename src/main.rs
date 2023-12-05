use std::{ffi::{c_void, CStr, c_char}, mem::size_of, fs,};

use erupt::{
    cstr,
    utils::{self, surface},
    DeviceLoader, EntryLoader, InstanceLoader, ObjectHandle, vk::{self, KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, BuildAccelerationStructureModeKHR, AccelerationStructureBuildRangeInfoKHR, AccelerationStructureKHR}, SmallVec, vk1_0::{CommandBuffer, SubmitInfo, DescriptorPoolSize, DescriptorPoolSizeBuilder, Offset3DBuilder, DependencyFlags},
};


use glm::{vec3, GenSquareMat};
use winit::{event_loop::{EventLoop, self, ControlFlow}, window::WindowBuilder, dpi::PhysicalSize, event::{StartCause, Event, WindowEvent, DeviceEvent, KeyboardInput, ElementState, VirtualKeyCode}};


unsafe extern "system" fn debug_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    eprintln!(
        "{}",
        CStr::from_ptr((*p_callback_data).p_message).to_string_lossy()
    );

    vk::FALSE
}

const FRAMES_IN_FLIGHT: usize = 2;
const VALIDATION_LAYERS: bool = true;
const LAYER_KHRONOS_VALIDATION: *const c_char = erupt::cstr!("VK_LAYER_KHRONOS_validation");

struct Camera {
    fov: f32,
	znear: f32, 
    zfar: f32,
	rotation: glm::Vec3,
	position: glm::Vec3, 
	view_pos: glm::Vec4,  
    perspective: glm::Mat4,
	view: glm::Mat4
}

const IDENTITY_MATRIX: glm::Mat4 = glm::Mat4 { c0: glm::Vector4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 },  c1: glm::Vector4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 },  c2: glm::Vector4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 },  c3: glm::Vector4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 }};
const IDENTITY_VEC4: glm::Vec4 = glm::Vec4 {w:1.0, x:1.0, y: 1.0, z: 1.0};

impl Camera {
    fn update_matricies(&mut self, width: f32, height: f32) {
        let mut rot_mat = IDENTITY_MATRIX;

		rot_mat = glm::ext::rotate(&rot_mat, glm::radians(self.rotation.x), glm::vec3(1.0, 0.0, 0.0));
		rot_mat = glm::ext::rotate(&rot_mat, glm::radians(self.rotation.y), glm::vec3(0.0, 1.0, 0.0));
		rot_mat = glm::ext::rotate(&rot_mat, glm::radians(self.rotation.z), glm::vec3(1.0, 0.0, 1.0));

		let translation = self.position;

		let trans_mat = glm::ext::translate(&IDENTITY_MATRIX, translation);

		self.view = rot_mat * trans_mat;
        self.perspective = glm::ext::perspective(self.fov, width/height, self.znear, self.zfar);
        self.view_pos = glm::vec4(self.position.x, self.position.y, self.position.z, 0.0) * glm::vec4(-1.0, 1.0, -1.0, 1.0);
    }

    fn new(fov: f32, position: glm::Vec3, rotation: glm::Vec3, size: PhysicalSize<u32>) -> Camera {
        let mut cam = Self { fov, znear: 0.1, zfar: 1000.0, rotation, position, view_pos: IDENTITY_VEC4, perspective: IDENTITY_MATRIX, view: IDENTITY_MATRIX};
        cam.update_matricies(size.width as f32, size.height as f32);
        cam
    }
}

struct UniformData {
    view_inverse: glm::Mat4,
    proj_inverse: glm::Mat4,
}

struct Vertex {
    pos: [f32; 3],
}

struct AccelerationStructure {
	handle: vk::AccelerationStructureKHR,
	device_address: u64,
	memory: vk::DeviceMemory,
	buffer: vk::Buffer,
}

struct RayTracingScratchBuffer
{
	device_address: u64,
	handle: vk::Buffer,
	memory: vk::DeviceMemory,
}

fn main() {
    let entry = EntryLoader::new().expect("Could not create EntryLoader");
    println!(
        "{} - Vulkan Instance {}.{}.{}",
        "Vulkan",
        vk::api_version_major(entry.instance_version()),
        vk::api_version_minor(entry.instance_version()),
        vk::api_version_patch(entry.instance_version())
    );
    let app_info = vk::ApplicationInfoBuilder::new()
        .api_version(vk::API_VERSION_1_1);

    
    let mut instance_layers = Vec::new();
    if VALIDATION_LAYERS {
        instance_layers.push(LAYER_KHRONOS_VALIDATION);
    }

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Vulkan")
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    let mut instance_extensions = erupt::utils::surface::enumerate_required_extensions(&window).unwrap();
    if VALIDATION_LAYERS {
        instance_extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    let instance_info = vk::InstanceCreateInfoBuilder::new()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers);

    let instance = unsafe {InstanceLoader::new(&entry, &instance_info).expect("could not create InstanceLoader")};  

    let messenger = if VALIDATION_LAYERS {
        let messenger_info = vk::DebugUtilsMessengerCreateInfoEXTBuilder::new()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE_EXT
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL_EXT
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION_EXT
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE_EXT,
            )
            .pfn_user_callback(Some(debug_callback));

        unsafe { instance.create_debug_utils_messenger_ext(&messenger_info, None) }.unwrap()
    } else {
        Default::default()
    };

    let device_extensions = vec![
        vk::KHR_SWAPCHAIN_EXTENSION_NAME, 
        vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, 
        vk::KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        vk::KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, 
        vk::KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, 
        vk::EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, 
        vk::KHR_SPIRV_1_4_EXTENSION_NAME,
        vk::KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
    ];
    
    let mut device_layers = Vec::new();
    if VALIDATION_LAYERS {
        device_layers.push(LAYER_KHRONOS_VALIDATION);
    }
    let surface = unsafe { surface::create_surface(&instance, &window, None) }.unwrap();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Physical_devices_and_queue_families
    let (physical_device, queue_family, swapchain_format, present_mode, device_properties) =
        unsafe { instance.enumerate_physical_devices(None) }
            .unwrap()
            .into_iter()
            .filter_map(|physical_device| unsafe {
                let queue_family = match instance
                    .get_physical_device_queue_family_properties(physical_device, None)
                    .into_iter()
                    .enumerate()
                    .position(|(i, queue_family_properties)| {
                        queue_family_properties
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS)
                            && instance
                                .get_physical_device_surface_support_khr(
                                    physical_device,
                                    i as u32,
                                    surface,
                                )
                                .unwrap()
                    }) {
                    Some(queue_family) => queue_family as u32,
                    None => return None,
                };

                let formats = instance
                    .get_physical_device_surface_formats_khr(physical_device, surface, None)
                    .unwrap();
                let format = match formats
                    .iter()
                    .find(|surface_format| {
                        (surface_format.format == vk::Format::B8G8R8A8_SRGB
                            || surface_format.format == vk::Format::R8G8B8A8_SRGB)
                            && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR
                    })
                    .or_else(|| formats.get(0))
                {
                    Some(surface_format) => *surface_format,
                    None => return None,
                };

                let present_mode = instance
                    .get_physical_device_surface_present_modes_khr(physical_device, surface, None)
                    .unwrap()
                    .into_iter()
                    .find(|present_mode| present_mode == &vk::PresentModeKHR::MAILBOX_KHR)
                    .unwrap_or(vk::PresentModeKHR::FIFO_KHR);

                let supported_device_extensions = instance
                    .enumerate_device_extension_properties(physical_device, None, None)
                    .unwrap();
                let device_extensions_supported =
                    device_extensions.iter().all(|device_extension| {
                        let device_extension = CStr::from_ptr(*device_extension);

                        supported_device_extensions.iter().any(|properties| {
                            CStr::from_ptr(properties.extension_name.as_ptr()) == device_extension
                        })
                    });

                if !device_extensions_supported {
                    return None;
                }

                let device_properties = instance.get_physical_device_properties(physical_device);
                Some((
                    physical_device,
                    queue_family,
                    format,
                    present_mode,
                    device_properties,
                ))
            })
            .max_by_key(|(_, _, _, _, properties)| match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 2,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 0,
            })
            .expect("No suitable physical device found");
    
    println!("Using physical device: {:?}", unsafe {
        CStr::from_ptr(device_properties.device_name.as_ptr())
    });

    let queue_info = vec![vk::DeviceQueueCreateInfoBuilder::new()
        .queue_family_index(queue_family)
        .queue_priorities(&[1.0])];
    let features = vk::PhysicalDeviceFeaturesBuilder::new();

    let device_info = vk::DeviceCreateInfoBuilder::new()
        .queue_create_infos(&queue_info)
        .enabled_features(&features)
        .enabled_extension_names(&device_extensions)
        .enabled_layer_names(&device_layers);

    let device = unsafe { DeviceLoader::new(&instance, physical_device, &device_info) }.unwrap();
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    let surface_caps =
    unsafe { instance.get_physical_device_surface_capabilities_khr(physical_device, surface) }
            .unwrap();
    let mut image_count = surface_caps.min_image_count + 1;
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }

    let swapchain_image_extent = match surface_caps.current_extent {
        vk::Extent2D {
            width: u32::MAX,
            height: u32::MAX,
        } => {
            let PhysicalSize { width, height } = window.inner_size();
            vk::Extent2D { width, height }
        }
        normal => normal,
    };

    let swapchain_info = vk::SwapchainCreateInfoKHRBuilder::new()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(swapchain_format.format)
        .image_color_space(swapchain_format.color_space)
        .image_extent(swapchain_image_extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let swapchain = unsafe { device.create_swapchain_khr(&swapchain_info, None) }.unwrap();
    let swapchain_images = unsafe { device.get_swapchain_images_khr(swapchain, None) }.unwrap();

    let swapchain_image_views: Vec<_> = swapchain_images
        .iter()
        .map(|swapchain_image| {
            let image_view_info = vk::ImageViewCreateInfoBuilder::new()
                .image(*swapchain_image)
                .view_type(vk::ImageViewType::_2D)
                .format(swapchain_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(
                    vk::ImageSubresourceRangeBuilder::new()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                );
            unsafe { device.create_image_view(&image_view_info, None) }.unwrap()
    })
    .collect();

    let attachments = vec![vk::AttachmentDescriptionBuilder::new()
        .format(swapchain_format.format)
        .samples(vk::SampleCountFlagBits::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

    let color_attachment_refs = vec![vk::AttachmentReferenceBuilder::new()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpasses = vec![vk::SubpassDescriptionBuilder::new()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)];
    let dependencies = vec![vk::SubpassDependencyBuilder::new()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

    let render_pass_info = vk::RenderPassCreateInfoBuilder::new()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    let render_pass = unsafe { device.create_render_pass(&render_pass_info, None) }.unwrap();


    let swapchain_framebuffers: Vec<_> = swapchain_image_views
        .iter()
        .map(|image_view| {
            let attachments = vec![*image_view];
            let framebuffer_info = vk::FramebufferCreateInfoBuilder::new()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(swapchain_image_extent.width)
                .height(swapchain_image_extent.height)
                .layers(1);

            unsafe { device.create_framebuffer(&framebuffer_info, None) }.unwrap()
        })
        .collect();

    let command_pool_info =
        vk::CommandPoolCreateInfoBuilder::new().queue_family_index(queue_family);
    let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }.unwrap();

    let cmd_buf_allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(swapchain_framebuffers.len() as _);
    let cmd_bufs = unsafe { device.allocate_command_buffers(&cmd_buf_allocate_info) }.unwrap();

    
    let semaphore_info = vk::SemaphoreCreateInfoBuilder::new();
    let image_available_semaphores: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap())
        .collect();
    let render_finished_semaphores: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap())
        .collect();

    let fence_info = vk::FenceCreateInfoBuilder::new().flags(vk::FenceCreateFlags::SIGNALED);
    let in_flight_fences: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| unsafe { device.create_fence(&fence_info, None) }.unwrap())
        .collect();

    let mut raytracing_pipline_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHRBuilder::new();
    let mut device_properties2 = vk::PhysicalDeviceProperties2Builder::new();
    device_properties2.p_next = &mut raytracing_pipline_properties as *mut _ as *mut c_void;
	unsafe { instance.get_physical_device_properties2(physical_device, &mut device_properties2); }

    let mut acceleration_structure_fetures = vk::PhysicalDeviceAccelerationStructureFeaturesKHRBuilder::new();
    let mut device_features2 = vk::PhysicalDeviceFeatures2Builder::new();
        device_features2.p_next = &mut acceleration_structure_fetures as *mut _ as *mut c_void;
    unsafe { instance.get_physical_device_features2(physical_device, &mut device_features2); }

 
    let mut vertices = vec![
        Vertex { pos: [  1.0,  1.0, 0.0 ] },
        Vertex { pos: [ -1.0,  1.0, 0.0 ] },
        Vertex { pos: [  0.0, -1.0, 0.0 ] }
    ];

    // Setup indices
    let mut indices = vec![0, 1, 2 ];

    // Setup identity transform matrix
    let mut transform_matrix = vk::TransformMatrixKHRBuilder::new().matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ]);

    let camera = Camera::new(100.0, vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), window.inner_size());

    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let bottom_level_acceleration_structure = create_bottom_level_acceleration_structure(&device, &memory_properties, &command_pool, &mut vertices.as_mut_slice(), &mut indices, &mut *transform_matrix);
    let top_level_acceleration_structure = create_top_level_acceleration_structure(&device, &memory_properties, &command_pool, &*transform_matrix, bottom_level_acceleration_structure.device_address);

    let storage_image = create_storage_image(&device, &swapchain_format.format, &memory_properties, &command_pool, &queue);
        
    let mut uniform_data = UniformData{ proj_inverse: camera.perspective.inverse().unwrap(), view_inverse: camera.view.inverse().unwrap()};

    let (uniform_buffer, uniform_memory) = create_buffer(&memory_properties, &device, vk::BufferUsageFlags::UNIFORM_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, size_of::<UniformData>() as u64, &mut uniform_data as *mut _ as *mut c_void);

    let mem_mapped = unsafe { device.map_memory( uniform_memory, 0, size_of::<UniformData>() as u64, vk::MemoryMapFlags::empty()) }.unwrap();

    let (pipeline, modules, shader_groups, descriptor_set_layout, pipeline_layout)  = create_ray_tracing_pipeline(&device);

    let ((raygen_shader_binding_table, raygen_shader_binding_table_mem), (miss_shader_binding_table, miss_shader_binding_table_mem), (hit_shader_binding_table, hit_shader_binding_table_mem)) = create_shader_binding_table(&device, &memory_properties, &pipeline, *raytracing_pipline_properties, shader_groups);
    
    let descriptor_set = create_descriptor_sets(&device, descriptor_set_layout, top_level_acceleration_structure.handle, storage_image.2, uniform_buffer);

    record_command_buffers(&device,
        &cmd_bufs,
        &descriptor_set,
        &swapchain_images,
       &storage_image.0,
        &raytracing_pipline_properties,
        &pipeline,
        &pipeline_layout,
        &[raygen_shader_binding_table, miss_shader_binding_table, hit_shader_binding_table],
        window.inner_size(),
    );

    let mut frame = 0;
    #[allow(clippy::collapsible_match, clippy::single_match)]
    event_loop.run(move |event, _, control_flow| match event {
        Event::NewEvents(StartCause::Init) => {
            *control_flow = ControlFlow::Poll;
        }
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            _ => (),
        },
        Event::DeviceEvent { event, .. } => match event {
            DeviceEvent::Key(KeyboardInput {
                virtual_keycode: Some(keycode),
                state,
                ..
            }) => match (keycode, state) {
                (VirtualKeyCode::Escape, ElementState::Released) => {
                    *control_flow = ControlFlow::Exit
                }
                _ => (),
            },
            _ => (),
        },
        Event::MainEventsCleared => {
            unsafe {
                device
                    .wait_for_fences(&[in_flight_fences[frame]], true, u64::MAX)
                    .unwrap();
            }

            let image_index = unsafe {
                device.acquire_next_image_khr(
                    swapchain,
                    u64::MAX,
                    image_available_semaphores[frame],
                    vk::Fence::null(),
                )
            }
            .unwrap();

            let wait_semaphores = vec![image_available_semaphores[frame]];
            let command_buffers = vec![cmd_bufs[image_index as usize]];
            let signal_semaphores = vec![render_finished_semaphores[frame]];
            let submit_info = vk::SubmitInfoBuilder::new()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            unsafe {
                let in_flight_fence = in_flight_fences[frame];
                device.reset_fences(&[in_flight_fence]).unwrap();
                device
                    .queue_submit(queue, &[submit_info], in_flight_fence)
                    .unwrap()
            }

            let swapchains = vec![swapchain];
            let image_indices = vec![image_index];
            let present_info = vk::PresentInfoKHRBuilder::new()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            unsafe { device.queue_present_khr(queue, &present_info) }.unwrap();

            frame = (frame + 1) % FRAMES_IN_FLIGHT;
        }
        Event::LoopDestroyed => unsafe {
            device.device_wait_idle().unwrap();

            for &semaphore in image_available_semaphores
                .iter()
                .chain(render_finished_semaphores.iter())
            {
                device.destroy_semaphore(semaphore, None);
            }

            for &fence in &in_flight_fences {
                device.destroy_fence(fence, None);
            }

            device.destroy_command_pool(command_pool, None);

            for &framebuffer in &swapchain_framebuffers {
                device.destroy_framebuffer(framebuffer, None);
            }

            device.destroy_render_pass(render_pass, None);

            for &image_view in &swapchain_image_views {
                device.destroy_image_view(image_view, None);
            }

            device.destroy_swapchain_khr(swapchain, None);

            for module in &modules {
                device.destroy_shader_module(*module, None);
            }

            device.destroy_device(None);

            instance.destroy_surface_khr(surface, None);

            if !messenger.is_null() {
                instance.destroy_debug_utils_messenger_ext(messenger, None);
            }

            instance.destroy_instance(None);
            println!("Exited cleanly");
        },
        _ => (),
    });

}


fn record_command_buffers(device: &DeviceLoader, draw_command_buffers: &SmallVec<vk::CommandBuffer>, descriptor_set: &vk::DescriptorSet, swapchain_images: &SmallVec<vk::Image>, storage_image: &vk::Image, raytracing_pipeline_properties: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR, pipeline: &vk::Pipeline, pipeline_layout: &vk::PipelineLayout, shader_binding_tabels: &[vk::Buffer; 3], size: PhysicalSize<u32>) {
    let cmd_buff_info = vk::CommandBufferBeginInfoBuilder::new()
        .flags(vk::CommandBufferUsageFlags::empty());
    let subresource_range = vk::ImageSubresourceRangeBuilder::new()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .level_count(1)
        .base_mip_level(0)
        .layer_count(1);
	for (i, cmd) in draw_command_buffers.iter().enumerate() {
        unsafe { device.begin_command_buffer(*cmd, &cmd_buff_info) }.unwrap();

        let handle_size_alinged = aligned_size(raytracing_pipeline_properties.shader_group_handle_size as u64, raytracing_pipeline_properties.shader_group_handle_alignment as u64);

        let get_region = move |i| {
            vk::StridedDeviceAddressRegionKHRBuilder::new()
                .device_address(get_buffer_device_address(device, &shader_binding_tabels[i]))
                .stride(handle_size_alinged)
                .size(handle_size_alinged)
        };

        let raygen_shader_sbt_entry = get_region(0);
        let miss_shader_sbt_entry = get_region(1);
        let hit_shader_sbt_entry = get_region(2);
        let callable_shader_sbt_entry = vk::StridedDeviceAddressRegionKHRBuilder::new();
        unsafe {
            device.cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::RAY_TRACING_KHR, *pipeline);
            device.cmd_bind_descriptor_sets(*cmd, vk::PipelineBindPoint::RAY_TRACING_KHR, *pipeline_layout, 0, &[*descriptor_set], &[0]);
            device.cmd_trace_rays_khr(*cmd, &raygen_shader_sbt_entry, &miss_shader_sbt_entry, &hit_shader_sbt_entry, &callable_shader_sbt_entry, size.width, size.height, 1);
            set_image_layout(device, cmd, &swapchain_images[i], vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &subresource_range);      
            set_image_layout(device, cmd, storage_image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, &subresource_range);       

            let copy_region = vk::ImageCopyBuilder::new()
                .src_subresource(*vk::ImageSubresourceLayersBuilder::new().aspect_mask(vk::ImageAspectFlags::COLOR).base_array_layer(0).mip_level(0).layer_count(1))
                .src_offset(*Offset3DBuilder::new().x(0).y(0).z(0))
                .dst_subresource(*vk::ImageSubresourceLayersBuilder::new().aspect_mask(vk::ImageAspectFlags::COLOR).base_array_layer(0).mip_level(0).layer_count(1))
                .dst_offset(*Offset3DBuilder::new().x(0).y(0).z(0))
                .extent(*vk::Extent3DBuilder::new().width(size.width).height(size.height).depth(1));
            device.cmd_copy_image(*cmd, *storage_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, swapchain_images[i], vk::ImageLayout::TRANSFER_SRC_OPTIMAL, &[copy_region]);

            set_image_layout(device, cmd, &swapchain_images[i], vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, &subresource_range);      
            set_image_layout(device, cmd, storage_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::GENERAL, &subresource_range);       

            device.end_command_buffer(*cmd).unwrap();
        }
    }
}

fn set_image_layout(device: &DeviceLoader, cmd: &vk::CommandBuffer, image: &vk::Image, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, subresource_range: &vk::ImageSubresourceRange) {
    let mut image_memory_barrier = vk::ImageMemoryBarrierBuilder::new()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .subresource_range(*subresource_range)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(*image);

        match old_layout {
            vk::ImageLayout::UNDEFINED => {image_memory_barrier = image_memory_barrier.src_access_mask(vk::AccessFlags::NONE);}
            vk::ImageLayout::PREINITIALIZED => {image_memory_barrier = image_memory_barrier.src_access_mask(vk::AccessFlags::HOST_WRITE);}
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => { image_memory_barrier = image_memory_barrier.src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);}
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => { image_memory_barrier = image_memory_barrier.src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);}
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => {image_memory_barrier = image_memory_barrier.src_access_mask(vk::AccessFlags::TRANSFER_READ);}
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => { image_memory_barrier = image_memory_barrier.src_access_mask(vk::AccessFlags::TRANSFER_WRITE);}
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => { image_memory_barrier = image_memory_barrier.src_access_mask(vk::AccessFlags::SHADER_READ);}
            _ => {todo!()}
        }

      
        match new_layout {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => {image_memory_barrier = image_memory_barrier.dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);}
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => {image_memory_barrier = image_memory_barrier.dst_access_mask(vk::AccessFlags::TRANSFER_READ);}
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => {image_memory_barrier = image_memory_barrier.dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);}
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {image_memory_barrier = image_memory_barrier.dst_access_mask(image_memory_barrier.dst_access_mask | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)}
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => { 
                if image_memory_barrier.src_access_mask == vk::AccessFlags::NONE {
                    image_memory_barrier = image_memory_barrier.src_access_mask(vk::AccessFlags::HOST_WRITE | vk::AccessFlags::TRANSFER_WRITE);
                }
                image_memory_barrier = image_memory_barrier.dst_access_mask(vk::AccessFlags::SHADER_READ);
            }
            _=>{todo!()}
        }

        unsafe { device.cmd_pipeline_barrier(*cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, DependencyFlags::empty(), &[] as &[vk::MemoryBarrierBuilder], &[] as &[vk::BufferMemoryBarrierBuilder], &[image_memory_barrier]) };
}

fn create_bottom_level_acceleration_structure(device: &DeviceLoader, memory_properties: &vk::PhysicalDeviceMemoryProperties, command_pool: &vk::CommandPool, vertices: &mut [Vertex], indices: &mut [i32], transform_matrix: &mut vk::TransformMatrixKHR) -> AccelerationStructure {
    let (vertex_buffer, vertex_memory) = create_buffer(
        memory_properties,
        device,
    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        (vertices.len() * size_of::<Vertex>()) as u64,
        &mut *vertices as *mut _ as *mut c_void);
    // Index buffer
    let (index_buffer, index_memory) = create_buffer(
        memory_properties,
        device,
    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        (size_of::<vk::TransformMatrixKHR>()) as u64,
        &mut *indices as *mut _ as *mut c_void);
    // Transform buffer
    let (transform_buffer, transform_memory) = create_buffer(
        memory_properties,
        device,
    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        size_of::<vk::TransformMatrixKHR>() as u64,
        &mut *transform_matrix as *mut _ as *mut c_void);

    let vertex_buffer_device_address = vk::DeviceOrHostAddressConstKHR{device_address: get_buffer_device_address(device, &vertex_buffer)};
    let index_buffer_device_address = vk::DeviceOrHostAddressConstKHR{device_address: get_buffer_device_address(device, &index_buffer)};
    let transform_buffer_device_address = vk::DeviceOrHostAddressConstKHR{device_address: get_buffer_device_address(device, &transform_buffer)};

    let acceleration_structure_geometry = vk::AccelerationStructureGeometryKHRBuilder::new()
        .geometry_type( vk::GeometryTypeKHR::TRIANGLES_KHR)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: vk::AccelerationStructureGeometryTrianglesDataKHR { 
                s_type: vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                vertex_format: vk::Format::R32G32B32_SFLOAT,
                vertex_data: vertex_buffer_device_address,
                vertex_stride: size_of::<Vertex>() as u64,
                max_vertex: (vertices.len() - 1) as u32,
                index_type: vk::IndexType::UINT32,
                index_data: index_buffer_device_address,
                transform_data: transform_buffer_device_address, 
                ..Default::default()
            }
        });

    let triangle_count = (indices.len()/3) as u32;

   create_acceleration_structure(device, memory_properties, &command_pool, &acceleration_structure_geometry, triangle_count, vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL_KHR)
}

fn create_buffer(memory_properties: &vk::PhysicalDeviceMemoryProperties, device: &DeviceLoader, usage_flags: vk::BufferUsageFlags, memory_property_flags: vk::MemoryPropertyFlags, size: vk::DeviceSize, data: *mut c_void) -> (vk::Buffer, vk::DeviceMemory) {
	// Create the buffer handle
    let buffer_info = vk::BufferCreateInfoBuilder::new()
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(usage_flags)
        .size(size);
    let buffer = unsafe{device.create_buffer(&buffer_info, None)}.unwrap();

    // Create the memory backing up the buffer handle
    let mem_requirements = unsafe {device.get_buffer_memory_requirements(buffer)};

    let mut mem_alloc = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(mem_requirements.size)
        .memory_type_index(get_memory_type(memory_properties, &mem_requirements.memory_type_bits, &memory_property_flags).unwrap());

    let mut alloc_flags_info = vk::MemoryAllocateFlagsInfoKHRBuilder::new();

    if usage_flags.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
        alloc_flags_info = alloc_flags_info.flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        mem_alloc.p_next = &mut alloc_flags_info as *mut _ as *mut c_void;
    }
    // If the buffer has VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT set we also need to enable the appropriate flag during allocation
    let memory = unsafe {device.allocate_memory(&mem_alloc, None)}.unwrap();
        
    // If a pointer to the buffer data has been passed, map the buffer and copy over the data
    let mapped = unsafe{device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())}.unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(data, mapped, size as usize);
    }
    
    // If host coherency hasn't been requested, do a manual flush to make writes visible
    if memory_property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
    {
        let mapped_range = vk::MappedMemoryRangeBuilder::new()
            .memory(memory)
            .offset(0)
            .size(size);
        unsafe {
            device.flush_mapped_memory_ranges(&[mapped_range]).unwrap();
        }
    }

    unsafe{
        device.unmap_memory(memory);
        device.bind_buffer_memory(buffer, memory, 0).unwrap();
    }

    return (buffer, memory);
}

fn get_memory_type(memory_properties: &vk::PhysicalDeviceMemoryProperties, type_bits: &u32, properties: &vk::MemoryPropertyFlags) -> Option<u32> {
    let mut type_bits = type_bits.clone();
    for i in 0..memory_properties.memory_type_count {
        if (type_bits & 1) == 1 {
            if (memory_properties.memory_types[i as usize].property_flags & *properties) == *properties {
                return Some(i);
            }
        }
        type_bits >>= 1;
    }
    None
}

fn get_buffer_device_address(device: &DeviceLoader, buffer: &vk::Buffer) -> u64 {
    let buffer_device_address_info = vk::BufferDeviceAddressInfoBuilder::new()
        .buffer(*buffer);
    return unsafe { device.get_buffer_device_address_khr(&buffer_device_address_info) };
}


fn create_acceleration_structure_buffer(device: &DeviceLoader, memory_properties: &vk::PhysicalDeviceMemoryProperties, build_size_info: vk::AccelerationStructureBuildSizesInfoKHR) -> (vk::Buffer, vk::DeviceMemory) {
    let buffer_create_info = vk::BufferCreateInfoBuilder::new()
        .size(build_size_info.acceleration_structure_size)
        .usage(vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR);

    let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }.unwrap();
    
    let mem_requriements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let mut memory_allocate_flag_info = vk::MemoryAllocateFlagsInfoKHRBuilder::new().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS_KHR);


    let mut memory_allocate_info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(mem_requriements.size)
        .memory_type_index(get_memory_type(memory_properties, &mem_requriements.memory_type_bits, &vk::MemoryPropertyFlags::DEVICE_LOCAL).unwrap());
    memory_allocate_info.p_next = &mut memory_allocate_flag_info as *mut _ as *mut c_void;
    
    let memory = unsafe { device.allocate_memory(&memory_allocate_info, None) }.unwrap();
    unsafe { device.bind_buffer_memory(buffer, memory, 0).unwrap() };
    (buffer, memory)
}

fn create_scratch_buffer(device: &DeviceLoader, memory_properties: &vk::PhysicalDeviceMemoryProperties, size: vk::DeviceSize) -> RayTracingScratchBuffer {

    let buffer_create_info = vk::BufferCreateInfoBuilder::new()
    .size(size)
    .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);

    let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }.unwrap();

    let mem_requriements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let mut memory_allocate_flag_info = vk::MemoryAllocateFlagsInfoKHRBuilder::new().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);

    let mut memory_allocate_info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(mem_requriements.size)
        .memory_type_index(get_memory_type(memory_properties, &mem_requriements.memory_type_bits, &vk::MemoryPropertyFlags::DEVICE_LOCAL).unwrap());
    memory_allocate_info.p_next = &mut memory_allocate_flag_info as *mut _ as *mut c_void;
    

    let memory = unsafe { device.allocate_memory(&memory_allocate_info, None) }.unwrap();
    unsafe { device.bind_buffer_memory(buffer, memory, 0).unwrap() };
    RayTracingScratchBuffer { device_address: unsafe { device.get_buffer_device_address(&vk::BufferDeviceAddressInfoBuilder::new().buffer(buffer)) }, handle: buffer, memory: memory }
}

fn create_command_buffer(device: &DeviceLoader, command_pool: &vk::CommandPool, count: u32) -> SmallVec<CommandBuffer> {
    let cmd_buf_allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(*command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(count);
    unsafe { device.allocate_command_buffers(&cmd_buf_allocate_info) }.unwrap()
}

fn flush_command_buffer(device: &DeviceLoader, command_buffer: &vk::CommandBuffer, queue: &vk::Queue, free: bool, command_pool: &vk::CommandPool) {
    unsafe { device.end_command_buffer(*command_buffer) }.unwrap();

    let command_buffers = &[*command_buffer];
    let submit_info = vk::SubmitInfoBuilder::new()
        .command_buffers(command_buffers);
    
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfoBuilder::new(), None) }.unwrap();

    unsafe { device.queue_submit(*queue, &[submit_info], fence) }.unwrap();

    unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }.unwrap();
    unsafe { device.destroy_fence(fence, None) };
    if free
    {
        unsafe { device.free_command_buffers(*command_pool, &[*command_buffer]) };
    }
}

fn create_acceleration_structure(device: &DeviceLoader, memory_properties: &vk::PhysicalDeviceMemoryProperties, command_pool: &vk::CommandPool, acceleration_structure_geometry: &vk::AccelerationStructureGeometryKHRBuilder, count: u32, _type: vk::AccelerationStructureTypeKHR) -> AccelerationStructure {
    let geometryies = &[*acceleration_structure_geometry];
    let acceleration_structure_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHRBuilder::new()
        ._type(_type)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE_KHR)
        .geometries(geometryies);

    let acceleration_structure_build_size = unsafe { device.get_acceleration_structure_build_sizes_khr(
            vk::AccelerationStructureBuildTypeKHR::DEVICE_KHR,
            &acceleration_structure_build_geometry_info,
            &[count],
            ) };

    let (buffer, memory) = create_acceleration_structure_buffer(device, memory_properties, acceleration_structure_build_size);
    
    let acceleration_structure_handle_create_info = vk::AccelerationStructureCreateInfoKHRBuilder::new()
        .buffer(buffer)
        .size(acceleration_structure_build_size.acceleration_structure_size)
        ._type(_type);
    let acceleration_structure = unsafe { device.create_acceleration_structure_khr(&acceleration_structure_handle_create_info, None) }.unwrap();

    // Create a small scratch buffer used during build of the bottom level acceleration structure
    let scratch_buffer = create_scratch_buffer(device, memory_properties, acceleration_structure_build_size.acceleration_structure_size);

    let acceleration_structure_geometries = &[*acceleration_structure_geometry];
    let acceleration_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHRBuilder::new()
        ._type(_type)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE_KHR)
        .mode(BuildAccelerationStructureModeKHR::BUILD_KHR)
        .dst_acceleration_structure(acceleration_structure)
        .geometries(acceleration_structure_geometries)
        .scratch_data(vk::DeviceOrHostAddressKHR{ device_address: scratch_buffer.device_address});

    let acceleration_structure_build_range_info = vk::AccelerationStructureBuildRangeInfoKHRBuilder::new()
        .primitive_count(count)
        .primitive_count(0)
        .first_vertex(0)
        .transform_offset(0);

    let command_buffer = create_command_buffer(device, command_pool, 1)[0];
        unsafe { device.cmd_build_acceleration_structures_khr(command_buffer, &[acceleration_build_geometry_info], &[&*acceleration_structure_build_range_info]) };

    // Build the acceleration structure on the device via a one-time command buffer submission
    // Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
    
    let acceleration_structure_address_info = vk::AccelerationStructureDeviceAddressInfoKHRBuilder::new()
        .acceleration_structure(acceleration_structure);
    let device_address = unsafe { device.get_acceleration_structure_device_address_khr(&acceleration_structure_address_info) };
    unsafe { device.free_memory(scratch_buffer.memory, None) };
    unsafe { device.destroy_buffer(scratch_buffer.handle, None) };
    AccelerationStructure { handle: acceleration_structure, device_address, memory, buffer: buffer }
}

fn create_top_level_acceleration_structure(device: &DeviceLoader, memory_properties: &vk::PhysicalDeviceMemoryProperties, command_pool: &vk::CommandPool, transform_matrix: &vk::TransformMatrixKHR, bottom_level_acceleration_structure_address: u64) -> AccelerationStructure {
    let mut instance = vk::AccelerationStructureInstanceKHRBuilder::new()
        .transform(*transform_matrix)
        .instance_custom_index(0)
        .mask(0xFF)
        .instance_shader_binding_table_record_offset(0)
        .flags(vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE_KHR)
        .acceleration_structure_reference(bottom_level_acceleration_structure_address);

    // Buffer for instance data
    let (buffer, memory) = create_buffer(memory_properties, device, vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, size_of::<vk::AccelerationStructureInstanceKHR>() as u64,  &mut instance as *mut _ as *mut c_void);

    let instance_address = get_buffer_device_address(device, &buffer);

    let acceleration_structure_geometry = vk::AccelerationStructureGeometryKHRBuilder::new()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES_KHR)
        .flags(vk::GeometryFlagsKHR::OPAQUE_KHR)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: *vk::AccelerationStructureGeometryInstancesDataKHRBuilder::new()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR { device_address: instance_address})
        });


    let acceleration_structure = create_acceleration_structure(device, memory_properties, command_pool, &acceleration_structure_geometry, 1, vk::AccelerationStructureTypeKHR::TOP_LEVEL_KHR);
    unsafe { device.destroy_buffer(buffer, None) };
    acceleration_structure
}

fn create_storage_image(device: &DeviceLoader, color_format: &vk::Format, memory_properties: &vk::PhysicalDeviceMemoryProperties, command_pool: &vk::CommandPool, queue: &vk::Queue) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
    let (image, memory) = create_image(device, color_format, memory_properties, vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE, vk::ImageLayout::UNDEFINED, &vk::MemoryPropertyFlags::DEVICE_LOCAL);
    let image_view = create_image_view(device, &image, color_format);
    
    transition_image_layout(device, command_pool, image, &vk::ImageLayout::UNDEFINED, &vk::ImageLayout::GENERAL, queue).unwrap();
    (image, memory, image_view)
}

fn create_image(device: &DeviceLoader, color_format: &vk::Format, memory_properties: &vk::PhysicalDeviceMemoryProperties, usage_flags: vk::ImageUsageFlags, initial_layout: vk::ImageLayout, properties: &vk::MemoryPropertyFlags) -> (vk::Image, vk::DeviceMemory) {
    let image_create_info = vk::ImageCreateInfoBuilder::new()
    .array_layers(1)
    .image_type(vk::ImageType::_2D)
    .format(*color_format)
    .mip_levels(1)
    .samples(vk::SampleCountFlagBits::_1)
    .tiling(vk::ImageTiling::OPTIMAL)
    .usage(usage_flags)
    .initial_layout(initial_layout);

    let image = unsafe { device.create_image(&image_create_info, None) }.unwrap();

    let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem_alloc_info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size( mem_reqs.size)
        .memory_type_index(get_memory_type(memory_properties, &mem_reqs.memory_type_bits, properties).unwrap());
    let memory =  unsafe { device.allocate_memory(&mem_alloc_info, None) }.unwrap();
    unsafe { device.bind_image_memory(image, memory, 0) }.unwrap();

    (image, memory)
}

fn create_image_view(device: &DeviceLoader, image: &vk::Image, format: &vk::Format) -> vk::ImageView {
    let image_view_info = vk::ImageViewCreateInfoBuilder::new()
        .image(*image)
        .view_type(vk::ImageViewType::_2D)
        .format(*format)
        .components(vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        })
        .subresource_range(
            vk::ImageSubresourceRangeBuilder::new()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        );
    unsafe { device.create_image_view(&image_view_info, None) }.unwrap()
}

fn transition_image_layout(device: &DeviceLoader, command_pool: &vk::CommandPool, image: vk::Image, old_layout: &vk::ImageLayout, new_layout: &vk::ImageLayout, queue: &vk::Queue) -> Result<(), String> {
        let command_buffer = create_command_buffer(device, command_pool, 1)[0];
        let mut barrier = vk::ImageMemoryBarrierBuilder::new()
            .old_layout(*old_layout)
            .new_layout(*new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRangeBuilder::new()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .base_mip_level(0)
                .level_count(1)
                .layer_count(1)
                .build());

        let source_stage: vk::PipelineStageFlags;
        let destination_stage: vk::PipelineStageFlags;

        if *old_layout == vk::ImageLayout::UNDEFINED && *new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL {
            barrier.src_access_mask = vk::AccessFlags::empty();
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;

            source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
            destination_stage = vk::PipelineStageFlags::TRANSFER;
        } else if *old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL && *new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask =  vk::AccessFlags::TRANSFER_READ;

            source_stage = vk::PipelineStageFlags::TRANSFER;
            destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        } else {
            return Err("Invalide transition".to_owned())
        }

        unsafe { device.cmd_pipeline_barrier(
                    command_buffer,
                    source_stage, destination_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier]
                ) };

        flush_command_buffer(device, &command_buffer, queue, true, command_pool);
        Ok(())
}

fn create_shader_module(device: &DeviceLoader, code_path: String) -> vk::ShaderModule {
    let code = fs::read(code_path).unwrap();
    let decoded_code = utils::decode_spv(code.as_slice()).unwrap();
    let create_info = vk::ShaderModuleCreateInfoBuilder::new()
        .code(&decoded_code);

    unsafe { device.create_shader_module(&create_info, None) }.unwrap()
} 

fn create_shader_stage(device: &DeviceLoader, stage: vk::ShaderStageFlagBits, path: String) -> (vk::PipelineShaderStageCreateInfoBuilder, vk::ShaderModule) {
    let module = create_shader_module(device, path);
    
    (vk::PipelineShaderStageCreateInfoBuilder::new()
        .stage(stage)
        .module(module), module)
}

fn create_ray_tracing_pipeline(device: &DeviceLoader) -> (vk::Pipeline, Vec<vk::ShaderModule>, Vec<vk::RayTracingShaderGroupCreateInfoKHRBuilder>, vk::DescriptorSetLayout, vk::PipelineLayout) {
    let create_descriptor_layout_binding = |binding, descriptor_type| {
        vk::DescriptorSetLayoutBindingBuilder::new()
        .binding(binding)
        .descriptor_type(descriptor_type)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
    };

    let acceleration_structure_layout_binding = create_descriptor_layout_binding(0, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR);
    let result_image_layout_binding = create_descriptor_layout_binding(1, vk::DescriptorType::STORAGE_IMAGE);
    let uniform_buffer_structure_layout_binding = create_descriptor_layout_binding(2, vk::DescriptorType::UNIFORM_BUFFER);
    let bindings = vec![acceleration_structure_layout_binding, result_image_layout_binding, uniform_buffer_structure_layout_binding];

    let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);
    let descriptor_set_layout =  unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None) }.unwrap();

    let layouts = &vec![descriptor_set_layout];
    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfoBuilder::new().set_layouts(&layouts);
    let pipline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }.unwrap();
    
    
    let mut shader_groupes = Vec::<vk::RayTracingShaderGroupCreateInfoKHRBuilder>::new();
    let mut shader_stages = Vec::<vk::PipelineShaderStageCreateInfoBuilder>::new();
    let mut shader_modules = Vec::<vk::ShaderModule>::new();

    {
        let (shader_stage, module) = create_shader_stage(device, vk::ShaderStageFlagBits::RAYGEN_KHR, "shaders/raygen".to_owned());
        shader_stages.push(shader_stage);
        shader_modules.push(module);
        let shader_group = vk::RayTracingShaderGroupCreateInfoKHRBuilder::new()
            ._type(vk::RayTracingShaderGroupTypeKHR::GENERAL_KHR)
            .general_shader((shader_groupes.len()-1) as u32)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR);
        shader_groupes.push(shader_group);        
    }

    {
        let (shader_stage, module) = create_shader_stage(device, vk::ShaderStageFlagBits::MISS_KHR, "shaders/miss".to_owned());
        shader_stages.push(shader_stage);
        shader_modules.push(module);
        let shader_group = vk::RayTracingShaderGroupCreateInfoKHRBuilder::new()
            ._type(vk::RayTracingShaderGroupTypeKHR::GENERAL_KHR)
            .general_shader((shader_groupes.len()-1) as u32)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR);
        shader_groupes.push(shader_group);
    }


    {
        let (shader_stage, module) = create_shader_stage(device, vk::ShaderStageFlagBits::CLOSEST_HIT_KHR, "shaders/closesthit".to_owned());
        shader_stages.push(shader_stage);
        shader_modules.push(module);
        let shader_group = vk::RayTracingShaderGroupCreateInfoKHRBuilder::new()
            ._type(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP_KHR)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader((shader_groupes.len()-1) as u32)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR);
        shader_groupes.push(shader_group);
    }


    let raytracing_pipeline_create_info = vk::RayTracingPipelineCreateInfoKHRBuilder::new()
        .stages(&shader_stages)
        .groups(&shader_groupes)
        .max_pipeline_ray_recursion_depth(1)
        .layout(pipline_layout);

    let pipeline = unsafe { device.create_ray_tracing_pipelines_khr(vk::DeferredOperationKHR::null(), vk::PipelineCache::null(), &vec![raytracing_pipeline_create_info], None) }.unwrap()[0];
    (pipeline, shader_modules, shader_groupes, descriptor_set_layout, pipline_layout)
}

fn aligned_size(value: u64, alignment: u64) -> u64{
    (value + alignment - 1) & !(alignment - 1)
}

fn create_shader_binding_table(device: &DeviceLoader, memory_properties: &vk::PhysicalDeviceMemoryProperties, pipeline: &vk::Pipeline, raytracing_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR, shader_groups: Vec<vk::RayTracingShaderGroupCreateInfoKHRBuilder>) -> ((vk::Buffer, vk::DeviceMemory), (vk::Buffer, vk::DeviceMemory), (vk::Buffer, vk::DeviceMemory)) {

    let handle_size = raytracing_pipeline_properties.shader_group_handle_size;
    let handle_size_alinged = aligned_size(handle_size as u64, raytracing_pipeline_properties.shader_group_handle_alignment as u64);
    let sbt_size = shader_groups.len() as u64 * handle_size_alinged;

    let mut shader_handle_storage = Vec::<u32>::new();
    shader_handle_storage.reserve(sbt_size as usize);

    unsafe { device.get_ray_tracing_shader_group_handles_khr(*pipeline, 0, shader_groups.len() as u32, sbt_size as usize, shader_handle_storage.as_mut_ptr() as *mut c_void) }.unwrap();

    let buffer_usage_flags = vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR; 
    let memory_usage_flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    let raygen_shader_binding_table = create_buffer(memory_properties, device, buffer_usage_flags, memory_usage_flags, handle_size as u64, shader_handle_storage.as_mut_ptr() as *mut c_void);
    let miss_shader_binding_table = create_buffer(memory_properties, device, buffer_usage_flags, memory_usage_flags, handle_size as u64, unsafe { (shader_handle_storage.as_mut_ptr() as *mut c_void).add(handle_size_alinged as usize) });
    let hit_shader_binding_table = create_buffer(memory_properties, device, buffer_usage_flags, memory_usage_flags, handle_size as u64, unsafe { (shader_handle_storage.as_mut_ptr() as *mut c_void).add(2 * handle_size_alinged as usize) });

    (raygen_shader_binding_table, miss_shader_binding_table, hit_shader_binding_table)
}

fn create_descriptor_sets(device: &DeviceLoader, descriptor_set_layout: vk::DescriptorSetLayout, handle: vk::AccelerationStructureKHR, storage_image_view: vk::ImageView, ubo: vk::Buffer) -> vk::DescriptorSet {
    let create_size = |_type| {DescriptorPoolSizeBuilder::new()._type(_type).descriptor_count(1)};

    let pool_sizes = vec![
            create_size(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR),
            create_size(vk::DescriptorType::STORAGE_IMAGE),
            create_size(vk::DescriptorType::UNIFORM_BUFFER),
        ];

    let descriptor_pool_create_info = vk::DescriptorPoolCreateInfoBuilder::new()
        .pool_sizes(&pool_sizes)
        .max_sets(1);
    let descriptor_pool =  unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None) }.unwrap();
    
    let descriptor_set_layouts = &vec![descriptor_set_layout];
    let descriptor_pool_alloc_ci = vk::DescriptorSetAllocateInfoBuilder::new()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&descriptor_set_layouts);
    let descriptor_set =  unsafe { device.allocate_descriptor_sets(&descriptor_pool_alloc_ci) }.unwrap()[0];

    let handels = &vec![handle];
    let mut descriptor_accleration_structure_info = vk::WriteDescriptorSetAccelerationStructureKHRBuilder::new()
        .acceleration_structures(handels);
    let mut acceleration_structure_write = vk::WriteDescriptorSetBuilder::new()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR);
        acceleration_structure_write.p_next = &mut descriptor_accleration_structure_info as *mut _ as *mut c_void;

    let storage_image_descriptor = vk::DescriptorImageInfoBuilder::new()
        .image_view(storage_image_view)
        .image_layout(vk::ImageLayout::GENERAL);

    let storage_image_descriptors = &[storage_image_descriptor];
    let result_image_write = vk::WriteDescriptorSetBuilder::new()
        .dst_set(descriptor_set)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .dst_binding(1)
        .image_info(storage_image_descriptors);

    let unifrom_buffer_descriptor = vk::DescriptorBufferInfoBuilder::new()
        .buffer(ubo)
        .offset(0);

    let unifrom_buffer_descriptors = &[unifrom_buffer_descriptor];
    let unifrom_buffer_write = vk::WriteDescriptorSetBuilder::new()
        .dst_set(descriptor_set)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .dst_binding(2)
        .buffer_info(unifrom_buffer_descriptors);

    let write_descriptor_sets = vec![
        acceleration_structure_write,
        result_image_write,
        unifrom_buffer_write,
    ];

    unsafe { device.update_descriptor_sets(&write_descriptor_sets, &[]) };
    descriptor_set
}