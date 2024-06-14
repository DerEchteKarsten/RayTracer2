use ash::{extensions::ext::{self, DebugUtils}, Entry, Instance};

use memoffset::offset_of;
use simple_logger::SimpleLogger;

use crate::{
    context::*,
    create_acceleration_structure,
    gltf::{self, Vertex}, RayTracingShaderCreateInfo,
};

use anyhow::Result;
use ash::extensions::khr;
use ash::vk;
use gpu_allocator::{vulkan::{Allocator, AllocatorCreateDesc}, AllocatorDebugSettings, MemoryLocation};

use glam::{vec3, Mat4, Vec3, Vec4};
use std::{default::Default, ffi::c_char, os::raw::c_void};
use std::ffi::{CStr, CString};
use std::os::unix::thread;
use std::slice::from_ref;
use std::time::{Duration, Instant};

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

use std::mem::{size_of, transmute};

use std::sync::Arc;

type ImageAV = (Image, vk::ImageView);

enum ResourceType {
    Buffer(Buffer),
    Image(ImageAV),
    Texture(ImageAV),
    Sampler(vk::Sampler),
    AccelerationStructure(AccelerationStructure),
}

struct Resource {
    pub _type: ResourceType,
    pub count: u32,
}

impl Resource {
    fn get_type(&self) -> vk::DescriptorType{
        match self._type {
            ResourceType::Buffer(_) => vk::DescriptorType::STORAGE_BUFFER,
            ResourceType::Image(_) => vk::DescriptorType::STORAGE_IMAGE,
            ResourceType::Texture(_) => vk::DescriptorType::SAMPLED_IMAGE,
            ResourceType::Sampler(_) => vk::DescriptorType::SAMPLER,
            ResourceType::AccelerationStructure(_) => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        }
    }
}

type ExternalResource = Arc<Resource>; 

enum RenderPassTypeCreateInfo<'a> {
    Compute(&'a str),
    Raytracing(&'a [RayTracingShaderCreateInfo<'a>]),
    Raster((&'a str, &'a str)),
}

type ResourceRef = usize;

enum RenderPassType {
    Compute,
    Raytracing(ShaderBindingTable),
    Raster(vk::RenderPass)
}

type RenderFunction = fn([&Resource], [ExternalResource]) -> ();

struct RenderPass {
    pub _type: RenderPassType,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub resources: Vec<ResourceRef>,
    pub external_resources: Vec<ResourceRef>,
    pub dependency: usize,
    pub name: String,
    pub execute: RenderFunction,
}

impl RenderPass {
    pub fn builder<'a>() -> RenderPassBuilder<'a> {
        RenderPassBuilder::default()
    }
}

struct TargetRef {
    pub res: ResourceRef,
    pub external: bool,
}

#[derive(Default)]
pub struct RenderPassBuilder<'a> {
    name: String,
    dependency: usize,
    recources: Vec<ResourceRef>,
    external_recources: Vec<ResourceRef>,
    targets: Vec<TargetRef>,
    _type: Option<RenderPassTypeCreateInfo<'a>>,
    execute: Option<RenderFunction>,
}

impl<'a> RenderPassBuilder<'a> {
    pub fn name(mut self, name: String) -> Self { self.name = name; self }
    pub fn dependency(mut self, dependency: usize) -> Self { self.dependency = dependency; self }
    pub fn resource(mut self, resource: ResourceRef) -> Self { self.recources.push(resource); self }
    pub fn external_resource(mut self, resource: ResourceRef) -> Self { self.external_recources.push(resource); self }
    pub fn target(mut self, target: TargetRef) -> Self { self.targets.push(target); self }

    pub fn _type(mut self, _type: RenderPassTypeCreateInfo) -> Self {
        self._type = Some(_type);
        self
    }

    pub fn execute(mut self, func: RenderFunction) -> Self {
        self.execute = Some(func);
        self
    }

    pub fn build(
        self,
        renderer: &mut Renderer,
    ) -> Result<RenderPass> {
        let dynamic_layout_bindings = self.recources.iter().enumerate().map(|(i, r)| {
            let resource = renderer.get_resource(*r);
            vk::DescriptorSetLayoutBinding::builder()
                .descriptor_count(resource.count)
                .binding(i as u32)
                .descriptor_type(resource.get_type())
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build()
        }).collect::<Vec<_>>();

        let static_layout_bindings = self.external_recources.iter().enumerate().map(|(i, er)| {
            let resource = renderer.get_external_resource(*er);
            vk::DescriptorSetLayoutBinding::builder()
                .descriptor_count(resource.count)
                .descriptor_type(resource.get_type())
                .binding(i as u32)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build()
        }).collect::<Vec<_>>();

        let dynamic_layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(dynamic_layout_bindings.as_slice()).build();
        let static_layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(static_layout_bindings.as_slice()).build();

        let dynamic_layout = unsafe { renderer.device.create_descriptor_set_layout(&dynamic_layout_info, None) }?;
        let static_layout = unsafe { renderer.device.create_descriptor_set_layout(&static_layout_info, None) }?;

        let layout_info = if matches!(self._type.unwrap(), RenderPassTypeCreateInfo::Raytracing(_)) || matches!(self._type.unwrap(), RenderPassTypeCreateInfo::Compute(_)) {
            let target_bidings = self.targets.iter().enumerate().map(|(i, e)| {
                let mut desc = vk::DescriptorSetLayoutBinding::builder()
                    .binding(i as u32);
                desc = if e.external {
                    let res = renderer.get_external_resource(e.res);
                    desc
                        .descriptor_count(res.count)
                        .descriptor_type(res.get_type())
                        .stage_flags(vk::ShaderStageFlags::ALL)
                } else {
                    let res = renderer.get_resource(e.res);
                    desc
                        .descriptor_count(res.count)
                        .descriptor_type(res.get_type())
                        .stage_flags(vk::ShaderStageFlags::ALL)
                };
                desc.build()
            }).collect::<Vec<_>>();
            let target_layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(target_bidings.as_slice()).build();
            let target_layout = unsafe { renderer.device.create_descriptor_set_layout(&target_layout_info, None) }?;
            vk::PipelineLayoutCreateInfo::builder().set_layouts(&[dynamic_layout, static_layout, target_layout])
        } else {
            vk::PipelineLayoutCreateInfo::builder().set_layouts(&[dynamic_layout, static_layout])
        };

        let layout = unsafe { renderer.device.create_pipeline_layout(&layout_info, None) }?;

        let (pipeline, type_data) = match self._type.unwrap() {
            RenderPassTypeCreateInfo::Compute(path) => {
                (create_compute_pipeline(&renderer, &layout, path)?, RenderPassType::Compute)
            },
            RenderPassTypeCreateInfo::Raster((vertex_path, fragment_path)) => {
                let render_pass_create_info = vk::RenderPassCreateInfo::builder().build();
                let render_pass = unsafe { renderer.device.create_render_pass(&render_pass_create_info, None) }?;
                todo!("renderpass");
                let pipeline = create_raster_pipeline(renderer, &layout, render_pass, vertex_path, fragment_path)?;
                (pipeline, RenderPassType::Raster(render_pass))
            },
            RenderPassTypeCreateInfo::Raytracing(shader_infos) => {
                let (pipeline, info) = create_ray_tracing_pipeline(&renderer, shader_infos, layout)?; 
                let shader_binding_table = ShaderBindingTable::new(renderer, &pipeline, info)?;
                
                (pipeline, RenderPassType::Raytracing(shader_binding_table))
            },
        };

        Ok(RenderPass {
            layout,
            pipeline,
            _type: type_data,
            dependency: self.dependency,
            name: self.name,
            resources: self.recources,
            execute: self.execute.unwrap(),
            external_resources: self.external_recources,
        })
    }
}

#[derive()]
pub struct Renderer {
    pub allocator: Allocator,
    pub command_pool: vk::CommandPool,
    pub graphics_queue: Queue,
    pub present_queue: Queue,
    pub compute_queue: Queue,
    pub device: ash::Device,
    pub physical_device: PhysicalDevice,
    pub instance: Instance,
    pub target: Target,
    _entry: Entry,
    pub render_passes: Vec<RenderPass>,
    pub ray_tracing_context: RayTracingContext,
    pub resources: Vec<Resource>,
    pub external_resources: Vec<ExternalResource>,
}

impl Renderer {
    // This method will help users to discover the builder
    pub fn builder<'a>() -> RenderBuilder<'a> {
        RenderBuilder::default()
    }

    pub fn get_resource<'a>(self, _ref: ResourceRef) -> & 'a Resource {
        &self.resources[_ref]
    }

    pub fn get_external_resource<'a>(self, _ref: ResourceRef) -> ExternalResource {
        self.external_resources[_ref].clone()
    }

    pub fn render(self) {
        for r in self.render_passes {
            let resources = r.resources.iter().map(|r| {
                self.get_resource(*r)
            }).collect::<Vec<_>>();
            let external_resources = r.resources.iter().map(|r| {
                self.get_external_resource(*r)
            }).collect::<Vec<_>>();
            (r.execute)(*resources.as_slice(), *external_resources.as_slice());
        }
    } 
}

#[derive(Default)]
pub struct RenderBuilder<'a> {
    render_passes: Vec<RenderPassBuilder<'a>>,
    required_extensions: &'a[&'static str],
    required_device_features: DeviceFeatures,
    app_name: &'a str,
    resources: Vec<Resource>,
    external_resources: Vec<ExternalResource>,
}

impl<'a> RenderBuilder<'a> {
    pub fn required_device_features(mut self, features: DeviceFeatures) -> Self {
        self.required_device_features = features;
        self
    }
    pub fn required_extensions(mut self, extensions: &'static[& 'static str]) -> Self {
        self.required_extensions = extensions;
        self
    }
    pub fn app_name(mut self, name: &'static str) -> Self {
        self.app_name = name;
        self
    }

    pub fn add_resource(mut self, _res: Resource) -> Self {
        self.resources.push(_res);
        self
    }

    pub fn add_external_resource(mut self, _res: ExternalResource) -> Self {
        self.external_resources.push(_res);
        self
    }

    pub fn pass(mut self, pass: RenderPassBuilder) -> Self {
        
        self.render_passes.push(pass);
        self
    }

    pub fn build(
        self,
        window: &winit::window::Window
    ) -> Result<Renderer> {
        SimpleLogger::default().env().init().unwrap();
        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);

        let layer_names = unsafe {
            [CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            )]
        };

        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let display_handle = window.raw_display_handle();
        let window_handle = window.raw_window_handle();

        let mut instance_extensions = ash_window::enumerate_required_extensions(display_handle)
            .unwrap()
            .to_vec();

        //#[cfg(debug_assertions)]
        instance_extensions.push(DebugUtils::name().as_ptr());

        let mut validation_features = vk::ValidationFeaturesEXT::builder()
            .enabled_validation_features(&[vk::ValidationFeatureEnableEXT::DEBUG_PRINTF])
            .build();

        let mut instance_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);

        //#[cfg(debug_assertions)]
        {
            instance_info = instance_info
                .enabled_layer_names(&layers_names_raw)
                .push_next(&mut validation_features);
        }

        let mut instance = unsafe { entry.create_instance(&instance_info, None)? };

        //#[cfg(debug_assertions)]
        {
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
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
            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            let debug_call_back = unsafe {
                debug_utils_loader
                    .create_debug_utils_messenger(&debug_info, None)
                    .unwrap()
            };
        }
        let vk_surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
        }
        .unwrap();
        let ash_surface = khr::Surface::new(&entry, &instance);

        let physical_devices =
            unsafe { enumerate_physical_devices(&instance, &ash_surface, &vk_surface)? };
        let (physical_device, graphics_queue_family, present_queue_family, compute_queue_family) =
            select_suitable_physical_device(
                physical_devices.as_slice(),
                self.required_extensions,
                &self.required_device_features,
            )?;

        let queue_families = [&graphics_queue_family, &present_queue_family, &compute_queue_family];

        let device = create_device(
            &instance,
            &physical_device,
            &queue_families,
            self.required_extensions,
        )?;

        let graphics_queue = Queue { handel: get_queue(&device, &graphics_queue_family, 0), family: graphics_queue_family};
        let present_queue = Queue { handel: get_queue(&device, &present_queue_family, 0), family: present_queue_family};
        let compute_queue = Queue { handel: get_queue(&device, &compute_queue_family, 0), family: compute_queue_family};


        let ray_tracing = {
            let ray_tracing = RayTracingContext::new(&instance, &physical_device.handel, &device);
            ray_tracing
        };

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
        })?;
        
        let window_size = window.inner_size();

        let swapchain = Swapchain::new(
            &instance,
            &device,
            vk_surface,
            &ash_surface,
            &physical_device,
            window_size.width,
            window_size.height,
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
        
        let target = Target {
            ash_surface_ext: ash_surface,
            frame_in_flight: 0,
            frames_in_flight: frames_in_flight,
            swapchain,
            vk_surface: vk_surface,
        };
        let mut renderer = Renderer {
            resources: self.resources,
            external_resources: self.external_resources,
            _entry: entry,
            allocator,
            command_pool,
            compute_queue,
            graphics_queue,
            present_queue,
            device,
            instance,
            physical_device,
            target,
            render_passes: vec![],
            ray_tracing_context: ray_tracing,
        };

        for r in self.render_passes {
            let render_pass = r.build(&mut renderer)?;
            renderer.render_passes.push(render_pass);
        }

        renderer.render_passes.sort_by(|e, f| {
            e.dependency.partial_cmp(&e.dependency).unwrap()
        });
        Ok(renderer)
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Flag;

    let message = CStr::from_ptr((*p_callback_data).p_message);
    match flag {
        Flag::VERBOSE => log::info!("{:?} - {:?}", typ, message),
        Flag::INFO => log::info!("{:?} - {:?}", typ, message),
        Flag::WARNING => log::warn!("{:?} - {:?}", typ, message),
        _ => log::error!("{:?} - {:?}", typ, message),
    }
    vk::FALSE
}


// pub fn render<F>(&mut self, func: F) -> Result<u64>
// where
//     F: FnOnce(&mut Context, u32),
// {
//     let before = Instant::now();

//     let (image_index, frame_index) = {
//         let frame_index: u64 = (self.frame + 1) % FRAMES_IN_FLIGHT as u64;
//         let frame = &self.frames_in_flight[frame_index as usize];
//         frame.fence.wait(&self.device, None)?;
//         frame.fence.reset(&self.device)?;

//         let image_index = match unsafe {
//             self.swapchain.ash_swapchain.acquire_next_image(
//                 self.swapchain.vk_swapchain,
//                 u64::MAX,
//                 frame.image_available,
//                 vk::Fence::null(),
//             )
//         } {
//             Err(err) => {
//                 if err == vk::Result::ERROR_OUT_OF_DATE_KHR || err == vk::Result::SUBOPTIMAL_KHR
//                 {
//                     debug!("Suboptimal");
//                 }
//                 0
//             }
//             Ok(v) => v.0,
//         };
//         let cmd = &self.cmd_buffs[image_index as usize];
//         begin_command_buffer(cmd, &self.device, None)?;
//         (image_index, frame_index)
//     };
//     func(self, image_index);

//     let frame = &self.frames_in_flight[frame_index as usize];
//     let cmd = &self.cmd_buffs[image_index as usize];
//     unsafe { self.device.end_command_buffer(*cmd) }.unwrap();
//     let submit_info = vk::SubmitInfo2::builder()
//         .wait_semaphore_infos(from_ref(
//             &vk::SemaphoreSubmitInfo::builder()
//                 .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT_KHR)
//                 .semaphore(frame.image_available)
//                 .build(),
//         ))
//         .command_buffer_infos(from_ref(
//             &vk::CommandBufferSubmitInfo::builder()
//                 .command_buffer(*cmd)
//                 .build(),
//         ))
//         .signal_semaphore_infos(from_ref(
//             &vk::SemaphoreSubmitInfo::builder()
//                 .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS_KHR)
//                 .semaphore(frame.render_finished)
//                 .build(),
//         ))
//         .build();

//     unsafe {
//         self.device
//             .queue_submit2(self.graphics_queue, &[submit_info], frame.fence.handel)?;
//     };

//     let rf = &[frame.render_finished];
//     let sc = &[self.swapchain.vk_swapchain];
//     let image_indices = vec![image_index];
//     let present_info = vk::PresentInfoKHR::builder()
//         .wait_semaphores(rf)
//         .swapchains(sc)
//         .image_indices(&image_indices);

//     match unsafe {
//         self.swapchain
//             .ash_swapchain
//             .queue_present(self.present_queue, &present_info)
//     } {
//         Err(err) => {
//             if err == vk::Result::ERROR_OUT_OF_DATE_KHR || err == vk::Result::SUBOPTIMAL_KHR {
//                 debug!("Suboptimal");
//             }
//         }
//         Ok(v) => {}
//     };
//     debug!("{:?}", Instant::now().duration_since(before));

//     Ok(frame_index)
// }