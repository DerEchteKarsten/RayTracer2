use ash::vk::{self, DescriptorType, ShaderStageFlags};

use crate::{context::*, shader_params::GConst, Model};

struct RayTracingPass {
    shader_binding_table: ShaderBindingTable,
    shader_group_info: RayTracingShaderGroupInfo,
    handle: vk::Pipeline,
}

struct ComputePass {
    handel: vk::Pipeline,
}

pub struct LightPasses {
    presample_lights_pass: ComputePass,
    presample_environment_map_pass: ComputePass,
    presample_re_gir: ComputePass,

    g_buffer_pass: RayTracingPass,

    di_generate_initial_samples_pass: RayTracingPass,
    di_fused_resampling_pass: RayTracingPass,
    di_gradients_pass: RayTracingPass,
    di_shade_samples_pass: RayTracingPass,

    brdf_ray_tracing_pass: RayTracingPass,
    gi_temporal_resampling_pass: RayTracingPass,
    gi_spatial_resampling_pass: RayTracingPass,
    gi_fused_resampling_pass: RayTracingPass,
    gi_final_shading_pass: RayTracingPass,

    layout: vk::PipelineLayout,
    static_set: vk::DescriptorSet,
    current_set: vk::DescriptorSet,
    prev_set: vk::DescriptorSet,

    uniform_buffer: Buffer,
    unifrom_data: GConst,
}

pub struct RenderResources {
    task_buffer: Buffer,
    primitive_light_buffer: Buffer,
    light_data_buffer: Buffer,
    geometry_instance_to_light_buffer: Buffer,
    light_index_mapping_buffer: Buffer,
    ris_buffer: Buffer,
    ris_light_data_buffer: Buffer,
    neighbor_offsets_buffer: Buffer,
    di_reservoir_buffer: Buffer,
    secondary_gbuffer: Buffer,
    environment_pdf_texture: ImageAndView,
    local_light_pdf_texture: ImageAndView,
    gi_reservoir_buffer: Buffer,
}

impl RenderResources {
    fn new(ctx: &mut Renderer) -> Self {


        
        Self {
            task_buffer: (),
            primitive_light_buffer: (),
            light_data_buffer: (),
            geometry_instance_to_light_buffer: (),
            light_index_mapping_buffer: (),
            ris_buffer: (),
            ris_light_data_buffer: (),
            neighbor_offsets_buffer: (),
            di_reservoir_buffer: (),
            secondary_gbuffer: (),
            environment_pdf_texture: (),
            local_light_pdf_texture: (),
            gi_reservoir_buffer: (),
        }
    }
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
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // Restir Luminance
        ];
        bindings
            .iter_mut()
            .enumerate()
            .map(|(i, b)| b.binding(i as u32).stage_flags(ShaderStageFlags::ALL));
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
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // Geometrie Infos
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // Vertex Buffer
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // Index Buffer
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER) // Textures
                .descriptor_count(num_textures),
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
                .descriptor_type(DescriptorType::STORAGE_IMAGE), // Gradients
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // GI Reservoirs
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // RIS Buffer
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(DescriptorType::STORAGE_BUFFER), // RIS Light Data
        ];
        bindings
            .iter_mut()
            .enumerate()
            .map(|(i, b)| b.binding(i as u32).stage_flags(ShaderStageFlags::ALL));
        bindings
    }

    fn new(ctx: &mut Renderer, model: &Model) -> Self {
        unsafe {
            let static_bindings = Self::get_static_descriptor_bindings(model.textures.len() as u32);
            let static_set_layout = ctx
                .create_descriptor_set_layout(&static_bindings, &[])
                .unwrap();

            let dynamic_bindings = Self::get_dynamic_descriptor_bindings();
            let dynamic_set_layout = ctx
                .create_descriptor_set_layout(&dynamic_bindings, &[])
                .unwrap();

            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .push_constant_ranges(&[vk::PushConstantRange {
                    offset: 0,
                    size: 4,
                    stage_flags: ShaderStageFlags::ALL,
                }])
                .set_layouts(&[static_set_layout, dynamic_set_layout, dynamic_set_layout]);

            let layout = ctx
                .device
                .create_pipeline_layout(&layout_info, None)
                .unwrap();

            Self {
                presample_lights_pass: (),
                presample_environment_map_pass: (),
                presample_re_gir: (),
                g_buffer_pass: (),
                di_generate_initial_samples_pass: (),
                di_fused_resampling_pass: (),
                di_gradients_pass: (),
                di_shade_samples_pass: (),
                brdf_ray_tracing_pass: (),
                gi_temporal_resampling_pass: (),
                gi_spatial_resampling_pass: (),
                gi_fused_resampling_pass: (),
                gi_final_shading_pass: (),
                layout,
                static_set: (),
                current_set: (),
                prev_set: (),
                uniform_buffer: (),
                di_reservoir_buffer: (),
                gi_reservoir_buffer: (),
            }
        }
    }
}
