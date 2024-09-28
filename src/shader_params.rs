use glam::UVec2;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PlanarViewConstants {
    pub mat_world_to_view: glam::Mat4,
    pub mat_view_to_clip: glam::Mat4,
    pub mat_world_to_clip: glam::Mat4,
    pub mat_clip_to_view: glam::Mat4,
    pub mat_view_to_world: glam::Mat4,
    pub mat_clip_to_world: glam::Mat4,

    pub viewport_origin: glam::Vec2,
    pub viewport_size: glam::Vec2,

    pub viewport_size_inv: glam::Vec2,
    pub pixel_offset: glam::Vec2,

    pub clip_to_window_scale: glam::Vec2,
    pub clip_to_window_bias: glam::Vec2,

    pub window_to_clip_scale: glam::Vec2,
    pub window_to_clip_bias: glam::Vec2,

    pub camera_direction_or_position: glam::Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RTXDI_RuntimeParameters {
    pub neighbor_offset_mask: u32,      // Spatial
    pub active_checkerboard_field: u32, // 0 - no checkerboard, 1 - odd pixels, 2 - even pixels
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReSTIRGI_TemporalResamplingParameters {
    pub depth_threshold: f32,
    pub normal_threshold: f32,
    pub enable_permutation_sampling: u32,
    pub max_history_length: u32,

    pub max_reservoir_age: u32,
    pub enable_boiling_filter: u32,
    pub boiling_filter_strength: f32,
    pub enable_fallback_sampling: u32,

    pub temporal_bias_correction_mode: u32, // = ResTIRGI_TemporalBiasCorrectionMode::Basic;
    pub uniform_random_number: u32,
    pub pad2: u32,
    pub pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
// See note for ReSTIRGI_TemporalResamplingParameters
pub struct ReSTIRGI_SpatialResamplingParameters {
    pub spatial_depth_threshold: f32,
    pub spatial_normal_threshold: f32,
    pub num_spatial_samples: u32,
    pub spatial_sampling_radius: f32,

    pub spatial_bias_correction_mode: u32, // = ResTIRGI_SpatialBiasCorrectionMode::Basic;
    pub pad1: u32,
    pub pad2: u32,
    pub pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReSTIRGI_FinalShadingParameters {
    pub enable_final_visibility: u32, // = true;
    pub enable_final_mis: u32,        // = true;
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReSTIRGI_BufferIndices {
    pub secondary_surface_re_stirdioutput_buffer_index: u32,
    pub temporal_resampling_input_buffer_index: u32,
    pub temporal_resampling_output_buffer_index: u32,
    pub spatial_resampling_input_buffer_index: u32,

    pub spatial_resampling_output_buffer_index: u32,
    pub final_shading_input_buffer_index: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RTXDI_ReservoirBufferParameters {
    pub reservoir_block_row_pitch: u32,
    pub reservoir_array_pitch: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReSTIRGI_Parameters {
    pub reservoir_buffer_params: RTXDI_ReservoirBufferParameters,
    pub buffer_indices: ReSTIRGI_BufferIndices,
    pub temporal_resampling_params: ReSTIRGI_TemporalResamplingParameters,
    pub spatial_resampling_params: ReSTIRGI_SpatialResamplingParameters,
    pub final_shading_params: ReSTIRGI_FinalShadingParameters,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RTXDI_LightBufferRegion {
    pub first_light_index: u32,
    pub num_lights: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RTXDI_EnvironmentLightBufferParameters {
    pub light_present: u32,
    pub light_index: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RTXDI_LightBufferParameters {
    pub local_light_buffer_region: RTXDI_LightBufferRegion,
    pub infinite_light_buffer_region: RTXDI_LightBufferRegion,
    pub environment_light_params: RTXDI_EnvironmentLightBufferParameters,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReSTIRDI_InitialSamplingParameters {
    pub num_primary_local_light_samples: u32,
    pub num_primary_infinite_light_samples: u32,
    pub num_primary_environment_samples: u32,
    pub num_primary_brdf_samples: u32,

    pub brdf_cutoff: f32,
    pub enable_initial_visibility: u32,
    pub environment_map_importance_sampling: u32, // Only used in InitialSamplingFunctions.hlsli via RAB_EvaluateEnvironmentMapSamplingPdf
    pub local_light_sampling_mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReSTIRDI_BufferIndices {
    pub initial_sampling_output_buffer_index: u32,
    pub temporal_resampling_input_buffer_index: u32,
    pub temporal_resampling_output_buffer_index: u32,
    pub spatial_resampling_input_buffer_index: u32,

    pub spatial_resampling_output_buffer_index: u32,
    pub shading_input_buffer_index: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReSTIRDI_TemporalResamplingParameters {
    pub temporal_depth_threshold: f32,
    pub temporal_normal_threshold: f32,
    pub max_history_length: u32,
    pub temporal_bias_correction: u32,

    pub enable_permutation_sampling: u32,
    pub permutation_sampling_threshold: f32,
    pub enable_boiling_filter: u32,
    pub boiling_filter_strength: f32,

    pub discard_invisible_samples: u32,
    pub uniform_random_number: u32,
    pub pad2: u32,
    pub pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReSTIRDI_SpatialResamplingParameters {
    pub spatial_depth_threshold: f32,
    pub spatial_normal_threshold: f32,
    pub spatial_bias_correction: u32,
    pub num_spatial_samples: u32,

    pub num_disocclusion_boost_samples: u32,
    pub spatial_sampling_radius: f32,
    pub neighbor_offset_mask: u32,
    pub discount_naive_samples: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReSTIRDI_ShadingParameters {
    pub enable_final_visibility: u32,
    pub reuse_final_visibility: u32,
    pub final_visibility_max_age: u32,
    pub final_visibility_max_distance: f32,

    pub enable_denoiser_input_packing: u32,
    pub pad1: u32,
    pub pad2: u32,
    pub pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReSTIRDI_Parameters {
    pub reservoir_buffer_params: RTXDI_ReservoirBufferParameters,
    pub buffer_indices: ReSTIRDI_BufferIndices,
    pub initial_sampling_params: ReSTIRDI_InitialSamplingParameters,
    pub temporal_resampling_params: ReSTIRDI_TemporalResamplingParameters,
    pub spatial_resampling_params: ReSTIRDI_SpatialResamplingParameters,
    pub shading_params: ReSTIRDI_ShadingParameters,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RTXDI_RISBufferSegmentParameters {
    pub buffer_offset: u32,
    pub tile_size: u32,
    pub tile_count: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GConst {
    pub view: PlanarViewConstants,
    pub prev_view: PlanarViewConstants,
    pub runtime_params: RTXDI_RuntimeParameters,
    pub light_buffer_params: RTXDI_LightBufferParameters,

    pub restir_gi: ReSTIRGI_Parameters,
    pub restir_di: ReSTIRDI_Parameters,
    pub local_lights_risbuffer_segment_params: RTXDI_RISBufferSegmentParameters,
    pub environment_light_risbuffer_segment_params: RTXDI_RISBufferSegmentParameters,

    pub environment_pdf_texture_size: UVec2,
    pub local_light_pdf_texture_size: UVec2,
}
