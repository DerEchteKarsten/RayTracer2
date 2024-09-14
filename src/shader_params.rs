use std::default;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PlanarViewConstants {
    pub matWorldToView: glam::Mat4,
    pub matViewToClip: glam::Mat4,
    pub matWorldToClip: glam::Mat4,
    pub matClipToView: glam::Mat4,
    pub matViewToWorld: glam::Mat4,
    pub matClipToWorld: glam::Mat4,

    pub viewportOrigin: glam::Vec2,
    pub viewportSize: glam::Vec2,

    pub viewportSizeInv: glam::Vec2,
    pub pixelOffset: glam::Vec2,

    pub clipToWindowScale: glam::Vec2,
    pub clipToWindowBias: glam::Vec2,

    pub windowToClipScale: glam::Vec2,
    pub windowToClipBias: glam::Vec2,

    pub cameraDirectionOrPosition: glam::Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RTXDI_RuntimeParameters {
    pub neighborOffsetMask: u32,      // Spatial
    pub activeCheckerboardField: u32, // 0 - no checkerboard, 1 - odd pixels, 2 - even pixels
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReSTIRGI_TemporalResamplingParameters {
    pub depthThreshold: f32,
    pub normalThreshold: f32,
    pub enablePermutationSampling: u32,
    pub maxHistoryLength: u32,

    pub maxReservoirAge: u32,
    pub enableBoilingFilter: u32,
    pub boilingFilterStrength: f32,
    pub enableFallbackSampling: u32,

    pub temporalBiasCorrectionMode: u32, // = ResTIRGI_TemporalBiasCorrectionMode::Basic;
    pub uniformRandomNumber: u32,
    pub pad2: u32,
    pub pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
// See note for ReSTIRGI_TemporalResamplingParameters
pub struct ReSTIRGI_SpatialResamplingParameters {
    pub spatialDepthThreshold: f32,
    pub spatialNormalThreshold: f32,
    pub numSpatialSamples: u32,
    pub spatialSamplingRadius: f32,

    pub spatialBiasCorrectionMode: u32, // = ResTIRGI_SpatialBiasCorrectionMode::Basic;
    pub pad1: u32,
    pub pad2: u32,
    pub pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReSTIRGI_FinalShadingParameters {
    pub enableFinalVisibility: u32, // = true;
    pub enableFinalMIS: u32,        // = true;
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReSTIRGI_BufferIndices {
    pub secondarySurfaceReSTIRDIOutputBufferIndex: u32,
    pub temporalResamplingInputBufferIndex: u32,
    pub temporalResamplingOutputBufferIndex: u32,
    pub spatialResamplingInputBufferIndex: u32,

    pub spatialResamplingOutputBufferIndex: u32,
    pub finalShadingInputBufferIndex: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RTXDI_ReservoirBufferParameters {
    pub reservoirBlockRowPitch: u32,
    pub reservoirArrayPitch: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReSTIRGI_Parameters {
    pub reservoirBufferParams: RTXDI_ReservoirBufferParameters,
    pub bufferIndices: ReSTIRGI_BufferIndices,
    pub temporalResamplingParams: ReSTIRGI_TemporalResamplingParameters,
    pub spatialResamplingParams: ReSTIRGI_SpatialResamplingParameters,
    pub finalShadingParams: ReSTIRGI_FinalShadingParameters,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GConst {
    pub view: PlanarViewConstants,
    pub prevView: PlanarViewConstants,
    pub runtimeParams: RTXDI_RuntimeParameters,

    pub restirGI: ReSTIRGI_Parameters,
}
