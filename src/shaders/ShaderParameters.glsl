#ifndef SHADER_PARAMETERS_H
#define SHADER_PARAMETERS_H

#include "rtxdi/ReSTIRDIParameters.h"
#include "rtxdi/ReSTIRGIParameters.h"

#define RTXDI_GRID_BUILD_GROUP_SIZE 256
#define RTXDI_SCREEN_SPACE_GROUP_SIZE 8

#define INSTANCE_MASK_OPAQUE 0x01
#define INSTANCE_MASK_ALPHA_TESTED 0x02
#define INSTANCE_MASK_TRANSPARENT 0x04
#define INSTANCE_MASK_ALL 0xFF

#define BACKGROUND_DEPTH 65504.f

struct PlanarViewConstants
{
    mat4    matWorldToView;
    mat4    matViewToClip;
    mat4    matWorldToClip;
    mat4    matClipToView;
    mat4    matViewToWorld;
    mat4    matClipToWorld;

    vec2      viewportOrigin;
    vec2      viewportSize;

    vec2      viewportSizeInv;
    vec2      pixelOffset;

    vec2      clipToWindowScale;
    vec2      clipToWindowBias;

    vec2      windowToClipScale;
    vec2      windowToClipBias;

    vec4      cameraDirectionOrPosition;
};

struct BrdfRayTracingConstants
{
    PlanarViewConstants view;

    uint frameIndex;
};

struct PrepareLightsConstants
{
    uint numTasks;
    uint currentFrameLightOffset;
    uint previousFrameLightOffset;
};

struct PrepareLightsTask
{
    uint instanceAndGeometryIndex; // low 12 bits are geometryIndex, mid 19 bits are instanceIndex, high bit is TASK_PRIMITIVE_LIGHT_BIT
    uint triangleCount;
    uint lightBufferOffset;
    int previousLightBufferOffset; // -1 means no previous data
};

struct PreprocessEnvironmentMapConstants
{
    uvec2 sourceSize;
    uint sourceMipLevel;
    uint numDestMipLevels;
};

struct GBufferConstants
{
    PlanarViewConstants view;
    PlanarViewConstants viewPrev;

    float roughnessOverride;
    float metalnessOverride;
    float normalMapScale;
    uint enableAlphaTestedGeometry;

    ivec2 materialReadbackPosition;
    uint materialReadbackBufferIndex;
    uint enableTransparentGeometry;

    float textureLodBias;
    float textureGradientScale; // 2^textureLodBias
};

struct GlassConstants
{
    PlanarViewConstants view;
    
    uint enableEnvironmentMap;
    uint environmentMapTextureIndex;
    float environmentScale;
    float environmentRotation;

    ivec2 materialReadbackPosition;
    uint materialReadbackBufferIndex;
    float normalMapScale;
};

struct CompositingConstants
{
    PlanarViewConstants view;
    PlanarViewConstants viewPrev;

    uint enableTextures;
    uint denoiserMode;
    uint enableEnvironmentMap;
    uint environmentMapTextureIndex;

    float environmentScale;
    float environmentRotation;
    float noiseMix;
    float noiseClampLow;

    float noiseClampHigh;
    uint checkerboard;
};


struct FilterGradientsConstants
{
    uvec2 viewportSize;
    int passIndex;
    uint checkerboard;
};

struct ConfidenceConstants
{
    uvec2 viewportSize;
    vec2 invGradientTextureSize;

    float darknessBias;
    float sensitivity;
    uint checkerboard;
    int inputBufferIndex;

    float blendFactor;
};

struct VisualizationConstants
{
    RTXDI_RuntimeParameters runtimeParams;
    RTXDI_ReservoirBufferParameters restirDIReservoirBufferParams;
    RTXDI_ReservoirBufferParameters restirGIReservoirBufferParams;

    ivec2 outputSize;
    vec2 resolutionScale;

    uint visualizationMode;
    uint inputBufferIndex;
    uint enableAccumulation;
};

struct SceneConstants
{
    uint enableEnvironmentMap; // Global. Affects BRDFRayTracing's GI code, plus RTXDI, ReGIR, etc.
    uint environmentMapTextureIndex; // Global
    float environmentScale;
    float environmentRotation;

    uint enableAlphaTestedGeometry;
    uint enableTransparentGeometry;
    uvec2 pad1;
};

struct ResamplingConstants
{
    PlanarViewConstants view;
    PlanarViewConstants prevView;
    RTXDI_RuntimeParameters runtimeParams;
    
    vec4 reblurDiffHitDistParams;
    vec4 reblurSpecHitDistParams;

    uint frameIndex;
    uint enablePreviousTLAS;
    uint denoiserMode;
    uint discountNaiveSamples;
    
    uint enableBrdfIndirect;
    uint enableBrdfAdditiveBlend;    
    uint enableAccumulation; // StoreShadingOutput
    uint pad1;

    SceneConstants sceneConstants;

    // Common buffer params
    RTXDI_LightBufferParameters lightBufferParams;
    RTXDI_RISBufferSegmentParameters localLightsRISBufferSegmentParams;
    RTXDI_RISBufferSegmentParameters environmentLightRISBufferSegmentParams;

    // Algo-specific params
    ReSTIRDI_Parameters restirDI;
    ReGIR_Parameters regir;
    ReSTIRGI_Parameters restirGI;

    uint visualizeRegirCells;
    uvec3 pad2;
    
    uvec2 environmentPdfTextureSize;
    uvec2 localLightPdfTextureSize;
};
// See TriangleLight.hlsli for encoding format
struct RAB_LightInfo
{
    // uint4[0]
    vec3 center;
    uint scalars; // 2x float16
    
    // uint4[1]
    uvec2 radiance; // fp16x4
    uint direction1; // oct-encoded
    uint direction2; // oct-encoded
};

#endif // SHADER_PARAMETERS_H