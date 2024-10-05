#ifndef SHADER_PARAMETERS_H
#define SHADER_PARAMETERS_H

#include "rtxdi/ReSTIRDIParameters.h"
#include "rtxdi/ReSTIRGIParameters.h"

#define INSTANCE_MASK_OPAQUE 0x01
#define INSTANCE_MASK_ALPHA_TESTED 0x02
#define INSTANCE_MASK_TRANSPARENT 0x04
#define INSTANCE_MASK_ALL 0xFF

#define BACKGROUND_DEPTH 100000.f

const uint kPolymorphicLightTypeShift = 24;
const uint kPolymorphicLightTypeMask = 0xf;
const uint kPolymorphicLightShapingEnableBit = 1 << 28;
const uint kPolymorphicLightIesProfileEnableBit = 1 << 29;
const float kPolymorphicLightMinLog2Radiance = -8.f;
const float kPolymorphicLightMaxLog2Radiance = 40.f;

const uint kSecondaryGBuffer_IsSpecularRay = 1;
const uint kSecondaryGBuffer_IsDeltaSurface = 2;
const uint kSecondaryGBuffer_IsEnvironmentMap = 4;

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

struct SecondaryGBufferData
{
    float3 worldPos;
    uint normal;

    uint2 throughputAndFlags;   // .x = throughput.rg as float16, .y = throughput.b as float16, flags << 16
    uint diffuseAlbedo;         // R11G11B10_UFLOAT
    uint specularAndRoughness;  // R8G8B8A8_Gamma_UFLOAT
    
    float3 emission;
    float pdf;
};

struct ResamplingConstants
{
    PlanarViewConstants view;
    PlanarViewConstants prevView;
    RTXDI_RuntimeParameters runtimeParams;
    RTXDI_LightBufferParameters lightBufferParams;

    ReSTIRGI_Parameters restirGI;
    ReSTIRDI_Parameters restirDI;
    RTXDI_RISBufferSegmentParameters localLightsRISBufferSegmentParams;
    RTXDI_RISBufferSegmentParameters environmentLightRISBufferSegmentParams;
    uvec2 environmentPdfTextureSize;
    uvec2 localLightPdfTextureSize;
};


#endif // SHADER_PARAMETERS_H