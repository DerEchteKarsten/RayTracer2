layout(location = 0) out vec4 outColor;
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_debug_printf: enable
#define RTXDI_GLSL
#define RTXDI_ENABLE_PRESAMPLING 0

#include "rtxdi/ReSTIRGIParameters.h"
#include "./ShaderParameters.glsl"

layout(binding = 0, set = 0) buffer TemporalReservoirBuffer {RTXDI_PackedGIReservoir reservoirs[];};
layout(binding = 1, set = 0) uniform Uniform {ResamplingConstants g_Const;};

layout(binding = 0, set = 1, r32f) uniform  image2D u_GBufferDepth;
layout(binding = 1, set = 1, r32ui) uniform  uimage2D u_GBufferNormals;
layout(binding = 2, set = 1, r32ui) uniform  uimage2D u_GBufferGeoNormals;
layout(binding = 3, set = 1, r32ui) uniform  uimage2D u_GBufferDiffuseAlbedo;
layout(binding = 4, set = 1, r32ui) uniform  uimage2D u_GBufferSpecularRough;
layout(binding = 5, set = 1, rgba32f) uniform  image2D u_MotionVectors;
layout(binding = 2, set = 0) uniform sampler2D skyBox;

layout( push_constant ) uniform Frame {
	uint frame;
} f;

#define square(x) x*x
const float kMinRoughness = 0.05f;
#define FINAL_SHADING
#include "RtxdiApplicationBridge.glsl"


#define RTXDI_GI_RESERVOIR_BUFFER reservoirs
#include "rtxdi/GIReservoir.hlsli"


#define PI 3.1415926

#define AGX_LOOK 0

// Mean error^2: 3.6705141e-06
vec3 agxDefaultContrastApprox(vec3 x) {
  vec3 x2 = x * x;
  vec3 x4 = x2 * x2;
  
  return + 15.5     * x4 * x2
         - 40.14    * x4 * x
         + 31.96    * x4
         - 6.868    * x2 * x
         + 0.4298   * x2
         + 0.1191   * x
         - 0.00232;
}

vec3 agx(vec3 val) {
  const mat3 agx_mat = mat3(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992,  0.878468636469772,  0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    
  const float min_ev = -12.47393f;
  const float max_ev = 4.026069f;

  // Input transform (inset)
  val = agx_mat * val;
  
  // Log2 space encoding
  val = clamp(log2(val), min_ev, max_ev);
  val = (val - min_ev) / (max_ev - min_ev);
  
  // Apply sigmoid function approximation
  val = agxDefaultContrastApprox(val);

  return val;
}

vec3 agxEotf(vec3 val) {
  const mat3 agx_mat_inv = mat3(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    
  // Inverse input transform (outset)
  val = agx_mat_inv * val;
  
  // sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
  // NOTE: We're linearizing the output here. Comment/adjust when
  // *not* using a sRGB render target
  val = pow(val, vec3(2.2));

  return val;
}

vec3 agxLook(vec3 val) {
  const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
  float luma = dot(val, lw);
  
  // Default
  vec3 offset = vec3(0.0);
  vec3 slope = vec3(1.0);
  vec3 power = vec3(1.0);
  float sat = 1.0;
 
#if AGX_LOOK == 1
  // Golden
  slope = vec3(1.0, 0.9, 0.5);
  power = vec3(0.8);
  sat = 0.8;
#elif AGX_LOOK == 2
  // Punchy
  slope = vec3(1.0);
  power = vec3(1.35, 1.35, 1.35);
  sat = 1.4;
#endif
  
  // ASC CDL
  val = pow(val * slope + offset, power);
  return luma + sat * (val - luma);
}

const float gamma = 2.2;

const float kMaxBrdfValue = 1e4;
const float kMISRoughness = 0.3;

float GetMISWeight(const SplitBrdf roughBrdf, const SplitBrdf trueBrdf, const float3 diffuseAlbedo)
{
    float3 combinedRoughBrdf = roughBrdf.demodulatedDiffuse * diffuseAlbedo + roughBrdf.specular;
    float3 combinedTrueBrdf = trueBrdf.demodulatedDiffuse * diffuseAlbedo + trueBrdf.specular;

    combinedRoughBrdf = clamp(combinedRoughBrdf, 1e-4, kMaxBrdfValue);
    combinedTrueBrdf = clamp(combinedTrueBrdf, 0, kMaxBrdfValue);

    const float initWeight = saturate(calcLuminance(combinedTrueBrdf) / calcLuminance(combinedTrueBrdf + combinedRoughBrdf));
    return initWeight * initWeight * initWeight;
}

void main() {
    vec3 col = vec3(0.0);
    ivec2 pixelPosition = ivec2(gl_FragCoord.xy);
    const RAB_Surface primarySurface = GetGBufferSurface(pixelPosition, g_Const.view);
    const uvec2 reservoirPosition = RTXDI_PixelPosToReservoirPos(pixelPosition, g_Const.runtimeParams.activeCheckerboardField);
    const RTXDI_GIReservoir reservoir = RTXDI_LoadGIReservoir(g_Const.restirGI.reservoirBufferParams, reservoirPosition, g_Const.restirGI.bufferIndices.finalShadingInputBufferIndex);
    // RTXDI_StoreGIReservoir(reservoir, g_Const.restirGI.reservoirBufferParams, pixelPosition, g_Const.restirGI.bufferIndices.temporalResamplingInputBufferIndex);

    if (RTXDI_IsValidGIReservoir(reservoir))
    {
        vec3 radiance = reservoir.radiance * reservoir.weightSum;
        
        const SplitBrdf brdf = EvaluateBrdf(primarySurface, reservoir.position);
		vec3 diffuse, specular;
        if (g_Const.restirGI.finalShadingParams.enableFinalMIS == 1)
        {
            const RTXDI_GIReservoir initialReservoir = RTXDI_LoadGIReservoir(g_Const.restirGI.reservoirBufferParams, reservoirPosition, g_Const.restirGI.bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex);//LoadInitialSampleReservoir(reservoirPosition, primarySurface);
            const SplitBrdf brdf0 = EvaluateBrdf(primarySurface, initialReservoir.position);

            RAB_Surface roughenedSurface = primarySurface;
            roughenedSurface.roughness = max(roughenedSurface.roughness, kMISRoughness);

            const SplitBrdf roughBrdf = EvaluateBrdf(roughenedSurface, reservoir.position);
            const SplitBrdf roughBrdf0 = EvaluateBrdf(roughenedSurface, initialReservoir.position);

            const float finalWeight = 1.0 - GetMISWeight(roughBrdf, brdf, primarySurface.diffuseAlbedo);
            const float initialWeight = GetMISWeight(roughBrdf0, brdf0, primarySurface.diffuseAlbedo);

            const float3 initialRadiance = initialReservoir.radiance * initialReservoir.weightSum;

            diffuse = brdf.demodulatedDiffuse * radiance * finalWeight 
                    + brdf0.demodulatedDiffuse * initialRadiance * initialWeight;

            specular = brdf.specular * radiance * finalWeight 
                     + brdf0.specular * initialRadiance * initialWeight;
        }else {
            diffuse = brdf.demodulatedDiffuse * radiance;
			      specular = brdf.specular * radiance;
        }

        specular = DemodulateSpecular(brdf.specular * radiance, primarySurface.specularF0);

        diffuse.rgb *= primarySurface.diffuseAlbedo;
        specular.rgb *= max(vec3(0.01), primarySurface.specularF0);

        col = diffuse + specular;
    } else {
        const vec2 pixelCenter = vec2(pixelPosition.xy) + vec2(0.5);
        const vec2 inUV = pixelCenter/vec2(g_Const.view.viewportSize);
        vec2 d = inUV * 2.0 - 1.0;
        vec2 dir = inUV * 2.0 - 1.0;
        vec4 target = g_Const.view.matClipToView * vec4(dir.x, dir.y, 1, 1) ;
        vec4 direction = g_Const.view.matViewToWorld*vec4(normalize(target.xyz), 0) ;

        float u = (0.5 + atan2(direction.z, direction.x)/(2*PI));
        float v = (0.5 - asin(direction.y)/PI);
		    col = texture(skyBox, vec2(u,v)).rgb;
    }

    col = agx(col);
    col = agxLook(col);
    col = agxEotf(col);
    vec3 gamma_cor = pow(col, vec3(1.0 / gamma));
    outColor = vec4(col, 1.0);
}