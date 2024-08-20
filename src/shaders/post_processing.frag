layout(location = 0) out vec4 outColor;
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#define RTXDI_GLSL
#include "rtxdi/ReSTIRGIParameters.h"
#include "./ShaderParameters.glsl"
#include "packing.glsl"
#define square(x) x*x
layout (binding = 0, set = 0) buffer TemporalReservoirBuffer {RTXDI_PackedGIReservoir reservoirs[];};
#define RTXDI_GI_RESERVOIR_BUFFER reservoirs
#include "rtxdi/GIReservoir.hlsli"

layout (binding = 1, set = 0) uniform Uniform {ResamplingConstants g_Const;};
layout (binding = 2, set = 0) uniform sampler2D skyBox;

layout(binding = 0, set = 1, r32f) uniform readonly image2D u_GBufferDepth;
layout(binding = 1, set = 1, r32ui) uniform readonly uimage2D u_GBufferNormals;
layout(binding = 2, set = 1, r32ui) uniform readonly uimage2D u_GBufferGeoNormals;
layout(binding = 3, set = 1, r32ui) uniform readonly uimage2D u_GBufferDiffuseAlbedo;
layout(binding = 4, set = 1, r32ui) uniform readonly uimage2D u_GBufferSpecularRough;
layout(binding = 5, set = 1, rgba32f) uniform readonly image2D u_MotionVectors;
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


vec3 DemodulateSpecular(vec3 surfaceSpecularF0, vec3 specular)
{
    return specular / max(vec3(0.01), surfaceSpecularF0);
}

struct RAB_Surface
{
    vec3 worldPos;
    vec3 viewDir;
    float viewDepth;
    vec3 normal;
    vec3 geoNormal;
    vec3 diffuseAlbedo;
    vec3 specularF0;
    float roughness;
    float diffuseProbability;
};

RAB_Surface RAB_EmptySurface()
{
    RAB_Surface surface;
    surface.viewDepth = BACKGROUND_DEPTH;
    return surface;
}


vec3 octToNdirSigned(vec2 p)
{
    // https://twitter.com/Stubbesaurus/status/937994790553227264
    vec3 n = vec3(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    float t = max(0, -n.z);

    n.xy += (n.x >= 0.0 || n.y >= 0.0) ? -t : t;
    return normalize(n);
}

vec3 octToNdirUnorm32(uint pUnorm)
{
    vec2 p;
    p.x = clamp(float(pUnorm & 0xffff) / float(0xfffel), 0, 1);
    p.y = clamp(float(pUnorm >> 16) / float(0xfffel), 0, 1);
    p = p * 2.0 - 1.0;
    return octToNdirSigned(p);
}

vec3 viewDepthToWorldPos(
    PlanarViewConstants view,
    ivec2 pixelPosition,
    float viewDepth)
{
    vec2 uv = (vec2(pixelPosition) + 0.5) * view.viewportSizeInv;
    vec4 clipPos = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1);
    vec4 viewPos = clipPos * view.matClipToView;
    viewPos.xy /= viewPos.z;
    viewPos.zw = vec2(1.0);
    viewPos.xyz *= viewDepth;
    return (viewPos * view.matViewToWorld).xyz;
}

float Schlick_Fresnel(float F0, float VdotH)
{
    return F0 + (1 - F0) * pow(max(1 - VdotH, 0), 5);
}

vec3 Schlick_Fresnel(vec3 F0, float VdotH)
{
    return F0 + (1 - F0) * pow(max(1 - VdotH, 0), 5);
}

float calcLuminance(vec3 color)
{
    return dot(color.xyz, vec3(0.299f, 0.587f, 0.114f));
}

float getSurfaceDiffuseProbability(RAB_Surface surface)
{
    float diffuseWeight = calcLuminance(surface.diffuseAlbedo);
    float specularWeight = calcLuminance(Schlick_Fresnel(surface.specularF0, dot(surface.viewDir, surface.normal)));
    float sumWeights = diffuseWeight + specularWeight;
    return sumWeights < 1e-7f ? 1.f : (diffuseWeight / sumWeights);
}

// Load a sample from the previous G-buffer.
RAB_Surface RAB_GetGBufferSurface(ivec2 pixelPosition, bool previousFrame)
{
    RAB_Surface surface = RAB_EmptySurface();

    // We do not have access to the current G-buffer in this sample because it's using
    // a single render pass with a fused resampling kernel, so just return an invalid surface.
    // This should never happen though, as the fused kernel doesn't call RAB_GetGBufferSurface(..., false)
    if (!previousFrame)
        return surface;

    const PlanarViewConstants view = g_Const.prevView;

    if (pixelPosition.x >= view.viewportSize.x || pixelPosition.y >= view.viewportSize.y)
        return surface;

    surface.viewDepth = imageLoad(u_GBufferDepth, pixelPosition).r;

    if(surface.viewDepth == BACKGROUND_DEPTH)
        return surface;

    surface.normal = octToNdirUnorm32(imageLoad(u_GBufferNormals, pixelPosition).r);
    surface.geoNormal = octToNdirUnorm32(imageLoad(u_GBufferGeoNormals, pixelPosition).r);
    surface.diffuseAlbedo = Unpack_R11G11B10_UFLOAT(imageLoad(u_GBufferDiffuseAlbedo, pixelPosition).r).rgb;
    vec4 specularRough = Unpack_R8G8B8A8_Gamma_UFLOAT(imageLoad(u_GBufferSpecularRough, pixelPosition).r);
    surface.specularF0 = specularRough.rgb;
    surface.roughness = specularRough.a;
    surface.worldPos = viewDepthToWorldPos(view, pixelPosition, surface.viewDepth);
    surface.viewDir = normalize(g_Const.view.cameraDirectionOrPosition.xyz - surface.worldPos);
    surface.diffuseProbability = getSurfaceDiffuseProbability(surface);

    return surface;
}

struct SplitBrdf
{
    float demodulatedDiffuse;
    vec3 specular;
};


float Lambert(vec3 normal, vec3 lightIncident)
{
    return max(0, -dot(normal, lightIncident)) / RTXDI_PI;
}

float G_Smith_over_NdotV(float roughness, float NdotV, float NdotL)
{
    float alpha = square(roughness);
    float g1 = NdotV * sqrt(square(alpha) + (1.0 - square(alpha)) * square(NdotL));
    float g2 = NdotL * sqrt(square(alpha) + (1.0 - square(alpha)) * square(NdotV));
    return 2.0 * NdotL / (g1 + g2);
}

vec3 GGX_times_NdotL(vec3 V, vec3 L, vec3 N, float roughness, vec3 F0)
{
    vec3 H = normalize(L + V);

    float NoL = saturate(dot(N, L));
    float VoH = saturate(dot(V, H));
    float NoV = saturate(dot(N, V));
    float NoH = saturate(dot(N, H));

    if (NoL > 0)
    {
        float G = G_Smith_over_NdotV(roughness, NoV, NoL);
        float alpha = square(roughness);
        float D = square(alpha) / (RTXDI_PI * square(square(NoH) * square(alpha) + (1 - square(NoH))));

        vec3 F = Schlick_Fresnel(F0, VoH);

        return F * (D * G / 4);
    }
    return vec3(0);
}
const float kMinRoughness = 0.05f;


SplitBrdf EvaluateBrdf(RAB_Surface surface, vec3 samplePosition)
{
    vec3 N = surface.normal;
    vec3 V = surface.viewDir;
    vec3 L = normalize(samplePosition - surface.worldPos);

    SplitBrdf brdf;
    brdf.demodulatedDiffuse = Lambert(surface.normal, -L);
    if (surface.roughness == 0)
        brdf.specular = vec3(0);
    else
        brdf.specular = GGX_times_NdotL(V, L, surface.normal, max(surface.roughness, kMinRoughness), surface.specularF0);
    return brdf;
}


void main() {
    vec3 col = vec3(0.0);
    ivec2 pixelPosition = ivec2(gl_FragCoord.xy);
    const RAB_Surface primarySurface = RAB_GetGBufferSurface(pixelPosition, false);
    
    const uvec2 reservoirPosition = RTXDI_PixelPosToReservoirPos(pixelPosition, g_Const.runtimeParams.activeCheckerboardField);
    const RTXDI_GIReservoir reservoir = RTXDI_LoadGIReservoir(g_Const.restirGI.reservoirBufferParams, reservoirPosition, g_Const.restirGI.bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex);

    if (RTXDI_IsValidGIReservoir(reservoir))
    {
        vec3 radiance = reservoir.radiance * reservoir.weightSum;

        const SplitBrdf brdf = EvaluateBrdf(primarySurface, reservoir.position);

        vec3 diffuse = brdf.demodulatedDiffuse * radiance;
        vec3 specular = DemodulateSpecular(brdf.specular * radiance, primarySurface.specularF0);

        diffuse.rgb *= primarySurface.diffuseAlbedo;
        specular.rgb *= max(vec3(0.01), primarySurface.specularF0);

        col = diffuse + specular;
    } else {
      vec2 uv = (vec2(pixelPosition) + 0.5) * g_Const.view.viewportSizeInv;
      vec4 clipPos = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, (1.0 / 256.0), 1);
      vec4 worldPos = clipPos * g_Const.view.matClipToWorld;
      worldPos.xyz /= worldPos.w;
      vec3 dir = normalize(worldPos.xyz - g_Const.view.cameraDirectionOrPosition.xyz);

      float u = (0.5 + atan2(dir.z, dir.x)/(2*PI));
      float v = (0.5 - asin(dir.y)/PI);
      col = texture(skyBox, vec2(u,v)).rgb;
    }

    col = agx(col);
    col = agxLook(col);
    col = agxEotf(col);
    vec3 gamma_cor = pow(col, vec3(1.0 / gamma));
    outColor = vec4(col, 1.0);
}