
#ifndef RTXDI_APPLICATION_BRIDGE_HLSLI
#define RTXDI_APPLICATION_BRIDGE_HLSLI

#include "ShaderParameters.glsl"
#include "GBufferHelpers.glsl"
#include "Helpers.glsl"
#include "packing.glsl"
#include "rtxdi/ReSTIRGIParameters.h"


layout(binding = 0, set = 0) uniform accelerationStructureEXT SceneBVH;
layout(binding = 1, set = 0) uniform Uniform {ResamplingConstants g_Const;};
layout(binding = 2, set = 0) buffer NeighborsBuffer {vec2 neighbors[];};
layout(binding = 3, set = 0) buffer TemporalReservoirBuffer {RTXDI_PackedGIReservoir reservoirs[];};

layout(binding = 1, set = 2, r32f) uniform readonly image2D t_PrevGBufferDepth;
layout(binding = 2, set = 2, r32ui) uniform readonly uimage2D t_PrevGBufferNormals;
layout(binding = 3, set = 2, r32ui) uniform readonly uimage2D t_PrevGBufferGeoNormals;
layout(binding = 4, set = 2, r32ui) uniform readonly uimage2D t_PrevGBufferDiffuseAlbedo;
layout(binding = 5, set = 2, r32ui) uniform readonly uimage2D t_PrevGBufferSpecularRough;


#define RTXDI_GI_RESERVOIR_BUFFER reservoirs
#include "rtxdi/GIReservoir.hlsli"

RTXDI_PackedDIReservoir light_reservoirs[1];
RAB_LightInfo t_LightDataBuffer[1];

#define RTXDI_NEIGHBOR_OFFSETS_BUFFER neighbors
#define RTXDI_LIGHT_RESERVOIR_BUFFER light_reservoirs


const float kMinRoughness = 0.05f;


// A surface with enough information to evaluate BRDFs
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

float Schlick_Fresnel(float F0, float VdotH)
{
    return F0 + (1 - F0) * pow(max(1 - VdotH, 0), 5);
}

vec3 Schlick_Fresnel(vec3 F0, float VdotH)
{
    return F0 + (1 - F0) * pow(max(1 - VdotH, 0), 5);
}

float getSurfaceDiffuseProbability(RAB_Surface surface)
{
    float diffuseWeight = calcLuminance(surface.diffuseAlbedo);
    float specularWeight = calcLuminance(Schlick_Fresnel(surface.specularF0, dot(surface.viewDir, surface.normal)));
    float sumWeights = diffuseWeight + specularWeight;
    return sumWeights < 1e-7f ? 1.f : (diffuseWeight / sumWeights);
}

RAB_Surface RAB_EmptySurface()
{
    RAB_Surface surface;
    surface.viewDepth = BACKGROUND_DEPTH;
    return surface;
}


struct RAB_LightSample
{
    vec3 position;
    vec3 normal;
    vec3 radiance;
    float solidAnglePdf;
};

RAB_LightInfo RAB_EmptyLightInfo()
{
    return RAB_LightInfo(vec3(0), 0, uvec2(0), 0 ,0);
}

RAB_LightSample RAB_EmptyLightSample()
{
    return RAB_LightSample(vec3(0), vec3(0) ,vec3(0), 0.0);
}

void RAB_GetLightDirDistance(RAB_Surface surface, RAB_LightSample lightSample,
    out vec3 o_lightDir,
    out float o_lightDistance)
{
    vec3 toLight = lightSample.position - surface.worldPos;
    o_lightDistance = length(toLight);
    o_lightDir = toLight / o_lightDistance;
}

bool RAB_IsAnalyticLightSample(RAB_LightSample lightSample)
{
    return false;
}

float RAB_LightSampleSolidAnglePdf(RAB_LightSample lightSample)
{
    return lightSample.solidAnglePdf;
}


vec2 RAB_GetEnvironmentMapRandXYFromDir(vec3 worldDir)
{
    return vec2(0);
}

float RAB_EvaluateEnvironmentMapSamplingPdf(vec3 L)
{
    // No Environment sampling
    return 0;
}

float RAB_EvaluateLocalLightSourcePdf(uint lightIndex)
{
    // Uniform pdf
    return 1.0 / g_Const.lightBufferParams.localLightBufferRegion.numLights;
}

RayDesc setupVisibilityRay(RAB_Surface surface, RAB_LightSample lightSample)
{
    vec3 L = lightSample.position - surface.worldPos;

    RayDesc ray;
    ray.TMin = 0.0001;
    ray.TMax = length(L) - 0.0001;
    ray.Direction = normalize(L);
    ray.Origin = surface.worldPos;

    return ray;
}

void trace(RayDesc ray) {
    traceRayEXT(SceneBVH, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, ray.Origin, ray.TMin, ray.Direction, ray.TMax, 0);
}

// Tests the visibility between a surface and a light sample.
// Returns true if there is nothing between them.
bool RAB_GetConservativeVisibility(RAB_Surface surface, RAB_LightSample lightSample)
{
    RayDesc ray = setupVisibilityRay(surface, lightSample);

    trace(ray);

    bool visible = p.missed;
    
    return visible;
}

bool RAB_GetConservativeVisibility(RAB_Surface surface, vec3 pos)
{
    RayDesc ray;
    ray.TMin = 0.0001;
    ray.TMax = 1000000.0;
    ray.Direction = normalize(pos - surface.worldPos);
    ray.Origin = surface.worldPos;

    trace(ray);

    bool visible = p.missed;
    
    return visible;
}

// Tests the visibility between a surface and a light sample on the previous frame.
// Since the scene is static in this sample app, it's equivalent to RAB_GetConservativeVisibility.
bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface, RAB_Surface previousSurface,
    RAB_LightSample lightSample)
{
    return RAB_GetConservativeVisibility(currentSurface, lightSample);
}

// This function is called in the spatial resampling passes to make sure that 
// the samples actually land on the screen and not outside of its boundaries.
// It can clamp the position or reflect it about the nearest screen edge.
// The simplest implementation will just return the input pixelPosition.
ivec2 RAB_ClampSamplePositionIntoView(ivec2 pixelPosition, bool previousFrame)
{
    return clamp(pixelPosition, ivec2(0), ivec2(g_Const.view.viewportSize) - 1);
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

    surface.viewDepth = imageLoad(t_PrevGBufferDepth, pixelPosition).r;

    if(surface.viewDepth == BACKGROUND_DEPTH)
        return surface;

    surface.normal = octToNdirUnorm32(imageLoad(t_PrevGBufferNormals, pixelPosition).r);
    surface.geoNormal = octToNdirUnorm32(imageLoad(t_PrevGBufferGeoNormals, pixelPosition).r);
    surface.diffuseAlbedo = Unpack_R11G11B10_UFLOAT(imageLoad(t_PrevGBufferDiffuseAlbedo, pixelPosition).r).rgb;
    vec4 specularRough = Unpack_R8G8B8A8_Gamma_UFLOAT(imageLoad(t_PrevGBufferSpecularRough, pixelPosition).r);
    surface.specularF0 = specularRough.rgb;
    surface.roughness = specularRough.a;
    surface.worldPos = viewDepthToWorldPos(view, pixelPosition, surface.viewDepth);
    surface.viewDir = normalize(g_Const.view.cameraDirectionOrPosition.xyz - surface.worldPos);
    surface.diffuseProbability = getSurfaceDiffuseProbability(surface);

    return surface;
}

bool RAB_IsSurfaceValid(RAB_Surface surface)
{
    return surface.viewDepth != BACKGROUND_DEPTH;
}

vec3 RAB_GetSurfaceWorldPos(RAB_Surface surface)
{
    return surface.worldPos;
}

// World space from surface to eye
vec3 RAB_GetSurfaceViewDir(RAB_Surface surface)
{
    return surface.viewDir;
}

vec3 RAB_GetSurfaceNormal(RAB_Surface surface)
{
    return surface.normal;
}

float RAB_GetSurfaceLinearDepth(RAB_Surface surface)
{
    return surface.viewDepth;
}

void ConstructONB(vec3 normal, out vec3 tangent, out vec3 bitangent)
{
    float sign = (normal.z >= 0) ? 1 : -1;
    float a = -1.0 / (sign + normal.z);
    float b = normal.x * normal.y * a;
    tangent = vec3(1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    bitangent = vec3(b, sign + normal.y * normal.y * a, -normal.y);
}

vec3 worldToTangent(RAB_Surface surface, vec3 w)
{
    // reconstruct tangent frame based off worldspace normal
    // this is ok for isotropic BRDFs
    // for anisotropic BRDFs, we need a user defined tangent
    vec3 tangent;
    vec3 bitangent;
    ConstructONB(surface.normal, tangent, bitangent);

    return vec3(dot(bitangent, w), dot(tangent, w), dot(surface.normal, w));
}

vec3 tangentToWorld(RAB_Surface surface, vec3 h)
{
    // reconstruct tangent frame based off worldspace normal
    // this is ok for isotropic BRDFs
    // for anisotropic BRDFs, we need a user defined tangent
    vec3 tangent;
    vec3 bitangent;
    ConstructONB(surface.normal, tangent, bitangent);

    return bitangent * h.x + tangent * h.y + surface.normal * h.z;
}

RAB_RandomSamplerState RAB_InitRandomSampler(uvec2 index, uint pass)
{
    return initRandomSampler(index, g_Const.frameIndex + pass * 13);
}

float RAB_GetNextRandom(inout RAB_RandomSamplerState rng)
{
    return sampleUniformRng(rng);
}


vec2 SampleDisk(vec2 random)
{
    float angle = 2 * RTXDI_PI * random.x;
    return vec2(cos(angle), sin(angle)) * sqrt(random.y);
}


vec3 SampleCosHemisphere(vec2 random, out float solidAnglePdf)
{
    vec2 tangential = SampleDisk(random);
    float elevation = sqrt(saturate(1.0 - random.y));

    solidAnglePdf = elevation / RTXDI_PI;

    return vec3(tangential.xy, elevation);
}

vec3 ImportanceSampleGGX(vec2 random, float roughness)
{
    float alpha = (roughness * roughness);

    float phi = 2 * RTXDI_PI * random.x;
    float cosTheta = sqrt((1 - random.y) / (1 + ((alpha * alpha) - 1) * random.y));
    float sinTheta = sqrt(1 - cosTheta * cosTheta);

    vec3 H;
    H.x = sinTheta * cos(phi);
    H.y = sinTheta * sin(phi);
    H.z = cosTheta;

    return H;
}

// Output an importanced sampled reflection direction from the BRDF given the view
// Return true if the returned direction is above the surface
bool RAB_GetSurfaceBrdfSample(RAB_Surface surface, inout RAB_RandomSamplerState rng, out vec3 dir)
{
    vec3 rand;
    rand.x = RAB_GetNextRandom(rng);
    rand.y = RAB_GetNextRandom(rng);
    rand.z = RAB_GetNextRandom(rng);
    if (rand.x < surface.diffuseProbability)
    {
        float pdf;
        vec3 h = SampleCosHemisphere(rand.yz, pdf);
        dir = tangentToWorld(surface, h);
    }
    else
    {
        vec3 h = ImportanceSampleGGX(rand.yz, max(surface.roughness, kMinRoughness));
        dir = reflect(-surface.viewDir, tangentToWorld(surface, h));
    }

    return dot(surface.normal, dir) > 0.f;
}

#define square(a) a * a

float ImportanceSampleGGX_VNDF_PDF(float roughness, vec3 N, vec3 V, vec3 L)
{
    vec3 H = normalize(L + V);
    float NoH = saturate(dot(N, H));
    float VoH = saturate(dot(V, H));

    float alpha = square(roughness);
    float D = square(alpha) / (RTXDI_PI * square(square(NoH) * square(alpha) + (1 - square(NoH))));
    return (VoH > 0.0) ? D / (4.0 * VoH) : 0.0;
}

// Return PDF wrt solid angle for the BRDF in the given dir
float RAB_GetSurfaceBrdfPdf(RAB_Surface surface, vec3 dir)
{
    float cosTheta = saturate(dot(surface.normal, dir));
    float diffusePdf = cosTheta / RTXDI_PI;
    float specularPdf = ImportanceSampleGGX_VNDF_PDF(max(surface.roughness, kMinRoughness), surface.normal, surface.viewDir, dir);
    float pdf = cosTheta > 0.f ? mix(specularPdf, diffusePdf, surface.diffuseProbability) : 0.f;
    return pdf;
}

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

// Evaluate the surface BRDF and compute the weighted reflected radiance for the given light sample
vec3 ShadeSurfaceWithLightSample(RAB_LightSample lightSample, RAB_Surface surface)
{
    // Ignore invalid light samples
    if (lightSample.solidAnglePdf <= 0)
        return vec3(0);

    vec3 L = normalize(lightSample.position - surface.worldPos);

    // Ignore light samples that are below the geometric surface (but above the normal mapped surface)
    if (dot(L, surface.geoNormal) <= 0)
        return vec3(0);


    vec3 V = surface.viewDir;
    
    // Evaluate the BRDF
    float diffuse = Lambert(surface.normal, -L);
    vec3 specular = GGX_times_NdotL(V, L, surface.normal, max(surface.roughness, kMinRoughness), surface.specularF0);

    vec3 reflectedRadiance = lightSample.radiance * (diffuse * surface.diffuseAlbedo + specular);

    return reflectedRadiance / lightSample.solidAnglePdf;
}

// Compute the target PDF (p-hat) for the given light sample relative to a surface
float RAB_GetLightSampleTargetPdfForSurface(RAB_LightSample lightSample, RAB_Surface surface)
{
    // Second-best implementation: the PDF is proportional to the reflected radiance.
    // The best implementation would be taking visibility into account,
    // but that would be prohibitively expensive.
    return calcLuminance(ShadeSurfaceWithLightSample(lightSample, surface));
}

// Compute the position on a triangle light given a pair of random numbers
RAB_LightSample RAB_SamplePolymorphicLight(RAB_LightInfo lightInfo, RAB_Surface surface, vec2 uv)
{
    return RAB_EmptyLightSample();
}

// Load the packed light information from the buffer.
// Ignore the previousFrame parameter as our lights are static in this sample.
RAB_LightInfo RAB_LoadLightInfo(uint index, bool previousFrame)
{
    return t_LightDataBuffer[index];
}

// Translate the light index between the current and previous frame.
// Do nothing as our lights are static in this sample.
int RAB_TranslateLightIndex(uint lightIndex, bool currentToPrevious)
{
    return int(lightIndex);
}

// Compare the materials of two surfaces to improve resampling quality.
// Just say that everything is similar for simplicity.
bool RAB_AreMaterialsSimilar(RAB_Surface a, RAB_Surface b)
{
    return true;
}

// uint getLightIndex(uint instanceID, uint geometryIndex, uint primitiveIndex)
// {
//     uint lightIndex = RTXDI_InvalidLightIndex;
//     InstanceData hitInstance = t_InstanceData[instanceID];
//     uint geometryInstanceIndex = hitInstance.firstGeometryInstanceIndex + geometryIndex;
//     lightIndex = t_GeometryInstanceToLight[geometryInstanceIndex];
//     if (lightIndex != RTXDI_InvalidLightIndex)
//       lightIndex += primitiveIndex;
//     return lightIndex;
// }


// Return true if anything was hit. If false, RTXDI will do environment map sampling
// o_lightIndex: If hit, must be a valid light index for RAB_LoadLightInfo, if no local light was hit, must be RTXDI_InvalidLightIndex
// randXY: The randXY that corresponds to the hit location and is the same used for RAB_SamplePolymorphicLight
bool RAB_TraceRayForLocalLight(vec3 origin, vec3 direction, float tMin, float tMax,
    out uint o_lightIndex, out vec2 o_randXY)
{
    o_lightIndex = RTXDI_InvalidLightIndex;
    o_randXY = vec2(0);
    o_randXY = vec2(0,0);

    return false;
}

bool RAB_ValidateGISampleWithJacobian(inout float jacobian)
{
    // Sold angle ratio is too different. Discard the sample.
    if (jacobian > 10.0 || jacobian < 1 / 10.0) {
        return false;
    }

    // clamp Jacobian.
    jacobian = clamp(jacobian, 1 / 3.0, 3.0);

    return true;
}


struct SplitBrdf
{
    float demodulatedDiffuse;
    vec3 specular;
};

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


// Computes the weight of the given GI sample when the given surface is shaded using that GI sample.
float RAB_GetGISampleTargetPdfForSurface(vec3 samplePosition, vec3 sampleRadiance, RAB_Surface surface)
{
    SplitBrdf brdf = EvaluateBrdf(surface, samplePosition);

    vec3 reflectedRadiance = sampleRadiance * (brdf.demodulatedDiffuse * surface.diffuseAlbedo + brdf.specular);

    return RTXDI_Luminance(reflectedRadiance);
}

bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface, RAB_Surface previousSurface, vec3 samplePosition)
{
    return RAB_GetConservativeVisibility(currentSurface, samplePosition);
}

#endif // RTXDI_APPLICATION_BRIDGE_HLSLI