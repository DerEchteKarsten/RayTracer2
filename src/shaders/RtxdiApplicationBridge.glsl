#ifndef RTXDI_APPLICATION_BRIDGE_HLSLI
#define RTXDI_APPLICATION_BRIDGE_HLSLI
#define RTXDI_GI_ALLOWED_BIAS_CORRECTION RTXDI_BIAS_CORRECTION_RAY_TRACED
const bool kSpecularOnly = false;

#include "ShaderParameters.glsl"
#include "GBufferHelpers.glsl"
#include "packing.glsl"
#include "rtxdi/ReSTIRGIParameters.h"

layout(binding = 0, set = 2, r32f) uniform readonly image2D t_PrevGBufferDepth;
layout(binding = 1, set = 2, r32ui) uniform readonly uimage2D t_PrevGBufferNormals;
layout(binding = 2, set = 2, r32ui) uniform readonly uimage2D t_PrevGBufferGeoNormals;
layout(binding = 3, set = 2, r32ui) uniform readonly uimage2D t_PrevGBufferDiffuseAlbedo;
layout(binding = 4, set = 2, r32ui) uniform readonly uimage2D t_PrevGBufferSpecularRough;

layout(binding = 0, set = 1, r32f) uniform  image2D u_GBufferDepth;
layout(binding = 1, set = 1, r32ui) uniform  uimage2D u_GBufferNormals;
layout(binding = 2, set = 1, r32ui) uniform  uimage2D u_GBufferGeoNormals;
layout(binding = 3, set = 1, r32ui) uniform  uimage2D u_GBufferDiffuseAlbedo;
layout(binding = 4, set = 1, r32ui) uniform  uimage2D u_GBufferSpecularRough;
layout(binding = 5, set = 1, rgba32f) uniform  image2D u_MotionVectors;

#ifndef EXCLUDE_PUSH_CONSTANT
layout( push_constant ) uniform Frame {
	uint frame;
} f;
#else
struct {
    uint frame;
} f = {0};
#endif

#if !FINAL_SHADING
    layout(binding = 8, set = 0) uniform sampler2D skyBox;
#else
    layout(binding = 2, set = 0) uniform sampler2D skyBox;
#endif

#include "PolymorphicLight.glsl"

#if !FINAL_SHADING && !PREPROSSES
    layout(binding = 0, set = 0) uniform accelerationStructureEXT SceneBVH;
    layout(binding = 1, set = 0) uniform Uniform {ResamplingConstants g_Const;};
    layout(binding = 2, set = 0) buffer NeighborsBuffer {vec2 neighbors[];};
    layout(binding = 3, set = 0) buffer ReservoirBuffer {RTXDI_PackedGIReservoir reservoirs[];};

    layout(binding = 9, set = 0) uniform sampler2D t_EnvironmentPdfTexture;
    layout(binding = 10, set = 0) uniform sampler2D t_LocalLightPdfTexture;
    layout(binding = 11, set = 0) buffer DIReservoirBuffer {RTXDI_PackedDIReservoir light_reservoirs[];};
    layout(binding = 12, set = 0) buffer LightInfoBuffer {RAB_LightInfo t_LightDataBuffer[];};
    layout(binding = 13, set = 0) buffer RisLightDataBuffer {uvec4 u_RisLightDataBuffer[];};
    layout(binding = 14, set = 0) buffer RisBuffer {uvec2 u_RisBuffer[];};
    layout(binding = 15, set = 0) buffer GeomToLight {uint t_GeometryInstanceToLight[];};
#endif

#if PREPROSSES
    RTXDI_PackedGIReservoir reservoirs[1];
    layout(binding = 0, set = 0) buffer DIReservoirBuffer {RTXDI_PackedDIReservoir light_reservoirs[];};
    layout(binding = 1, set = 0) buffer LightInfoBuffer {RAB_LightInfo t_LightDataBuffer[];};
    layout(binding = 2, set = 0) buffer RisLightDataBuffer {uvec4 u_RisLightDataBuffer[];};
    layout(binding = 3, set = 0) buffer RisBuffer {uvec2 u_RisBuffer[];};
    layout(binding = 4, set = 0) uniform sampler2D t_EnvironmentPdfTexture;
    layout(binding = 5, set = 0) uniform sampler2D t_LocalLightPdfTexture;
    layout(binding = 6, set = 0) uniform Uniform {ResamplingConstants g_Const;};
#endif

#if FINAL_SHADING
    layout(binding = 0, set = 0) buffer TemporalReservoirBuffer {RTXDI_PackedGIReservoir reservoirs[];};
    layout(binding = 1, set = 0) uniform Uniform {ResamplingConstants g_Const;};
    layout(binding = 3, set = 0) buffer DIReservoirBuffer {RTXDI_PackedDIReservoir light_reservoirs[];};
    layout(binding = 4, set = 0) buffer LightInfoBuffer {RAB_LightInfo t_LightDataBuffer[];};
    layout(binding = 5, set = 0) buffer RisLightDataBuffer {uvec4 u_RisLightDataBuffer[];};
    layout(binding = 6, set = 0) buffer RisBuffer {uvec2 u_RisBuffer[];};
#endif


#define RTXDI_RIS_BUFFER u_RisBuffer 

#define RTXDI_GI_RESERVOIR_BUFFER reservoirs
#include "rtxdi/GIReservoir.hlsli"

#define RTXDI_NEIGHBOR_OFFSETS_BUFFER neighbors
#define RTXDI_LIGHT_RESERVOIR_BUFFER light_reservoirs

#include "Helpers.glsl"

// A surface with enough information to evaluate BRDFs
void trace(RayDesc ray) {
    #if !PREPROSSES
    #if !FINAL_SHADING
    traceRayEXT(SceneBVH, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, ray.Origin, ray.TMin, ray.Direction, ray.TMax, 0);
    #endif
    #endif
}


const bool environmentMapImportanceSampling = true;

struct RAB_Surface
{
    float3 worldPos;
    float3 viewDir;
    float viewDepth;
    float3 normal;
    float3 geoNormal;
    float3 diffuseAlbedo;
    float3 specularF0;
    float roughness;
    float diffuseProbability;
};

struct RAB_LightSample
{
    float3 position;
    float3 normal;
    float3 radiance;
    float solidAnglePdf;
    PolymorphicLightType lightType;
};

#define PolymorphicLightInfo RAB_LightInfo;
#define RandomSamplerState RAB_RandomSamplerState;


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


float getSurfaceDiffuseProbability(RAB_Surface surface)
{
    float diffuseWeight = calcLuminance(surface.diffuseAlbedo);
    float specularWeight = calcLuminance(Schlick_Fresnel(surface.specularF0, dot(surface.viewDir, surface.normal)));
    float sumWeights = diffuseWeight + specularWeight;
    return sumWeights < 1e-7f ? 1.f : (diffuseWeight / sumWeights);
}

struct SplitBrdf
{
    float demodulatedDiffuse;
    float3 specular;
};

SplitBrdf EvaluateBrdf(RAB_Surface surface, float3 samplePosition)
{
    float3 N = surface.normal;
    float3 V = surface.viewDir;
    float3 L = normalize(samplePosition - surface.worldPos);

    SplitBrdf brdf;
    brdf.demodulatedDiffuse = Lambert(surface.normal, -L);
    if (surface.roughness == 0)
        brdf.specular = vec3(0);
    else
        brdf.specular = GGX_times_NdotL(V, L, surface.normal, max(surface.roughness, kMinRoughness), surface.specularF0);
    return brdf;
}

RAB_Surface RAB_EmptySurface()
{
    RAB_Surface surface;
    surface.viewDepth = BACKGROUND_DEPTH;
    return surface;
}

RAB_LightInfo RAB_EmptyLightInfo()
{
    RAB_LightInfo info;
    return info;
}

RAB_LightSample RAB_EmptyLightSample()
{
    RAB_LightSample s;
    return s;
}

struct RayPayload
{
    float3 throughput;
    float committedRayT;
    uint instanceID;
    uint geometryIndex;
    uint primitiveIndex;
    bool frontFace;
    float2 barycentrics;
};

RayDesc setupVisibilityRay(RAB_Surface surface, float3 samplePosition)
{
    float offset = 0.001;
    float3 L = samplePosition - surface.worldPos;

    RayDesc ray;
    ray.TMin = offset;
    ray.TMax = max(offset, length(L) - offset * 2);
    ray.Direction = normalize(L);
    ray.Origin = surface.worldPos;

    return ray;
}


RayDesc setupVisibilityRay(RAB_Surface surface, float3 samplePosition, float offset)
{
    float3 L = samplePosition - surface.worldPos;

    RayDesc ray;
    ray.TMin = offset;
    ray.TMax = max(offset, length(L) - offset * 2);
    ray.Direction = normalize(L);
    ray.Origin = surface.worldPos;

    return ray;
}
#if !FINAL_SHADING
bool GetConservativeVisibility(RAB_Surface surface, float3 samplePosition)
{
    RayDesc ray = setupVisibilityRay(surface, samplePosition);
    trace(ray);

    #if !PREPROSSES
    return p.missed;
    #else
    return false;
    #endif
}
// Traces a cheap visibility ray that returns approximate, conservative visibility
// between the surface and the light sample. Conservative means if unsure, assume the light is visible.
// Significant differences between this conservative visibility and the final one will result in more noise.
// This function is used in the spatial resampling functions for ray traced bias correction.
bool RAB_GetConservativeVisibility(RAB_Surface surface, RAB_LightSample lightSample)
{
    return GetConservativeVisibility(surface, lightSample.position);
}

// Same as RAB_GetConservativeVisibility but for temporal resampling.
// When the previous frame TLAS and BLAS are available, the implementation should use the previous position and the previous AS.
// When they are not available, use the current AS. That will result in transient bias.

bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface, RAB_Surface previousSurface, RAB_LightSample lightSample)
{
    return GetConservativeVisibility(previousSurface, lightSample.position);
}
#endif


// This function is called in the spatial resampling passes to make sure that 
// the samples actually land on the screen and not outside of its boundaries.
// It can clamp the position or reflect it across the nearest screen edge.
// The simplest implementation will just return the input pixelPosition.
int2 RAB_ClampSamplePositionIntoView(int2 pixelPosition, bool previousFrame)
{
    int width = int(g_Const.view.viewportSize.x);
    int height = int(g_Const.view.viewportSize.y);

    // Reflect the position across the screen edges.
    // Compared to simple clamping, this prevents the spread of colorful blobs from screen edges.
    if (pixelPosition.x < 0) pixelPosition.x = -pixelPosition.x;
    if (pixelPosition.y < 0) pixelPosition.y = -pixelPosition.y;
    if (pixelPosition.x >= width) pixelPosition.x = 2 * width - pixelPosition.x - 1;
    if (pixelPosition.y >= height) pixelPosition.y = 2 * height - pixelPosition.y - 1;

    return pixelPosition;
}
#if !FINAL_SHADING
RAB_Surface GetPrevGBufferSurface(
    ivec2 pixelPosition, 
    PlanarViewConstants view
)
{
    RAB_Surface surface = RAB_EmptySurface();

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
    surface.viewDir = normalize(view.cameraDirectionOrPosition.xyz - surface.worldPos);
    surface.diffuseProbability = getSurfaceDiffuseProbability(surface);

    return surface;
}
#endif


RAB_Surface GetGBufferSurface(
    ivec2 pixelPosition, 
    PlanarViewConstants view
)
{
    RAB_Surface surface = RAB_EmptySurface();

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
    surface.viewDir = normalize(view.cameraDirectionOrPosition.xyz - surface.worldPos);
    surface.diffuseProbability = getSurfaceDiffuseProbability(surface);

    return surface;
}



// Reads the G-buffer, either the current one or the previous one, and returns a surface.
// If the provided pixel position is outside of the viewport bounds, the surface
// should indicate that it's invalid when RAB_IsSurfaceValid is called on it.
RAB_Surface RAB_GetGBufferSurface(int2 pixelPosition, bool previousFrame)
{
    #if !FINAL_SHADING
    if(previousFrame)
    {
        return GetPrevGBufferSurface(
            pixelPosition,
            g_Const.prevView
        );
    }
    else
    #endif
    {
        return GetGBufferSurface(
            pixelPosition, 
            g_Const.view
        );
    }
}

// Checks if the given surface is valid, see RAB_GetGBufferSurface.
bool RAB_IsSurfaceValid(RAB_Surface surface)
{
    return surface.viewDepth != BACKGROUND_DEPTH;
}

// Returns the world position of the given surface
float3 RAB_GetSurfaceWorldPos(RAB_Surface surface)
{
    return surface.worldPos;
}

// Returns the world shading normal of the given surface
float3 RAB_GetSurfaceNormal(RAB_Surface surface)
{
    return surface.normal;
}

// Returns the linear depth of the given surface.
// It doesn't have to be linear depth in a strict sense (i.e. viewPos.z),
// and can be distance to the camera or primary path length instead.
// Just make sure that the motion vectors' .z component follows the same logic.
float RAB_GetSurfaceLinearDepth(RAB_Surface surface)
{
    return surface.viewDepth;
}

// Initialized the random sampler for a given pixel or tile index.
// The pass parameter is provided to help generate different RNG sequences
// for different resampling passes, which is important for image quality.
// In general, a high quality RNG is critical to get good results from ReSTIR.
// A table-based blue noise RNG dose not provide enough entropy, for example.
#if !FINAL_SHADING
RAB_RandomSamplerState RAB_InitRandomSampler(uint2 index, uint pass)
{
    return initRandomSampler(index, f.frame + pass * 13);
}
#endif

// Draws a random number X from the sampler, so that (0 <= X < 1).
float RAB_GetNextRandom(inout RAB_RandomSamplerState rng)
{
    return sampleUniformRng(rng);
}

float2 RAB_GetEnvironmentMapRandXYFromDir(float3 worldDir)
{
    float2 uv = directionToEquirectUV(worldDir); 
    return uv;
}

// Computes the probability of a particular direction being sampled from the environment map
// relative to all the other possible directions, based on the environment map pdf texture.
#ifndef REUSE
#if !FINAL_SHADING
float RAB_EvaluateEnvironmentMapSamplingPdf(float3 L)
{
    if (!environmentMapImportanceSampling)
        return 1.0;

    float2 uv = RAB_GetEnvironmentMapRandXYFromDir(L);

    uint2 pdfTextureSize = g_Const.environmentPdfTextureSize.xy;
    uint2 texelPosition = uint2(pdfTextureSize * uv);
    float texelValue = texture(t_EnvironmentPdfTexture, ivec2(texelPosition)).r;
    
    int lastMipLevel = max(0, int(floor(log2(max(pdfTextureSize.x, pdfTextureSize.y)))));
    float averageValue = textureLod(t_EnvironmentPdfTexture, ivec2(0, 0), lastMipLevel).x;

    // The single texel in the last mip level is effectively the average of all texels in mip 0,
    // padded to a square shape with zeros. So, in case the PDF texture has a 2:1 aspect ratio,
    // that texel's value is only half of the true average of the rectangular input texture.
    // Compensate for that by assuming that the input texture is square.
    float sum = averageValue * float(square(1u << uint(lastMipLevel)));

    return texelValue / sum;
}
// Evaluates pdf for a particular light
float RAB_EvaluateLocalLightSourcePdf(uint lightIndex)
{
    uint2 pdfTextureSize = g_Const.localLightPdfTextureSize.xy;
    uint2 texelPosition = RTXDI_LinearIndexToZCurve(lightIndex);
    float texelValue = texture(t_LocalLightPdfTexture, ivec2(texelPosition)).r;

    int lastMipLevel = max(0, int(floor(log2(max(pdfTextureSize.x, pdfTextureSize.y)))));
    float averageValue = textureLod(t_LocalLightPdfTexture, ivec2(0, 0), lastMipLevel).x;

    // See the comment at 'sum' in RAB_EvaluateEnvironmentMapSamplingPdf.
    // The same texture shape considerations apply to local lights.
    float sum = averageValue * float(square(1u << uint(lastMipLevel)));

    return texelValue / sum;
}
#endif
#endif


bool RAB_GetSurfaceBrdfSample(RAB_Surface surface, inout RAB_RandomSamplerState rng, out float3 dir)
{
    float3 rand;
    rand.x = RAB_GetNextRandom(rng);
    rand.y = RAB_GetNextRandom(rng);
    rand.z = RAB_GetNextRandom(rng);
    if (rand.x < surface.diffuseProbability)
    {
        if (kSpecularOnly)
            return false;

        float pdf;
        float3 h = SampleCosHemisphere(rand.yz, pdf);
        dir = tangentToWorld(surface, h);
    }
    else
    {
        float3 Ve = normalize(worldToTangent(surface, surface.viewDir));
        float3 h = ImportanceSampleGGX_VNDF(rand.yz, max(surface.roughness, kMinRoughness), Ve, 1.0);
        h = normalize(h);
        dir = reflect(-surface.viewDir, tangentToWorld(surface, h));
    }

    return dot(surface.normal, dir) > 0.f;
}

float RAB_GetSurfaceBrdfPdf(RAB_Surface surface, float3 dir)
{
    float cosTheta = saturate(dot(surface.normal, dir));
    float diffusePdf = kSpecularOnly ? 0.f : (cosTheta / RTXDI_PI);
    float specularPdf = ImportanceSampleGGX_VNDF_PDF(max(surface.roughness, kMinRoughness), surface.normal, surface.viewDir, dir);
    float pdf = cosTheta > 0.f ? mix(specularPdf, diffusePdf, surface.diffuseProbability) : 0.f;
    return pdf;
}

// Computes the weight of the given light samples when the given surface is
// shaded using that light sample. Exact or approximate BRDF evaluation can be
// used to compute the weight. ReSTIR will converge to a correct lighting result
// even if all samples have a fixed weight of 1.0, but that will be very noisy.
// Scaling of the weights can be arbitrary, as long as it's consistent
// between all lights and surfaces.
float RAB_GetLightSampleTargetPdfForSurface(RAB_LightSample lightSample, RAB_Surface surface)
{
    if (lightSample.solidAnglePdf <= 0)
        return 0;

    float3 L = normalize(lightSample.position - surface.worldPos);

    if (dot(L, surface.geoNormal) <= 0)
        return 0;
    
    float3 V = surface.viewDir;

    float d = Lambert(surface.normal, -L);
    float3 s;
    if (surface.roughness == 0)
        s = vec3(0);
    else
        s = GGX_times_NdotL(V, L, surface.normal, max(surface.roughness, kMinRoughness), surface.specularF0);

    float3 reflectedRadiance = lightSample.radiance * (d * surface.diffuseAlbedo + s);
    
    return calcLuminance(reflectedRadiance) / lightSample.solidAnglePdf;
}

// Computes the weight of the given light for arbitrary surfaces located inside 
// the specified volume. Used for world-space light grid construction.
float RAB_GetLightTargetPdfForVolume(RAB_LightInfo light, float3 volumeCenter, float volumeRadius)
{
    return getWeightForVolume(light, volumeCenter, volumeRadius);
}

// Samples a polymorphic light relative to the given receiver surface.
// For most light types, the "uv" parameter is just a pair of uniform random numbers, originally
// produced by the RAB_GetNextRandom function and then stored in light reservoirs.
// For importance sampled environment lights, the "uv" parameter has the texture coordinates
// in the PDF texture, normalized to the (0..1) range.
RAB_LightSample RAB_SamplePolymorphicLight(RAB_LightInfo lightInfo, RAB_Surface surface, float2 uv)
{
    PolymorphicLightSample pls = calcSample(lightInfo, uv, surface.worldPos);

    RAB_LightSample lightSample;
    lightSample.position = pls.position;
    lightSample.normal = pls.normal;
    lightSample.radiance = pls.radiance;
    lightSample.solidAnglePdf = pls.solidAnglePdf;
    lightSample.lightType = getLightType(lightInfo);
    return lightSample;
}

void RAB_GetLightDirDistance(RAB_Surface surface, RAB_LightSample lightSample,
    out float3 o_lightDir,
    out float o_lightDistance)
{
    if (lightSample.lightType == kEnvironment)
    {
        o_lightDir = -lightSample.normal;
        o_lightDistance = DISTANT_LIGHT_DISTANCE;
    }
    else
    {
        float3 toLight = lightSample.position - surface.worldPos;
        o_lightDistance = length(toLight);
        o_lightDir = toLight / o_lightDistance;
    }
}

bool RAB_IsAnalyticLightSample(RAB_LightSample lightSample)
{
    return lightSample.lightType != kTriangle && 
        lightSample.lightType != kEnvironment;
}

float RAB_LightSampleSolidAnglePdf(RAB_LightSample lightSample)
{
    return lightSample.solidAnglePdf;
}

// Loads polymorphic light data from the global light buffer.
RAB_LightInfo RAB_LoadLightInfo(uint index, bool previousFrame)
{
    return t_LightDataBuffer[index];
}

// Loads triangle light data from a tile produced by the presampling pass.
RAB_LightInfo RAB_LoadCompactLightInfo(uint linearIndex)
{
    uvec4 packedData1, packedData2;
    packedData1 = u_RisLightDataBuffer[linearIndex * 2 + 0];
    packedData2 = u_RisLightDataBuffer[linearIndex * 2 + 1];
    return unpackCompactLightInfo(packedData1, packedData2);
}

// Stores triangle light data into a tile.
// Returns true if this light can be stored in a tile (i.e. compacted).
// If it cannot, for example it's a shaped light, this function returns false and doesn't store.
// A basic implementation can ignore this feature and always return false, which is just slower.
bool RAB_StoreCompactLightInfo(uint linearIndex, RAB_LightInfo lightInfo)
{
    uvec4 data1, data2;
    if (!packCompactLightInfo(lightInfo, data1, data2))
        return false;

    u_RisLightDataBuffer[linearIndex * 2 + 0] = data1;
    u_RisLightDataBuffer[linearIndex * 2 + 1] = data2;

    return true;
}

// Translates the light index from the current frame to the previous frame (if currentToPrevious = true)
// or from the previous frame to the current frame (if currentToPrevious = false).
// Returns the new index, or a negative number if the light does not exist in the other frame.
int RAB_TranslateLightIndex(uint lightIndex, bool currentToPrevious)
{
    return int(lightIndex);
}

// Forward declare the SDK function that's used in RAB_AreMaterialsSimilar
bool RTXDI_CompareRelativeDifference(float reference, float candidate, float threshold);

// Compares the materials of two surfaces, returns true if the surfaces
// are similar enough that we can share the light reservoirs between them.
// If unsure, just return true.
bool RAB_AreMaterialsSimilar(RAB_Surface a, RAB_Surface b)
{
    const float roughnessThreshold = 0.5;
    const float reflectivityThreshold = 0.25;
    const float albedoThreshold = 0.25;

    if (!RTXDI_CompareRelativeDifference(a.roughness, b.roughness, roughnessThreshold))
        return false;

    if (abs(calcLuminance(a.specularF0) - calcLuminance(b.specularF0)) > reflectivityThreshold)
        return false;
    
    if (abs(calcLuminance(a.diffuseAlbedo) - calcLuminance(b.diffuseAlbedo)) > albedoThreshold)
        return false;

    return true;
}

#ifndef REUSE
float3 GetEnvironmentRadiance(float3 direction)
{
    float2 uv = directionToEquirectUV(direction);
    vec3 environmentRadiance = texture(skyBox, uv).rgb;

    return environmentRadiance;
}
#endif

#if !FINAL_SHADING
#if !PREPROSSES
uint getLightIndex(uint geometryIndex, uint primitiveIndex)
{
    uint lightIndex = RTXDI_InvalidLightIndex;
    uint geometryInstanceIndex = geometryIndex;
    lightIndex = t_GeometryInstanceToLight[geometryInstanceIndex];
    if (lightIndex != RTXDI_InvalidLightIndex)
      lightIndex += primitiveIndex;
    return lightIndex;
}
#endif
// Return true if anything was hit. If false, RTXDI will do environment map sampling
// o_lightIndex: If hit, must be a valid light index for RAB_LoadLightInfo, if no local light was hit, must be RTXDI_InvalidLightIndex
// randXY: The randXY that corresponds to the hit location and is the same used for RAB_SamplePolymorphicLight
bool RAB_TraceRayForLocalLight(float3 origin, float3 direction, float tMin, float tMax,
    out uint o_lightIndex, out float2 o_randXY)
{
    o_randXY = vec2(0);

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;

    float2 hitUV;
    bool hitAnything;

    trace(ray);
    #if !PREPROSSES
    #ifndef REUSE
    hitUV = p.uv;
    o_lightIndex = getLightIndex(p.geometryIndex, p.primitiveId);
    #endif
    #endif

    if (o_lightIndex != RTXDI_InvalidLightIndex)
    {
        o_randXY = randomFromBarycentric(hitUVToBarycentric(hitUV));
    }

    return hitAnything;
}
#endif

// Check if the sample is fine to be used as a valid spatial sample.
// This function also be able to clamp the value of the Jacobian.
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

// Computes the weight of the given GI sample when the given surface is shaded using that GI sample.
float RAB_GetGISampleTargetPdfForSurface(float3 samplePosition, float3 sampleRadiance, RAB_Surface surface)
{
    SplitBrdf brdf = EvaluateBrdf(surface, samplePosition);

    float3 reflectedRadiance = sampleRadiance * (brdf.demodulatedDiffuse * surface.diffuseAlbedo + brdf.specular);

    return RTXDI_Luminance(reflectedRadiance);
}

// Traces a cheap visibility ray that returns approximate, conservative visibility
// between the surface and the light sample. Conservative means if unsure, assume the light is visible.
// Significant differences between this conservative visibility and the final one will result in more noise.
// This function is used in the spatial resampling functions for ray traced bias correction.
#if !FINAL_SHADING

bool RAB_GetConservativeVisibility(RAB_Surface surface, float3 samplePosition)
{
    return GetConservativeVisibility(surface, samplePosition);
}

// Same as RAB_GetConservativeVisibility but for temporal resampling.
// When the previous frame TLAS and BLAS are available, the implementation should use the previous position and the previous AS.
// When they are not available, use the current AS. That will result in transient bias.
bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface, RAB_Surface previousSurface, float3 samplePosition)
{
    return GetConservativeVisibility(previousSurface, samplePosition);
}
#endif

#endif // RTXDI_APPLICATION_BRIDGE_HLSLI