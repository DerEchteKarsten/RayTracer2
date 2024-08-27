#include "Helpers.glsl"
#include "GBufferHelpers.glsl"
#define square(x) x*x
const float kMinRoughness = 0.05f;

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

float Lambert(vec3 normal, vec3 lightIncident)
{
    return max(0, -dot(normal, lightIncident)) / RTXDI_PI;
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

vec3 DemodulateSpecular(vec3 surfaceSpecularF0, vec3 specular)
{
    return specular / max(vec3(0.01), surfaceSpecularF0);
}