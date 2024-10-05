#ifndef HELPER_FUNCTIONS_HLSLI
#define HELPER_FUNCTIONS_HLSLI

#include "rtxdi/RtxdiMath.hlsli"


struct RAB_RandomSamplerState
{
    uint seed;
    uint index;
};

RAB_RandomSamplerState initRandomSampler(uvec2 pixelPos, uint frameIndex)
{
    RAB_RandomSamplerState state;

    uint linearPixelIndex = RTXDI_ZCurveToLinearIndex(pixelPos);

    state.index = 1;
    state.seed = RTXDI_JenkinsHash(linearPixelIndex) + frameIndex;

    return state;
}

uint murmur3(inout RAB_RandomSamplerState r)
{
#define ROT32(x, y) ((x << y) | (x >> (32 - y)))

    // https://en.wikipedia.org/wiki/MurmurHash
    uint c1 = 0xcc9e2d51;
    uint c2 = 0x1b873593;
    uint r1 = 15;
    uint r2 = 13;
    uint m = 5;
    uint n = 0xe6546b64;

    uint hash = r.seed;
    uint k = r.index++;
    k *= c1;
    k = ROT32(k, r1);
    k *= c2;

    hash ^= k;
    hash = ROT32(hash, r2) * m + n;

    hash ^= 4;
    hash ^= (hash >> 16);
    hash *= 0x85ebca6b;
    hash ^= (hash >> 13);
    hash *= 0xc2b2ae35;
    hash ^= (hash >> 16);

#undef ROT32

    return hash;
}

float sampleUniformRng(inout RAB_RandomSamplerState r)
{
    uint v = murmur3(r);
    const uint one = asuint(1.f);
    const uint mask = (1 << 23) - 1;
    return asfloat((mask & v) | one) - 1.f;
}

vec3 sampleTriangle(vec2 rndSample)
{
    float sqrtx = sqrt(rndSample.x);

    return vec3(
        1 - sqrtx,
        sqrtx * (1 - rndSample.y),
        sqrtx * rndSample.y);
}

vec3 hitUVToBarycentric(vec2 hitUV)
{
    return vec3(1 - hitUV.x - hitUV.y, hitUV.x, hitUV.y);
}

// Inverse of sampleTriangle
vec2 randomFromBarycentric(vec3 barycentric)
{
    float sqrtx = 1 - barycentric.x;
    return vec2(sqrtx * sqrtx, barycentric.z / sqrtx);
}

// For converting an area measure pdf to solid angle measure pdf
float pdfAtoW(float pdfA, float distance_, float cosTheta)
{
    return pdfA * (distance_ * distance_) / cosTheta;
}

float calcLuminance(vec3 color)
{
    return dot(color.xyz, vec3(0.299f, 0.587f, 0.114f));
}

vec3 basicToneMapping(vec3 color, float bias)
{
    float lum = calcLuminance(color);

    if (lum > 0)
    {
        float newlum = lum / (bias + lum);
        color *= newlum / lum;
    }

    return color;
}

void ConstructONB(vec3 normal, out vec3 tangent, out vec3 bitangent)
{
    float sign = (normal.z >= 0) ? 1 : -1;
    float a = -1.0 / (sign + normal.z);
    float b = normal.x * normal.y * a;
    tangent = vec3(1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    bitangent = vec3(b, sign + normal.y * normal.y * a, -normal.y);
}


vec2 SampleDisk(vec2 random)
{
    float angle = 2 * RTXDI_PI * random.x;
    return vec2(cos(angle), sin(angle)) * sqrt(random.y);
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

vec3 ImportanceSampleGGX_VNDF(float2 random, float roughness, vec3 Ve, float ndf_trim)
{
    float alpha = square(roughness);

    vec3 Vh = normalize(vec3(alpha * Ve.x, alpha * Ve.y, Ve.z));

    float lensq = square(Vh.x) + square(Vh.y);
    vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) * (1 / sqrt(lensq)) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(Vh, T1);

    float r = sqrt(random.x * ndf_trim);
    float phi = 2.0 * RTXDI_PI * random.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - square(t1)) + s * t2;

    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - square(t1) - square(t2))) * Vh;

    vec3 H;
    H.x = alpha * Nh.x;
    H.y = alpha * Nh.y;
    H.z = max(0.0, Nh.z);

    return H;
}

float3 SampleCosHemisphere(vec2 random, out float solidAnglePdf)
{
    vec2 tangential = SampleDisk(random);
    float elevation = sqrt(saturate(1.0 - random.y));

    solidAnglePdf = elevation / RTXDI_PI;

    return float3(tangential.xy, elevation);
}


float ImportanceSampleGGX_VNDF_PDF(float roughness, vec3 N, vec3 V, vec3 L)
{
    vec3 H = normalize(L + V);
    float NoH = saturate(dot(N, H));
    float VoH = saturate(dot(V, H));

    float alpha = square(roughness);
    float D = square(alpha) / (RTXDI_PI * square(square(NoH) * square(alpha) + (1 - square(NoH))));
    return (VoH > 0.0) ? D / (4.0 * VoH) : 0.0;
}


float Schlick_Fresnel(float F0, float VdotH)
{
    return F0 + (1 - F0) * pow(max(1 - VdotH, 0), 5);
}

vec3 Schlick_Fresnel(vec3 F0, float VdotH)
{
    return F0 + (1 - F0) * pow(max(1 - VdotH, 0), 5);
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


float Lambert(vec3 normal, vec3 lightIncident)
{
    return max(0, -dot(normal, lightIncident)) / RTXDI_PI;
}


float2 directionToEquirectUV(float3 normalizedDirection)
{
    float elevation = asin(normalizedDirection.y);
    float azimuth = 0;
    if (abs(normalizedDirection.y) < 1.0)
        azimuth = atan2(normalizedDirection.z, normalizedDirection.x);

    float2 uv;
    uv.x = azimuth / (2 * RTXDI_PI) - 0.25;
    uv.y = 0.5 - elevation / RTXDI_PI;

    return uv;
}


vec2 octWrap(vec2 v)
{
    return (1.f - abs(v.yx)) * ((v.x >= 0.f || v.y >= 0.f) ? 1.f : -1.f);
}

vec2 ndirToOctSigned(vec3 n)
{
    // Project the sphere onto the octahedron (|x|+|y|+|z| = 1) and then onto the xy-plane
    vec2 p = n.xy * (1.f / (abs(n.x) + abs(n.y) + abs(n.z)));
    return (n.z < 0.f) ? octWrap(p) : p;
}

uint ndirToOctUnorm32(vec3 n)
{
    vec2 p = ndirToOctSigned(n);
    p = saturate(p.xy * 0.5 + 0.5);
    return uint(p.x * 0xfffe) | (uint(p.y * 0xfffe) << 16);
}


void branchlessONB(in vec3 n, out vec3 b1, out vec3 b2)
{
    float sign = n.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (sign + n.z);
    float b = n.x * n.y * a;
    b1 = vec3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = vec3(b, sign + n.y * n.y * a, -n.y);
}


vec3 sampleGGX_VNDF(vec3 Ve, float roughness, vec2 random)
{
    float alpha = square(roughness);

    vec3 Vh = normalize(vec3(alpha * Ve.x, alpha * Ve.y, Ve.z));

    float lensq = square(Vh.x) + square(Vh.y);
    vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(Vh, T1);

    float r = sqrt(random.x);
    float phi = 2.0 * RTXDI_PI * random.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - square(t1)) + s * t2;

    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - square(t1) - square(t2))) * Vh;

    // Tangent space H
    vec3 Ne = vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z));
    return Ne;
}

float G1_Smith(float roughness, float NdotL)
{
    float alpha = square(roughness);
    return 2.0 * NdotL / (NdotL + sqrt(square(alpha) + (1.0 - square(alpha)) * square(NdotL)));
}


vec3 DemodulateSpecular(vec3 surfaceSpecularF0, vec3 specular)
{
    return specular / max(vec3(0.01), surfaceSpecularF0);
}

float3 Unpack_R8G8B8_UFLOAT(uint rgb)
{
    float r = Unpack_R8_UFLOAT(rgb);
    float g = Unpack_R8_UFLOAT(rgb >> 8);
    float b = Unpack_R8_UFLOAT(rgb >> 16);
    return float3(r, g, b);
}

uint Pack_R8G8B8_UFLOAT(float3 rgb)
{
    float3 d = float3(0.5f, 0.5f, 0.5f);
    uint r = Pack_R8_UFLOAT(rgb.r, d.r);
    uint g = Pack_R8_UFLOAT(rgb.g, d.g) << 8;
    uint b = Pack_R8_UFLOAT(rgb.b, d.b) << 16;
    return r | g | b;
}

float3 equirectUVToDirection(float2 uv, out float cosElevation)
{
    float azimuth = (uv.x + 0.25) * (2 * RTXDI_PI);
    float elevation = (0.5 - uv.y) * RTXDI_PI;
    cosElevation = cos(elevation);

    return float3(
        cos(azimuth) * cosElevation,
        sin(elevation),
        sin(azimuth) * cosElevation
    );
}

float3 sampleSphere(float2 rand, out float solidAnglePdf)
{
    // See (6-8) in https://mathworld.wolfram.com/SpherePointPicking.html

    rand.y = rand.y * 2.0 - 1.0;

    float2 tangential = SampleDisk(float2(rand.x, 1.0 - square(rand.y)));
    float elevation = rand.y;

    solidAnglePdf = 0.25f / RTXDI_PI;

    return float3(tangential.xy, elevation);
}

#endif // HELPER_FUNCTIONS_HLSLI