#extension GL_GOOGLE_include_directive : enable

#include "./common.glsl"

struct Sample {
    vec3 origin;
    vec3 originNormal;
    vec3 samplePoint;
    vec3 samplePointNormal;
    vec3 radiance;
    uint random;
};

struct Reservoir {
    Sample s;
    float weightOfReservoir;
    uint numCandidateSamples;
    float weightOfSample;
};

void update_reservoir(inout Reservoir self, Sample new_sample, float new_weight, inout uint state) {
    self.weightOfReservoir += new_weight;
    self.numCandidateSamples += 1;

    if (RandomValue(state) < new_weight / self.weightOfReservoir) {
        self.s = new_sample;
    } 
}

void merge_reservoir(inout Reservoir self, Reservoir other, float target_pdf, inout uint state) {
    uint s = self.numCandidateSamples;
    update_reservoir(self, other.s, target_pdf * other.weightOfSample * other.numCandidateSamples, state);
    self.numCandidateSamples = s + other.numCandidateSamples;
}

struct Surface
{
    vec3 worldPos;
    vec3 viewDir;
    float viewDepth;
    vec3 normal;
    vec3 diffuseAlbedo;
    float metallic;
    float roughness;
    float diffuseProbability;
};

struct SecondaryGBufferData
{
    vec3 worldPos;
    uint normal;

    uvec2 throughputAndFlags;   // .x = throughput.rg as float16, .y = throughput.b as float16, flags << 16
    uint diffuseAlbedo;         // R11G11B10_UFLOAT
    uint specularAndRoughness;  // R8G8B8A8_Gamma_UFLOAT
    
    vec3 emission;
    float pdf;
};

struct LightSample
{
    vec3 position;
    vec3 normal;
    vec3 radiance;
    float solidAnglePdf;
};

float calcLuminance(vec3 color)
{
    return dot(color.xyz, vec3(0.299f, 0.587f, 0.114f));
}

float calcLuminance(float color) {
    return dot(vec3(color), vec3(0.299f, 0.587f, 0.114f));
}

float saturate(float v) {
	return clamp(v, 0.0, 1.0);
}


float square(float x) {return x*x;}

float G1_Smith(float roughness, float NdotL)
{
    float alpha = square(roughness);
    return 2.0 * NdotL / (NdotL + sqrt(square(alpha) + (1.0 - square(alpha)) * square(NdotL)));
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
        float D = square(alpha) / (PI * square(square(NoH) * square(alpha) + (1 - square(NoH))));

        vec3 F = Schlick_Fresnel(F0, VoH);

        return F * (D * G / 4);
    }
    return vec3(0);
}

vec2 sampleDisk(vec2 rand)
{
    float angle = 2 * PI * rand.x;
    return vec2(cos(angle), sin(angle)) * sqrt(rand.y);
}

float Lambert(vec3 N, vec3 L)
{
  vec3 nrmN = normalize(N);
  vec3 nrmL = normalize(L);
  float result = dot(nrmN, nrmL);
  return max(result, 0.0);
}

vec3 sampleCosHemisphere(vec2 rand, out float solidAnglePdf)
{
    vec2 tangential = sampleDisk(rand);
    float elevation = sqrt(saturate(1.0 - rand.y));

    solidAnglePdf = elevation / PI;

    return vec3(tangential.xy, elevation);
}


vec3 sampleGGX_VNDF(vec3 Ve, float roughness, vec2 random)
{
    float alpha = square(roughness);

    vec3 Vh = normalize(vec3(alpha * Ve.x, alpha * Ve.y, Ve.z));

    float lensq = square(Vh.x) + square(Vh.y);
    vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(Vh, T1);

    float r = sqrt(random.x);
    float phi = 2.0 * PI * random.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - square(t1)) + s * t2;

    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - square(t1) - square(t2))) * Vh;

    // Tangent space H
    vec3 Ne = vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z));
    return Ne;
}

vec3 fresnelSchlick(vec3 F0, float cosTheta)
{
	return F0 + (vec3(1.0) - F0) * pow(1.0 - cosTheta, 5.0);
}

float fresnelSchlick(float F0, float cosTheta)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float getSurfaceDiffuseProbability(Surface surface)
{
    float diffuseWeight = calcLuminance(surface.diffuseAlbedo);
    float specularWeight = calcLuminance(fresnelSchlick(surface.metallic, dot(surface.viewDir, surface.normal)));
    float sumWeights = diffuseWeight + specularWeight;
    return sumWeights < 1e-7f ? 1.f : (diffuseWeight / sumWeights);
}

struct SplitBrdf
{
    float demodulatedDiffuse;
    vec3 specular;
};

SplitBrdf EvaluateBrdf(Surface surface, vec3 samplePosition)
{
    vec3 N = surface.normal;
    vec3 V = surface.viewDir;
    vec3 L = normalize(samplePosition - surface.worldPos);

    SplitBrdf brdf;
    brdf.demodulatedDiffuse = Lambert(surface.normal, -L);
    if (surface.roughness == 0)
        brdf.specular = vec3(0);
    else
        brdf.specular = GGX_times_NdotL(V, L, surface.normal, max(surface.roughness, 0.05), vec3(surface.metallic));
    return brdf;
}

float ImportanceSampleGGX_VNDF_PDF(float roughness, vec3 N, vec3 V, vec3 L)
{
    vec3 H = normalize(L + V);
    float NoH = clamp(dot(N, H), 0, 1);
    float VoH = clamp(dot(V, H), 0, 1);

    float alpha = square(roughness);
    float D = square(alpha) / (PI * square(square(NoH) * square(alpha) + (1 - square(NoH))));
    return (VoH > 0.0) ? D / (4.0 * VoH) : 0.0;
}