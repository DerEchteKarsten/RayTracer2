
#ifndef POLYMORPHIC_LIGHT_HLSLI
#define POLYMORPHIC_LIGHT_HLSLI
#include "Helpers.glsl"

const uint kSphere = 0;
const uint kCylinder = 1;
const uint kDisk = 2;
const uint kRect = 3;
const uint kTriangle = 4;
const uint kDirectional = 5;
const uint kEnvironment = 6;
const uint kPoint = 7;

#define PolymorphicLightType uint

// Stores shared light information (type) and specific light information
// See PolymorphicLight.hlsli for encoding format
struct RAB_LightInfo
{
    // uint4[0]
    float3 center;
    uint colorTypeAndFlags; // RGB8 + uint8 (see the kPolymorphicLight... constants above)

    // uint4[1]
    uint direction1; // oct-encoded
    uint direction2; // oct-encoded
    uint scalars; // 2x float16
    uint logRadiance; // uint16

    // uint4[2] -- optional, contains only shaping data
    uint iesProfileIndex;
    uint primaryAxis; // oct-encoded
    uint cosConeAngleAndSoftness; // 2x float16
    uint padding;
};

#include "Helpers.glsl"
#include "LightShaping.glsl"
#include "./rtxdi/RtxdiHelpers.hlsli"

#define LIGHT_SAMPING_EPSILON 1e-10
#define DISTANT_LIGHT_DISTANCE 10000.0

#ifndef ENVIRONMENT_SAMPLER
#define ENVIRONMENT_SAMPLER s_EnvironmentSampler
#endif

struct PolymorphicLightSample
{
    float3 position;
    float3 normal;
    float3 radiance;
    float solidAnglePdf;
};

PolymorphicLightType getLightType(RAB_LightInfo lightInfo)
{
    uint typeCode = (lightInfo.colorTypeAndFlags >> kPolymorphicLightTypeShift) 
        & kPolymorphicLightTypeMask;

    return typeCode;
}

float unpackLightRadiance(uint logRadiance)
{
    return (logRadiance == 0) ? 0 : exp2((float(logRadiance - 1) / 65534.0) * (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance) + kPolymorphicLightMinLog2Radiance);
}

float3 unpackLightColor(RAB_LightInfo lightInfo)
{
    float3 color = Unpack_R8G8B8_UFLOAT(lightInfo.colorTypeAndFlags);
    float radiance = unpackLightRadiance(lightInfo.logRadiance & 0xffff);
    return color * radiance.xxx;
}

void packLightColor(float3 radiance, inout RAB_LightInfo lightInfo)
{   
    float intensity = max(radiance.r, max(radiance.g, radiance.b));

    if (intensity > 0.0)
    {
        float logRadiance = saturate((log2(intensity) - kPolymorphicLightMinLog2Radiance) 
            / (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance));
        uint packedRadiance = min(uint32_t(ceil(logRadiance * 65534.0)) + 1, 0xffffu);
        float unpackedRadiance = unpackLightRadiance(packedRadiance);

        float3 normalizedRadiance = saturate(radiance.rgb / unpackedRadiance.xxx);

        lightInfo.logRadiance |= packedRadiance;
        lightInfo.colorTypeAndFlags |= Pack_R8G8B8_UFLOAT(normalizedRadiance);
    }
}

bool packCompactLightInfo(RAB_LightInfo lightInfo, out uvec4 res1, out uvec4 res2)
{
    if (unpackLightShaping(lightInfo).isSpot == 1)
    {
        res1 = uvec4(0);
        res2 = uvec4(0);
        return false;
    }

    res1.xyz = asuint(lightInfo.center.xyz);
    res1.w = lightInfo.colorTypeAndFlags;

    res2.x = lightInfo.direction1;
    res2.y = lightInfo.direction2;
    res2.z = lightInfo.scalars;
    res2.w = lightInfo.logRadiance;
    return true;
}

RAB_LightInfo unpackCompactLightInfo(const uvec4 data1, const uvec4 data2)
{
    RAB_LightInfo lightInfo;
    lightInfo.center.xyz = asfloat(data1.xyz);
    lightInfo.colorTypeAndFlags = data1.w;
    lightInfo.direction1 = data2.x;
    lightInfo.direction2 = data2.y;
    lightInfo.scalars = data2.z;
    lightInfo.logRadiance = data2.w;
    return lightInfo;
}

// Computes estimated distance between a given point in space and a random point inside
// a spherical volume. Since the geometry of this solution is spherically symmetric,
// only the distance from the volume center to the point and the volume radius matter here.
float getAverageDistanceToVolume(float distanceToCenter, float volumeRadius)
{
    // The expression and factor are fitted to a Monte Carlo estimated curve.
    // At distanceToCenter == 0, this function returns (0.75 * volumeRadius) which is analytically accurate.
    // At infinity, the result asymptotically approaches distanceToCenter.

    const float nonlinearFactor = 1.1547;

    return distanceToCenter + volumeRadius * square(volumeRadius) 
        / square(distanceToCenter + volumeRadius * nonlinearFactor);
}

// Point light is a sphere light with zero radius.
// On the host side, they are both created from LightType_Point, depending on the radius.
// The values returned from all interface methods of PointLight are the same as SphereLight
// would produce in the limit when radius approaches zero, with some exceptions in calcSample.
struct PointLight
{
    float3 position;
    float3 flux;
    LightShaping shaping;

    // Interface methods
};

PolymorphicLightSample calcPointLightSample(in const float3 viewerPosition, inout PointLight self)
{
    const float3 lightVector = self.position - viewerPosition;
    
    PolymorphicLightSample lightSample;

    // We cannot compute finite values for radiance and solidAnglePdf for a point light,
    // so return the limit of (radiance / solidAnglePdf) with radius --> 0 as radiance.
    lightSample.position = self.position;
    lightSample.normal = normalize(-lightVector);
    lightSample.radiance = self.flux / dot(lightVector, lightVector);
    lightSample.solidAnglePdf = 1.0;

    return lightSample;
}

float getPointLightPower(PointLight self)
{
    return 4.0 * RTXDI_PI * calcLuminance(self.flux) * getShapingFluxFactor(self.shaping);
}

float getPointLightWeightForVolume(in const float3 volumeCenter, in const float volumeRadius, inout PointLight self)
{
    if (!testSphereIntersectionForShapedLight(self.position, 0, self.shaping, volumeCenter, volumeRadius))
        return 0.0;

    float distance = length(volumeCenter - self.position);
    distance = getAverageDistanceToVolume(distance, volumeRadius);

    return calcLuminance(self.flux) / square(distance);
}

static PointLight CreatePointLight(in const RAB_LightInfo lightInfo)
{
    PointLight pointLight;

    pointLight.position = lightInfo.center;
    pointLight.flux = unpackLightColor(lightInfo);
    pointLight.shaping = unpackLightShaping(lightInfo);

    return pointLight;
}

struct DirectionalLight
{
    float3 direction;
    float cosHalfAngle; // Note: Assumed to be != 1 to avoid delta light special case
    float sinHalfAngle;
    float solidAngle;
    float3 radiance;

};
// Interface methods

PolymorphicLightSample calcDirectionalLightSample(in const float2 random, in const float3 viewerPosition, inout DirectionalLight self)
{
    const float2 diskSample = SampleDisk(random);

    float3 tangent, bitangent;
    branchlessONB(self.direction, tangent, bitangent);

    const float3 distantDirectionSample = self.direction 
        + tangent * diskSample.x * self.sinHalfAngle
        + bitangent * diskSample.y * self.sinHalfAngle;

    // Calculate sample position on the distant light
    // Since there is no physical distant light to hit (as it is at infinity), this simply uses a large
    // number far enough away from anything in the world.

    const float3 distantPositionSample = viewerPosition - distantDirectionSample * DISTANT_LIGHT_DISTANCE;
    const float3 distantNormalSample = self.direction;

    // Create the light sample

    PolymorphicLightSample lightSample;

    lightSample.position = distantPositionSample;
    lightSample.normal = distantNormalSample;
    lightSample.radiance = self.radiance;
    lightSample.solidAnglePdf = 1.0 / self.solidAngle;
    
    return lightSample;
}

// Helper methods

static DirectionalLight CreateDirectionalLight(in const RAB_LightInfo lightInfo)
{
    DirectionalLight directionalLight;

    directionalLight.direction = octToNdirUnorm32(lightInfo.direction1);

    float halfAngle = f16tof32(lightInfo.scalars);
    sincos(halfAngle, directionalLight.sinHalfAngle, directionalLight.cosHalfAngle);
    directionalLight.solidAngle = f16tof32(lightInfo.scalars >> 16);
    directionalLight.radiance = unpackLightColor(lightInfo);

    return directionalLight;
}

struct TriangleLight
{
    float3 base;
    float3 edge1;
    float3 edge2;
    float3 radiance;
    float3 normal;
    float surfaceArea;

    // Interface methods
};

float calcTriangleSolidAnglePdf(in const float3 viewerPosition,
                        in const float3 lightSamplePosition,
                        in const float3 lightSampleNormal,
                        inout TriangleLight self)
{
    float3 L = lightSamplePosition - viewerPosition;
    float Ldist = length(L);
    L /= Ldist;

    const float areaPdf = 1.0 / self.surfaceArea;
    const float sampleCosTheta = saturate(dot(L, -lightSampleNormal));

    return pdfAtoW(areaPdf, Ldist, sampleCosTheta);
}

PolymorphicLightSample calcTriangleSample(in const float2 random, in const float3 viewerPosition, inout TriangleLight self)
{
    PolymorphicLightSample result;

    float3 bary = sampleTriangle(random);
    result.position = self.base + self.edge1 * bary.y + self.edge2 * bary.z;
    result.normal = self.normal;

    result.solidAnglePdf = calcTriangleSolidAnglePdf(viewerPosition, result.position, result.normal, self);

    result.radiance = self.radiance;

    return result;   
}


float getTrianglePower(inout TriangleLight self)
{
    return self.surfaceArea * RTXDI_PI * calcLuminance(self.radiance);
}

float getTriangleWeightForVolume(in const float3 volumeCenter, in const float volumeRadius, inout TriangleLight self)
{
    float distanceToPlane = dot(volumeCenter - self.base, self.normal);
    if (distanceToPlane < -volumeRadius)
        return 0; // Cull - the entire volume is below the light's horizon

    float3 barycenter = self.base + (self.edge1 + self.edge2) / 3.0;
    float distance = length(barycenter - volumeCenter);
    distance = getAverageDistanceToVolume(distance, volumeRadius);

    float approximateSolidAngle = self.surfaceArea / square(distance);
    approximateSolidAngle = min(approximateSolidAngle, 2 * RTXDI_PI);

    return approximateSolidAngle * calcLuminance(self.radiance);
}

// Helper methods

static TriangleLight CreateTriangleLight(in const RAB_LightInfo lightInfo)
{
    TriangleLight triLight;

    triLight.edge1 = octToNdirUnorm32(lightInfo.direction1) * f16tof32(lightInfo.scalars);
    triLight.edge2 = octToNdirUnorm32(lightInfo.direction2) * f16tof32(lightInfo.scalars >> 16);
    triLight.base = lightInfo.center - (triLight.edge1 + triLight.edge2) / 3.0;
    triLight.radiance = unpackLightColor(lightInfo);

    float3 lightNormal = cross(triLight.edge1, triLight.edge2);
    float lightNormalLength = length(lightNormal);

    if(lightNormalLength > 0.0)
    {
        triLight.surfaceArea = 0.5 * lightNormalLength;
        triLight.normal = lightNormal / lightNormalLength;
    }
    else
    {
        triLight.surfaceArea = 0.0;
        triLight.normal = vec3(0.0); 
    }

    return triLight;
}

RAB_LightInfo StoreTriangleLight(inout TriangleLight self)
{
    RAB_LightInfo lightInfo;

    packLightColor(self.radiance, lightInfo);
    lightInfo.center = self.base + (self.edge1 + self.edge2) / 3.0;
    lightInfo.direction1 = ndirToOctUnorm32(normalize(self.edge1));
    lightInfo.direction2 = ndirToOctUnorm32(normalize(self.edge2));
    lightInfo.scalars = f32tof16(length(self.edge1)) | (f32tof16(length(self.edge2)) << 16);
    lightInfo.colorTypeAndFlags |= uint(kTriangle) << kPolymorphicLightTypeShift;
    
    return lightInfo;
}

struct EnvironmentLight
{
    int textureIndex;
    bool importanceSampled;
    float3 radianceScale;
    float rotation;
    uint2 textureSize;
};

PolymorphicLightSample calcEnvironmentLightSample(in const float2 random, in const float3 viewerPosition, inout EnvironmentLight self)
{
    PolymorphicLightSample lightSample;

    float2 textureUV;
    float3 sampleDirection;
    if (self.importanceSampled)
    {
        float2 directionUV = random;
        directionUV.x += self.rotation;

        float cosElevation;
        sampleDirection = equirectUVToDirection(directionUV, cosElevation);

        // Inverse of the solid angle of one texel of the environment map using the equirectangular projection.
        lightSample.solidAnglePdf = (self.textureSize.x * self.textureSize.y) / (2 * RTXDI_PI * RTXDI_PI * cosElevation);
        textureUV = random;
    }
    else
    {
        sampleDirection = sampleSphere(random, lightSample.solidAnglePdf);
        textureUV = directionToEquirectUV(sampleDirection);
        textureUV.x -= self.rotation;
    }

    float3 sampleRadiance = self.radianceScale;
    if (self.textureIndex >= 0)
    {
        sampleRadiance *= textureLod(skyBox, textureUV, 0).xyz;
    }

    // Inf / NaN guard.
    // Sometimes EXR files might contain those values (e.g. when saved by Photoshop).
    float radianceSum = sampleRadiance.r + sampleRadiance.g + sampleRadiance.b;
    if (isinf(radianceSum) || isnan(radianceSum))
        sampleRadiance = vec3(0);

    lightSample.position = viewerPosition + sampleDirection * DISTANT_LIGHT_DISTANCE;
    lightSample.normal = -sampleDirection;
    lightSample.radiance = sampleRadiance;

    return lightSample;
}

// Helper methods

static EnvironmentLight CreateEnvironmentLight(in const RAB_LightInfo lightInfo)
{
    EnvironmentLight envLight;

    envLight.textureIndex = int(lightInfo.direction1);
    envLight.rotation = f16tof32(lightInfo.scalars);
    envLight.importanceSampled = ((lightInfo.scalars >> 16) != 0);
    envLight.radianceScale = unpackLightColor(lightInfo);
    envLight.textureSize.x = lightInfo.direction2 & 0xffff;
    envLight.textureSize.y = lightInfo.direction2 >> 16;

    return envLight;
}


PolymorphicLightSample calcSample(
    in const RAB_LightInfo lightInfo, 
    in const float2 random, 
    in const float3 viewerPosition)
{
    PolymorphicLightSample lightSample;

    switch (getLightType(lightInfo))
    {
        case kPoint:       
            PointLight plight = CreatePointLight(lightInfo);
            lightSample = calcPointLightSample(viewerPosition, plight); 
        break;
        case kTriangle:    
            TriangleLight tlight = CreateTriangleLight(lightInfo);
            lightSample = calcTriangleSample(random, viewerPosition, tlight); 
        break;
        case kDirectional: 
            DirectionalLight dlight = CreateDirectionalLight(lightInfo);
            lightSample = calcDirectionalLightSample(random, viewerPosition, dlight); 
        break;
        case kEnvironment: 
            EnvironmentLight elight = CreateEnvironmentLight(lightInfo);
            lightSample = calcEnvironmentLightSample(random, viewerPosition, elight); 
        break;
    }

    if (lightSample.solidAnglePdf > 0)
    {
        lightSample.radiance *= evaluateLightShaping(unpackLightShaping(lightInfo),
            viewerPosition, lightSample.position);
    }

    return lightSample;
}

float getPower(
    in const RAB_LightInfo lightInfo)
{
    switch (getLightType(lightInfo))
    {
    case kPoint:       
        PointLight plight = CreatePointLight(lightInfo);
        return getPointLightPower(plight);
    case kTriangle:    
        TriangleLight tlight = CreateTriangleLight(lightInfo);
        return getTrianglePower(tlight);
    case kDirectional: 
    
        return 0; // infinite lights don't go into the local light PDF map
    case kEnvironment: 
    
        return 0;
    default: return 0;
    }
}

float getWeightForVolume(
    in const RAB_LightInfo lightInfo, 
    in const float3 volumeCenter,
    in const float volumeRadius)
{
    switch (getLightType(lightInfo))
    {
    case kPoint:       
        PointLight plight = CreatePointLight(lightInfo);
    return getPointLightWeightForVolume(volumeCenter, volumeRadius, plight);
    case kTriangle:    
        TriangleLight tlight = CreateTriangleLight(lightInfo);
    return getTriangleWeightForVolume(volumeCenter, volumeRadius, tlight);
    case kDirectional: return 0; // infinite lights do not affect volume sampling
    case kEnvironment: return 0;
    default: return 0;
    }
}

#endif