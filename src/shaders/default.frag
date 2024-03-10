#version 450
#extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "./common.glsl"

layout(binding = 1, set = 0) uniform sampler2D textures[];
layout(binding = 2, set = 0) readonly buffer GeometryInfos { GeometryInfo g[]; } geometryInfos;

layout( push_constant ) uniform Index {
	uint index;
} g;

layout(location = 0) in vec4 inNormal;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec4 outColor;

void main() {
    GeometryInfo geometryInfo = geometryInfos.g[g.index];
    vec4 Color;
    if (geometryInfo.baseColorTextureIndex > -1) {
        Color = texture(textures[geometryInfo.baseColorTextureIndex], inUV) * geometryInfo.baseColor * inColor;
    } else {
        Color = inColor * geometryInfo.baseColor;
    }
    outColor = pack(Color, normalize(geometryInfo.transform * inNormal), geometryInfo.roughness, geometryInfo.metallicFactor, geometryInfo.emission);
}