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

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inNormal;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outPosition;
layout(location = 2) out vec4 outNormal;

void main() {
    GeometryInfo geometryInfo = geometryInfos.g[g.index];
    if (geometryInfo.baseColorTextureIndex > -1) {
        outColor = texture(textures[geometryInfo.baseColorTextureIndex], inUV) * inColor;
    } else {
        outColor = inColor;
    }
    outPosition = vec4(inPosition.xyz, 1.0);
    outNormal = vec4(inNormal.xyz, 1.0);
}