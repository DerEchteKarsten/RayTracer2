#version 450
#extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : enable

#include "./common.glsl"
layout(binding = 0, set = 0) uniform CameraProperties 
{
	mat4 view;
	mat4 proj;
    mat4 model;
} cam;

layout(binding = 2, set = 0) readonly buffer GeometryInfos { GeometryInfo g[]; } geometryInfos;
layout( push_constant ) uniform Index {
	uint index;
} g;


layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inNormal;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec4 outNormal;
layout(location = 1) out vec4 outColor;
layout(location = 2) out vec2 outUV;


void main() {
    gl_Position = cam.proj * cam.view * cam.model * geometryInfos.g[g.index].transform * vec4(inPosition.xyz, 1.0);
    outColor = inColor;
    outNormal = inNormal;
    outUV = inUV;
}