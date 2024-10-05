#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_tracing : require

#include "common.glsl"

layout(location = 0) rayPayloadInEXT Payload p;

#include "RtxdiApplicationBridge.glsl"

void main()
{   
	p.uv = vec2(0.0, 0.0);
	p.geometryIndex = ~0u;
	p.primitiveId = ~0u;
}