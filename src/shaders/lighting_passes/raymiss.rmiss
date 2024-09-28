#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_tracing : require

#include "../common.glsl"

layout(location = 0) rayPayloadInEXT Payload p;

#include "RtxdiApplicationBridge.glsl"

void main()
{   
	vec3 d = gl_WorldRayDirectionEXT;
	float u = (0.5 + atan2(d.z, d.x)/(2*PI));
    float v = (0.5 - asin(d.y)/PI);
	vec4 tex =  texture(SkyBox, vec2(u,v));
	p.color = tex;
    p.missed = true;
	p.emission = vec3(0.0);
}