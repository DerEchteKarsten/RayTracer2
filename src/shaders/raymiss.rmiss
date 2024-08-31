#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive: enable

#include "./common.glsl"

layout(location = 0) rayPayloadInEXT Payload p;


layout(binding = 8, set = 0) uniform sampler2D skyBox;

float atan2(in float y, in float x)
{
	if (x > 0) {
		return atan(y/x);
	}
	if (x < 0 && y >= 0){
		return atan(y/x) + PI;
	}
	if (x < 0 && y < 0) {
		return atan(y/x) - PI;
	}
	if (x == 0 && y > 0) {
		return PI/2;
	}
	if (x == 0 && y < 0) {
		return -PI/2;
	}
	if (x == 0 && y == 0) {
		return 0;
	}
	return 0;
}

void main()
{   
	vec3 d = gl_WorldRayDirectionEXT;
	float u = (0.5 + atan2(d.z, d.x)/(2*PI));
    float v = (0.5 - asin(d.y)/PI);
	vec4 tex =  texture(skyBox, vec2(u,v));
	p.color = tex;
    p.missed = true;
	p.emission = vec3(0.0);
}