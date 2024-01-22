#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadInEXT Payload {
	bool missed;
	float metallicFactor;
  	float roughness;
	vec3 color;
  	vec3 emission;
	vec3 hitPoint;
	vec3 hitNormal;
} p;


layout(binding = 8, set = 0) uniform sampler2D skyBox;

#define PI 3.141592653589793238

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

    p.color = texture(skyBox, vec2(u,v)).xyz;
    p.missed = true;
    //vec4(0.392, 0.5843, 0.92941, 1.0);
}