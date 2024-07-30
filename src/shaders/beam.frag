#version 450
#extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
// #extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
// #extension GL_EXT_shader_explicit_arithmetic_types_float32 : enable
// #extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable

#include "./oct_tree.glsl"


layout( push_constant ) uniform PushConstants {
    float uOriginSize;
};

layout(binding = 1, set = 0) uniform CameraProperties 
{
	mat4 viewInverse;
	mat4 projInverse;
    vec4 controlls;
} cam;

layout(location = 0) out float FragColor;


vec3 GenCoarseRay(in vec2 bias) {
	vec2 uv = (vec2(ivec2(gl_FragCoord.xy)) + bias) / vec2(cam.controlls.z, cam.controlls.w) * uBeamSize;
	vec2 d = uv * 2.0 - 1.0;
	vec4 target = cam.projInverse * vec4(d.x, d.y, 1, 1) ;
	vec4 direction = cam.viewInverse*vec4(normalize(target.xyz), 0);
	return direction.xyz;
}

float RayTangent(in vec3 x, in vec3 y) {
	float c = dot(x, y);
	return sqrt(1.0 - c * c) / abs(c);
}


void main() {
    const vec2 pixelCenter = vec2(gl_FragCoord.xy) + vec2(0.5);
	const vec2 inUV = pixelCenter/vec2(cam.controlls.zw);
	vec2 d = inUV * 2.0 - 1.0;
	vec4 origin = cam.viewInverse * vec4(0,0,0,1);
	vec4 target = cam.projInverse * vec4(d.x, d.y, 1, 1) ;
	vec4 direction = cam.viewInverse*vec4(normalize(target.xyz), 0) ;
	vec3 dir = direction.xyz;
	vec3 o = origin.xyz;

	float t0 = RayTangent(dir, GenCoarseRay(vec2(0.5, 0.5)));
	float t1 = RayTangent(dir, GenCoarseRay(vec2(-0.5, 0.5)));
	float t2 = RayTangent(dir, GenCoarseRay(vec2(0.5, -0.5)));
	float t3 = RayTangent(dir, GenCoarseRay(vec2(-0.5, -0.5)));
	float dir_sz = 2.0 * max(max(t0, t1), max(t2, t3)), t, size;

    FragColor = Octree_RayMarchCoarse(origin.xyz, direction.xyz, uOriginSize, dir_sz, t, size) ? max(0.0, t - size) : 1e10;
}