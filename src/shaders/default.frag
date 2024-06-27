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

#include "./restir.glsl"

struct Octant {
    uint32_t children[8];
    uint32_t empty_color;
};

layout(binding = 0, set = 0) uniform CameraProperties 
{
	mat4 viewInverse;
	mat4 projInverse;
    vec4 controlls;
} cam;

struct Gizzmo {
    vec4 color;
    vec3 pos;
    float radius;
};

layout(binding = 2, set = 0) uniform sampler2D skybox;

layout(std430, set = 0, binding = 3) readonly buffer uGizzmoBuffer { Gizzmo GizzmoBuffer[]; };

layout( push_constant ) uniform Frame {
	uint frame;
} f;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outDepth;


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

float raySphereIntersect(vec3 r0, vec3 rd, vec3 s0, float sr) {
    // - r0: ray origin
    // - rd: normalized ray direction
    // - s0: sphere center
    // - sr: sphere radius
    // - Returns distance from r0 to first intersecion with sphere,
    //   or -1.0 if no intersection.
    float a = dot(rd, rd);
    vec3 s0_r0 = r0 - s0;
    float b = 2.0 * dot(rd, s0_r0);
    float c = dot(s0_r0, s0_r0) - (sr * sr);
    if (b*b - 4.0*a*c < 0.0) {
        return -1.0;
    }
    return (-b - sqrt((b*b) - 4.0*a*c))/(2.0*a);
}

const int max_rays = 3;

void main() {

    const vec2 pixelCenter = vec2(gl_FragCoord.xy) + vec2(0.5);
	const vec2 inUV = pixelCenter/vec2(cam.controlls.zw);
	vec2 d = inUV * 2.0 - 1.0;
	vec4 origin = cam.viewInverse * vec4(0,0,0,1);
	vec4 target = cam.projInverse * vec4(d.x, d.y, 1, 1) ;
	vec4 direction = cam.viewInverse*vec4(normalize(target.xyz), 0) ;

    for(int i = 0; i<10; i++) {
        Gizzmo current = GizzmoBuffer[i];
        if(current.radius < 0.0) {
            continue;
        }

        if(raySphereIntersect(origin.xyz, direction.xyz, current.pos, current.radius) != -1) {
            outColor = current.color;
            outDepth = 1.0;
            return;
        }
    }

    vec3 o_pos, o_color, o_normal, o_pos2, o_color2, o_normal2;
    float depth = ray_cast(origin.xyz, direction.xyz, o_pos, o_normal, o_color);

    if (depth >= INFINITY) {
        float u = (0.5 + atan2(direction.z, direction.x)/(2*PI));
        float v = (0.5 - asin(direction.y)/PI);
        outColor = texture(skybox, vec2(u, v));
        outDepth = INFINITY;
        return;
    }

    uint rngState = uint((gl_FragCoord.y * cam.controlls.z) + gl_FragCoord.x) + uint(f.frame) * 23145;
    Reservoir reservoir = Reservoir(0, 0.0, 0, 0.0);
    outColor = vec4(RIS(reservoir, o_pos, o_normal, o_color, rngState), 0.0);
    // const vec3 light_dir =  polar_form(ligth_rotation, light_hight);
    // float depth2 = ray_cast(o_pos + o_normal * 0.0001, light_dir, o_pos2, o_normal2, o_color2);
    // outColor = vec4(max(dot(o_normal, light_dir), 0.3) * (depth2 == INFINITY ? o_color : o_color * 0.4), 1.0); 

    outDepth = depth;
}