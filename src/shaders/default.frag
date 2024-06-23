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
#include "./common.glsl"
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


layout(location = 0) out vec4 outColor;
layout(location = 1) out float outDepth;

const float ligth_rotation = PI/1.5; 
const float light_hight = PI/4; 

const vec3 light_dir = vec3(sin(ligth_rotation)*cos(light_hight), sin(ligth_rotation)*sin(light_hight), cos(ligth_rotation));

float ray_plane(vec3 origin, vec3 direction, vec3 normal, vec3 center) {
    float denom = dot(normal, direction);
    if (abs(denom) > 0.0001f)
    {
        float t = dot((center - origin), normal) / denom;
        if (t >= 0) return t;
    }
    return 1.0/0.0;
}
    const vec3 plane_normal = vec3(0.0, 1.0, 0.0);


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

    vec3 o_pos, o_pos2;
    vec3 o_color, o_color2;
    vec3 o_normal, o_normal2;
    bool hit = Octree_RayMarchLeaf(origin.xyz, direction.xyz, o_pos, o_color, o_normal);
    float plane_depth = ray_plane(origin.xyz, direction.xyz, plane_normal, vec3(0.0, 0.99999, 0.0));
    float depth = hit ? length(origin.xyz - o_pos) : 1.0/0.0;

    if (!hit || depth > plane_depth) {
        o_pos = origin.xyz + direction.xyz * plane_depth;
        o_normal = plane_normal;
        o_color = vec3(min(max(plane_depth / 100.0, 0.3), 0.8));
    }

    bool shadow_hit = Octree_RayMarchLeaf(o_pos, light_dir, o_pos2, o_color2, o_normal2);
    // * (abs(dot(o_normal, light_dir)) + 0.2)



    if (min(depth, plane_depth) < 1.0 / 0.0) {
        if (shadow_hit) {
            o_color *= 0.35;
        }
        outColor = vec4(o_color * max(dot(o_normal, light_dir), 0.25), 0.0);
    } else {
        float u = (0.5 + atan2(direction.z, direction.x)/(2*PI));
        float v = (0.5 - asin(direction.y)/PI);
        outColor = texture(skybox, vec2(u, v));
    }
    outDepth = min(depth, plane_depth);
}