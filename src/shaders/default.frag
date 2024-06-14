#version 450
#extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_ARB_gpu_shader_int64 : enable

#extension GL_EXT_shader_explicit_arithmetic_types : enable
// #extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
// #extension GL_EXT_shader_explicit_arithmetic_types_float32 : enable
// #extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable

#include "./common.glsl"
struct Octant {
    bool empty;
    uint32_t color;
    vec3 lb, rt;
    uint32_t children[8];
};

layout(binding = 0, set = 0) uniform CameraProperties 
{
	mat4 viewInverse;
	mat4 projInverse;
    vec4 controlls;
} cam;

#define MAX_DEPTH 6

#define MAX_SIZE 1000.0
#define EMPTY 0

layout(binding = 1, set = 0) readonly buffer OctTree {Octant[] pool; } oct_tree;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outDepth;

struct HitInfo {
    bool hit;
    float closest_dist;
    uint8_t color;
};

struct Ray {
    vec3 dir;
    vec3 org;
};

HitInfo bounding_box(Ray r, vec3 lb, vec3 rt) {
    HitInfo hit;
    hit.hit = false;

    // r.dir is unit direction vector of ray
    vec3 dirfrac;
    dirfrac.x = 1.0f / r.dir.x;
    dirfrac.y = 1.0f / r.dir.y;
    dirfrac.z = 1.0f / r.dir.z;
    // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
    // r.org is origin of ray
    float t1 = (lb.x - r.org.x)*dirfrac.x;
    float t2 = (rt.x - r.org.x)*dirfrac.x;
    float t3 = (lb.y - r.org.y)*dirfrac.y;
    float t4 = (rt.y - r.org.y)*dirfrac.y;
    float t5 = (lb.z - r.org.z)*dirfrac.z;
    float t6 = (rt.z - r.org.z)*dirfrac.z;

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
    if (tmax < 0)
    {
        hit.closest_dist = tmax;
        return hit;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (tmin > tmax)
    {
        hit.closest_dist = tmax;
        return hit;
    }

    hit.closest_dist = tmin;
    hit.hit = true;
    return hit;
}

HitInfo traverse(Ray ray) {
    Octant node_stack[100];
    int stack_index = 0;
    node_stack[stack_index++] = oct_tree.pool[0];
    HitInfo state;
    state.hit = false;
    state.color = uint8_t(0);
    state.closest_dist = 1.0 / 0.0;
    
    while(stack_index > 0) {
        Octant node = node_stack[--stack_index];
        HitInfo bounds_hit = bounding_box(ray, node.lb, node.rt);
        if (bounds_hit.hit && !node.empty) {
            if (node.color != 0) {
                state.hit = bounds_hit.hit;
                state.closest_dist = bounds_hit.closest_dist;
                state.color = uint8_t(node.color);
            }else {
                for(int i = 0; i < 8; i++) {
                    node_stack[stack_index++] = oct_tree.pool[node.children[i]];
                }
            }
        }
    }
    return state;
}

vec4 hex_to_decimal(uint8_t col) {
    float r = float((col >> 6) & 0xFF) / 255.0;
    float g = float((col >> 4) & 0xFF) / 255.0;
    float b = float((col >> 2) & 0xFF) / 255.0;
    float a = float(col & 0xFF) / 255.0;
    return vec4(r,g,b,a);
}

void main() {

    const vec2 pixelCenter = vec2(gl_FragCoord.xy) + vec2(0.5);
	const vec2 inUV = pixelCenter/vec2(cam.controlls.zw);
	vec2 d = inUV * 2.0 - 1.0;
	vec4 origin = cam.viewInverse * vec4(0,0,0,1);
	vec4 target = cam.projInverse * vec4(d.x, d.y, 1, 1) ;
	vec4 direction = cam.viewInverse*vec4(normalize(target.xyz), 0) ;

    Ray ray = Ray(direction.xyz, origin.xyz);

    HitInfo hit = traverse(ray);

    if (hit.hit) {
        outColor = vec4(hex_to_decimal(hit.color).rgb, 1.0);
    } else {
        outColor = direction;
    }
    outDepth = hit.closest_dist;
}