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

#define MAX_DEPTH 6

#define MAX_SIZE 1000.0
#define EMPTY 0

layout(std430, binding = 1, set = 0) readonly buffer OctTree {Octant[] pool; } oct_tree;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outDepth;

struct HitInfo {
    bool hit;
    float closest_dist;
    vec3 color;
};

struct Ray {
    vec3 dir;
    vec3 org;
};

HitInfo bounding_box(Ray r, vec3 pos, vec3 bounds) {

    vec3 lb = vec3(pos.x, pos.y, pos.z);
    vec3 rt = vec3(pos.x + bounds.x, pos.y + bounds.y, pos.z + bounds.z);
    HitInfo hit = HitInfo(false, 1.0/0.0, vec3(0.0));

    vec3 dirfrac;
    dirfrac.x = 1.0f / r.dir.x;
    dirfrac.y = 1.0f / r.dir.y;
    dirfrac.z = 1.0f / r.dir.z;

    float t1 = (lb.x - r.org.x)*dirfrac.x;
    float t2 = (rt.x - r.org.x)*dirfrac.x;
    float t3 = (lb.y - r.org.y)*dirfrac.y;
    float t4 = (rt.y - r.org.y)*dirfrac.y;
    float t5 = (lb.z - r.org.z)*dirfrac.z;
    float t6 = (rt.z - r.org.z)*dirfrac.z;

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    if (tmax < 0) {
        hit.closest_dist = tmax;
        return hit;
    }

    if (tmin > tmax) {
        hit.closest_dist = tmax;
        return hit;
    }

    hit.closest_dist = tmin;
    hit.hit = true;
    return hit;
}

int closest_child(vec3 point, vec3 node_center) {
    vec3 oriented_point = point - node_center;
    int x_test = int(oriented_point.x >= 0.0);
    int y_test = int(oriented_point.y >= 0.0);
    int z_test = int(oriented_point.z >= 0.0);
    return x_test | (y_test << 1) | (z_test << 2); 
}

bool box_point(vec3 point, vec3 min, vec3 max) {
  return (
    point.x >= min.x &&
    point.x <= max.x &&
    point.y >= min.y &&
    point.y <= max.y &&
    point.z >= min.z &&
    point.z <= max.z
  );
}

HitInfo ray_plane(Ray ray, vec3 normal, vec3 center, float side_length) {
    HitInfo hit = HitInfo(false, 1.0/0.0, vec3(0.0));
    float denominator = dot(normal, ray.dir);
    
    if (abs(denominator) < 1e-6) {
        return hit;
    }
    
    float d = dot(normal, (center - ray.org)) / denominator;
    if (d < 0.0) {
        return hit;
    }
    
    vec3 intersection_point = ray.org + d * ray.dir;
    
    vec3 plane_center_to_point = intersection_point - center;
    vec3 u = plane_center_to_point - dot(plane_center_to_point, normal) * normal;
    
    if (all(lessThanEqual(abs(u), vec3(side_length / 2.0)))) {
        hit.hit = true;
        hit.closest_dist = d;
        return hit;
    } else {
        return hit;
    }
}

int KthBit(int n, int k)
{
    return ((n & (1 << (k - 1))) >> (k - 1));
}

vec3 calc_new_pos(vec3 pos, int i, float move) {
    float x_test = KthBit(i, 1) == 1 ? 1.0 : -1.0;
    float y_test = KthBit(i, 2) == 1 ? 1.0 : -1.0;
    float z_test = KthBit(i, 3) == 1 ? 1.0 : -1.0;
    return vec3(pos.x + move * x_test, pos.y + move * y_test, pos.z + move * z_test);
}


vec4 hex_to_decimal(uint32_t col) {
    float r = float((col) & 0xFF) / 255.0;
    float g = float((col >> 8) & 0xFF) / 255.0;
    float b = float((col >> 16) & 0xFF) / 255.0;
    float a = float((col >> 24) & 0xFF) / 255.0;
    return vec4(r, g, b, a);
}

HitInfo traverse(Ray ray) {
    Octant node_stack[100];
    vec4 bounds_stack[100];
    int stack_index = 0;
    int bounds_index = 0;
    node_stack[stack_index++] = oct_tree.pool[0];
    bounds_stack[bounds_index++] = vec4(0.0, 0.0, 0.0, 4.0);
    HitInfo state;
    state.hit = false;
    state.color = vec3(0);
    state.closest_dist = 1.0 / 0.0;
    HitInfo bounds_hit = bounding_box(ray, vec3(0), vec3(4));

    while(stack_index > 0) {
        Octant node = node_stack[--stack_index];
        vec4 pos_b = bounds_stack[--bounds_index];
        float bounds = pos_b.a;
        vec3 pos = pos_b.xyz;
        HitInfo bounds_hit = bounding_box(ray, pos, vec3(bounds));
        if (bounds_hit.hit && node.empty_color != 0 && state.closest_dist > bounds_hit.closest_dist) {
            bool leaf = true;
            for(int i = 0; i < 8; i++) {
                leaf = leaf && (node.children[i] == 0); 
            }
            if (leaf) {
                state.hit = bounds_hit.hit;
                state.closest_dist = bounds_hit.closest_dist;
                state.color = hex_to_decimal(node.empty_color).rgb;
            }else {
                for(int i = 0; i < 8; i++) {
                    node_stack[stack_index++] = oct_tree.pool[uint32_t(node.children[i])];
                    vec3 center = pos + bounds/4.0;
                    bounds_stack[bounds_index++] = vec4(calc_new_pos(center, i, bounds/4.0), bounds / 2.0);
                }
            }
        }
    }
    return state;
}


int closest_plane_hit(HitInfo plane_hits[3]) {
    int closest = 0;
    for (int i = 0; i < 3; i++) {
        if (((plane_hits[i].closest_dist < plane_hits[closest].closest_dist) && plane_hits[i].hit == true) || plane_hits[closest].hit == false) {
            closest = i;
        }
    }
    if (!plane_hits[closest].hit) {
        return -1;
    }else {
        return closest;
    }
}

HitInfo traverse2(Ray ray) {
    Octant node_stack[200];
    vec4 bounds_stack[200];
    HitInfo plane_hits[3];
    int stack_index = 0;
    int bounds_index = 0;
    node_stack[stack_index++] = oct_tree.pool[0];
    bounds_stack[bounds_index++] = vec4(0.0, 0.0, 0.0, 4.0);
    HitInfo state = HitInfo(false, 1.0/0.0, vec3(0.0));
    
    while(stack_index > 0) {
        Octant node = node_stack[--stack_index];
        vec4 pos_b = bounds_stack[--bounds_index];
        float bounds = pos_b.a;
        vec3 pos = pos_b.xyz;

        HitInfo bounds_hit = bounding_box(ray, pos, vec3(bounds));
        if(!bounds_hit.hit || state.closest_dist < bounds_hit.closest_dist) {
            continue;
        } 

        bool leaf = true;
        for(int i = 0; i < 8; i++) {
            leaf = leaf && (node.children[i] == 0); 
        }
        if (leaf) {
            state.hit = true;
            state.closest_dist = bounds_hit.closest_dist;
            state.color = hex_to_decimal(node.empty_color).rgb;
        }else {
            vec3 center = pos + vec3(bounds/2);
            int closest_node_index = closest_child(ray.org + ray.dir * bounds_hit.closest_dist, center);

            plane_hits[0] = ray_plane(ray, vec3(1.0, 0.0, 0.0), center, bounds);
            plane_hits[1] = ray_plane(ray, vec3(0.0, 1.0, 0.0), center, bounds);
            plane_hits[2] = ray_plane(ray, vec3(0.0, 0.0, 1.0), center, bounds);

            for(int i = 0; i < 4; ++i) {
                uint32_t child_node = node.children[closest_node_index];
                Octant child = oct_tree.pool[child_node];
                if(child.empty_color != 0) {
                    node_stack[stack_index++] = child;
                    bounds_stack[bounds_index++] = vec4(calc_new_pos(pos + vec3(bounds/4), closest_node_index, bounds/4.0), bounds / 2.0);
                }
                if (state.hit) {
                    break;
                }
                int plane_index = closest_plane_hit(plane_hits);
                if (plane_index == -1) {
                    break;
                }
                closest_node_index ^= 0x1 << plane_index;
                plane_hits[plane_index].hit = false;
            }
        }
    }
    return state;
}

void main() {

    const vec2 pixelCenter = vec2(gl_FragCoord.xy) + vec2(0.5);
	const vec2 inUV = pixelCenter/vec2(cam.controlls.zw);
	vec2 d = inUV * 2.0 - 1.0;
	vec4 origin = cam.viewInverse * vec4(0,0,0,1);
	vec4 target = cam.projInverse * vec4(d.x, d.y, 1, 1) ;
	vec4 direction = cam.viewInverse*vec4(normalize(target.xyz), 0) ;
    // if (gl_FragCoord.x == 0.5 && gl_FragCoord.y == 0.5)
    // {for(int i = 0; i < 20; i++) {
    //     Octant node = oct_tree.pool[i];
    //     if (node.empty_color != 0) {
    //         continue;
    //     }
    //     debugPrintfEXT("children: %u, %u, %u, %u, %u, %u, %u, %u,", 
    //         node.children[0].x,
    //         node.children[1].x,
    //         node.children[2].x,
    //         node.children[3].x,
    //         node.children[4].x,
    //         node.children[5].x,
    //         node.children[6].x,
    //         node.children[7].x);
    //     debugPrintfEXT("color: %u", node.empty_color);
    //     debugPrintfEXT("i: %d", i);
    // }}

    Ray ray = Ray(direction.xyz, origin.xyz);

    HitInfo hit = traverse2(ray);

    if (hit.hit) {
        outColor = vec4(hit.color, 1.0);
    } else {
        outColor = direction;
    }
    outDepth = hit.closest_dist;
}