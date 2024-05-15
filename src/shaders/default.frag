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

float sdfBox( vec3 p, float b )
{
  return length(max(abs(p)-vec3(b),0.0));
}

// bool get_color_or_indirection(in Octant node, in uint index, out uint32_t color_or_indirection) {
//     uint64_t data = node.data[index / 2];
//     uint32_t node_data;
//     switch (index % 2) {
//         case 0:
//             node_data = uint32_t(data >> 32); 
//         case 1:
//             node_data = uint32_t(data << 32);
//     }
//     bool is_leave = bool(node_data << 31);
//     color_or_indirection = (node_data >> 1) << 1;
//     return is_leave;
// }

// uint pos_index_to_lin(in ivec3 pos_index) {
//     return 16 * pos_index.y + 4 * pos_index.z + pos_index.x;
// }

// uint32_t get_leave_from_position(vec3 position, out Octant root, out ivec3 pos_index) {
//     root = oct_tree.pool[0];
//     uint32_t next_root;
//     for(int i == 0; i<MAX_DEPTH; i++) {
//         ivec3 pos_index = ivec3(position) / ivec3(4);
//         uint32_t index =
//         bool is_leave = get_color_or_indirection(root, index, next_root);
//         if (is_leave) {
//             return next_root;
//         } else {
//             position -= pos_index * (MAX_SIZE / (4*i));
//             root = oct_tree.pool[next_root];
//         }
//     }
//     return next_root;
// }

// float trace(vec3 direction, vec3 origin) {
//     float depth = max(sdfBox(origin, MAX_SIZE), 0.0);
//     origin += direction * (depth+0.01);
//     Octant root;
//     ivec3 pos_index;
//     uint32_t current_octant_value = get_leave_from_position(origin, root, pos_index);
//     if (current_octant_value != EMPTY) {
//         return depth;
//     }
//     for(int i = 0; i < 64; i++) {
//         uint index = pos_index_to_lin(pos_index);
//         if (index > 64) {
//             break;
//         }
//         uint32_t val;
//         get_color_or_indirection(root, index, val);
//         if (val != EMPTY) {
//             break;
//         }else {
            
//         }
//     }
// }

int closest_child(vec3 origin, float bounds, vec3 center) {
    vec3 oriented_point = origin - center;
    
    bool x_test = oriented_point.x >= 0.0;
    bool y_test = oriented_point.y >= 0.0;
    bool z_test = oriented_point.z >= 0.0;

    return x_test | (y_test << 1) | (z_test >= 0.0 << 2);
} 

bool plane_collision(vec3 origin, vec3 direction, vec3 center, vec3 normal) {
    float denom = normal.dot(direction);
    if (abs(denom) > 0.0001f) // your favorite epsilon
    {
        float t = (center - origin).dot(normal) / denom;
        if (t >= 0) return true; // you might want to allow an epsilon here too
    }
    return false;
}

bool trace(vec3 direction, vec3 origin, Octant node, int depth, vec3 center) {
    float bounds = MAX_SIZE / (depth * 2);

    float depth = sdfBox(origin, bounds);
    if(depth > 0.0) {
        return false;
    }

    if(node.color != 0) {
        return true;
    }else {
        int closest_child_index = closest_child(origin, bounds, center);

        //change to bit mask
        bool[3] plane_hits = {
            plane_collision(origin, direction, center, vec3(1.0, 0.0, 0.0)),
            plane_collision(origin, direction, center, vec3(0.0, 1.0, 0.0)),
            plane_collision(origin, direction, center, vec3(0.0, 0.0, 1.0))
        }

        for (int i = 0; i < 4; i++) {
            uint32_t child_adress = node.children[closest_child_index]; 
            if (child_adress != 0) {
                trace(direction, origin, oct_tree.pool[child_adress], depth-1, center+(((vec3((bounds) * ((closest_child_index >> 0) & 1), (bounds) * ((closest_child_index >> 1) & 1), (bounds) * ((closest_child_index >> 2) & 1))))))
            }
        }
    }
}

void main() {

    const vec2 pixelCenter = vec2(gl_FragCoord.xy) + vec2(0.5);
	const vec2 inUV = pixelCenter/vec2(cam.controlls.zw);
	vec2 d = inUV * 2.0 - 1.0;
	vec4 origin = cam.viewInverse * vec4(0,0,0,1);
	vec4 target = cam.projInverse * vec4(d.x, d.y, 1, 1) ;
	vec4 direction = cam.viewInverse*vec4(normalize(target.xyz), 0) ;

    outColor = direction;

    if(cam.controlls.x == 1.0) {
        outColor = vec4(1.0, 0.0, 1.0,1.0);
    }
    outDepth = 1.0;
}