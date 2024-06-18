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

#define MAX_DEPTH 6

#define MAX_SIZE 1000.0
#define EMPTY 0

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outDepth;

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

    vec3 o_pos;
    vec3 o_color;
    vec3 o_normal;
    bool hit = Octree_RayMarchLeaf(origin.xyz, direction.xyz, o_pos, o_color, o_normal);

    if (hit) {
        outColor = vec4(o_color, 1.0);
    } else {
        outColor = direction;
    }
    outDepth = 1.0f;
}