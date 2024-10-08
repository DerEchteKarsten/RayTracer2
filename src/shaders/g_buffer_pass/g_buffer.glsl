#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

layout (binding = 0, set = 1, r32ui) uniform uimage2D inputImage;

layout(binding = 0, set = 0) readonly buffer uuOctree { uint uOctree[]; };
layout(binding = 1, set = 0) buffer uHashMapBuffer { uint[khashmapCapacity] keys; ivec3[khashmapCapacity] values; uint[khashmapCapacity] total_sampels; uint64_t[khashmapCapacity] last_seen;};
layout(binding = 2, set = 0) uniform GConst g_Const;

#include "oct_tree.glsl"
#include "common.glsl"
#include "hash_map.glsl"
