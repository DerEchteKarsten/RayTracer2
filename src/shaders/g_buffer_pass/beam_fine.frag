#version 450
#extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : enable

#include "g_buffer.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;  

void main() {
    const vec2 pixelCenter = vec2(gl_FragCoord.xy) + vec2(0.5);
	const vec2 inUV = pixelCenter/vec2(gl_NumWorkGroups.xy * 8);
	vec2 dir = inUV * 2.0 - 1.0;
	vec4 origin = g_Const.viewInverse * vec4(0,0,0,1);
	vec4 target = g_Const.projInverse * vec4(dir.x, dir.y, 1, 1) ;
	vec4 direction = g_Const.viewInverse*vec4(normalize(target.xyz), 0) ;
    vec3 o = origin.xyz;
    vec3 d = direction.xyz;

    HitInfo hit_info;
    float beam;
    ivec2 beam_coord = ivec2(gl_GlobalInvocationID.xy / uBeamSize);
    beam = nonuniformEXT(min(min(uintBitsToFloat(imageLoad(inputImage, beam_coord).r), uintBitsToFloat(imageLoad(inputImage, beam_coord + ivec2(1, 0)).r)),
                min(uintBitsToFloat(imageLoad(inputImage, beam_coord + ivec2(0, 1)).r),
                    uintBitsToFloat(imageLoad(inputImage, beam_coord + ivec2(1, 1)).r))));
    o += d * beam;

    bool hit = ray_cast(o, d, hit_info);
    uint face_id = 0;
    if (hit_info.normal.x > 0)
        face_id |= 0x1;
    if (hit_info.normal.y > 0)
        face_id |= 0x2;
    if (hit_info.normal.z > 0)
        face_id |= 0x4;

    out_voxel_id = (face_id << 28) | hit_info.voxel_id;

    uint rngState = uint((gl_GlobalInvocationID.y * g_Const.controlls.z) + gl_GlobalInvocationID.x) + uint(f.frame) * 23145;
    if (!hit) {
        out_voxel_id = SKYBOX;
    }else {
        gpu_hashmap_insert(out_voxel_id, f.frame, hit_info, rngState);
    }
}