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


struct Octant {
    uint32_t children[8];
    uint32_t empty_color;
};

struct Gizzmo {
    vec4 color;
    vec3 pos;
    float radius;
};

layout(binding = 1, set = 0) uniform CameraProperties 
{
	mat4 viewInverse;
	mat4 projInverse;
    vec4 controlls;
} cam;
#define khashmapCapacity 100000

layout(binding = 2, set = 0) uniform sampler2D skybox;
layout(binding = 0, set = 1) buffer uHashMapBuffer { uint[khashmapCapacity] keys; ivec3[khashmapCapacity] values; uint[khashmapCapacity] total_sampels; };
layout(std430, set = 0, binding = 3) readonly buffer uGizzmoBuffer { Gizzmo GizzmoBuffer[]; };

layout( push_constant ) uniform Frame {
	uint frame;
} f;

layout(location = 0) out uint out_voxel_id;
#include "./common.glsl"
#include "./restir.glsl"
#include "./hash_map.glsl"


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

    // uint index = uint(gl_FragCoord.y) * uint(cam.controlls.z) + uint(gl_FragCoord.x);
    // if (index < khashmapCapacity && hashmap[index].key != kEmpty) {
    //     // debugPrintfEXT("not empty, %i", hashmap[index].key);
    // }

    // for(int i = 0; i<10; i++) {
    //     Gizzmo current = GizzmoBuffer[i];
    //     if(current.radius < 0.0) {
    //         continue;
    //     }

    //     if(raySphereIntersect(origin.xyz, direction.xyz, current.pos, current.radius) != -1) {
    //         outColor = current.color;
    //         outDepth = 1.0;
    //         return;
    //     }
    // }

    HitInfo hit_info;
    bool hit = ray_cast(origin.xyz, direction.xyz, hit_info);
    uint face_id = 0;
    if (hit_info.normal.x > 0)
        face_id |= 0x1;
    if (hit_info.normal.y > 0)
        face_id |= 0x2;
    if (hit_info.normal.z > 0)
        face_id |= 0x4;

    out_voxel_id = (face_id << 28) | hit_info.voxel_id;


    if (!hit) {
        out_voxel_id = SKYBOX;//texture(skybox, vec2(u, v));
    }else {
        vec3 radiance = vec3(0.0);
        vec3 color = hit_info.color;

        uint rngState = uint((gl_FragCoord.y * cam.controlls.z) + gl_FragCoord.x) + uint(f.frame) * 23145;
        radiance += RIS(hit_info.pos, hit_info.normal, color, rngState);

        // vec3 dir = normalize(hit_info.normal + RandomDirection(rngState));
        // HitInfo hit_info2;
        // hit = ray_cast(hit_info.pos + hit_info.normal * 0.0001, dir, hit_info2);

        // if (!hit) {
        //     vec2 uv = get_uv(dir);
        //     radiance += texture(skybox, uv).rgb;
        // }else {
        //     color *= hit_info2.color;

        //     radiance += vec3(0.0);//RIS(hit_info2.pos, hit_info2.normal, color, rngState);
        // }

        gpu_hashmap_insert(out_voxel_id, radiance);
    }

    // outColor = vec4(, 0.0);
}