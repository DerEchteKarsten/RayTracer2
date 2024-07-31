#define kEmpty 0
#define MAX_AGE 100

#extension GL_EXT_shader_atomic_float : require


uint hash(uint x)
{
    x ^= x >> 17;
    x *= 0xed5ad4bbU;
    x ^= x >> 11;
    x *= 0xac4c1b51U;
    x ^= x >> 15;
    x *= 0x31848babU;
    x ^= x >> 14;
    return x;
}


uint mod_u32( uint u32_bas , uint u32_div ){

    float   flt_res =  mod( float(u32_bas), float(u32_div));
    uint    u32_res = uint( flt_res );
    return( u32_res );
}

float evalJacobian(vec3 secondary, vec3 secondary_normal, vec3 primary_one, vec3 primary_tow)
{
    vec3 dir1 = primary_one - secondary;
    vec3 dir2 = primary_tow - secondary; 

    float theta_one = dot(dir1, secondary_normal);
    float theta_tow = dot(dir2, secondary_normal);

    float denom1 = cos(theta_tow);
    float denom2 = pow(length(dir1), 2);

    denom1 == 0 ? EPS : denom1;
    denom2 == 0 ? EPS : denom2;

    return (cos(theta_one) / denom1) * (pow(length(dir2), 2) / denom2);
}
const vec3 light_dir = vec3(0.6123724, 0.6123724, -0.50000006);
const float LightCone = 0.3;

void gpu_hashmap_insert(uint key, uint64_t now, HitInfo hit_info, inout uint rngState) {
    uint slot = mod_u32(hash(key), (khashmapCapacity-1));
    while(true)
    {
        if (atomicCompSwap(keys[slot], kEmpty, key) == kEmpty || keys[slot] == key) {
            // last_seen[slot] = now;
            // uint prev = atomicAdd(total_sampels[slot], 1);
            // bool is_direct_sample = prev < 300; 
            // if (prev > 10 && length(vec3(values[slot])) < 200) {
            //     is_direct_sample = false;
            // }

            // vec3 radiance = vec3(0.0);
            // vec3 dir;
            // if(is_direct_sample) {
            //     dir = getConeSample(rngState, light_dir, LightCone);
            // }else {
            //     dir = RandomDirection(rngState);
            //     dir *= sign(dot(hit_info.normal, dir));
            // }

            // vec3 color = hit_info.color;
            // HitInfo hit_info2;
            // bool hit = ray_cast(hit_info.pos + hit_info.normal * 0.0001, dir, hit_info2);
            // float pdf = is_direct_sample == true ? (5.0/PI) : (1.0/(2.*PI));
            
            // if (!hit) {
            //     if(dot(dir, light_dir) > 0.94 || is_direct_sample) {
            //         radiance += vec3(10.0) * (color / PI) * dot(hit_info.normal, dir) * (1.0 / pdf); 
            //     }else {
            //         radiance += vec3(0.5) * (color / PI) * dot(hit_info.normal, dir) * (1.0 / pdf); //texture(skybox, uv).rgb * color;
            //     }
            // }else {
            //     vec3 dir2 = normalize(hit_info2.normal + RandomDirection(rngState));

            //     color *= hit_info2.color;
            //     HitInfo hit_info3;
            //     hit = ray_cast(hit_info2.pos + hit_info2.normal * 0.0001, dir2, hit_info3);
            //     if(!hit) {
            //         if(dot(dir2, light_dir) > 0.94) {
            //             radiance += vec3(10.0) * color; 
            //         }else {
            //             radiance += vec3(0.5) * color;
            //         }
            //     }
            // }

            values[slot] = ivec3(hit_info.color * 100.0);
            // atomicAdd(values[slot].r, int(radiance.r * 100.0));
            // atomicAdd(values[slot].g, int(radiance.g * 100.0));
            // atomicAdd(values[slot].b, int(radiance.b * 100.0));
            return;

        }

        slot = mod_u32(slot +1, (khashmapCapacity-1));
    }
    // debugPrintfEXT("full");
}

bool gpu_hashmap_get(uint key, uint64_t now, out vec3 radiance, out uint M)
{
    uint slot = mod_u32(hash(key), (khashmapCapacity-1));
    while (true)
    {
        if (keys[slot] == key) {
            radiance = vec3(values[slot]) / 100.0;
            M = total_sampels[slot];
            last_seen[slot] = now;
            return true;
        }else if (keys[slot] == kEmpty) {
            return false;
        }
        slot = mod_u32(slot +1, (khashmapCapacity-1));
    }
}