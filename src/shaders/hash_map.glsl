#define kEmpty 0
#define MAX_AGE 10000
#define MAX_ACCUM 100000

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

void gpu_hashmap_insert(uint key, uint64_t now, vec3 radiance)
{
    uint slot = mod_u32(hash(key), (khashmapCapacity-1));
    while (true)
    {
        bool insert = false;
        if(now - last_seen[slot] >= MAX_AGE ) {
            keys[slot] = key;
            values[slot] = ivec3(0);
            last_seen[slot] = now;
            total_sampels[slot] = 0;
            insert = true;
        }else {
            insert = atomicCompSwap(keys[slot], kEmpty, key) == kEmpty || keys[slot] == key;
        }
        
        if (insert) {
            last_seen[slot] = now;

            atomicAdd(values[slot].r, int(radiance.r * 100));
            atomicAdd(values[slot].g, int(radiance.g * 100));
            atomicAdd(values[slot].b, int(radiance.b * 100));

            atomicAdd(total_sampels[slot], 1);
            return;
        }

        slot = mod_u32(slot +1, (khashmapCapacity-1));
    }
    // debugPrintfEXT("full");
}

bool gpu_hashmap_get(uint key, uint64_t now, out vec3 radiance, out uint total_sampel)
{
    uint slot = mod_u32(hash(key), (khashmapCapacity-1));
    while (true)
    {
        if (keys[slot] == key) {
            radiance = values[slot] / 100.0;
            total_sampel = total_sampels[slot];
            last_seen[slot] = now;
            return true;
        }else if (keys[slot] == kEmpty) {
            return false;
        }
        slot = mod_u32(slot +1, (khashmapCapacity-1));
    }
}