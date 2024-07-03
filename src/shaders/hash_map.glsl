#define kEmpty 0
#define vEmpty RISReservoir(0, 0.0, 0.0, 0.0)
// #define vEmpty GIReservoir(Sample(vec3(0), vec3(0), vec3(0), vec3(0), vec3(0)), 0.0, 0, 0.0)
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
void gpu_hashmap_insert(uint key, RISReservoir qn, inout uint rngState)
{
    uint slot = mod_u32(hash(key), (khashmapCapacity-1));
    while (true)
    {
        uint prev = atomicCompSwap(keys[slot], kEmpty, key);
        if (prev == kEmpty || keys[slot] == key) {
            if (values[slot] == vEmpty) {
               values[slot] = qn;
            //    total_sampels[slot] = 1;
            } else {
                // GIReservoir q = values[slot];
                // float Jacobian = evalJacobian(qn.z.sampled_point, qn.z.sampled_normal, q.z.primary_point, qn.z.primary_point);
                // float p_hat = length(qn.z.radiance_sampled) / Jacobian;
                // Merge(q, qn, p_hat, rngState);
                // if(length(q.z.radiance_sampled) > EPS) {
                //     atomicAdd(total_sampels[slot], qn.M);
                // }
                // q.W = q.w/(total_sampels[slot]*length(q.z.radiance_sampled));
                // values[slot] = q;
                values[slot] = qn;
            }
            return;
        }
        slot = mod_u32(slot +1, (khashmapCapacity-1));
    }
    // debugPrintfEXT("full");
}

bool gpu_hashmap_get(uint key, out RISReservoir reservoir)
{
    uint slot = mod_u32(hash(key), (khashmapCapacity-1));
    while (true)
    {
        if (keys[slot] == key) {
            reservoir = values[slot];
            return true;
        }else if (keys[slot] == kEmpty) {
            return false;
        }
        slot = mod_u32(slot +1, (khashmapCapacity-1));
    }
}