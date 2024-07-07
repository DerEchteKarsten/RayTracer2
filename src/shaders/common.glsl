#extension GL_AMD_gpu_shader_half_float : enable
#define PI 3.1415926
#include "./oct_tree.glsl"

#define INFINITY 1.0 / 0.0

// float RandomValue(inout uint state) {
// 	float res = fract(sin(dot(vec2(state, state), vec2(12.9898, 78.233))) * 43758.5453);
// 	state = uint(res);
// 	return res / 4294967295.0;
// }

#define SKYBOX 4294967295

uint NextRandom(inout uint state) {
	state = state * 747796405 + 2891336453;
	uint result = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
	result = (result >> 22) ^ result;
	return result;
}

float RandomValue(inout uint state) {
	return NextRandom(state) / 4294967295.0;
}

float ValueNormalDistribution(float x) {
	float theta = 2 * PI * x;
	float rho = sqrt(-2 * log(x));
	return rho * cos(theta);
}

float RandomValueNormalDistribution(inout uint state) {
	float theta = 2 * 3.1415926 * RandomValue(state);
	float rho = sqrt(-2 * log(RandomValue(state)));
	return rho * cos(theta);
}

vec3 RandomDirection(inout uint state) {
	float x = RandomValueNormalDistribution(state);
	float y = RandomValueNormalDistribution(state);
	float z = RandomValueNormalDistribution(state);
	return normalize(vec3(x,y,z));
}

vec2 RandomDirection2(inout uint state) {
	float x = RandomValueNormalDistribution(state);
	float y = RandomValueNormalDistribution(state);
	return normalize(vec2(x,y));
}

float FresnelSchlickRoughness( float cosTheta, float F0, float roughness ) {
    return F0 + (max((1. - roughness), F0) - F0) * pow(abs(1. - cosTheta), 5.0);
}


vec3 modifyDirectionWithRoughness( const vec3 normal, const vec3 n, const float roughness, inout uint state ) {
    vec2 r = vec2(RandomValue(state), RandomValue(state));
    
	vec3  uu = normalize(cross(n, abs(n.y) > .5 ? vec3(1.,0.,0.) : vec3(0.,1.,0.)));
	vec3  vv = cross(uu, n);
	
    float a = roughness*roughness;
    
	float rz = sqrt(abs((1.0-r.y) / clamp(1.+(a - 1.)*r.y,.00001,1.)));
	float ra = sqrt(abs(1.-rz*rz));
	float rx = ra*cos(6.28318530718*r.x); 
	float ry = ra*sin(6.28318530718*r.x);
	vec3  rr = vec3(rx*uu + ry*vv + rz*n);
    
    vec3 ret = normalize(rr);
    return dot(ret,normal) > 0. ? ret : n;
}

float roughness_to_perceptual_roughness(float r) {
    return sqrt(r);
}

vec4 pack(in vec4 albedo, in vec4 normal, in float roughness, in float metalness, in vec4 emissive) {
    vec4 res = 0.0.xxxx;
    res.x = uintBitsToFloat(packUnorm4x8(albedo));
    res.y = uintBitsToFloat(packUnorm4x8(normal));

    vec4 roughness_metalness = vec4(roughness_to_perceptual_roughness(roughness), metalness, 0, 0);
    res.z = uintBitsToFloat(packUnorm4x8(roughness_metalness));
    res.w = uintBitsToFloat(packUnorm4x8(emissive));

   return res;
}


void unpack(in vec4 g_buffer, out vec4 albedo, out vec4 normal, out float roughness, out float metalness, out vec4 emissive) {
    albedo = unpackUnorm4x8(floatBitsToUint(g_buffer.x));
	normal = unpackUnorm4x8(floatBitsToUint(g_buffer.y));
    vec4 roughness_metalness = unpackUnorm4x8(floatBitsToUint(g_buffer.z));
	roughness = roughness_metalness.x;
	metalness = roughness_metalness.y;
    emissive = unpackUnorm4x8(floatBitsToUint(g_buffer.w));
}

float ray_plane(vec3 origin, vec3 direction, vec3 normal, vec3 center) {
    float denom = dot(normal, direction);
    if (abs(denom) > 0.0001f)
    {
        float t = dot((center - origin), normal) / denom;
        if (t >= 0) return t;
    }
    return 1.0/0.0;
}
const vec3 plane_normal = vec3(0.0, 1.0, 0.0);


struct HitInfo {
	vec3 pos;
	vec3 normal;
	vec3 color;
	uint voxel_id;
	float depth;
};

bool ray_cast(in vec3 org, in vec3 dir, out HitInfo hit_info) {
	vec3 o_pos, o_color, o_normal;
	uint o_voxel_id;
	bool hit = Octree_RayMarchLeaf(org, dir, o_pos, o_color, o_normal, o_voxel_id);
	float hit_depth = length(org - o_pos);
    // float plane_depth = ray_plane(org, dir, plane_normal, vec3(0.0, 0.99999, 0.0));

	// float depth = min(hit ? hit_depth : INFINITY, plane_depth);

    // if (depth == plane_depth) {
    //     o_pos = org + dir * plane_depth;
    //     o_normal = plane_normal;
    //     o_color = vec3(min(max(plane_depth / 100.0, 0.3), 0.8));
	// 	o_voxel_id = PLANE;
    // }
	hit_info = HitInfo(o_pos, o_normal, o_color, o_voxel_id, hit_depth);
	return hit;
}



mat3 angleAxis3x3(float angle, vec3 axis)
{
    float c, s;
    s = sin(angle);
    c = cos(angle);

    float t = 1 - c;
    float x = axis.x;
    float y = axis.y;
    float z = axis.z;

    return mat3(
        t * x * x + c,      t * x * y - s * z,  t * x * z + s * y,
        t * x * y + s * z,  t * y * y + c,      t * y * z - s * x,
        t * x * z - s * y,  t * y * z + s * x,  t * z * z + c
    );
}

vec3 getConeSample(inout uint randSeed, vec3 direction, float coneAngle) {
    float cosAngle = cos(coneAngle);

    // Generate points on the spherical cap around the north pole [1].
    // [1] See https://math.stackexchange.com/a/205589/81266
    float z = RandomValue(randSeed) * (1.0f - cosAngle) + cosAngle;
    float phi = RandomValue(randSeed) * 2.0f * PI;

    float x = sqrt(1.0f - z * z) * cos(phi);
    float y = sqrt(1.0f - z * z) * sin(phi);
    vec3 north = vec3(0.f, 0.f, 1.f);

    // Find the rotation axis `u` and rotation angle `rot` [1]
    vec3 axis = normalize(cross(north, normalize(direction)));
    float angle = acos(dot(normalize(direction), north));

    // Convert rotation axis and angle to 3x3 rotation matrix [2]
    mat3 R = angleAxis3x3(angle, axis);

    return vec3(x, y, z) * R;
}
