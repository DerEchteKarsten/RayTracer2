#extension GL_AMD_gpu_shader_half_float : enable
#define square(a) a*a
const float kMinRoughness = 0.05f;

struct Vertex {
  vec3 pos;
  vec3 normal;
  vec3 color;
  vec2 uvs;
};

struct GeometryInfo {
  mat4 transform;
  vec4 baseColor;
  int baseColorTextureIndex;
  float metallicFactor;
  uint indexOffset;
  uint vertexOffset;
  vec4 emission;
  float roughness;
};

struct Payload {
	float depth;
	vec2 uv;
	uint geometryIndex;
  	uint primitiveId;
};


#define PI 3.1415926

// float RandomValue(inout uint state) {
// 	float res = fract(sin(dot(vec2(state, state), vec2(12.9898, 78.233))) * 43758.5453);
// 	state = uint(res);
// 	return res / 4294967295.0;
// }

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