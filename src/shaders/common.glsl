
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
	bool missed;
	float metallicFactor;
  float roughness;
	vec4 color;
  vec3 emission;
	vec3 hitPoint;
	vec3 hitNormal;
};

const float tmin = 0.1;
const float tmax = 10000.0;