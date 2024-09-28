#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_tracing : require

#include "../common.glsl"

layout(location = 0) rayPayloadInEXT Payload p;

#include "RtxdiApplicationBridge.glsl"

//layout(location = 1) rayPayloadEXT bool isShadowed;
hitAttributeEXT vec2 attribs;

void main()
{
  GeometryInfo geometryInfo = geometryInfos.g[nonuniformEXT(gl_GeometryIndexEXT)];

  uint vertexOffset = geometryInfo.vertexOffset;
  uint indexOffset = geometryInfo.indexOffset + (3 * gl_PrimitiveID);

  uint i0 = vertexOffset + indices.i[nonuniformEXT(indexOffset)];
  uint i1 = vertexOffset + indices.i[nonuniformEXT(indexOffset + 1)];
  uint i2 = vertexOffset + indices.i[nonuniformEXT(indexOffset + 2)];

  Vertex v0 = vertices.v[nonuniformEXT(i0)];
	Vertex v1 = vertices.v[nonuniformEXT(i1)];
	Vertex v2 = vertices.v[nonuniformEXT(i2)];

	const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
	vec3 normal = normalize(v0.normal * barycentricCoords.x + v1.normal * barycentricCoords.y + v2.normal * barycentricCoords.z);
  normal = normalize(geometryInfo.transform * vec4(normal, 0.0)).xyz;

  if (dot(normal, gl_WorldRayDirectionEXT) > 0) {
    normal *= -1;
  }

  vec2 uvs = v0.uvs * barycentricCoords.x + v1.uvs * barycentricCoords.y + v2.uvs * barycentricCoords.z;

  vec3 vertexColor = v0.color * barycentricCoords.x + v1.color * barycentricCoords.y + v2.color * barycentricCoords.z;
  vec3 baseColor = geometryInfo.baseColor.xyz;
  vec4 color = vec4(baseColor * vertexColor, 1.0);

  if (geometryInfo.baseColorTextureIndex > -1) {
    color = color * texture(nonuniformEXT(textures[geometryInfo.baseColorTextureIndex]), uvs);
  }

  vec3 origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

  p.missed = false;
  p.metallicFactor = geometryInfo.metallicFactor;
  p.roughness = geometryInfo.roughness;
  p.hitPoint = origin;
  p.hitNormal = normal;
  p.emission = geometryInfo.emission.xyz;
  p.color = color;
  p.depth = gl_HitTEXT;
  p.uv = uvs;
  p.geometryIndex = gl_GeometryIndexEXT;
  p.primitiveId = gl_PrimitiveID;
}