
void GetGeometryFromHit(
    in uint primitiveID,
    in uint geometryIndex,
    in vec2 attribs,
    out vec3 normal,
    out vec3 specularF0,
    out float roughness,
    out vec3 color,
    out vec3 emission
) {
    GeometryInfo geometryInfo = GeometryInfos[nonuniformEXT(geometryIndex)];

    uint vertexOffset = geometryInfo.vertexOffset;
    uint indexOffset = geometryInfo.indexOffset + (3 * primitiveID);

    uint i0 = vertexOffset + Indices[nonuniformEXT(indexOffset)];
    uint i1 = vertexOffset + Indices[nonuniformEXT(indexOffset + 1)];
    uint i2 = vertexOffset + Indices[nonuniformEXT(indexOffset + 2)];

    Vertex v0 = Vertices[nonuniformEXT(i0)];
    Vertex v1 = Vertices[nonuniformEXT(i1)];
    Vertex v2 = Vertices[nonuniformEXT(i2)];

    const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    normal = normalize(v0.normal * barycentricCoords.x + v1.normal * barycentricCoords.y + v2.normal * barycentricCoords.z);
    normal = normalize(geometryInfo.transform * vec4(normal, 0.0)).xyz;

    vec2 uvs = v0.uvs * barycentricCoords.x + v1.uvs * barycentricCoords.y + v2.uvs * barycentricCoords.z;

    vec3 vertexColor = v0.color * barycentricCoords.x + v1.color * barycentricCoords.y + v2.color * barycentricCoords.z;
    vec3 baseColor = geometryInfo.baseColor.xyz;
    color = baseColor * vertexColor;

    if (geometryInfo.baseColorTextureIndex > -1) {
        color = color * texture(nonuniformEXT(textures[geometryInfo.baseColorTextureIndex]), uvs).rgb;
    }

    specularF0 = mix(vec3(0.0), color, geometryInfo.metallicFactor);
    roughness = 1.0;//geometryInfo.roughness;
    emission = geometryInfo.emission.xyz * 12.0;
}

RAB_Surface GetSurface(RayDesc ray, out vec3 emission) {
    trace(ray);

    RAB_Surface surface = RAB_EmptySurface();

    surface.viewDepth = p.depth;
    if(surface.viewDepth == BACKGROUND_DEPTH){
        emission = GetEnvironmentRadiance(ray.Direction);
        return surface;
    }

    GetGeometryFromHit(
        p.primitiveId, 
        p.geometryIndex,
        p.uv, 
        surface.normal,
        surface.specularF0,
        surface.roughness,
        surface.diffuseAlbedo,
        emission
    );

    surface.geoNormal = surface.normal;
    surface.worldPos = ray.Origin + ray.Direction * p.depth;
    surface.viewDir = ray.Direction;
    surface.diffuseProbability = getSurfaceDiffuseProbability(surface);
    return surface;
}