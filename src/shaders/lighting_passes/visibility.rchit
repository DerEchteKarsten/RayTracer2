#version 460

#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadInEXT VisibilityCheck {
    bool missed;
    uint geometryIndex;
  	uint primitiveId;
} p;

void main() {
    p.missed = false;
    p.geometryIndex = gl_GeometryIndexEXT;
    p.primitiveId = gl_PrimitiveID;
}