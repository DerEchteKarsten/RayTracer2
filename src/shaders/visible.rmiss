#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive: enable

#include "./common.glsl"

layout(location = 0) rayPayloadInEXT VisibilityCheck {
    float depth;
    bool visible;
} p;

void main() {
    p.visible = false;
}