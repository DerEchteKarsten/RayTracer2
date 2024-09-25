#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive: enable

layout(location = 0) rayPayloadInEXT VisibilityCheck {
    bool missed;
} p;

void main() {
    p.missed = false;
}