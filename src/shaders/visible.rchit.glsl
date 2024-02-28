#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : enable

layout(location = 0) rayPayloadInEXT VisibilityCheck {
    float depth;
    bool missed;
} p;

void main() {
    if (gl_HitTEXT == p.depth) {
        p.missed = false;
    } else {
        p.missed = true;
    }
}