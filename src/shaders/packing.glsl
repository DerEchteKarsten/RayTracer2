#define RTXDI_GLSL
#include "rtxdi/RtxdiTypes.h"
#define PACK_UFLOAT_TEMPLATE(size)                      \
uint Pack_R ## size ## _UFLOAT(float r) \
{                                                       \
     float d = 0.5;                                     \
    const uint mask = (1U << size) - 1U;                \
                                                        \
    return uint(floor(r * mask + d)) & mask;            \
}                                                       \
                                                        \
float Unpack_R ## size ## _UFLOAT(uint r)               \
{                                                       \
    const uint mask = (1U << size) - 1U;                \
                                                        \
    return float(r & mask) / float(mask);             \
}

PACK_UFLOAT_TEMPLATE(8)
PACK_UFLOAT_TEMPLATE(10)
PACK_UFLOAT_TEMPLATE(11)
PACK_UFLOAT_TEMPLATE(16)


#define PACK_UFLOAT_TEMPLATE_OVERLOAD(size)                      \
uint Pack_R ## size ## _UFLOAT(float r, float d) \
{                                                        \
    const uint mask = (1U << size) - 1U;                \
                                                        \
    return uint(floor(r * mask + d)) & mask;            \
}                                                       \

PACK_UFLOAT_TEMPLATE_OVERLOAD(8)
PACK_UFLOAT_TEMPLATE_OVERLOAD(10)
PACK_UFLOAT_TEMPLATE_OVERLOAD(11)
PACK_UFLOAT_TEMPLATE_OVERLOAD(16)

vec3 Unpack_R11G11B10_UFLOAT(uint rgb)
{
    float r = Unpack_R11_UFLOAT(rgb);
    float g = Unpack_R11_UFLOAT(rgb >> 11);
    float b = Unpack_R10_UFLOAT(rgb >> 22);
    return vec3(r, g, b);
}

uint Pack_R11G11B10_UFLOAT(vec3 rgb)
{
    vec3 d = vec3(0.5f, 0.5f, 0.5f);
    uint r = Pack_R11_UFLOAT(rgb.r, d.r);
    uint g = Pack_R11_UFLOAT(rgb.g, d.g) << 11;
    uint b = Pack_R10_UFLOAT(rgb.b, d.b) << 22;
    return r | g | b;
}


uint Pack_R8G8B8A8_Gamma_UFLOAT(vec4 rgba)
{

    float gamma = 2.2; vec4 d = vec4(0.5f, 0.5f, 0.5f, 0.5f);
    rgba = pow(clamp(rgba, vec4(0), vec4(1.0)), vec4(1.0 / gamma));
    uint r = Pack_R8_UFLOAT(rgba.r, d.r);
    uint g = Pack_R8_UFLOAT(rgba.g, d.g) << 8;
    uint b = Pack_R8_UFLOAT(rgba.b, d.b) << 16;
    uint a = Pack_R8_UFLOAT(rgba.a, d.a) << 24;
    return r | g | b | a;
}


vec4 Unpack_R8G8B8A8_Gamma_UFLOAT(uint rgba)
{
    float gamma = 2.2;
    float r = Unpack_R8_UFLOAT(rgba);
    float g = Unpack_R8_UFLOAT(rgba >> 8);
    float b = Unpack_R8_UFLOAT(rgba >> 16);
    float a = Unpack_R8_UFLOAT(rgba >> 24);
    vec4 v = vec4(r, g, b, a);
    v = pow(saturate(v), vec4(gamma));
    return v;
}

uint Pack_R8G8B8A8_Gamma_UFLOAT(vec4 rgba, float gamma, vec4 d)
{
    rgba = pow(saturate(rgba), vec4(1.0 / gamma));
    uint r = Pack_R8_UFLOAT(rgba.r, d.r);
    uint g = Pack_R8_UFLOAT(rgba.g, d.g) << 8;
    uint b = Pack_R8_UFLOAT(rgba.b, d.b) << 16;
    uint a = Pack_R8_UFLOAT(rgba.a, d.a) << 24;
    return r | g | b | a;
}


uint Pack_R16G16_FLOAT(vec2 rg)
{
    uint r = f32tof16(rg.r);
    uint g = f32tof16(rg.g) << 16;
    return r | g;
}

uvec2 Pack_R16G16B16A16_FLOAT(vec4 rgba)
{
    return uvec2(Pack_R16G16_FLOAT(rgba.rg), Pack_R16G16_FLOAT(rgba.ba));
}

vec2 Unpack_R16G16_FLOAT(uint rg)
{
    uvec2 d = uvec2(rg, rg >> 16);
    return unpackHalf2x16(d.x);
}

vec4 Unpack_R16G16B16A16_FLOAT(uvec2 rgba)
{
    return vec4(Unpack_R16G16_FLOAT(rgba.x), Unpack_R16G16_FLOAT(rgba.y));
}


vec3 octToNdirSigned(vec2 p)
{
    // https://twitter.com/Stubbesaurus/status/937994790553227264
    vec3 n = vec3(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    float t = max(0, -n.z);

    n.xy += (n.x >= 0.0 || n.y >= 0.0) ? -t : t;
    return normalize(n);
}

vec3 octToNdirUnorm32(uint pUnorm)
{
    vec2 p;
    p.x = clamp(float(pUnorm & 0xffff) / float(0xfffel), 0, 1);
    p.y = clamp(float(pUnorm >> 16) / float(0xfffel), 0, 1);
    p = p * 2.0 - 1.0;
    return octToNdirSigned(p);
}
