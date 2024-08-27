#ifndef SHADER_PARAMETERS_H
#define SHADER_PARAMETERS_H

#include "rtxdi/ReSTIRDIParameters.h"
#include "rtxdi/ReSTIRGIParameters.h"

#define RTXDI_GRID_BUILD_GROUP_SIZE 256
#define RTXDI_SCREEN_SPACE_GROUP_SIZE 8

#define INSTANCE_MASK_OPAQUE 0x01
#define INSTANCE_MASK_ALPHA_TESTED 0x02
#define INSTANCE_MASK_TRANSPARENT 0x04
#define INSTANCE_MASK_ALL 0xFF

#define BACKGROUND_DEPTH 1000000000000.f

struct PlanarViewConstants
{
    mat4    matWorldToView;
    mat4    matViewToClip;
    mat4    matWorldToClip;
    mat4    matClipToView;
    mat4    matViewToWorld;
    mat4    matClipToWorld;

    vec2      viewportOrigin;
    vec2      viewportSize;

    vec2      viewportSizeInv;
    vec2      pixelOffset;

    vec2      clipToWindowScale;
    vec2      clipToWindowBias;

    vec2      windowToClipScale;
    vec2      windowToClipBias;

    vec4      cameraDirectionOrPosition;
};

struct ResamplingConstants
{
    PlanarViewConstants view;
    PlanarViewConstants prevView;
    RTXDI_RuntimeParameters runtimeParams;

    ReSTIRGI_Parameters restirGI;
};


#endif // SHADER_PARAMETERS_H