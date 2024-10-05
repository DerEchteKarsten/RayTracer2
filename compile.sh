#!/bin/sh

base=$(dirname "$0")

function compile {
    ./glslangValidator -I$base/src/shaders --glsl-version 460 -V -o $base/src/shaders/bin/$2.spv $1 --target-env vulkan1.3
}

compile $base/src/shaders/lighting_passes/raymiss.rmiss raymiss
compile $base/src/shaders/lighting_passes/rayhit.rchit rayhit

compile $base/src/shaders/lighting_passes/brdf_rays.rgen brdf_rays
compile $base/src/shaders/lighting_passes/di_fused_resampling.rgen di_fused_resampling
compile $base/src/shaders/lighting_passes/gi_final_shading.rgen gi_final_shading
compile $base/src/shaders/lighting_passes/shade_secondary_surfaces.rgen shade_secondary_surfaces
compile $base/src/shaders/lighting_passes/g_buffer.rgen g_buffer

compile $base/src/shaders/lighting_passes/spatial_resampling.rgen spatial_resampling
compile $base/src/shaders/lighting_passes/temporal_resampling.rgen temporal_resampling

compile $base/src/shaders/lighting_passes/presample_environment.comp presample_environment
compile $base/src/shaders/lighting_passes/presample_locallights.comp presample_locallights



compile $base/src/shaders/post_processing.comp post_processing

compile $base/src/shaders/generate_mip_levels/mip_levels.comp mip_levels
compile $base/src/shaders/generate_mip_levels/env_mip_levels.comp env_mip_levels

compile $base/src/shaders/prepare_lights.comp prepare_lights