#!/bin/sh

base=$(dirname "$0")

function compile {
    ./glslangValidator --glsl-version 460 -V -o $1.spv $1 --target-env vulkan1.3
}

compile $base/src/shaders/raymiss.rmiss
compile $base/src/shaders/raygen.rgen
compile $base/src/shaders/rayhit.rchit

compile $base/src/shaders/post_processing.vert
compile $base/src/shaders/post_processing.frag

compile $base/src/shaders/spatial_resampling.rgen
compile $base/src/shaders/temporal_resampling.rgen

compile $base/src/shaders/visibility.rmiss
compile $base/src/shaders/visibility.rchit

compile $base/src/shaders/presample_environment.comp
compile $base/src/shaders/presample_locallights.comp

compile $base/src/shaders/mip_levels.comp
compile $base/src/shaders/env_mip_levels.comp

compile $base/src/shaders/prepare_lights.comp