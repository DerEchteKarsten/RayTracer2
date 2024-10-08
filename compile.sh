#!/bin/sh

base=$(dirname "$0")

function compile {
    ./glslangValidator -I$base/src/shaders --glsl-version 460 -V -o $base/src/shaders/bin/$2.spv $1 --target-env vulkan1.3
}

compile $base/src/shaders/g_buffer_pass/beam_coarse.comp beam_coarse
compile $base/src/shaders/g_buffer_pass/beam_fine.comp beam_fine

compile $base/src/shaders/post_processing_pass/post_processing.comp beam_fine

