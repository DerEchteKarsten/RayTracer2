#!/bin/sh

base=$(dirname "$0")

function compile {
    glslangValidator --glsl-version 460 -V -o $1.spv $1 --target-env spirv1.6
}

compile $base/src/shaders/post_processing.vert
compile $base/src/shaders/post_processing.frag
compile $base/src/shaders/default.frag
compile $base/src/shaders/default.vert
compile $base/src/shaders/temp_reuse.comp
