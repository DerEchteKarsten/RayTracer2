#!/bin/sh

base=$(dirname "$0")

function compile {
    ./glslangValidator --glsl-version 460 -V -o $1.spv $1 --target-env vulkan1.3
}

compile $base/src/shaders/raymiss.rmiss
compile $base/src/shaders/raygen.rgen
compile $base/src/shaders/rayhit.rchit
compile $base/src/shaders/rayint.rint
compile $base/src/shaders/post_processing.vert
compile $base/src/shaders/post_processing.frag
