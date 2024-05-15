#!/bin/sh

base=$(dirname "$0")

function compile {
    ./glslangValidator --glsl-version 460 -V -o $1.spv $1 --target-env vulkan1.3
}

compile $base/src/shaders/post_processing.vert
compile $base/src/shaders/post_processing.frag
compile $base/src/shaders/default.frag
compile $base/src/shaders/default.vert
