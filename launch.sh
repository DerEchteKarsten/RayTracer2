#!/bin/sh
./compile.sh
VK_LAYER_PRINTF_BUFFER_SIZE=10000 cargo run
