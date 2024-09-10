#!/bin/sh
./compile.sh
VK_LAYER_PRINTF_BUFFER_SIZE=1000000 cargo run
