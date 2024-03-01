#version 450
#extension GL_EXT_debug_printf : enable

layout(binding = 1) uniform sampler2D texSampler[];

layout(location = 0) in vec4 Color;
layout(location = 1) in vec3 Position;
layout(location = 2) in vec3 Normal;
layout(location = 3) in vec2 UV;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outPosition;
layout(location = 2) out vec4 outNormal;

void main() {
    outColor = vec4(1.0, 0.0, 0.0, 1.0); //* texture(texSampler[0], UV) * Color;
    outPosition = vec4(Position, 1.0);
    outNormal = vec4(Normal, 1.0);
}