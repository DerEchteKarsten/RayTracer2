#version 450

layout(binding = 1) uniform sampler2D texSampler[];

layout(location = 0) out vec3 Position;
layout(location = 1) out vec2 Color;
layout(location = 2) out vec2 Normal;
layout(location = 3) out vec2 UV;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outPosition;
layout(location = 2) out vec4 outNormal;

void main() {
    outColor = texture(texSampler[0], UV) * Color;
    outPosition = Position;
    outNormal = Normal;
}