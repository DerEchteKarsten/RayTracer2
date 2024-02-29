#version 450

layout(binding = 2, set = 0) uniform CameraProperties 
{
	mat4 viewInverse;
	mat4 projInverse;
} cam;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec3 Position;
layout(location = 1) out vec2 Color;
layout(location = 2) out vec2 Normal;
layout(location = 3) out vec2 UV;

void main() {
    vec4 pos = cam.viewInverse * ubo.projInverse * vec4(inPosition, 1.0);
    gl_Position = pos;
     
    Position = pos;
    Color = inColor;
    Normal = inNormal;
    UV = inUV;
}