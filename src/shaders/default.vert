#version 450
#extension GL_EXT_debug_printf : enable

layout(binding = 0, set = 0) uniform CameraProperties 
{
	mat4 viewInverse;
	mat4 projInverse;
} cam;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec4 Color;
layout(location = 1) out vec3 Position;
layout(location = 2) out vec3 Normal;
layout(location = 3) out vec2 UV;


void main() {
    gl_Position = vec4(inPosition, 1.0) * cam.viewInverse * cam.projInverse;
    Position = gl_Position.xyz;
    Color = vec4(inColor, 1.0);
    Normal = inNormal;
    UV = inUV;
}