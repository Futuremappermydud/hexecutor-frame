#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;

layout(push_constant) uniform Camera {
    mat4 viewProjection;
} camera;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec3 outWorldPos;

void main() {
    gl_Position = camera.viewProjection * vec4(inPosition, 1.0);
    gl_PointSize = 2.5;
    outColor = inColor;
    outWorldPos = inPosition;
}
