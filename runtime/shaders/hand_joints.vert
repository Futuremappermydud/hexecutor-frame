#version 450
// Lit hand joint spheres and bone cylinders in the OpenXR LOCAL space.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;

layout(push_constant) uniform Camera {
    mat4 viewProjection;
    vec4 eye;  // xyz: eye position in LOCAL space
} camera;

layout(location = 0) out vec3 normal;
layout(location = 1) out vec3 toEye;
layout(location = 2) out vec4 color;

void main() {
    gl_Position = camera.viewProjection * vec4(inPosition, 1.0);
    normal = inNormal;
    toEye = camera.eye.xyz - inPosition;
    color = inColor;
}
