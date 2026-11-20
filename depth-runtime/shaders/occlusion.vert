#version 450
layout(location=0) in vec3 worldPoint;
layout(location=1) in float supportLifetime;
layout(push_constant) uniform Params {
    mat4 viewProjection;
    vec4 depthRange; // minDepth, maxDepth, nearZ, farZ
    vec4 viewport;   // width, height, depth offset x, y
    vec4 tuning;     // display age from capture end, depth bias, unused, unused
} p;
layout(location=1) out float realZ;
layout(location=2) flat out float triangleSupportLifetime;
void main() {
    vec4 clip = p.viewProjection * vec4(worldPoint, 1);
    realZ = clip.w;
    triangleSupportLifetime = supportLifetime;
    gl_Position = clip;
}
