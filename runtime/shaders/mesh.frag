#version 450

layout(location = 0) in vec4 inColor;
layout(location = 1) in vec3 inWorldPos;

layout(location = 0) out vec4 outColor;

void main() {
    // Screen-space geometric normal calculation for flat-shaded 3D triangles
    vec3 dX = dFdx(inWorldPos);
    vec3 dY = dFdy(inWorldPos);
    vec3 normal = cross(dX, dY);
    float len = length(normal);
    if (len > 1e-6) {
        normal = normal / len;
    } else {
        normal = vec3(0.0, 1.0, 0.0);
    }

    if (!gl_FrontFacing) {
        normal = -normal;
    }

    // Directional key light from upper-front + ambient base
    vec3 lightDir = normalize(vec3(0.2, 0.9, 0.4));
    float diff = max(dot(normal, lightDir), 0.0);
    float ambient = 0.35;
    float lighting = clamp(ambient + 0.65 * diff, 0.2, 1.25);

    outColor = vec4(inColor.rgb * lighting, inColor.a);
}
