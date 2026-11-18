#version 450
layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 toEye;
layout(location = 2) in vec4 color;
layout(location = 0) out vec4 outColor;

void main() {
    // Headlight shading: reads as 3D without a scene light direction.
    const vec3 n = normalize(normal);
    const vec3 v = normalize(toEye);
    const float facing = max(dot(n, v), 0.0);
    const float rim = pow(1.0 - facing, 3.0);
    const vec3 lit = color.rgb * (0.35 + 0.65 * facing) + vec3(0.25) * rim;
    // Premultiplied alpha for the alpha-blended passthrough projection layer.
    outColor = vec4(lit * color.a, color.a);
}
