#version 450
layout(binding=0) uniform sampler2D appDepth;
layout(push_constant) uniform Params {
    mat4 viewProjection;
    vec4 depthRange;
    vec4 viewport;
    vec4 tuning;
} p;
layout(location=1) in float realZ;
layout(location=2) flat in float triangleSupportLifetime;
layout(location=0) out vec4 maskedColor;
void main() {
    // All three worker vertices carry the earliest support deadline of the cell.
    // Repeated draws and reprojection never refresh this capture-relative age.
    if (p.tuning.x>=0 && (isnan(triangleSupportLifetime) || isinf(triangleSupportLifetime) ||
                         triangleSupportLifetime<=p.tuning.x)) discard;
    if (realZ<=0) discard;
    ivec2 pixel = ivec2(gl_FragCoord.xy) + ivec2(p.viewport.zw);
    float d = texelFetch(appDepth, pixel, 0).r;
    float t = (d-p.depthRange.x)/(p.depthRange.y-p.depthRange.x);
    if (isnan(t) || isinf(t) || t<0 || t>1) discard;
    float n=p.depthRange.z, f=p.depthRange.w;
    float virtualZ;
    if (isinf(f)) virtualZ = n / (1-t);
    else if (isinf(n)) virtualZ = f / t;
    else virtualZ = n*f / (f*(1-t)+n*t);
    if (isnan(virtualZ) || virtualZ<=0 || realZ+p.tuning.y>=virtualZ) discard;
    maskedColor=vec4(0); // premultiplied transparent passthrough hole
}
