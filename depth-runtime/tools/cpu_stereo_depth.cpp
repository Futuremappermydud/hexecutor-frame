#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <chrono>

const int RAW_W = 2464;
const int RAW_H = 2464;
const int OUT_W = 616;
const int OUT_H = 616;
const int MAX_DISP = 64;

const float F_RECT = 380.0f;
const float CX_RECT = 308.0f;
const float CY_RECT = 308.0f;
const float BASELINE_M = 0.06396f; // 63.96 mm baseline
const float F_B = F_RECT * BASELINE_M; // 24.3048

// Left Camera (factory calibration)
const float FX_L = 1477.53f, FY_L = 1479.15f, CX_L = 1082.43f, CY_L = 1243.37f;
const float K1_L = -0.027497f, K2_L = -0.16945f, K3_L = 0.10018f, K4_L = -0.01506f;

// Right Camera (factory calibration)
const float FX_R = 1476.21f, FY_R = 1478.26f, CX_R = 1078.65f, CY_R = 1238.10f;
const float K1_R = -0.031552f, K2_R = -0.14942f, K3_R = 0.07644f, K4_R = -0.00628f;

struct Point2D { float x, y; };

static Point2D projectKB(float x, float y, float z, float fx, float fy, float cx, float cy, float k1, float k2, float k3, float k4) {
    float r_xy = sqrtf(x * x + y * y);
    if (r_xy < 1e-6f) return { cx, cy };

    float theta = atan2f(r_xy, z);
    float t2 = theta * theta;
    float t4 = t2 * t2;
    float t6 = t4 * t2;
    float t8 = t4 * t4;

    float r_dist = theta * (1.0f + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8);
    float scale = r_dist / r_xy;

    return { cx + fx * x * scale, cy + fy * y * scale };
}

static inline float sampleBilinear(const uint8_t *img, float px, float py) {
    if (px < 0.0f || px >= RAW_W - 1 || py < 0.0f || py >= RAW_H - 1) return 0.0f;
    int x0 = (int)px;
    int y0 = (int)py;
    float wx = px - (float)x0;
    float wy = py - (float)y0;

    float p00 = img[y0 * RAW_W + x0];
    float p10 = img[y0 * RAW_W + x0 + 1];
    float p01 = img[(y0 + 1) * RAW_W + x0];
    float p11 = img[(y0 + 1) * RAW_W + x0 + 1];

    return (p00 * (1.0f - wx) + p10 * wx) * (1.0f - wy) + (p01 * (1.0f - wx) + p11 * wx) * wy;
}

int main() {
    printf("=== Fast CPU Stereo Rectification & Depth Pipeline ===\n");

    // 1. Load captured PPM images
    auto loadPPM = [](const char *path) -> std::vector<uint8_t> {
        FILE *fp = fopen(path, "rb");
        if (!fp) { perror(path); return {}; }
        char header[256];
        fgets(header, sizeof(header), fp); // P5
        int w, h, maxv;
        fscanf(fp, "%d %d\n%d\n", &w, &h, &maxv);
        std::vector<uint8_t> data(w * h);
        fread(data.data(), 1, w * h, fp);
        fclose(fp);
        return data;
    };

    auto leftRaw = loadPPM("/home/steamos/test_openvr/left_view.ppm");
    auto rightRaw = loadPPM("/home/steamos/test_openvr/right_view.ppm");
    if (leftRaw.empty() || rightRaw.empty()) {
        printf("Failed to load raw images\n");
        return 1;
    }
    printf("Loaded left and right 2464x2464 raw frames.\n");

    // 2. Precompute Rectification LUT
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<Point2D> lutL(OUT_W * OUT_H);
    std::vector<Point2D> lutR(OUT_W * OUT_H);

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < OUT_H; y++) {
        for (int x = 0; x < OUT_W; x++) {
            float rayX = ((float)x - CX_RECT) / F_RECT;
            float rayY = ((float)y - CY_RECT) / F_RECT;

            // Left
            lutL[y * OUT_W + x] = projectKB(rayX, rayY, 1.0f, FX_L, FY_L, CX_L, CY_L, K1_L, K2_L, K3_L, K4_L);

            // Right: sensor mounted 180 deg inverted relative to Left
            lutR[y * OUT_W + x] = projectKB(-rayX, -rayY, 1.0f, FX_R, FY_R, CX_R, CY_R, K1_R, K2_R, K3_R, K4_R);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("LUT precomputation: %.2f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    // 3. Rectify Images
    std::vector<uint8_t> rectL(OUT_W * OUT_H);
    std::vector<uint8_t> rectR(OUT_W * OUT_H);

    t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < OUT_H; y++) {
        for (int x = 0; x < OUT_W; x++) {
            int idx = y * OUT_W + x;
            rectL[idx] = (uint8_t)clamp(sampleBilinear(leftRaw.data(), lutL[idx].x, lutL[idx].y), 0.0f, 255.0f);
            rectR[idx] = (uint8_t)clamp(sampleBilinear(rightRaw.data(), lutR[idx].x, lutR[idx].y), 0.0f, 255.0f);
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    printf("Rectification execution: %.2f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Save rectified left & right PPMs
    auto savePPM = [](const char *path, const uint8_t *img, int w, int h) {
        FILE *fp = fopen(path, "wb");
        if (fp) {
            fprintf(fp, "P5\n%d %d\n255\n", w, h);
            fwrite(img, 1, w * h, fp);
            fclose(fp);
            printf("Saved %s\n", path);
        }
    };
    savePPM("/home/steamos/test_openvr/rect_left_cpu.ppm", rectL.data(), OUT_W, OUT_H);
    savePPM("/home/steamos/test_openvr/rect_right_cpu.ppm", rectR.data(), OUT_W, OUT_H);

    // 4. Census Transform (7x7 window -> 48 bits in uint64_t)
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<uint64_t> censusL(OUT_W * OUT_H, 0);
    std::vector<uint64_t> censusR(OUT_W * OUT_H, 0);

    #pragma omp parallel for collapse(2)
    for (int y = 3; y < OUT_H - 3; y++) {
        for (int x = 3; x < OUT_W - 3; x++) {
            uint8_t cL = rectL[y * OUT_W + x];
            uint8_t cR = rectR[y * OUT_W + x];
            uint64_t bL = 0, bR = 0;
            int bit = 0;

            for (int dy = -3; dy <= 3; dy++) {
                for (int dx = -3; dx <= 3; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int qIdx = (y + dy) * OUT_W + (x + dx);
                    if (rectL[qIdx] > cL) bL |= (1ULL << bit);
                    if (rectR[qIdx] > cR) bR |= (1ULL << bit);
                    bit++;
                }
            }
            censusL[y * OUT_W + x] = bL;
            censusR[y * OUT_W + x] = bR;
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    printf("Census Transform: %.2f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    // 5. Stereo Matching across 64 disparities
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> depthMap(OUT_W * OUT_H, 0.0f);
    std::vector<float> dispMap(OUT_W * OUT_H, 0.0f);
    int validCount = 0;
    float sumDepth = 0.0f, minDepth = 1e9f, maxDepth = -1e9f;

    #pragma omp parallel for collapse(2) reduction(+:validCount, sumDepth) reduction(min:minDepth) reduction(max:maxDepth)
    for (int y = 3; y < OUT_H - 3; y++) {
        for (int x = 3; x < OUT_W - 3; x++) {
            uint64_t cL = censusL[y * OUT_W + x];
            uint32_t minCost = 999999;
            int bestD = 0;
            uint32_t costs[MAX_DISP];

            for (int d = 0; d < MAX_DISP; d++) {
                int rx = x - d;
                if (rx < 3) { costs[d] = 48; continue; }
                uint64_t cR = censusR[y * OUT_W + rx];
                uint32_t cost = (uint32_t)__builtin_popcountll(cL ^ cR);
                costs[d] = cost;
                if (cost < minCost) {
                    minCost = cost;
                    bestD = d;
                }
            }

            // Subpixel refinement
            float subD = (float)bestD;
            if (bestD > 0 && bestD < MAX_DISP - 1) {
                float c0 = (float)costs[bestD - 1];
                float c1 = (float)costs[bestD];
                float c2 = (float)costs[bestD + 1];
                float denom = 2.0f * (c0 - 2.0f * c1 + c2);
                if (fabsf(denom) > 1e-4f) {
                    subD += (c0 - c2) / denom;
                }
            }

            // Confidence threshold
            if (subD > 1.5f && minCost < 22) {
                float Z = F_B / subD;
                if (Z >= 0.25f && Z <= 8.0f) {
                    depthMap[y * OUT_W + x] = Z;
                    dispMap[y * OUT_W + x] = subD;
                    validCount++;
                    sumDepth += Z;
                    if (Z < minDepth) minDepth = Z;
                    if (Z > maxDepth) maxDepth = Z;
                }
            }
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    printf("Stereo Matching: %.2f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    printf("=== Reconstructed Depth Map Statistics ===\n");
    printf("  Valid 3D Points: %d / %d (%.1f%%)\n", validCount, OUT_W * OUT_H, validCount * 100.0f / (OUT_W * OUT_H));
    if (validCount > 0) {
        printf("  Min Depth: %.2f m\n", minDepth);
        printf("  Max Depth: %.2f m\n", maxDepth);
        printf("  Avg Depth: %.2f m\n", sumDepth / validCount);
    }

    // 6. Export Depth Colormap PPM
    FILE *fpColor = fopen("/home/steamos/test_openvr/metric_depth_color.ppm", "wb");
    if (fpColor) {
        fprintf(fpColor, "P6\n%d %d\n255\n", OUT_W, OUT_H);
        for (int y = 0; y < OUT_H; y++) {
            for (int x = 0; x < OUT_W; x++) {
                float d = depthMap[y * OUT_W + x];
                uint8_t rgb[3] = { 0, 0, 0 };
                if (d > 0.2f && d < 6.0f) {
                    float norm = 1.0f - (d - 0.3f) / (3.5f - 0.3f);
                    norm = fmaxf(0.0f, fminf(1.0f, norm));
                    // Turbo / Jet style: red = close, blue = far
                    rgb[0] = (uint8_t)(255.0f * norm);
                    rgb[1] = (uint8_t)(255.0f * (1.0f - fabsf(norm - 0.5f) * 2.0f));
                    rgb[2] = (uint8_t)(255.0f * (1.0f - norm));
                }
                fwrite(rgb, 1, 3, fpColor);
            }
        }
        fclose(fpColor);
        printf("Saved /home/steamos/test_openvr/metric_depth_color.ppm!\n");
    }

    // 7. Export 3D Point Cloud PLY
    FILE *plyFp = fopen("/home/steamos/test_openvr/real_dense_mesh.ply", "w");
    if (plyFp) {
        fprintf(plyFp, "ply\nformat ascii 1.0\nelement vertex %d\n", validCount);
        fprintf(plyFp, "property float x\nproperty float y\nproperty float z\n");
        fprintf(plyFp, "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n");

        for (int y = 0; y < OUT_H; y++) {
            for (int x = 0; x < OUT_W; x++) {
                float Z = depthMap[y * OUT_W + x];
                if (Z > 0.2f && Z <= 8.0f) {
                    float X = ((float)x - CX_RECT) * Z / F_RECT;
                    float Y = -((float)y - CY_RECT) * Z / F_RECT; // Camera up
                    uint8_t intensity = rectL[y * OUT_W + x];
                    fprintf(plyFp, "%.4f %.4f %.4f %u %u %u\n", X, Y, -Z, intensity, intensity, intensity);
                }
            }
        }
        fclose(plyFp);
        printf("Saved 3D point cloud to /home/steamos/test_openvr/real_dense_mesh.ply!\n");
    }

    return 0;
}
