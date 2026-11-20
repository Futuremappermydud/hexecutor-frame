#include "calibration.hpp"
#include "ir_dot_refine.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main() {
    using namespace FrameSpatial;
    IrDotRefine refiner;

    // Simulate an RGB frame: 616x616
    constexpr int W = RECT_WIDTH, H = RECT_HEIGHT;
    constexpr size_t N = size_t(W) * H;

    // 1. Flat luma image (e.g. blank wall with luma = 120 everywhere)
    std::vector<uint8_t> luma(N, 120);

    // 2. Bulging neural depth map:
    // True wall is at z = 2.0 m.
    // Neural model predicts center bulge (z = 1.82 m) and edge sag (z = 2.30 m).
    std::vector<float> depth(N);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float dx = (x - RECT_CX) / float(W);
            float dy = (y - RECT_CY) / float(H);
            float r2 = dx * dx + dy * dy;
            // Bulge: 1.82 at center, 2.30 at corners
            depth[y * W + x] = 1.82f + 1.0f * r2;
        }
    }

    // Measure initial dome error against true flat plane z = 2.0 m
    float maxInitialDev = 0.f;
    for (size_t i = 0; i < N; ++i) {
        maxInitialDev = std::max(maxInitialDev, std::abs(depth[i] - 2.0f));
    }
    std::cout << "[Test] Initial neural dome max deviation: " << maxInitialDev * 1000.f << " mm" << std::endl;
    assert(maxInitialDev > 0.15f); // At least 150 mm bulge

    // 3. Test camera poses:
    // Pose 0: Camera at origin looking along +Z
    TsdfPose pose0; // Identity rotation, zero translation

    // Simulate an observed plane: wall at z = 2.0 m (normal = [0, 0, 1], c = 2.0)
    // We test IrDotRefine apply by creating a dummy patch with a world plane.
    // In order to directly test apply with WorldPlane, we can test that the plane math
    // matches analytical planar rays.

    // Let's verify reprojection math:
    // Rotate camera by 20 degrees yaw around Y:
    TsdfPose poseRotated;
    float angle = 20.0f * float(M_PI) / 180.0f;
    poseRotated.rotation[0][0] = std::cos(angle);
    poseRotated.rotation[0][2] = std::sin(angle);
    poseRotated.rotation[2][0] = -std::sin(angle);
    poseRotated.rotation[2][2] = std::cos(angle);
    poseRotated.translation[0] = 0.1f; // 10 cm right shift
    poseRotated.translation[1] = 0.0f;
    poseRotated.translation[2] = 0.0f;

    // Plane in world is Z = 2.0 (n_world = [0, 0, 1], c_world = 2.0)
    // In rotated camera:
    // n_cam = R^T * n_world = [-sin(20), 0, cos(20)]
    // c_cam = c_world - n_world . t = 2.0 - 0 = 2.0
    float n_cam[3] = {
        poseRotated.rotation[0][0] * 0.f + poseRotated.rotation[1][0] * 0.f + poseRotated.rotation[2][0] * 1.f,
        poseRotated.rotation[0][1] * 0.f + poseRotated.rotation[1][1] * 0.f + poseRotated.rotation[2][1] * 1.f,
        poseRotated.rotation[0][2] * 0.f + poseRotated.rotation[1][2] * 0.f + poseRotated.rotation[2][2] * 1.f
    };
    float c_cam = 2.0f - (0.f * poseRotated.translation[0] + 0.f * poseRotated.translation[1] + 1.f * poseRotated.translation[2]);

    assert(std::abs(n_cam[0] - (-std::sin(angle))) < 1e-5f);
    assert(std::abs(n_cam[2] - std::cos(angle)) < 1e-5f);
    assert(std::abs(c_cam - 2.0f) < 1e-5f);

    // Any camera ray: r = [(x - cx)/f, (y - cy)/f, 1]
    // Point P_cam = z * r
    // P_world = R * P_cam + t
    // P_world.z = R[2][0]*P_cam.x + R[2][1]*P_cam.y + R[2][2]*P_cam.z + t.z
    // We want P_world.z == 2.0
    for (int y = 50; y < H - 50; y += 100) {
        for (int x = 50; x < W - 50; x += 100) {
            float rx = (x - RECT_CX) / RECT_FOCAL;
            float ry = (y - RECT_CY) / RECT_FOCAL;
            float denom = n_cam[0] * rx + n_cam[1] * ry + n_cam[2];
            float z_planar = c_cam / denom;

            float P_cam[3] = {rx * z_planar, ry * z_planar, z_planar};
            float P_world_z = poseRotated.rotation[2][0] * P_cam[0] +
                              poseRotated.rotation[2][1] * P_cam[1] +
                              poseRotated.rotation[2][2] * P_cam[2] + poseRotated.translation[2];
            assert(std::abs(P_world_z - 2.0f) < 1e-4f);
        }
    }
    std::cout << "[Test] World plane motion reprojection: PASS (all rays intersect Z_world = 2.000 m exact)" << std::endl;

    // 4. Test IrDotRefine::apply() directly on the bulging neural depth map:
    IrDotRefine::WorldPlane testPlane;
    testPlane.n[0] = 0.f;
    testPlane.n[1] = 0.f;
    testPlane.n[2] = 1.f;
    testPlane.c = 2.0f; // wall at Z = 2.0 m
    testPlane.centroid[0] = 0.f;
    testPlane.centroid[1] = 0.f;
    testPlane.centroid[2] = 2.0f;
    testPlane.radius = 2.5f; // broad wall coverage
    testPlane.inliers = 15000;
    testPlane.valid = true;
    testPlane.timestamp = std::chrono::steady_clock::now();
    testPlane.fillDiskFootprint();

    refiner.setTestWorldPlane(testPlane);

    // Call apply with pose0 (camera looking at the wall)
    refiner.apply(depth, luma, &pose0);

    // Verify:
    // Across the flat blank wall, every pixel should now have depth = 2.000 m (flat!)
    float maxPostDev = 0.f;
    int overwrittenCount = 0;
    for (int y = 2; y < H - 2; ++y) {
        for (int x = 2; x < W - 2; ++x) {
            size_t i = size_t(y) * W + x;
            maxPostDev = std::max(maxPostDev, std::abs(depth[i] - 2.0f));
            if (std::abs(depth[i] - 2.0f) < 1e-4f) {
                ++overwrittenCount;
            }
        }
    }

    std::cout << "[Test] Post-refinement wall max deviation: " << maxPostDev * 1000.f << " mm" << std::endl;
    std::cout << "[Test] Overwritten planar pixels: " << overwrittenCount << " / " << (W-4)*(H-4) << std::endl;
    assert(maxPostDev < 1e-4f); // Bulge eliminated completely!
    assert(overwrittenCount == (W - 4) * (H - 4));

    // 5. Foreground must survive the plane correction.
    std::vector<uint8_t> texturedLuma = luma;
    for (int y = 100; y < 150; ++y) {
        for (int x = 100; x < 150; ++x) {
            texturedLuma[y * W + x] = ((x + y) % 2 == 0) ? 0 : 255;
        }
    }
    const size_t texturedPixel = size_t(125) * W + 125;
    std::vector<float> texturedDepth(N, 1.70f);  // 30 cm in front: a textured object, inside the dome band
    refiner.apply(texturedDepth, texturedLuma, &pose0);
    assert(texturedDepth[texturedPixel] == 1.70f);
    // Elsewhere the same depth on blank luma is dome and is corrected.
    assert(std::abs(texturedDepth[size_t(300) * W + 300] - 2.0f) < 1e-3f);
    std::cout << "[Test] Textured foreground in the dome band preserved: PASS" << std::endl;

    std::vector<float> plainNear(N, 1.40f);  // 60 cm in front: plain sofa, past the dome band
    refiner.apply(plainNear, luma, &pose0);
    assert(plainNear[size_t(300) * W + 300] == 1.40f);
    std::cout << "[Test] Plain foreground past the dome band preserved: PASS" << std::endl;

    std::vector<float> opening(N, 2.80f);  // 80 cm behind: doorway, not model error
    refiner.apply(opening, luma, &pose0);
    assert(opening[size_t(300) * W + 300] == 2.80f);
    std::cout << "[Test] Opening behind the plane preserved: PASS" << std::endl;

    // Dots seen in front of the plane mark their cells; blank RGB there stays put.
    IrDotRefine blockedRefiner;
    IrDotRefine::WorldPlane blockedPlane = testPlane;
    for (int v = 28; v < 36; ++v)
        for (int u = 28; u < 36; ++u) blockedPlane.footprint[size_t(v) * kIrFootprintSize + u] = kIrCellBlocked;
    blockedRefiner.setTestWorldPlane(blockedPlane);
    std::vector<float> chair(N, 1.75f);
    blockedRefiner.apply(chair, luma, &pose0);
    assert(chair[size_t(308) * W + 308] == 1.75f);  // on the axis: cell (32, 32)
    assert(std::abs(chair[size_t(100) * W + 100] - 2.0f) < 1e-3f);
    std::cout << "[Test] Blocked footprint cell preserved: PASS" << std::endl;

    // 6. Spatial bounding (prevent attaching to walls that don't exist):
    IrDotRefine boundedRefiner;
    IrDotRefine::WorldPlane boundedPlane;
    boundedPlane.n[0] = 0.f; boundedPlane.n[1] = 0.f; boundedPlane.n[2] = 1.f;
    boundedPlane.c = 2.0f;
    boundedPlane.centroid[0] = -0.5f; // 50 cm left
    boundedPlane.centroid[1] = 0.f;
    boundedPlane.centroid[2] = 2.0f;
    boundedPlane.radius = 0.35f; // 35 cm radius
    boundedPlane.inliers = 2000;
    boundedPlane.valid = true;
    boundedPlane.timestamp = std::chrono::steady_clock::now();
    boundedPlane.fillDiskFootprint();
    boundedRefiner.setTestWorldPlane(boundedPlane);

    std::vector<float> testBoundingDepth(N, 1.85f);
    boundedRefiner.apply(testBoundingDepth, luma, &pose0);
    size_t rightPixel = size_t(300) * W + 550;
    assert(testBoundingDepth[rightPixel] == 1.85f); // Retained original depth!
    size_t leftPixel = size_t(308) * W + 210;       // about 50 cm left at 2 m
    assert(std::abs(testBoundingDepth[leftPixel] - 2.0f) < 1e-3f); // Flattened!
    std::cout << "[Test] Spatial bounding prevents phantom walls: PASS" << std::endl;

    // 7. RGB holes are left for real IR observations.
    std::vector<float> holeDepth(N, 2.0f);
    holeDepth[leftPixel] = 0.0f;
    boundedRefiner.apply(holeDepth, luma, &pose0);
    assert(holeDepth[leftPixel] == 0.0f);
    std::cout << "[Test] RGB holes are not stamped from the plane: PASS" << std::endl;

    // 8. Plane correction needs a capture-time pose, and expires.
    IrDotRefine idleRefiner;
    IrDotRefine::WorldPlane stalePlane = testPlane;
    idleRefiner.setTestWorldPlane(stalePlane);
    std::vector<float> noPose(N, 1.75f);
    idleRefiner.apply(noPose, luma, nullptr);
    assert(noPose[size_t(300) * W + 300] == 1.75f);
    stalePlane.timestamp = std::chrono::steady_clock::now() - std::chrono::seconds(3);
    idleRefiner.setTestWorldPlane(stalePlane);
    std::vector<float> stale(N, 1.75f);
    idleRefiner.apply(stale, luma, &pose0);
    assert(stale[size_t(300) * W + 300] == 1.75f);
    IrDotRefine::WorldPlane emptyPlane;
    emptyPlane.valid = false;
    idleRefiner.setTestWorldPlane(emptyPlane);
    std::vector<float> preserveDepth(N, 1.75f);
    idleRefiner.apply(preserveDepth, luma, &pose0);
    assert(preserveDepth[size_t(300) * W + 300] == 1.75f);
    std::cout << "[Test] No pose, stale or missing plane changes nothing: PASS" << std::endl;

    // 9. The plane stays world-fixed when the camera translates.
    TsdfPose raised = pose0;
    raised.translation[1] = 0.4f;
    IrDotRefine movedRefiner;
    movedRefiner.setTestWorldPlane(testPlane);
    std::vector<float> moved(N, 1.8f);
    movedRefiner.apply(moved, luma, &raised);
    assert(std::abs(moved[size_t(300) * W + 300] - 2.0f) < 1e-3f);
    std::cout << "[Test] Translation keeps the plane world-fixed: PASS" << std::endl;

    // 10. dotScore on recorded frames: lit pairs stand out and odd frames are dark.
    const std::string raw = "experiments/room-reconstruction/20261104T004533Z-e10a/raw/";
    const auto load = [&](const std::string& name) {
        std::ifstream f(raw + name, std::ios::binary);
        std::vector<uint8_t> px(size_t(IrPairCapture::Width) * IrPairCapture::Height);
        return f.read(reinterpret_cast<char*>(px.data()), std::streamsize(px.size())) ? px : std::vector<uint8_t>{};
    };
    const auto unlit = load("motion/e4/video13-39700.u8"), lit = load("motion/e4/video13-39704.u8");
    const auto odd = load("rec-lit/video13-48805.u8");
    if (!unlit.empty() && !lit.empty() && !odd.empty()) {
        const float a = IrPairCapture::dotScore(unlit), b = IrPairCapture::dotScore(lit), c = IrPairCapture::dotScore(odd);
        std::cout << "[Test] dotScore unlit " << a << " lit " << b << " odd " << c << std::endl;
        assert(b > a * 1.5f);
        assert(c < a * 0.3f);
    } else {
        std::cout << "[Test] dotScore recordings absent, skipped" << std::endl;
    }
    std::cout << "All IR dot plane tests PASS" << std::endl;
    return 0;
}
