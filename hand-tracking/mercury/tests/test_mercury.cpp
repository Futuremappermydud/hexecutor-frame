// Portable checks for the Mercury port: preprocessing, decoding and an
// end-to-end stereo run whose network outputs are generated from a known 3D
// hand. No neural network or device is involved.
#include "mercury/tracker.hpp"

#include "../src/model_io.hpp"

#include <Eigen/Geometry>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mercury;
using namespace mercury::detail;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        failures++;
    }
}

float map_ranges(float v, float a, float b, float c, float d) { return (v - a) * (d - c) / (b - a) + c; }

// Right hand seen from behind, fingers up (+Y), thumb toward -X, in a plane.
const float kRightHand[21][2] = {
    {0.000f, 0.000f},                                                                   // wrist
    {-0.025f, 0.020f}, {-0.045f, 0.040f}, {-0.060f, 0.060f}, {-0.070f, 0.080f},         // thumb
    {-0.022f, 0.090f}, {-0.025f, 0.130f}, {-0.026f, 0.155f}, {-0.027f, 0.175f},         // index
    {0.000f, 0.095f},  {0.000f, 0.140f},  {0.000f, 0.168f},  {0.000f, 0.190f},          // middle
    {0.020f, 0.090f},  {0.022f, 0.130f},  {0.023f, 0.155f},  {0.024f, 0.175f},          // ring
    {0.038f, 0.080f},  {0.042f, 0.110f},  {0.044f, 0.128f},  {0.046f, 0.145f},          // little
};

// Joints of hand `h` (0 left, 1 right) in left-camera axes (+Y up, -Z forward).
Joints21 make_hand(int h, Eigen::Vector3f wrist) {
    Joints21 j;
    const float mirror = h == 1 ? 1.0f : -1.0f;
    for (int i = 0; i < 21; i++)
        j.col(i) = wrist + Eigen::Vector3f(mirror * kRightHand[i][0], kRightHand[i][1], 0.0f);
    return j;
}

struct Scene {
    StereoCalibration calibration;
    Joints21 hands[2];
    bool present[2] = {true, true};
};

Eigen::Vector3f to_view(const Scene& s, int view, const Eigen::Vector3f& p) {
    if (view == 0) return p;
    const Pose& t = s.calibration.left_in_right;
    Eigen::Quaternionf q(t.orientation.w, t.orientation.x, t.orientation.y, t.orientation.z);
    return q * p + Eigen::Vector3f(t.position.x, t.position.y, t.position.z);
}

// Generates network outputs that an ideal model would produce for the scene.
class SyntheticNetworks : public Networks {
public:
    explicit SyntheticNetworks(const Scene& scene) : scene_(scene) {}
    int detections = 0, keypoint_calls = 0;

    void detect(const DetectionInput* in, DetectionOutput* out, size_t count) override {
        for (size_t k = 0; k < count; k++) {
            detections++;
            const int view = in[k].view;
            const PinholeCamera& cam = view ? scene_.calibration.right : scene_.calibration.left;
            const float scale = std::fmin(float(kDetectionSize) / cam.width, float(kDetectionSize) / cam.height);
            const float tx = (kDetectionSize - cam.width * scale) / 2, ty = (kDetectionSize - cam.height * scale) / 2;
            DetectionOutput& o = out[k];
            for (int h = 0; h < 2; h++) {
                o.hand_exists[size_t(h)] = 0.0f;
                o.cx[size_t(h)] = o.cy[size_t(h)] = o.size[size_t(h)] = 0.0f;
                if (!scene_.present[h]) continue;
                float mnx = 1e9f, mny = 1e9f, mxx = -1e9f, mxy = -1e9f;
                for (int i = 0; i < 21; i++) {
                    Eigen::Vector3f p = to_view(scene_, view, scene_.hands[h].col(i));
                    float u, v;
                    flip_and_project(cam, p.x(), p.y(), p.z(), u, v);
                    mnx = std::fmin(mnx, u), mny = std::fmin(mny, v), mxx = std::fmax(mxx, u), mxy = std::fmax(mxy, v);
                }
                const float cu = (mnx + mxx) / 2, cv = (mny + mxy) / 2;
                const float side = std::fmax(mxx - mnx, mxy - mny);
                o.hand_exists[size_t(h)] = 0.99f;
                o.cx[size_t(h)] = map_ranges(cu * scale + tx, 0, kDetectionSize, -1, 1);
                o.cy[size_t(h)] = map_ranges(cv * scale + ty, 0, kDetectionSize, -1, 1);
                // Mercury: size_px = size * 160 * 2 / scale. Ask for a box about the hand's extent.
                o.size[size_t(h)] = side * scale / (kDetectionSize * 2.0f);
            }
        }
    }

    void keypoints(const KeypointInput* in, KeypointOutput* out, size_t count) override {
        for (size_t k = 0; k < count; k++) {
            keypoint_calls++;
            const KeypointInput& req = in[k];
            KeypointOutput& o = out[k];
            o.heatmap_xy.fill(0.0f);
            o.heatmap_depth.fill(0.0f);
            o.scalar_extras.fill(0.0f);
            o.curls.fill(0.0f);
            o.scalar_extras[0] = 10.0f;  // confidently a hand
            for (int i = 5; i < 10; i++) o.curls[size_t(i)] = 1.0f;
            Eigen::Quaternionf look(req.look_dir.w, req.look_dir.x, req.look_dir.y, req.look_dir.z);
            Joints21 p;
            for (int i = 0; i < 21; i++) p.col(i) = to_view(scene_, req.view, scene_.hands[req.hand].col(i));
            // kine_lm compares depth with the index proximal joint (Joint21::INDX_PXM)
            // and normalizes by its hand size, wrist to middle proximal.
            const float hand_size = (Eigen::Vector3f(p.col(0)) - Eigen::Vector3f(p.col(9))).norm();
            const float mid = Eigen::Vector3f(p.col(5)).norm();
            for (int i = 0; i < 21; i++) {
                Eigen::Vector3f d = look.conjugate() * Eigen::Vector3f(p.col(i)).normalized();
                const float sx = d.x() / (1 - d.z()) / req.stereographic_radius;
                const float sy = d.y() / (1 - d.z()) / req.stereographic_radius;
                // Inverse of Mercury's heatmap decoding (hand 1 is mirrored).
                const float lx = req.hand == 0 ? (sx + 1) * 11 : (1 - sx) * 11;
                const float ly = (1 - sy) * 11;
                float* plane = &o.heatmap_xy[size_t(i) * kHeatmapSize * kHeatmapSize];
                for (int y = 0; y < kHeatmapSize; y++)
                    for (int x = 0; x < kHeatmapSize; x++) {
                        const float dx = x + 0.5f - lx, dy = y + 0.5f - ly;
                        plane[y * kHeatmapSize + x] = std::exp(-(dx * dx + dy * dy) / (2 * 1.2f * 1.2f));
                    }
                // Depth bins whose center of mass is the true relative depth.
                const float rel = (Eigen::Vector3f(p.col(i)).norm() - mid) / hand_size;
                const float avg = (rel / 3 + 0.5f) * 22 - 0.5f;
                const int lo = std::max(0, std::min(20, int(std::floor(avg))));
                const float frac = std::fmin(1.0f, std::fmax(0.0f, avg - float(lo)));
                o.heatmap_depth[size_t(i * 22 + lo)] = 1 - frac;
                o.heatmap_depth[size_t(i * 22 + lo + 1)] = frac;
            }
        }
    }

private:
    const Scene& scene_;
};

void test_preprocessing() {
    // An identity warp reproduces the image; a 2x downscale averages 2x2 corners
    // with OpenCV's pixel-center convention.
    std::vector<uint8_t> img(16 * 8);
    for (int i = 0; i < 16 * 8; i++) img[size_t(i)] = uint8_t(i * 7 % 251);
    GrayImage g{img.data(), 16, 8, 16};
    std::vector<uint8_t> out(16 * 8);
    warp_affine_u8(g, {1, 0, 0, 0, 1, 0}, out.data(), 16, 8);
    check(out == img, "identity warp");

    std::vector<uint8_t> gray(320 * 200, 100);
    GrayImage wide{gray.data(), 320, 200, 320};
    std::vector<uint8_t> boxed(160 * 160);
    Affine back = letterbox(wide, 160, boxed.data());
    check(boxed[0] == 0 && boxed[80 * 160 + 80] == 100, "letterbox bars and interior");
    xrt_vec2 corner = apply(back, {0.0f, 30.0f});
    check(std::fabs(corner.x) < 1e-4f && std::fabs(corner.y) < 1e-4f, "letterbox inverse maps the bar edge to 0,0");

    std::vector<uint8_t> checker(616 * 616);
    for (int y = 0; y < 616; ++y)
        for (int x = 0; x < 616; ++x) checker[size_t(y) * 616 + size_t(x)] = (x + y) % 2 ? 255 : 0;
    GrayImage high_res{checker.data(), 616, 616, 616};
    letterbox(high_res, 160, boxed.data());
    check(boxed[80 * 160 + 80] >= 120 && boxed[80 * 160 + 80] <= 135,
          "high-resolution detection input is antialiased");

    std::vector<uint8_t> ramp(128 * 128);
    for (size_t i = 0; i < ramp.size(); i++) ramp[i] = uint8_t(i % 200);
    std::vector<float> norm(ramp.size());
    check(normalize_gray(ramp.data(), ramp.size(), norm.data()), "normalize succeeds");
    double m = 0, s2 = 0;
    for (float f : norm) m += f;
    m /= double(norm.size());
    for (float f : norm) s2 += (f - m) * (f - m);
    check(std::fabs(m - 0.5) < 1e-4 && std::fabs(std::sqrt(s2 / double(norm.size())) - 0.25) < 1e-4,
          "normalized mean 0.5, std 0.25");
    std::vector<uint8_t> flat(64, 9);
    check(!normalize_gray(flat.data(), flat.size(), norm.data()), "constant image rejected");

    PinholeCamera cam{400, 410, 320, 240, 640, 480};
    xrt_vec3 dir;
    unproject_and_flip(cam, 100.0f, 50.0f, dir);
    float u, v;
    check(flip_and_project(cam, dir.x * 2, dir.y * 2, dir.z * 2, u, v) && std::fabs(u - 100) < 1e-3f &&
              std::fabs(v - 50) < 1e-3f,
          "pinhole round trip");
    check(dir.z < 0 && dir.y > 0, "Monado camera axes: forward -Z, image up is +Y");
}

void test_decode() {
    KeypointOutput o{};
    o.heatmap_xy.fill(0);
    o.heatmap_depth.fill(0);
    o.scalar_extras.fill(0);
    o.curls.fill(0);
    o.scalar_extras[0] = 5;
    for (int i = 0; i < 21; i++) {
        o.heatmap_xy[size_t(i * 484 + 5 * 22 + 16)] = 1;  // row 5, col 16
        o.heatmap_depth[size_t(i * 22 + 10)] = 0.5f;
        o.heatmap_depth[size_t(i * 22 + 11)] = 0.5f;
    }
    hm::one_frame_one_view view{};
    check(decode_keypoints(o, 0, 0.97f, view), "logit 5 is a hand");
    check(std::fabs(view.keypoints_in_scaled_stereographic[3].pos_2d.x - (16.5f / 11 - 1)) < 1e-5f &&
              std::fabs(view.keypoints_in_scaled_stereographic[3].pos_2d.y - (1 - 5.5f / 11)) < 1e-5f,
          "heatmap peak decodes to its cell center");
    check(std::fabs(view.keypoints_in_scaled_stereographic[0].depth_relative_to_midpxm) < 1e-5f,
          "centered depth bins decode to zero");
    hm::one_frame_one_view mirrored{};
    decode_keypoints(o, 1, 0.97f, mirrored);
    check(std::fabs(mirrored.keypoints_in_scaled_stereographic[3].pos_2d.x +
                    view.keypoints_in_scaled_stereographic[3].pos_2d.x) < 1e-5f,
          "right-hand crops are mirrored");
    o.scalar_extras[0] = 2;
    check(!decode_keypoints(o, 0, 0.97f, view), "logit 2 (0.88) is not a hand");
}

const int kMercury21[21] = {1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25};

double mean_joint_error(const Hand& hand, const Joints21& truth) {
    double mean = 0;
    for (int i = 0; i < 21; i++) {
        const Vec3& p = hand.joints[size_t(kMercury21[i])].pose.position;
        mean += (Eigen::Vector3f(p.x, p.y, p.z) - Eigen::Vector3f(truth.col(i))).norm();
    }
    return mean / 21;
}

Joints21 joints21(const Hand& hand) {
    Joints21 j;
    for (int i = 0; i < 21; i++) {
        const Vec3& p = hand.joints[size_t(kMercury21[i])].pose.position;
        j.col(i) = Eigen::Vector3f(p.x, p.y, p.z);
    }
    return j;
}

void test_stereo_tracking() {
    Scene scene;
    const int w = 640, h = 480;
    scene.calibration.left = {380, 380, 320, 240, w, h};
    scene.calibration.right = scene.calibration.left;
    scene.calibration.left_in_right.position = {-0.064f, 0, 0};

    std::mt19937 rng(7);
    std::vector<uint8_t> pixels[2];
    for (auto& p : pixels) {
        p.resize(size_t(w * h));
        for (auto& x : p) x = uint8_t(rng() & 255);
    }
    GrayImage images[2] = {{pixels[0].data(), w, h, size_t(w)}, {pixels[1].data(), w, h, size_t(w)}};

    SyntheticNetworks nets(scene);
    Tracker tracker(scene.calibration, nets);
    Result r;
    double worst = 0;
    int frames_active = 0;
    for (int f = 0; f < 40; f++) {
        // Hands about 45 cm ahead, drifting slowly sideways.
        const float drift = 0.0015f * float(f);
        scene.hands[0] = make_hand(0, {-0.12f + drift, -0.10f, -0.45f});
        scene.hands[1] = make_hand(1, {0.10f + drift, -0.10f, -0.42f});
        r = tracker.process(images[0], images[1], int64_t(f + 1) * 16'666'667);
        if (r.hands[0].active && r.hands[1].active) frames_active++;
        if (f < 20) continue;
        for (int hand = 0; hand < 2; hand++) {
            check(r.hands[size_t(hand)].tracked, "hand " + std::to_string(hand) + " tracked at frame " + std::to_string(f));
            worst = std::fmax(worst, mean_joint_error(r.hands[size_t(hand)], scene.hands[hand]));
        }
    }
    std::printf("stereo synthetic: worst mean joint error %.1f mm over frames 20-39, %d frames with both active, "
                "%d detections, %d keypoint inferences, hand size %.3f m\n",
                worst * 1000, frames_active, nets.detections, nets.keypoint_calls, tracker.target_hand_size());
    check(frames_active >= 25, "both hands displayed after Mercury's warm-up");
    // The planar test hand is not anatomically representable by Mercury's
    // kinematic model (its wrist sits about 12 mm off), so this bound is loose.
    check(worst < 0.030, "mean joint error below 30 mm for the planar hand");
    check(nets.detections < 20, "tracking stops calling the detector once both hands are held");

    // Self-consistency: a hand Mercury's model produced must be recovered almost
    // exactly from ideal observations. Convention errors in projection, mirroring,
    // stereographic crops, heatmap decoding or stereo extrinsics break this.
    const Joints21 fitted[2] = {joints21(r.hands[0]), joints21(r.hands[1])};
    SyntheticNetworks exact_nets(scene);
    Tracker exact(scene.calibration, exact_nets);
    double worst_px = 0, worst_centroid = 0, worst_reprojection = 0;
    for (int f = 0; f < 40; f++) {
        const Eigen::Vector3f shift(0.001f * float(f), 0.0005f * float(f), 0.0f);
        for (int hand = 0; hand < 2; hand++) scene.hands[hand] = fitted[hand].colwise() + shift.array();
        r = exact.process(images[0], images[1], int64_t(f + 1) * 16'666'667);
        if (f < 20) continue;
        for (int hand = 0; hand < 2; hand++) {
            check(r.hands[size_t(hand)].active, "fitted hand active");
            const Joints21 est = joints21(r.hands[size_t(hand)]);
            worst_reprojection = std::fmax(worst_reprojection, r.hands[size_t(hand)].reprojection_error);
            worst_centroid = std::fmax(worst_centroid,
                (est.rowwise().mean() - scene.hands[hand].rowwise().mean()).matrix().norm());
            for (int view = 0; view < 2; view++) {
                const PinholeCamera& cam = view ? scene.calibration.right : scene.calibration.left;
                double px = 0;
                for (int i = 0; i < 21; i++) {
                    Eigen::Vector3f a = to_view(scene, view, est.col(i)), b = to_view(scene, view, scene.hands[hand].col(i));
                    float ua, va, ub, vb;
                    flip_and_project(cam, a.x(), a.y(), a.z(), ua, va);
                    flip_and_project(cam, b.x(), b.y(), b.z(), ub, vb);
                    px += std::hypot(ua - ub, va - vb) / 21;
                }
                worst_px = std::fmax(worst_px, px);
            }
        }
    }
    std::printf("stereo self-consistent: worst mean image error %.2f px, centroid %.1f mm, reprojection %.4f\n",
                worst_px, worst_centroid * 1000, worst_reprojection);
    // Ideal rays from a 6.4 cm baseline leave joint depth along each ray weakly
    // determined, so 3D per-joint error is not asserted; what both views observe is.
    check(worst_reprojection < 0.01, "observations are fitted in both views");
    check(worst_px < 2.0, "joints reproject within 2 px in both views");
    check(worst_centroid < 0.010, "hand centroid within 10 mm");

    // Losing a hand from the images ends its track and re-detection brings it back.
    // The detector still reports hand 1; the keypoint model rejects it.
    struct Hide : Networks {
        SyntheticNetworks& inner;
        explicit Hide(SyntheticNetworks& n) : inner(n) {}
        void detect(const DetectionInput* i, DetectionOutput* o, size_t c) override { inner.detect(i, o, c); }
        void keypoints(const KeypointInput* i, KeypointOutput* o, size_t c) override {
            inner.keypoints(i, o, c);
            for (size_t k = 0; k < c; k++)
                if (i[k].hand == 1) o[k].scalar_extras[0] = -10;
        }
    } hide(nets);
    Tracker second(scene.calibration, hide);
    for (int f = 0; f < 15; f++) r = second.process(images[0], images[1], int64_t(f + 1) * 16'666'667);
    check(r.hands[0].active && !r.hands[1].tracked, "a hand the keypoint model rejects is dropped");

    bool threw = false;
    try {
        second.process(images[0], images[1], 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "non-increasing timestamps are rejected");
}

}  // namespace

int main() {
    test_preprocessing();
    test_decode();
    test_stereo_tracking();
    if (failures) {
        std::fprintf(stderr, "%d mercury check(s) failed\n", failures);
        return 1;
    }
    std::puts("mercury tests passed");
    return 0;
}
