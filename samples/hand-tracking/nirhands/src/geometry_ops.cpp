// Detector pre/post-processing, stereo seeding, keypoint crops and decode: ports of
// stereo_nir_hands nirhands/training/detector.py, samples.py, adapter/rectified_crop.py and
// adapter/photometric.py. Comments name the Python function each block mirrors.
#include "nirhands/pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nirhands {
namespace {

constexpr double kPi = 3.14159265358979323846;

// OpenCV computeResizeAreaTab: (source index, weight) per destination index.
std::vector<std::vector<std::pair<int, float>>> area_table(int ssize, int dsize) {
    const double scale = double(ssize) / dsize;
    std::vector<std::vector<std::pair<int, float>>> tab(dsize);
    for (int dx = 0; dx < dsize; ++dx) {
        const double fsx1 = dx * scale, fsx2 = fsx1 + scale;
        int sx1 = int(std::ceil(fsx1)), sx2 = int(std::floor(fsx2));
        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);
        const double cell = std::min(scale, ssize - fsx1);
        if (sx1 - fsx1 > 1e-3) tab[dx].push_back({sx1 - 1, float((sx1 - fsx1) / cell)});
        for (int sx = sx1; sx < sx2; ++sx) tab[dx].push_back({sx, float(1.0 / cell)});
        if (fsx2 - sx2 > 1e-3) tab[dx].push_back({sx2, float(std::min(std::min(fsx2 - sx2, 1.0), cell) / cell)});
    }
    return tab;
}

uint8_t round_u8(double v) { return uint8_t(std::clamp<long>(std::lrint(v), 0, 255)); }

Vec3 camera_center(const KbCamera& c) { return c.center(); }

VirtualCamera make_virtual(const Vec3& center, const Mat3& R, double f, const Vec3& hand, int size) {
    const Vec3 p = R * (hand - center);
    return {center, R, f, {f * p[0] / p[2] - size / 2.0, f * p[1] / p[2] - size / 2.0}, size};
}

struct StereoPair {
    VirtualCamera left, right;
    double baseline_m = 0, baseline_angle_deg = 0;
};

// rectified_crop.stereo_crop_pair
StereoPair stereo_crop_pair(const Rig& rig, const Vec3& h, int size, double radius, double fill) {
    const Vec3 cl = camera_center(rig.left), cr = camera_center(rig.right);
    const Vec3 b = cr - cl;
    const double baseline = norm(b);
    const Vec3 x = (1.0 / baseline) * b;
    const Vec3 z0 = h - 0.5 * (cl + cr);
    const double angle = std::acos(std::clamp(std::abs(dot(z0, x)) / norm(z0), 0.0, 1.0)) * 180.0 / kPi;
    const Vec3 z = unit(z0 - dot(z0, x) * x);
    const Vec3 y = cross(z, x);
    const Mat3 R = Mat3::rows(x, y, z);
    const double z_hand = dot(z, h - cl);
    const double f = fill * (size / 2.0) * z_hand / radius;
    return {make_virtual(cl, R, f, h, size), make_virtual(cr, R, f, h, size), baseline, angle};
}

// rectified_crop.monocular_crop
VirtualCamera monocular_crop(const KbCamera& cam, const Vec3& h, const Vec3& up, int size, double radius, double fill) {
    const Vec3 c = camera_center(cam);
    const double z_hand = norm(h - c);
    const Vec3 z = (1.0 / z_hand) * (h - c);
    const Vec3 y = unit(-1.0 * (up - dot(up, z) * z));
    const Vec3 x = cross(y, z);
    const double f = fill * (size / 2.0) * z_hand / radius;
    return make_virtual(c, Mat3::rows(x, y, z), f, h, size);
}

// samples._rig_down
Vec3 rig_down(const Rig& rig, const Vec3& baseline_dir) {
    Vec3 forward = rig.left.R_cw.row(2) + rig.right.R_cw.row(2);
    forward = forward - dot(forward, baseline_dir) * baseline_dir;
    return cross(unit(forward), baseline_dir);
}

// samples._monocular_view
VirtualCamera monocular_view(const KbCamera& cam, const Vec3& center, const Vec3& down, const Vec3& baseline_dir,
                             int size, double radius, double fill) {
    const Vec3 view = unit(center - camera_center(cam));
    const Vec3 up = norm(cross(down, view)) > 1e-3 ? -1.0 * down : -1.0 * cross(view, baseline_dir);
    return monocular_crop(cam, center, up, size, radius, fill);
}

// photometric.robust_normalize: 2nd/98th percentiles of valid pixels -> 0/1, clipped to [-0.5, 1.5].
void robust_normalize(const uint8_t* crop, const uint8_t* valid, int n, float* out) {
    std::vector<double> vals;
    vals.reserve(n);
    for (int i = 0; i < n; ++i) if (valid[i]) vals.push_back(float(crop[i]) / 255.0f);
    if (vals.empty()) for (int i = 0; i < n; ++i) vals.push_back(float(crop[i]) / 255.0f);
    std::sort(vals.begin(), vals.end());
    auto pct = [&](double p) {  // numpy "linear"
        const double idx = p / 100.0 * (vals.size() - 1);
        const size_t lo = size_t(std::floor(idx)), hi = std::min(lo + 1, vals.size() - 1);
        return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo);
    };
    const double lo = pct(2.0), hi = pct(98.0);
    if (hi - lo < 1e-3 * std::max(1.0, std::abs(hi))) {
        std::fill(out, out + n, 0.5f);
        return;
    }
    for (int i = 0; i < n; ++i)
        out[i] = float(std::clamp((double(float(crop[i]) / 255.0f) - lo) / (hi - lo), -0.5, 1.5));
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

}  // namespace

// ------------------------------------------------------------------------------------------- detector

void detector_preprocess(const Image& view, int size, float* out) {
    const auto tx = area_table(view.width, size), ty = area_table(view.height, size);
    std::vector<float> row(view.width);
    std::vector<uint8_t> small(size_t(size) * size);
    std::vector<double> acc(size);
    for (int dy = 0; dy < size; ++dy) {
        std::fill(acc.begin(), acc.end(), 0.0);
        for (const auto& [sy, wy] : ty[dy]) {
            const uint8_t* src = view.data + size_t(sy) * view.width;
            for (int dx = 0; dx < size; ++dx) {
                float s = 0;
                for (const auto& [sx, wx] : tx[dx]) s += wx * src[sx];
                acc[dx] += double(wy) * s;
            }
        }
        for (int dx = 0; dx < size; ++dx) small[size_t(dy) * size + dx] = round_u8(acc[dx]);
    }
    // detector.standardize (float32)
    const size_t n = small.size();
    double mean = 0;
    for (auto v : small) mean += float(v) / 255.0f;
    mean /= n;
    double var = 0;
    for (auto v : small) { const double d = float(v) / 255.0f - mean; var += d * d; }
    const double sd = std::max(std::sqrt(var / n), 0.02);
    for (size_t i = 0; i < n; ++i) out[i] = float((float(small[i]) / 255.0f - mean) / sd);
}

// detector.decode_view
std::vector<Detection> decode_view(const float* maps, int h, int w, int image_w, int image_h, double threshold,
                                   int top_k, double merge_fraction) {
    const double norm_wh = std::sqrt(double(image_w) * image_h);
    std::vector<Detection> found;
    const size_t plane = size_t(h) * w;
    std::vector<double> heat(plane);
    for (int side = 0; side < 2; ++side) {
        for (size_t i = 0; i < plane; ++i) heat[i] = sigmoid(maps[side * plane + i]);
        std::vector<int> peaks;
        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j) {
                const double v = heat[size_t(i) * w + j];
                bool peak = true;
                for (int di = -1; di <= 1 && peak; ++di)
                    for (int dj = -1; dj <= 1; ++dj) {
                        const int a = i + di, b = j + dj;
                        const double n = (a < 0 || b < 0 || a >= h || b >= w) ? -1.0 : heat[size_t(a) * w + b];
                        if (n > v) { peak = false; break; }
                    }
                if (peak) peaks.push_back(i * w + j);
            }
        std::stable_sort(peaks.begin(), peaks.end(), [&](int a, int b) { return heat[a] > heat[b]; });
        for (int k = 0; k < std::min<int>(top_k, peaks.size()); ++k) {
            const int idx = peaks[k], i = idx / w, j = idx % w;
            const double score = heat[idx];
            if (score < threshold) break;
            const float* r = maps + (2 + 4 * side) * plane + idx;
            const double xn = (j + r[0]) / w, yn = (i + r[plane]) / h;
            found.push_back({side, score, xn * image_w - 0.5, yn * image_h - 0.5,
                             std::exp(double(r[2 * plane])) * norm_wh, std::exp(double(r[3 * plane]))});
        }
    }
    std::stable_sort(found.begin(), found.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    for (const auto& d : found) {
        bool ok = true;
        for (const auto& k : kept)
            if (std::hypot(d.u - k.u, d.v - k.v) < merge_fraction * std::max(d.radius_px, k.radius_px)) { ok = false; break; }
        if (ok) kept.push_back(d);
    }
    return kept;
}

// detector.seed_hands
std::vector<Seed> seed_hands(const std::array<std::vector<Detection>, 2>& detections, const Rig& rig,
                             const SeedConfig& cfg) {
    const KbCamera* cams[2] = {&rig.left, &rig.right};
    std::array<std::vector<std::pair<Vec3, Vec3>>, 2> rays;
    for (int v = 0; v < 2; ++v)
        for (const auto& d : detections[v]) {
            const Vec3 dir = unit(cams[v]->R_cw.tmul(cams[v]->unproject({d.u, d.v})));
            rays[v].push_back({cams[v]->center(), dir});
        }
    struct Cand { double cost; int i, j; double gap; Vec3 mid; };
    std::vector<Cand> cands;
    for (size_t i = 0; i < detections[0].size(); ++i)
        for (size_t j = 0; j < detections[1].size(); ++j) {
            const auto& [c1, d1] = rays[0][i];
            const auto& [c2, d2] = rays[1][j];
            const Vec3 w0 = c1 - c2;
            const double b = dot(d1, d2), d = dot(d1, w0), e = dot(d2, w0), denom = 1.0 - b * b;
            if (denom < 1e-9) continue;
            const double s = (b * e - d) / denom, t = (e - b * d) / denom;
            const Vec3 p1 = c1 + s * d1, p2 = c2 + t * d2;
            const double gap = norm(p1 - p2);
            const Vec3 mid = 0.5 * (p1 + p2);
            if (!(cfg.depth_lo <= s && s <= cfg.depth_hi && cfg.depth_lo <= t && t <= cfg.depth_hi) || gap > cfg.max_ray_gap_m)
                continue;
            const auto& a = detections[0][i];
            const auto& bb = detections[1][j];
            if (std::abs(s - a.distance_m) > cfg.max_distance_ratio * a.distance_m ||
                std::abs(t - bb.distance_m) > cfg.max_distance_ratio * bb.distance_m)
                continue;
            cands.push_back({gap + (a.side != bb.side ? cfg.side_mismatch_cost_m : 0.0), int(i), int(j), gap, mid});
        }
    std::stable_sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        return a.cost != b.cost ? a.cost < b.cost : (a.i != b.i ? a.i < b.i : a.j < b.j);
    });
    std::vector<bool> used0(detections[0].size()), used1(detections[1].size());
    std::vector<Seed> seeds;
    for (const auto& c : cands) {
        if (used0[c.i] || used1[c.j]) continue;
        used0[c.i] = used1[c.j] = true;
        const auto& a = detections[0][c.i];
        const auto& b = detections[1][c.j];
        double side_score[2] = {0, 0};
        side_score[a.side] += a.score;
        side_score[b.side] += b.score;
        seeds.push_back({c.mid, side_score[1] > side_score[0], (a.score + b.score) / 2, {0, 1}, {a, b}, c.gap});
    }
    for (int v = 0; v < 2; ++v) {
        const auto& used = v == 0 ? used0 : used1;
        for (size_t k = 0; k < detections[v].size(); ++k) {
            const auto& d = detections[v][k];
            if (used[k] || d.score < cfg.single_view_threshold) continue;
            const auto& [c, dir] = rays[v][k];
            seeds.push_back({c + d.distance_m * dir, d.is_right(), d.score, {v}, {d}, NAN});
        }
    }
    std::stable_sort(seeds.begin(), seeds.end(), [](const Seed& a, const Seed& b) {
        return a.views.size() != b.views.size() ? a.views.size() > b.views.size() : a.score > b.score;
    });
    std::vector<Seed> kept;
    for (const auto& s : seeds) {
        bool ok = true;
        for (const auto& k : kept) if (norm(s.center - k.center) <= cfg.dedupe_m) { ok = false; break; }
        if (ok) kept.push_back(s);
    }
    std::stable_sort(kept.begin(), kept.end(), [](const Seed& a, const Seed& b) { return a.score > b.score; });
    if (int(kept.size()) > cfg.max_hands) kept.resize(cfg.max_hands);
    return kept;
}

// ------------------------------------------------------------------------------------------- crops

// VirtualCamera.resample: bilinear (OpenCV remap, 1/32-pixel fixed point, zero border) and the valid mask
// (inside the image and within 95 degrees of the optical axis).
void resample(const VirtualCamera& v, const Image& image, const KbCamera& camera, uint8_t* crop, uint8_t* valid) {
    const double max_theta = 95.0 * kPi / 180.0;
    const int S = v.size, W = image.width, H = image.height;
    for (int r = 0; r < S; ++r)
        for (int c = 0; c < S; ++c) {
            const Vec3 ray{(c + v.offset[0]) / v.f, (r + v.offset[1]) / v.f, 1.0};
            const Vec3 dir = unit(v.R.tmul(ray));
            const Vec3 cd = camera.R_cw * dir;
            const double theta = std::acos(std::clamp(cd[2], -1.0, 1.0));
            const Vec2 px = camera.project_camera(cd);
            const size_t o = size_t(r) * S + c;
            const bool inside = px[0] >= 0 && px[0] <= W - 1 && px[1] >= 0 && px[1] <= H - 1;
            valid[o] = inside && theta < max_theta;
            // cv2.remap (OpenCV 5.0, as in the training venv): float32 map coordinates, float32 bilinear
            // blend, round to nearest. This reproduces it bit-exactly.
            const float fx = float(px[0]), fy = float(px[1]);
            if (!std::isfinite(fx) || !std::isfinite(fy) || std::abs(fx) > 1e6f || std::abs(fy) > 1e6f) { crop[o] = 0; continue; }
            const long x0 = long(std::floor(fx)), y0 = long(std::floor(fy));
            const float ax = fx - float(x0), ay = fy - float(y0);
            auto px_at = [&](long x, long y) -> float {
                return (x < 0 || y < 0 || x >= W || y >= H) ? 0.0f : float(image.at(int(x), int(y)));
            };
            const float top = px_at(x0, y0) + ax * (px_at(x0 + 1, y0) - px_at(x0, y0));
            const float bottom = px_at(x0, y0 + 1) + ax * (px_at(x0 + 1, y0 + 1) - px_at(x0, y0 + 1));
            crop[o] = round_u8(double(top + ay * (bottom - top)));
        }
}

// samples.build_sample (inference part: crops, network input and decode geometry).
Sample build_sample(const std::array<Image, 2>& images, const Rig& rig, const Vec3& center, bool is_right, int size,
                    double radius, double fill) {
    constexpr double min_valid = 0.5, min_angle = 20.0;
    const KbCamera* cams[2] = {&rig.left, &rig.right};
    const StereoPair pair = stereo_crop_pair(rig, center, size, radius, fill);
    const VirtualCamera virtuals[2] = {pair.left, pair.right};
    const size_t n = size_t(size) * size;
    std::vector<uint8_t> s_crop[2], s_valid[2], m_crop[2], m_valid[2];
    VirtualCamera m_cams[2];
    const Vec3 baseline_dir = (1.0 / pair.baseline_m) * (pair.right.center - pair.left.center);
    const Vec3 down = rig_down(rig, baseline_dir);
    double rect_frac[2], mono_frac[2];
    for (int i = 0; i < 2; ++i) {
        s_crop[i].resize(n); s_valid[i].resize(n); m_crop[i].resize(n); m_valid[i].resize(n);
        resample(virtuals[i], images[i], *cams[i], s_crop[i].data(), s_valid[i].data());
        m_cams[i] = monocular_view(*cams[i], center, down, baseline_dir, size, radius, fill);
        resample(m_cams[i], images[i], *cams[i], m_crop[i].data(), m_valid[i].data());
        rect_frac[i] = std::accumulate(s_valid[i].begin(), s_valid[i].end(), 0.0) / n;
        mono_frac[i] = std::accumulate(m_valid[i].begin(), m_valid[i].end(), 0.0) / n;
    }
    const bool both = std::min(rect_frac[0], mono_frac[0]) >= min_valid && std::min(rect_frac[1], mono_frac[1]) >= min_valid;
    const bool stereo = both && pair.baseline_angle_deg >= min_angle;
    int order[2] = {is_right ? 1 : 0, is_right ? 0 : 1};
    const double* frac = stereo ? rect_frac : mono_frac;
    if (frac[order[0]] < min_valid && frac[order[1]] >= min_valid) std::swap(order[0], order[1]);
    auto mirror = [&](std::vector<uint8_t>& a) {
        for (int r = 0; r < size; ++r) std::reverse(a.begin() + size_t(r) * size, a.begin() + size_t(r + 1) * size);
    };
    if (is_right) for (int i = 0; i < 2; ++i) { mirror(s_crop[i]); mirror(s_valid[i]); mirror(m_crop[i]); mirror(m_valid[i]); }
    Sample out;
    out.input.assign(2 * n, 0.0f);
    out.second_valid = stereo;
    Geometry& g = out.geometry;
    g.baseline_m = pair.baseline_m;
    g.primary_view = order[0];
    g.mirrored = is_right;
    g.stereo = stereo;
    if (stereo) {
        g.camera = virtuals[order[0]];
        g.center_depth_m = pair.left.depth(center);
        robust_normalize(s_crop[order[0]].data(), s_valid[order[0]].data(), int(n), out.input.data());
        robust_normalize(s_crop[order[1]].data(), s_valid[order[1]].data(), int(n), out.input.data() + n);
    } else {
        g.camera = m_cams[order[0]];
        g.center_depth_m = g.camera.depth(center);
        robust_normalize(m_crop[order[0]].data(), m_valid[order[0]].data(), int(n), out.input.data());
    }
    return out;
}

// samples.decode_prediction
Joints decode_prediction(const Geometry& g, const float* keypoints, const float* disparity) {
    const VirtualCamera& cam = g.camera;
    const double size = cam.size;
    Joints out{};
    for (int j = 0; j < kJoints; ++j) {
        double pu = double(keypoints[2 * j]) * (size / 2) + size / 2;
        const double pv = double(keypoints[2 * j + 1]) * (size / 2) + size / 2;
        if (g.mirrored) pu = (size - 1) - pu;
        const double dc = double(disparity[j]) * size / 8.0;
        const double z = 1.0 / (dc / (cam.f * g.baseline_m) + 1.0 / g.center_depth_m);
        const Vec3 p{(pu + cam.offset[0]) * z / cam.f, (pv + cam.offset[1]) * z / cam.f, z};
        out[j] = cam.R.tmul(p) + cam.center;
    }
    return out;
}

}  // namespace nirhands
