// Native model execution and the tracker: a port of HandPipeline in hand-tracking/tools/nirhands_pipeline.py.
#include "nirhands/pipeline.hpp"
#include "nirhands/skeleton.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace nirhands {

// ------------------------------------------------------------------------------------------- models

namespace {
int manifest_size(const std::string& text, const std::string& model) {
    // "detector": {..., "size": 224, ...}; the manifest is ours, so a regex is enough.
    const std::regex re("\"" + model + "\"\\s*:\\s*\\{[^}]*\"size\"\\s*:\\s*(\\d+)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) throw std::runtime_error("manifest.json has no size for " + model);
    return std::stoi(m[1]);
}

SkeletonProfile manifest_profile(const std::string& text, bool& skeleton) {
    const std::regex arch_re("\"keypoints\"\\s*:\\s*\\{[^}]*\"arch\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    const bool has_arch = std::regex_search(text, match, arch_re);
    skeleton = has_arch && match[1] == "skeleton";
    if (has_arch && !skeleton && match[1] != "keypoint")
        throw std::runtime_error("manifest.json has unknown keypoint architecture " + match[1].str());
    SkeletonProfile profile{};
    if (!skeleton) return profile;
    const std::regex profile_re("\"keypoints\"\\s*:\\s*\\{[^}]*\"mean_profile\"\\s*:\\s*\\[([^\\]]*)\\]");
    if (!std::regex_search(text, match, profile_re))
        throw std::runtime_error("manifest.json has no skeleton mean_profile");
    const std::string values = match[1];
    const std::regex number("[-+]?([0-9]+(\\.[0-9]*)?|\\.[0-9]+)([eE][-+]?[0-9]+)?");
    int count = 0;
    for (std::sregex_iterator i(values.begin(), values.end(), number), end; i != end; ++i) {
        if (count >= kSkeletonProfileValues) throw std::runtime_error("skeleton mean_profile is too long");
        profile[count++] = std::stof(i->str());
    }
    if (count != kSkeletonProfileValues) throw std::runtime_error("skeleton mean_profile must have 33 values");
    return profile;
}

bool manifest_tight_crop(const std::string& text) {
    const std::regex re("\"keypoints\"\\s*:\\s*\\{[^}]*\"tight_crop\"\\s*:\\s*(true|false)");
    std::smatch m;
    return std::regex_search(text, m, re) && m[1] == "true";
}
}  // namespace

struct Models::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "nirhands"};
    std::unique_ptr<Ort::Session> detector;
    std::vector<std::unique_ptr<Ort::Session>> keypoints;  // one per concurrent hand
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int det_size = 0, kp_size = 0;
    bool skeleton = false;
    bool tight_crop = false;
    SkeletonProfile mean_profile{};
    Run run;
};

Models::Models(const std::string& dir, int threads) : impl_(new Impl) {
    std::ifstream f(dir + "/manifest.json");
    if (!f) throw std::runtime_error("cannot read " + dir + "/manifest.json");
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string manifest = ss.str();
    impl_->det_size = manifest_size(manifest, "detector");
    impl_->kp_size = manifest_size(manifest, "keypoints");
    impl_->mean_profile = manifest_profile(manifest, impl_->skeleton);
    impl_->tight_crop = manifest_tight_crop(manifest);
    Ort::SessionOptions o;
    o.SetIntraOpNumThreads(threads);
    o.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    impl_->detector = std::make_unique<Ort::Session>(impl_->env, (dir + "/detector.onnx").c_str(), o);
    for (int i = 0; i < 2; ++i)
        impl_->keypoints.push_back(std::make_unique<Ort::Session>(impl_->env, (dir + "/keypoints.onnx").c_str(), o));
}

Models::Models(const std::string& dir, Run run) : impl_(new Impl) {
    if (!run) throw std::invalid_argument("NIR NPU runner is empty");
    std::ifstream f(dir + "/manifest.json");
    if (!f) throw std::runtime_error("cannot read " + dir + "/manifest.json");
    std::stringstream ss; ss << f.rdbuf();
    const std::string manifest = ss.str();
    impl_->det_size = manifest_size(manifest, "detector");
    impl_->kp_size = manifest_size(manifest, "keypoints");
    impl_->mean_profile = manifest_profile(manifest, impl_->skeleton);
    impl_->tight_crop = manifest_tight_crop(manifest);
    impl_->run = std::move(run);
}

Models::~Models() = default;
int Models::detector_size() const { return impl_->det_size; }
int Models::keypoint_size() const { return impl_->kp_size; }
bool Models::tight_crop() const { return impl_->tight_crop; }
bool Models::is_skeleton() const { return impl_->skeleton; }

void Models::detect(const float* input, std::vector<float>& maps) {
    const int S = impl_->det_size;
    if (impl_->run) {
        const auto out = impl_->run(0, {{input, size_t(2) * S * S * sizeof(float)}}, nullptr);
        const int M = S / 4, C = 10;
        if (out.size() != 1 || out[0].size() != size_t(2) * M * M * C * sizeof(float))
            throw std::runtime_error("NIR NPU detector output size mismatch");
        maps.resize(size_t(2) * C * M * M);
        const auto* source = reinterpret_cast<const float*>(out[0].data());
        for (int b=0;b<2;++b) for (int y=0;y<M;++y) for (int x=0;x<M;++x) for (int c=0;c<C;++c)
            maps[((b*C+c)*M+y)*M+x] = source[((b*M+y)*M+x)*C+c];
        return;
    }
    const int64_t shape[4] = {2, 1, S, S};
    Ort::Value in = Ort::Value::CreateTensor<float>(impl_->mem, const_cast<float*>(input), size_t(2) * S * S, shape, 4);
    const char* in_names[] = {"image"};
    const char* out_names[] = {"maps"};
    auto out = impl_->detector->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    const size_t count = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const float* p = out[0].GetTensorData<float>();
    maps.assign(p, p + count);
}

void Models::keypoints(int slot, const float* input, bool second_valid, const Geometry& geometry,
                       float* kp, float* disp, float* vis_logit, float* kp_hm_out, float* disp_hm_out) {
    const int K = impl_->kp_size;
    if (impl_->run) {
        (void)slot;
        std::vector<float> nhwc(size_t(2)*K*K);
        for (int y=0;y<K;++y) for (int x=0;x<K;++x) for (int c=0;c<2;++c)
            nhwc[(y*K+x)*2+c] = input[(c*K+y)*K+x];
        float sv = second_valid ? 1.0f : 0.0f;
        const auto out = impl_->run(1, {{nhwc.data(), nhwc.size()*sizeof(float)}, {&sv, sizeof(sv)}}, &geometry);
        if (impl_->skeleton) {
            // QNN orders the three skeleton outputs differently from one compile to the next, and
            // their sizes are distinct, so take scale (1), palm points (18) and angles (30) by size.
            const float *scale = nullptr, *palm = nullptr, *angles = nullptr;
            if (out.size()!=6 || out[0].size()!=2*kJoints*sizeof(float) ||
                out[1].size()!=kJoints*sizeof(float) || out[2].size()!=kJoints*sizeof(float))
                throw std::runtime_error("NIR NPU skeleton output size mismatch");
            for (size_t i=3;i<6;++i) {
                const auto* values = reinterpret_cast<const float*>(out[i].data());
                const float** slot = out[i].size()==sizeof(float) ? &scale : out[i].size()==18*sizeof(float) ? &palm :
                                     out[i].size()==30*sizeof(float) ? &angles : nullptr;
                if (!slot || *slot) throw std::runtime_error("NIR NPU skeleton output size mismatch");
                *slot = values;
            }
            const auto* kp_hm = reinterpret_cast<const float*>(out[0].data());
            const auto* disp_hm = reinterpret_cast<const float*>(out[1].data());
            std::memcpy(vis_logit,out[2].data(),out[2].size());
            if (kp_hm_out) std::memcpy(kp_hm_out,out[0].data(),out[0].size());
            if (disp_hm_out) std::memcpy(disp_hm_out,out[1].data(),out[1].size());
            decode_skeleton(geometry, impl_->mean_profile, kp_hm, disp_hm, palm, angles, *scale, kp, disp);
            return;
        }
        if (out.size()!=3 || out[0].size()!=2*kJoints*sizeof(float) ||
            out[1].size()!=kJoints*sizeof(float) || out[2].size()!=kJoints*sizeof(float))
            throw std::runtime_error("NIR NPU keypoint output size mismatch");
        std::memcpy(kp,out[0].data(),out[0].size());
        std::memcpy(disp,out[1].data(),out[1].size());
        std::memcpy(vis_logit,out[2].data(),out[2].size());
        return;
    }
    const int64_t shape[4] = {1, 2, K, K}, sshape[1] = {1};
    float sv = second_valid ? 1.0f : 0.0f;
    Ort::Value in[2] = {
        Ort::Value::CreateTensor<float>(impl_->mem, const_cast<float*>(input), size_t(2) * K * K, shape, 4),
        Ort::Value::CreateTensor<float>(impl_->mem, &sv, 1, sshape, 1)};
    const char* in_names[] = {"crops", "second_valid"};
    if (impl_->skeleton) {
        const char* out_names[] = {"keypoints_hm", "disparity_hm", "visibility_logit", "palm_points", "angles", "scale"};
        auto out = impl_->keypoints.at(slot)->Run(Ort::RunOptions{nullptr}, in_names, in, 2, out_names, 6);
        std::copy_n(out[2].GetTensorData<float>(), kJoints, vis_logit);
        if (kp_hm_out) std::copy_n(out[0].GetTensorData<float>(), 2 * kJoints, kp_hm_out);
        if (disp_hm_out) std::copy_n(out[1].GetTensorData<float>(), kJoints, disp_hm_out);
        decode_skeleton(geometry, impl_->mean_profile, out[0].GetTensorData<float>(), out[1].GetTensorData<float>(),
                        out[3].GetTensorData<float>(), out[4].GetTensorData<float>(), *out[5].GetTensorData<float>(), kp, disp);
        return;
    }
    const char* out_names[] = {"keypoints", "disparity", "visibility_logit"};
    auto out = impl_->keypoints.at(slot)->Run(Ort::RunOptions{nullptr}, in_names, in, 2, out_names, 3);
    std::copy_n(out[0].GetTensorData<float>(), 2 * kJoints, kp);
    std::copy_n(out[1].GetTensorData<float>(), kJoints, disp);
    std::copy_n(out[2].GetTensorData<float>(), kJoints, vis_logit);
}

// ------------------------------------------------------------------------------------------- smoothing

Joints OneEuro::operator()(const Joints& x, double t) {
    if (!init_ || t <= t_) {
        x_ = x;
        for (auto& d : dx_) d = {0, 0, 0};
        t_ = t;
        init_ = true;
        return x;
    }
    const double dt = t - t_;
    auto alpha = [dt](double cutoff) { const double tau = 1.0 / (2 * M_PI * cutoff); return 1.0 / (1.0 + tau / dt); };
    const double a_d = alpha(d_cutoff_);
    for (int j = 0; j < kJoints; ++j) {
        const Vec3 dx = (1.0 / dt) * (x[j] - x_[j]);
        dx_[j] = a_d * dx + (1 - a_d) * dx_[j];
        const double a = alpha(min_cutoff_ + beta_ * norm(dx_[j]));
        x_[j] = a * x[j] + (1 - a) * x_[j];
    }
    t_ = t;
    return x_;
}

// ------------------------------------------------------------------------------------------- tracking

struct Pipeline::Track {
    int id = 0;
    Vec3 center{};
    double side_vote = 0, t = 0;
    int hits = 1, missed = 0, weak = 0, age = 0;
    Vec3 velocity{0, 0, 0};
    bool has_joints = false;
    Joints joints{}, smoothed{};  // joints: skeleton output (tracking); smoothed: shown joints, filtered
    Joints shown{};               // unsmoothed shown joints (TrackerConfig::output)
    bool combined = false;
    std::array<float, kJoints> visibility{};
    bool stereo = false;
    double score = 0;
    std::vector<int> views;
    std::unique_ptr<OneEuro> filter;
    bool confirmed = false;
    double last_hit_t = 0;
    std::deque<double> spans;
    bool is_right() const { return side_vote > 0; }
    double vis_mean() const {
        double s = 0;
        for (float v : visibility) s += v;
        return s / kJoints;
    }
};

struct Pipeline::Impl {
    std::vector<std::unique_ptr<Track>> tracks;
    struct Recent { int id; bool is_right; Vec3 center; double t; };
    std::vector<Recent> recent;
    int frame = 0, next_id = 1;
    double t = 0;
    std::vector<float> det_input, maps;
    ProfileEstimator profiles;
};

namespace {
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

Pipeline::Pipeline(const Rig& rig, std::shared_ptr<Models> models, TrackerConfig cfg)
    : rig_(rig), models_(std::move(models)), cfg_(cfg), impl_(new Impl) {
    impl_->profiles = ProfileEstimator(cfg_.profile_window, cfg_.profile_min_samples);
}
Pipeline::~Pipeline() = default;

void Pipeline::reset() {
    impl_->profiles.reset();
    impl_->tracks.clear();
    impl_->recent.clear();
    impl_->frame = 0;
    impl_->next_id = 1;
}

FrameResult Pipeline::step(const std::array<Image, 2>& images, double t) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    Impl& s = *impl_;
    const TrackerConfig& cfg = cfg_;
    FrameResult result;
    s.t = t;
    s.recent.erase(std::remove_if(s.recent.begin(), s.recent.end(),
                                  [&](const Impl::Recent& r) { return t - r.t > cfg.reuse_id_s; }), s.recent.end());
    for (auto& tr : s.tracks) {
        const double dt = std::min(std::max(t - tr->t, 0.0), 0.2);
        Vec3 v = tr->velocity;
        const double speed = norm(v);
        if (speed > cfg.max_speed_mps) v = (cfg.max_speed_mps / speed) * v;
        tr->center = tr->center + dt * v;
    }

    auto retire = [&](Track& tr) {
        if (tr.confirmed) s.recent.push_back({tr.id, tr.is_right(), tr.center, s.t});
    };
    auto confirmed_count = [&]() {
        int n = 0;
        for (auto& tr : s.tracks) n += tr->confirmed;
        return n;
    };

    // ---- detection (Detector.__call__ + _need_detect) ----
    bool need = confirmed_count() < cfg.max_hands;
    for (auto& tr : s.tracks) need = need || !tr->confirmed || tr->weak;
    need = need || s.frame % cfg.detect_every == 0;
    if (need) {
        const int S = models_->detector_size();
        s.det_input.resize(size_t(2) * S * S);
        for (int v = 0; v < 2; ++v) detector_preprocess(images[v], S, s.det_input.data() + size_t(v) * S * S);
        models_->detect(s.det_input.data(), s.maps);
        const int h = S / 4, w = S / 4;
        std::array<std::vector<Detection>, 2> cands;
        for (int v = 0; v < 2; ++v) {
            cands[v] = decode_view(s.maps.data() + size_t(v) * 10 * h * w, h, w, images[v].width, images[v].height,
                                   cfg.candidate_threshold, cfg.top_k);
            for (const auto& d : cands[v]) if (d.score >= cfg.detect_threshold) result.detections[v].push_back(d);
        }
        SeedConfig sc;
        sc.single_view_threshold = cfg.single_view_threshold;
        result.seeds = seed_hands(cands, rig_, sc);
        result.detected = true;
        result.detect_ms = ms_since(t0);

        // ---- _associate ----
        const auto& seeds = result.seeds;
        auto cost = [&](const Seed& seed, const Vec3& center) {
            if (seed.views.size() == 2) return norm(seed.center - center) / cfg.associate_m;
            const KbCamera& cam = seed.views[0] == 0 ? rig_.left : rig_.right;
            const Detection& det = seed.detections[0];
            if (cam.to_camera(center)[2] <= 0) return std::numeric_limits<double>::infinity();
            const Vec2 uv = cam.project(center);
            return std::hypot(uv[0] - det.u, uv[1] - det.v) / (cfg.associate_radius * std::max(det.radius_px, 20.0));
        };
        struct Pair { double cost; size_t i, k; };
        std::vector<Pair> pairs;
        for (size_t i = 0; i < s.tracks.size(); ++i)
            for (size_t k = 0; k < seeds.size(); ++k) pairs.push_back({cost(seeds[k], s.tracks[i]->center), i, k});
        std::stable_sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
            return a.cost != b.cost ? a.cost < b.cost : (a.i != b.i ? a.i < b.i : a.k < b.k);
        });
        std::vector<bool> used_t(s.tracks.size()), used_s(seeds.size());
        for (const auto& p : pairs) {
            if (p.cost > 1.0 || used_t[p.i] || used_s[p.k]) continue;
            used_t[p.i] = used_s[p.k] = true;
            Track& tr = *s.tracks[p.i];
            const Seed& sd = seeds[p.k];
            tr.hits += 1;
            tr.missed = 0;
            tr.score = sd.score;
            tr.views = sd.views;
            tr.last_hit_t = t;
            if (tr.weak) tr.center = sd.center;
            tr.side_vote = 0.8 * tr.side_vote + (sd.is_right ? sd.score : -sd.score);
            if (!tr.confirmed && tr.hits >= cfg.confirm_hits && tr.has_joints && confirmed_count() < cfg.max_hands &&
                tr.vis_mean() >= cfg.confirm_visibility && int(tr.spans.size()) >= cfg.confirm_span_frames &&
                median({tr.spans.begin(), tr.spans.end()}) >= cfg.confirm_span_m)
                tr.confirmed = true;
        }
        std::vector<std::unique_ptr<Track>> kept;
        for (size_t i = 0; i < s.tracks.size(); ++i) {
            Track& tr = *s.tracks[i];
            if (!used_t[i]) {
                tr.missed += 1;
                if (!tr.confirmed && tr.missed >= 2) continue;
                const bool clear = tr.has_joints && tr.vis_mean() >= cfg.visible_mean;
                const double unseen = clear ? cfg.max_unseen_visible_s : cfg.max_unseen_s;
                if (tr.missed >= cfg.max_missed && t - tr.last_hit_t >= unseen) { retire(tr); continue; }
            }
            if (!tr.confirmed && tr.age >= cfg.max_tentative_frames) continue;
            kept.push_back(std::move(s.tracks[i]));
        }
        s.tracks = std::move(kept);
        std::vector<size_t> order(seeds.size());
        for (size_t k = 0; k < order.size(); ++k) order[k] = k;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return seeds[a].score > seeds[b].score; });
        for (size_t k : order) {
            const Seed& sd = seeds[k];
            if (used_s[k] || int(s.tracks.size()) >= cfg.max_hands + cfg.max_tentative) continue;
            if (sd.score < (sd.views.size() == 2 ? cfg.start_stereo_score : cfg.start_mono_score)) continue;
            bool near = false;
            for (auto& tr : s.tracks) near = near || cost(sd, tr->center) <= 1.0;
            if (near) continue;
            auto tr = std::make_unique<Track>();
            tr->id = s.next_id;
            tr->center = sd.center;
            tr->side_vote = sd.is_right ? sd.score : -sd.score;
            tr->t = t;
            tr->score = sd.score;
            tr->views = sd.views;
            tr->last_hit_t = t;
            tr->confirmed = cfg.confirm_hits <= 1;
            int reuse = -1;
            for (int r = int(s.recent.size()) - 1; r >= 0 && reuse < 0; --r) {
                const auto& rc = s.recent[r];
                bool free = true;
                for (auto& o : s.tracks) free = free && o->id != rc.id;
                if (rc.is_right == sd.is_right && norm(rc.center - sd.center) < cfg.reuse_id_m && free) reuse = r;
            }
            if (reuse >= 0 && confirmed_count() < cfg.max_hands) {
                tr->id = s.recent[reuse].id;
                tr->confirmed = true;
                s.recent.erase(s.recent.begin() + reuse);
            } else {
                s.next_id += 1;
            }
            s.tracks.push_back(std::move(tr));
        }
    }

    // ---- keypoints: one session per hand, run concurrently ----
    const auto t1 = Clock::now();
    const int K = models_->keypoint_size();
    struct Out {
        bool ok = false; Joints joints{}; std::array<float, kJoints> vis{}; bool stereo = false;
        bool has_heatmap = false; Joints heatmap{};  // skeleton models: decoded auxiliary heatmap keypoints
    };
    const bool want_heatmap = models_->is_skeleton() && cfg.output != TrackerConfig::Output::Skeleton;
    std::vector<Out> outs(s.tracks.size());
    auto run_one = [&](size_t i, int slot) {
        const Track& tr = *s.tracks[i];
        Vec3 anchor = tr.center;
        double radius = 0.10;
        if (models_->tight_crop() && tr.has_joints) {
            const Vec3 shift = tr.center - tr.joints[kRoot];
            Vec3 min_pt = tr.joints[0] + shift, max_pt = min_pt;
            for (int j = 1; j < kJoints; ++j) {
                const Vec3 p = tr.joints[j] + shift;
                for (int c = 0; c < 3; ++c) {
                    min_pt[c] = std::min(min_pt[c], p[c]);
                    max_pt[c] = std::max(max_pt[c], p[c]);
                }
            }
            anchor = 0.5 * (min_pt + max_pt);
            double max_d = 0;
            for (int j = 0; j < kJoints; ++j) {
                const Vec3 p = tr.joints[j] + shift;
                max_d = std::max(max_d, norm(p - anchor));
            }
            double half = max_d * 1.3;
            half = std::min(std::max(half, 0.06), 0.16);
            radius = half * 0.8;
        }
        const Sample sample = build_sample(images, rig_, anchor, tr.is_right(), K, radius, 0.8);
        float kp[2 * kJoints], disp[kJoints], vl[kJoints], kp_hm[2 * kJoints], disp_hm[kJoints];
        models_->keypoints(slot, sample.input.data(), sample.second_valid, sample.geometry, kp, disp, vl,
                           want_heatmap ? kp_hm : nullptr, want_heatmap ? disp_hm : nullptr);
        Out o;
        o.joints = decode_prediction(sample.geometry, kp, disp);
        o.ok = true;
        for (const auto& p : o.joints) o.ok = o.ok && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
        if (want_heatmap) {
            o.heatmap = decode_prediction(sample.geometry, kp_hm, disp_hm);
            o.has_heatmap = true;
            for (const auto& p : o.heatmap)
                o.has_heatmap = o.has_heatmap && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
        }
        for (int j = 0; j < kJoints; ++j) o.vis[j] = float(1.0 / (1.0 + std::exp(-double(vl[j]))));
        o.stereo = sample.geometry.stereo;
        outs[i] = o;
    };
    for (size_t i = 0; i < s.tracks.size(); i += 2) {
        std::future<void> second;
        if (i + 1 < s.tracks.size()) second = std::async(std::launch::async, run_one, i + 1, 1);
        run_one(i, 0);
        if (second.valid()) second.get();
    }
    result.keypoints_ms = ms_since(t1);

    std::vector<std::unique_ptr<Track>> survivors;
    for (size_t i = 0; i < s.tracks.size(); ++i) {
        Track& tr = *s.tracks[i];
        const Out& o = outs[i];
        if (!o.ok) continue;
        const Vec3 root = o.joints[kRoot];
        if (norm(root - tr.center) > cfg.max_jump_m) {
            tr.weak += 1;
            tr.velocity = {0, 0, 0};
            if (tr.has_joints) tr.center = tr.joints[kRoot];
            if (tr.weak >= cfg.max_weak || !tr.has_joints) { retire(tr); continue; }
            survivors.push_back(std::move(s.tracks[i]));
            continue;
        }
        const double dt = t - tr.t;
        if (tr.has_joints && dt > 1e-4) tr.velocity = 0.5 * tr.velocity + (0.5 / dt) * (root - tr.joints[kRoot]);
        tr.joints = o.joints;
        tr.has_joints = true;
        tr.visibility = o.vis;
        tr.stereo = o.stereo;
        tr.t = t;
        tr.center = root;
        tr.weak = tr.vis_mean() < cfg.min_visibility ? tr.weak + 1 : 0;
        if (tr.weak >= cfg.max_weak) { retire(tr); continue; }
        // Shown joints (TrackerConfig::output). Tracking above keeps using the skeleton output.
        tr.shown = tr.joints;
        tr.combined = false;
        if (o.has_heatmap && cfg.output == TrackerConfig::Output::Heatmap) {
            tr.shown = o.heatmap;
        } else if (o.has_heatmap && cfg.output == TrackerConfig::Output::Combined) {
            const bool right = tr.is_right();
            if (o.stereo && tr.vis_mean() >= cfg.profile_min_visibility) s.profiles.add(o.heatmap, right);
            SkeletonProfile profile;
            if (s.profiles.profile(right, profile))
                tr.combined = combine_skeleton(tr.joints, o.heatmap, right, profile, cfg.heatmap_angle_weight, tr.shown);
        }
        if (!tr.filter) tr.filter = std::make_unique<OneEuro>(cfg.min_cutoff, cfg.beta, cfg.d_cutoff);
        tr.smoothed = (*tr.filter)(tr.shown, t);
        tr.spans.push_back(norm(tr.joints[5] - tr.joints[17]));
        if (int(tr.spans.size()) > cfg.span_window) tr.spans.pop_front();
        if (tr.confirmed && int(tr.spans.size()) == cfg.span_window &&
            median({tr.spans.begin(), tr.spans.end()}) < cfg.keep_span_m) { retire(tr); continue; }
        tr.age += 1;
        survivors.push_back(std::move(s.tracks[i]));
    }

    // ---- _dedupe ----
    std::stable_sort(survivors.begin(), survivors.end(), [](const auto& a, const auto& b) {
        if (a->confirmed != b->confirmed) return a->confirmed;
        if (a->hits != b->hits) return a->hits > b->hits;
        return a->vis_mean() > b->vis_mean();
    });
    std::vector<std::unique_ptr<Track>> kept;
    for (auto& tr : survivors) {
        bool dup = false;
        for (auto& k : kept) {
            if (norm(tr->center - k->center) >= cfg.duplicate_m) continue;
            if (!tr->has_joints || !k->has_joints || norm(tr->joints[0] - k->joints[0]) < cfg.duplicate_m) { dup = true; break; }
        }
        if (dup) { retire(*tr); continue; }
        kept.push_back(std::move(tr));
    }
    std::vector<std::unique_ptr<Track>> confirmed, tentative;
    for (auto& tr : kept) (tr->confirmed ? confirmed : tentative).push_back(std::move(tr));
    std::stable_sort(confirmed.begin(), confirmed.end(), [](const auto& a, const auto& b) { return a->id < b->id; });
    if (int(confirmed.size()) > cfg.max_hands) confirmed.resize(cfg.max_hands);
    if (int(tentative.size()) > cfg.max_tentative) tentative.resize(cfg.max_tentative);
    s.tracks = std::move(confirmed);
    for (auto& tr : tentative) s.tracks.push_back(std::move(tr));
    s.frame += 1;

    for (auto& tr : s.tracks) {
        if (!tr->confirmed) continue;
        HandResult h;
        h.id = tr->id;
        h.is_right = tr->is_right();
        h.joints = tr->smoothed;
        h.raw = tr->shown;
        h.combined = tr->combined;
        h.visibility = tr->visibility;
        h.stereo = tr->stereo;
        h.score = tr->score;
        h.age = tr->age;
        result.hands.push_back(h);
    }
    result.total_ms = ms_since(t0);
    return result;
}

}  // namespace nirhands
