// Copyright 2022-2026, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
//
// Ported from Monado's src/xrt/tracking/hand/mercury/hg_sync.cpp and
// hg_model.cpp at 3acbda6b7dacbe014749020e8543fded54ef076b (authors Jakob
// Bornecrantz, Moshi Turner, Charlton Rodda). Changes for Frame Spatial
// Runtime: synchronous calls with batched network requests through
// mercury::Networks, pinhole cameras, no debug sinks, masks or u_var, and a
// two-entry history in place of HistoryBuffer.
#include "mercury/tracker.hpp"

#include "model_io.hpp"

#include "math/m_api.h"
#include "math/m_eigen_interop.hpp"
#include "math/m_vec3.h"
#include "mercury/hg_hand_size_opt.hpp"
#include "mercury/kine_lm/lm_interface.hpp"
#include "util/u_box_iou.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace mercury {

using namespace detail;
namespace lm = hm::lm;
using xrt::auxiliary::math::map_quat;
using xrt::auxiliary::math::map_vec3;

namespace {

enum class Provenance { Detection, Prediction };

struct Roi {
    Provenance provenance = Provenance::Detection;
    xrt_vec2 center_px{};
    float size_px = 0;
    bool found = false;
    float confidence = 0;
};

// Keeps the two most recent entries; age 0 is the newest.
template <typename T>
struct History2 {
    T items[2]{};
    size_t count = 0, head = 0;
    void push(const T& v) {
        head = (head + 1) % 2;
        items[head] = v;
        count = std::min<size_t>(count + 1, 2);
    }
    const T& age(size_t a) const { return items[(head + 2 - a) % 2]; }
    void clear() { count = 0; }
};

const xrt_hand_joint joints_5x5_to_26[5][5] = {
    {XRT_HAND_JOINT_WRIST, XRT_HAND_JOINT_THUMB_METACARPAL, XRT_HAND_JOINT_THUMB_PROXIMAL,
     XRT_HAND_JOINT_THUMB_DISTAL, XRT_HAND_JOINT_THUMB_TIP},
    {XRT_HAND_JOINT_INDEX_METACARPAL, XRT_HAND_JOINT_INDEX_PROXIMAL, XRT_HAND_JOINT_INDEX_INTERMEDIATE,
     XRT_HAND_JOINT_INDEX_DISTAL, XRT_HAND_JOINT_INDEX_TIP},
    {XRT_HAND_JOINT_MIDDLE_METACARPAL, XRT_HAND_JOINT_MIDDLE_PROXIMAL, XRT_HAND_JOINT_MIDDLE_INTERMEDIATE,
     XRT_HAND_JOINT_MIDDLE_DISTAL, XRT_HAND_JOINT_MIDDLE_TIP},
    {XRT_HAND_JOINT_RING_METACARPAL, XRT_HAND_JOINT_RING_PROXIMAL, XRT_HAND_JOINT_RING_INTERMEDIATE,
     XRT_HAND_JOINT_RING_DISTAL, XRT_HAND_JOINT_RING_TIP},
    {XRT_HAND_JOINT_LITTLE_METACARPAL, XRT_HAND_JOINT_LITTLE_PROXIMAL, XRT_HAND_JOINT_LITTLE_INTERMEDIATE,
     XRT_HAND_JOINT_LITTLE_DISTAL, XRT_HAND_JOINT_LITTLE_TIP},
};

float box_iou(const Roi& a, const Roi& b) {
    if (!a.found || !b.found) return -1;
    namespace bi = xrt::auxiliary::util::box_iou;
    return bi::boxIOU(bi::Box(a.center_px, a.size_px), bi::Box(b.center_px, b.size_px));
}

// Monado's u_hand_joints_apply_joint_width (util/u_hand_tracking.c).
void apply_joint_width(xrt_hand_joint_set& set) {
    xrt_hand_joint_value* gr = set.values.hand_joint_set_default;
    static const float finger_joint_size[5] = {0.022f, 0.021f, 0.022f, 0.021f, 0.02f};
    static const float hand_finger_size[4] = {1.0f, 1.0f, 0.83f, 0.75f};
    static const float thumb_size[4] = {0.016f, 0.014f, 0.012f, 0.012f};
    for (int i = XRT_HAND_JOINT_THUMB_METACARPAL; i <= XRT_HAND_JOINT_THUMB_TIP; i++)
        gr[i].radius = thumb_size[i - XRT_HAND_JOINT_THUMB_METACARPAL];
    for (int finger = 0; finger < 4; finger++)
        for (int joint = 0; joint < 5; joint++)
            gr[finger * 5 + joint + XRT_HAND_JOINT_INDEX_METACARPAL].radius =
                finger_joint_size[joint] * hand_finger_size[finger] * 0.5f;
    gr[XRT_HAND_JOINT_PALM].radius = 0.032f * 0.5f;
    gr[XRT_HAND_JOINT_WRIST].radius = 0.040f * 0.5f;
}

void joint_set_to_21(const xrt_hand_joint_set& set, Joints21& out) {
    int acc = 0;
    out.col(acc++) = map_vec3(set.values.hand_joint_set_default[XRT_HAND_JOINT_WRIST].relation.pose.position);
    for (int finger = 0; finger < 5; finger++)
        for (int joint = 1; joint < 5; joint++)
            out.col(acc++) =
                map_vec3(set.values.hand_joint_set_default[joints_5x5_to_26[finger][joint]].relation.pose.position);
}

void validate_camera(const PinholeCamera& c, const char* name) {
    if (!(c.fx > 0 && c.fy > 0 && std::isfinite(c.cx) && std::isfinite(c.cy) && c.width > 0 && c.height > 0))
        throw std::invalid_argument(std::string("mercury: invalid ") + name + " camera");
}

}  // namespace

struct Tracker::State {
    StereoCalibration calibration;
    Networks& networks;
    Options options;
    xrt_pose left_in_right{};
    PinholeCamera cameras[2];

    Roi rois[2][2];  // [view][hand]
    hm::one_frame_input keypoint_outputs[2];
    bool hand_seen_before[2] = {};
    bool last_frame_hand_detected[2] = {};
    bool this_frame_hand_detected[2] = {};
    History2<Joints21> history_hands[2];
    History2<int64_t> history_timestamps;
    uint64_t hand_tracked_for_num_frames[2] = {};
    Joints21 pose_predicted_keypoints[2];
    int detection_counter = 0;
    float target_hand_size = hm::STANDARD_HAND_SIZE;
    int64_t current_timestamp = 0;
    bool have_timestamp = false;
    hm::HandSizeRefinement hand_size_refinement{};
    lm::KinematicHandLM* kinematic_hands[2] = {nullptr, nullptr};
    GrayImage images[2];

    State(const StereoCalibration& c, Networks& n, const Options& o) : calibration(c), networks(n), options(o) {
        validate_camera(c.left, "left");
        validate_camera(c.right, "right");
        cameras[0] = c.left;
        cameras[1] = c.right;
        left_in_right.orientation = {c.left_in_right.orientation.x, c.left_in_right.orientation.y,
                                     c.left_in_right.orientation.z, c.left_in_right.orientation.w};
        math_quat_normalize(&left_in_right.orientation);
        left_in_right.position = {c.left_in_right.position.x, c.left_in_right.position.y,
                                  c.left_in_right.position.z};
        const u_logging_level level = o.log_debug ? U_LOGGING_DEBUG : U_LOGGING_WARN;
        lm::optimizer_create(left_in_right, false, level, &kinematic_hands[0]);
        lm::optimizer_create(left_in_right, true, level, &kinematic_hands[1]);
    }

    ~State() {
        lm::optimizer_destroy(&kinematic_hands[0]);
        lm::optimizer_destroy(&kinematic_hands[1]);
    }

    bool outside_view(int view, xrt_vec2 kp) const {
        return kp.y > float(cameras[view].height) || kp.y < 0 || kp.x > float(cameras[view].width) || kp.x < 0;
    }

    // Mercury back_project(): project 21 joints (left-camera axes) into both views.
    void back_project(const Joints21& pts, int hand_idx, int num_outside[2]) {
        for (int view = 0; view < 2; view++) {
            Eigen::Vector3f p = Eigen::Vector3f::Zero();
            Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
            if (view == 1) {
                p = map_vec3(left_in_right.position);
                q = map_quat(left_in_right.orientation);
            }
            bool invalid[21] = {};
            xrt_vec2 kp[21];
            for (int i = 0; i < 21; i++) {
                Eigen::Vector3f r = q * Eigen::Vector3f(pts.col(i)) + p;
                if (r.z() > 0) invalid[i] = true;
                invalid[i] = !flip_and_project(cameras[view], r.x(), r.y(), r.z(), kp[i].x, kp[i].y) || invalid[i];
                invalid[i] = invalid[i] || outside_view(view, kp[i]);
            }
            if (!num_outside) continue;
            num_outside[view] = int(std::count(std::begin(invalid), std::end(invalid), true));
            xrt_vec2 mn = kp[0], mx = kp[0];
            for (int i = 0; i < 21; i++) {
                mn.x = std::fmin(kp[i].x, mn.x);
                mn.y = std::fmin(kp[i].y, mn.y);
                mx.x = std::fmax(kp[i].x, mx.x);
                mx.y = std::fmax(kp[i].y, mx.y);
            }
            xrt_vec2 center = {(mn.x + mx.x) * 0.5f, (mn.y + mx.y) * 0.5f};
            float r = std::fmax(center.x - mn.x, center.y - mn.y);
            rois[view][hand_idx].center_px = center;
            rois[view][hand_idx].size_px = r * 2;
        }
    }

    int dispatch_and_process_hand_detections() {
        if (options.always_run_detection_model) {
            for (int h = 0; h < 2; h++) {
                this_frame_hand_detected[h] = false;
                history_hands[h].clear();
            }
        }
        Roi infos[2][2];  // [view][hand], all not found
        const bool no_hands_detected_last_frame = !(this_frame_hand_detected[0] || this_frame_hand_detected[1]);
        const int active_camera = detection_counter++ % 2;
        const bool both = options.always_run_detection_model || hand_size_refinement.isOptimizing() ||
                          options.detection_model_in_both_views;
        int views[2] = {active_camera, 1 - active_camera};
        const int num_views = both ? 2 : 1;

        DetectionInput inputs[2];
        DetectionOutput outputs[2];
        Affine go_back[2];
        std::vector<uint8_t> boxed(size_t(kDetectionSize) * kDetectionSize);
        size_t batch = 0;
        int batch_view[2] = {};
        for (int k = 0; k < num_views; k++) {
            const int v = views[k];
            go_back[v] = letterbox(images[v], kDetectionSize, boxed.data());
            // Mercury ignores normalization failure here and runs the model anyway.
            normalize_gray(boxed.data(), boxed.size(), inputs[batch].image.data());
            inputs[batch].view = v;
            batch_view[batch++] = v;
        }
        networks.detect(inputs, outputs, batch);
        for (size_t b = 0; b < batch; b++) {
            const int v = batch_view[b];
            std::array<Region, 2> regions;
            decode_detection(outputs[b], go_back[v], options.min_detection_confidence, regions);
            for (int h = 0; h < 2; h++) {
                infos[v][h].provenance = Provenance::Detection;
                infos[v][h].found = regions[size_t(h)].found;
                infos[v][h].confidence = regions[size_t(h)].confidence;
                infos[v][h].center_px = regions[size_t(h)].center_px;
                infos[v][h].size_px = regions[size_t(h)].size_px;
            }
        }

        for (int h = 0; h < 2; h++) {
            const float confidence = (infos[0][h].confidence + infos[1][h].confidence) / float(num_views);
            if (confidence < options.detection_confidence_sum) continue;
            if (options.always_run_detection_model || !last_frame_hand_detected[h]) {
                bool good = true;
                for (int v = 0; v < 2 && good; v++) {
                    if (no_hands_detected_last_frame) {
                        // Reject double detections: both hands in the same place.
                        if (box_iou(infos[v][h], infos[v][1 - h]) > options.mpiou_double_detection) good = false;
                    } else if (box_iou(infos[v][h], rois[v][1 - h]) > options.mpiou_single_detection) {
                        // Reject a new hand where the other hand is already tracked.
                        good = false;
                    }
                }
                if (good) {
                    rois[0][h] = infos[0][h];
                    rois[1][h] = infos[1][h];
                }
            }
            this_frame_hand_detected[h] = true;
        }
        return int(batch);
    }

    void predict_new_regions_of_interest() {
        for (int h = 0; h < 2; h++) {
            auto& hh = history_hands[h];
            if (hh.count < 2) continue;
            this_frame_hand_detected[h] = last_frame_hand_detected[h];
            const double t2 = double(history_timestamps.age(1)), t1 = double(history_timestamps.age(0));
            const double dt_past = (t1 - t2) / 1e9, dt_now = (double(current_timestamp) - t1) / 1e9;
            Joints21 add = hh.age(0) - hh.age(1);
            add *= float((dt_now * options.amount_to_lerp_prediction) / dt_past);
            pose_predicted_keypoints[h] = hh.age(0) + add;
            int num_outside[2];
            back_project(pose_predicted_keypoints[h], h, num_outside);
            for (int v = 0; v < 2; v++) {
                Roi& roi = rois[v][h];
                if (num_outside[v] < options.max_num_outside_view) {
                    roi.provenance = Provenance::Prediction;
                    roi.found = true;
                } else {
                    roi.found = false;
                }
            }
        }
    }

    void stop_if_hands_overlap() {
        for (int v = 0; v < 2; v++) {
            if (!rois[v][0].found || !rois[v][1].found) continue;
            if (box_iou(rois[v][0], rois[v][1]) > options.mpiou_any) {
                for (auto& view : rois)
                    for (auto& roi : view) roi.found = false;
                return;
            }
        }
    }

    // Mercury run_keypoint_estimation_unsafe() up to the inference, for one crop.
    bool prepare_keypoint(int view, int hand, KeypointInput& input, Projection& projection) {
        const Roi& roi = rois[view][hand];
        const float twist = 0;
        input.last_keypoints.fill(0.0f);
        input.use_last_keypoints = 0.0f;
        if (roi.provenance == Provenance::Detection) {
            xrt_vec3 center, edges[4];
            unproject_and_flip(cameras[view], roi.center_px.x, roi.center_px.y, center);
            const float s = roi.size_px / 2;
            const xrt_vec2 corners[4] = {{roi.center_px.x + s, roi.center_px.y + s},
                                         {roi.center_px.x - s, roi.center_px.y + s},
                                         {roi.center_px.x - s, roi.center_px.y - s},
                                         {roi.center_px.x + s, roi.center_px.y - s}};
            float angle = 0;
            for (int i = 0; i < 4; i++) {
                unproject_and_flip(cameras[view], corners[i].x, corners[i].y, edges[i]);
                angle = std::fmax(angle, m_vec3_angle(center, edges[i]));
            }
            make_projection_instructions_angular(center, hand != 0, angle, options.after_detection_fac, twist,
                                                 projection);
        } else {
            Joints21 kp;
            if (view == 0) {
                kp = pose_predicted_keypoints[hand];
            } else {
                const Eigen::Quaternionf ori = map_quat(left_in_right.orientation);
                for (int i = 0; i < 21; i++)
                    kp.col(i) = ori * Eigen::Vector3f(pose_predicted_keypoints[hand].col(i)) +
                                map_vec3(left_in_right.position);
            }
            hand21_2d predicted;
            make_projection_instructions(hand != 0, options.dyn_radii_fac, twist, kp, projection, predicted);
            if (options.enable_pose_predicted_input) {
                for (int j = 0; j < 21; j++) {
                    input.last_keypoints[size_t(j * 2)] = predicted[size_t(j)].pos_2d.x;
                    input.last_keypoints[size_t(j * 2 + 1)] = predicted[size_t(j)].pos_2d.y;
                }
                input.use_last_keypoints = 1.0f;
            }
        }
        input.view = view;
        input.hand = hand;
        input.look_dir = {projection.rot_quat.x(), projection.rot_quat.y(), projection.rot_quat.z(),
                          projection.rot_quat.w()};
        input.stereographic_radius = projection.stereographic_radius;
        uint8_t crop[kKeypointSize * kKeypointSize];
        stereographic_project_image(cameras[view], projection, images[view], crop);
        return normalize_gray(crop, sizeof crop, input.image.data());
    }

    int run_keypoints() {
        KeypointInput inputs[4];
        KeypointOutput outputs[4];
        Projection projections[4];
        bool normalized[4] = {};
        int slot_view[4], slot_hand[4];
        size_t count = 0;
        for (int h = 0; h < 2; h++) {
            for (int v = 0; v < 2; v++) {
                if (!rois[v][h].found) continue;
                normalized[count] = prepare_keypoint(v, h, inputs[count], projections[count]);
                slot_view[count] = v;
                slot_hand[count] = h;
                count++;
            }
        }
        if (count) networks.keypoints(inputs, outputs, count);
        for (size_t k = 0; k < count; k++) {
            hm::one_frame_one_view& out = keypoint_outputs[slot_hand[k]].views[slot_view[k]];
            map_quat(out.look_dir) = projections[k].rot_quat;
            out.stereographic_radius = projections[k].stereographic_radius;
            const bool is_hand =
                decode_keypoints(outputs[k], slot_hand[k], options.min_hand_probability, out);
            out.active = normalized[k] && is_hand;
        }
        return int(count);
    }

    Result process(const GrayImage& left, const GrayImage& right, int64_t timestamp_ns) {
        for (int v = 0; v < 2; v++) {
            const GrayImage& img = v ? right : left;
            if (!img.data || img.width != cameras[v].width || img.height != cameras[v].height ||
                img.stride < size_t(img.width))
                throw std::invalid_argument("mercury: image does not match its calibration");
        }
        if (have_timestamp && timestamp_ns <= current_timestamp)
            throw std::invalid_argument("mercury: timestamps must increase");
        images[0] = left;
        images[1] = right;
        current_timestamp = timestamp_ns;
        have_timestamp = true;

        Result result;
        result.timestamp_ns = timestamp_ns;
        xrt_hand_joint_set sets[2] = {};

        const bool saw_both = last_frame_hand_detected[0] && last_frame_hand_detected[1];
        if (!saw_both || options.always_run_detection_model) result.detections_run = dispatch_and_process_hand_detections();
        stop_if_hands_overlap();
        for (int h = 0; h < 2; h++)
            if (!(rois[0][h].found || rois[1][h].found)) this_frame_hand_detected[h] = false;

        result.keypoints_run = run_keypoints();

        const std::array<hm::HandSizeRefinement::ViewData, hm::kNumViews> view_data{
            hm::HandSizeRefinement::ViewData{{{rois[0][0].found}, {rois[0][1].found}}},
            hm::HandSizeRefinement::ViewData{{{rois[1][0].found}, {rois[1][1].found}}},
        };
        const std::array<hm::HandSizeRefinement::HandData, hm::kNumHands> hand_data{
            hm::HandSizeRefinement::HandData{this_frame_hand_detected[0], hand_seen_before[0]},
            hm::HandSizeRefinement::HandData{this_frame_hand_detected[1], hand_seen_before[1]},
        };
        bool optimize_hand_size = hand_size_refinement.framePreOptimizer(view_data, hand_data);
        target_hand_size = hand_size_refinement.getTargetHandSize();
        optimize_hand_size = optimize_hand_size && options.optimize_hand_size;

        for (int h = 0; h < 2; h++) {
            for (int v = 0; v < 2; v++)
                if (rois[v][h].found && !keypoint_outputs[h].views[v].active) this_frame_hand_detected[h] = false;
            if (!this_frame_hand_detected[h]) continue;
            for (int v = 0; v < 2; v++)
                if (!rois[v][h].found) keypoint_outputs[h].views[v].active = false;

            xrt_hand_joint_set& set = sets[h];
            float out_hand_size = 0, reprojection_error = 0;
            // The optimizer mutates its observation, as in Mercury.
            lm::optimizer_run(kinematic_hands[h], keypoint_outputs[h], !last_frame_hand_detected[h],
                              options.opt_smooth_factor, optimize_hand_size, target_hand_size,
                              hand_size_refinement.getHandSizeErrorMul(), options.amt_use_depth, set,
                              out_hand_size, reprojection_error);
            result.hands[size_t(h)].reprojection_error = reprojection_error;
            if (reprojection_error > options.max_reprojection_error) {
                this_frame_hand_detected[h] = false;
                continue;
            }
            if (m_vec3_len(set.values.hand_joint_set_default[XRT_HAND_JOINT_PALM].relation.pose.position) >
                options.max_hand_dist) {
                this_frame_hand_detected[h] = false;
                continue;
            }
            hand_size_refinement.frameHandPostOptimizer(out_hand_size, reprojection_error, keypoint_outputs[h]);
            apply_joint_width(set);
            Joints21 asf;
            joint_set_to_21(set, asf);
            back_project(asf, h, nullptr);
            history_hands[h].push(asf);
            hand_tracked_for_num_frames[h]++;
            result.hands[size_t(h)].tracked = true;
        }
        history_timestamps.push(current_timestamp);
        hand_size_refinement.framePost();

        bool active[2];
        for (int h = 0; h < 2; h++) {
            active[h] = this_frame_hand_detected[h];
            last_frame_hand_detected[h] = this_frame_hand_detected[h];
            hand_seen_before[h] = hand_seen_before[h] || this_frame_hand_detected[h];
            if (!last_frame_hand_detected[h]) {
                rois[0][h].found = false;
                rois[1][h].found = false;
                history_hands[h].clear();
                hand_tracked_for_num_frames[h] = 0;
            }
        }
        if (!options.always_run_detection_model) {
            predict_new_regions_of_interest();
            for (int h = 0; h < 2; h++) active[h] = rois[0][h].found || rois[1][h].found;
        }
        for (int h = 0; h < 2; h++) {
            if (hand_tracked_for_num_frames[h] < uint64_t(options.num_frames_before_display)) active[h] = false;
            Hand& out = result.hands[size_t(h)];
            out.active = active[h] && out.tracked;
            if (!out.tracked) continue;
            for (int j = 0; j < kJointCount; j++) {
                const xrt_hand_joint_value& jv = sets[h].values.hand_joint_set_default[j];
                const xrt_pose& p = jv.relation.pose;
                out.joints[size_t(j)].pose = {{p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w},
                                              {p.position.x, p.position.y, p.position.z}};
                out.joints[size_t(j)].radius = jv.radius;
            }
        }
        return result;
    }

    void reset() {
        for (auto& view : rois)
            for (auto& roi : view) roi = {};
        for (int h = 0; h < 2; h++) {
            keypoint_outputs[h] = {};
            hand_seen_before[h] = last_frame_hand_detected[h] = this_frame_hand_detected[h] = false;
            history_hands[h].clear();
            hand_tracked_for_num_frames[h] = 0;
        }
        history_timestamps.clear();
        have_timestamp = false;
        detection_counter = 0;
        hand_size_refinement.newUserEvent();
        target_hand_size = hand_size_refinement.getTargetHandSize();
    }
};

Tracker::Tracker(const StereoCalibration& calibration, Networks& networks, const Options& options)
    : state_(std::make_unique<State>(calibration, networks, options)) {}

Tracker::~Tracker() = default;

Result Tracker::process(const GrayImage& left, const GrayImage& right, int64_t timestamp_ns) {
    return state_->process(left, right, timestamp_ns);
}

void Tracker::reset() { state_->reset(); }

float Tracker::target_hand_size() const { return state_->target_hand_size; }

}  // namespace mercury
