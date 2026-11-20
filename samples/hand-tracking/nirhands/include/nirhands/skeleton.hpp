// Host-side decode for SkeletonHandNet. The CDSP produces auxiliary crop keypoints and skeleton parameters;
// forward kinematics, the small palm Procrustes solve and crop re-encoding run here.
#pragma once
#include "nirhands/pipeline.hpp"

#include <deque>

namespace nirhands {

constexpr int kSkeletonProfileValues = 33;
using SkeletonProfile = std::array<float, kSkeletonProfileValues>;

void decode_skeleton(const Geometry& geometry, const SkeletonProfile& mean_profile,
                     const float* keypoints_hm, const float* disparity_hm,
                     const float* palm_points, const float* angles, float scale,
                     float* keypoints, float* disparity);

// ---- combined output: the skeleton structure fitted to the auxiliary heatmap keypoints ----
// Port of stereo_nir_hands nirhands.training.skeleton (skeleton_profile, inverse_kinematics,
// forward_kinematics_local, pose_joints). The skeleton code uses a left-hand convention: right hands are mirrored
// (x negated, any rigid frame works) before and after.
constexpr int kSkeletonAngles = 30;  // 15 bones x (flex, side), radians
using SkeletonAngles = std::array<double, kSkeletonAngles>;

// Profile (palm points in palm coordinates + 15 bone lengths) of the given hand. Joints in left-hand convention.
SkeletonProfile measure_profile(const Joints& joints);
// Angles that reproduce `joints` on `profile` (exact when the profile is the hand's own). Left-hand convention.
SkeletonAngles inverse_kinematics(const Joints& joints, const SkeletonProfile& profile);
// Joints in palm coordinates (wrist at the origin). Left-hand convention.
Joints forward_kinematics_profile(const SkeletonProfile& profile, const SkeletonAngles& angles);

// The combined hand: `profile`'s bones, angles blended (1 - w) skeleton + w heatmap, palm placed onto the heatmap
// palm. `skeleton` and `heatmap` are the two decoded outputs in the same (world) frame. Returns false (and leaves
// `out` unchanged) if either input is degenerate.
bool combine_skeleton(const Joints& skeleton, const Joints& heatmap, bool is_right, const SkeletonProfile& profile,
                      double heatmap_weight, Joints& out);

// The wearer's profile per side: per-value median of measure_profile() over the last `window` accepted heatmap hands.
class ProfileEstimator {
public:
    explicit ProfileEstimator(int window = 300, int min_samples = 20) : window_(window), min_samples_(min_samples) {}
    void add(const Joints& heatmap_joints, bool is_right);
    // False until `min_samples` hands of that side have been added.
    bool profile(bool is_right, SkeletonProfile& out);
    int samples(bool is_right) const { return int(history_[is_right].size()); }
    void reset();

private:
    int window_, min_samples_;
    std::array<std::deque<SkeletonProfile>, 2> history_;
    std::array<SkeletonProfile, 2> cached_{};
    std::array<bool, 2> dirty_{true, true};
};

}  // namespace nirhands
