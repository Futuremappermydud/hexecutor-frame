#include "hand_tracking/viewer_geometry.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void check(bool value) { if (!value) throw std::runtime_error("hand viewer geometry test failed"); }
int main() {
    using namespace hand_tracking;
    // Project a generated metric hand at a known camera translation and
    // recover every joint in camera coordinates.
    Hand hand{};
    for (int i=0;i<21;++i) {
        const float x=(i%5-2)*.015f,y=(i/5-2)*.018f,z=(i%3-1)*.006f;
        hand.model_world_landmarks[i]={x,y,z};
        hand.landmarks[i]={(380*(x+.03f)/(z+.5f)+308)/616,(380*(y-.02f)/(z+.5f)+308)/616,0};
    }
    std::array<Point,21> joints{};
    check(viewer::camera_joints(hand,616,616,380,308,308,joints));
    for (int i=0;i<21;++i) {
        const auto w=hand.model_world_landmarks[i];
        check(std::abs(joints[i].x-(w.x+.03f))<1e-4f && std::abs(joints[i].y-(w.y-.02f))<1e-4f &&
              std::abs(joints[i].z-(w.z+.5f))<1e-4f);
    }
    // Every bone references valid, distinct joints and every joint is used.
    std::array<int,21> used{};
    for (const auto& bone : viewer::bones) {
        check(bone[0]>=0 && bone[0]<21 && bone[1]>=0 && bone[1]<21 && bone[0]!=bone[1]);
        used[bone[0]]++; used[bone[1]]++;
    }
    for (int count : used) check(count>0);

    // OpenXR / Mercury 26-joint skeleton topology:
    // 25 bones connecting 26 joints. Wrist is joint 1, palm is joint 0.
    std::array<int,26> mercury_used{};
    for (const auto& bone : viewer::mercury_bones) {
        check(bone[0]>=0 && bone[0]<26 && bone[1]>=0 && bone[1]<26 && bone[0]!=bone[1]);
        mercury_used[bone[0]]++; mercury_used[bone[1]]++;
    }
    for (int count : mercury_used) check(count>0);
    // Metacarpals and palm must attach to the wrist (1).
    int wrist_connections = 0;
    for (const auto& bone : viewer::mercury_bones) {
        if (bone[0] == 1 || bone[1] == 1) wrist_connections++;
    }
    check(wrist_connections == 6); // palm + thumb + index + middle + ring + little

    // Degenerate (all landmarks on one ray) and nonfinite inputs are rejected.
    Hand flat=hand;
    for (auto& p : flat.landmarks) p={.5f,.5f,0};
    check(!viewer::camera_joints(flat,616,616,380,308,308,joints));
    hand.landmarks[3].x=std::numeric_limits<float>::quiet_NaN();
    check(!viewer::camera_joints(hand,616,616,380,308,308,joints));

    // Quaternion rotation helper checks.
    const float kSqrt2_2 = 0.70710678f;
    // Identity rotation.
    const std::array<float, 4> q_ident{0.0f, 0.0f, 0.0f, 1.0f};
    const std::array<float, 3> v0{1.0f, 2.0f, 3.0f};
    const auto r0 = viewer::rotate(q_ident, v0);
    check(std::abs(r0[0] - v0[0]) < 1e-6f && std::abs(r0[1] - v0[1]) < 1e-6f && std::abs(r0[2] - v0[2]) < 1e-6f);

    // 90 deg around Z: (1, 0, 0) -> (0, 1, 0).
    const std::array<float, 4> q_z90{0.0f, 0.0f, kSqrt2_2, kSqrt2_2};
    const auto rz = viewer::rotate(q_z90, {1.0f, 0.0f, 0.0f});
    check(std::abs(rz[0]) < 1e-5f && std::abs(rz[1] - 1.0f) < 1e-5f && std::abs(rz[2]) < 1e-5f);

    // 90 deg around X: (0, 1, 0) -> (0, 0, 1).
    const std::array<float, 4> q_x90{kSqrt2_2, 0.0f, 0.0f, kSqrt2_2};
    const auto rx = viewer::rotate(q_x90, {0.0f, 1.0f, 0.0f});
    check(std::abs(rx[0]) < 1e-5f && std::abs(rx[1]) < 1e-5f && std::abs(rx[2] - 1.0f) < 1e-5f);

    // World-locking transformation: joint in head space moves into local space
    // using the head pose at capture time.
    const std::array<float, 3> p_head{0.05f, -0.02f, -0.40f};
    const std::array<float, 3> head_pos{0.1f, 1.6f, -0.5f};
    const auto p_world = viewer::rotate(q_z90, p_head);
    const std::array<float, 3> p_local{p_world[0] + head_pos[0], p_world[1] + head_pos[1], p_world[2] + head_pos[2]};
    check(std::abs(p_local[0] - (head_pos[0] - p_head[1])) < 1e-5f);
    check(std::abs(p_local[1] - (head_pos[1] + p_head[0])) < 1e-5f);
    check(std::abs(p_local[2] - (head_pos[2] + p_head[2])) < 1e-5f);

    std::cout << "Hand joint geometry: camera fit, 21-joint and 26-joint topologies, degenerate, rotation and world-locking PASS\n";
}
