#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

struct HandSkeleton {
    // MediaPipe uses 21 joints; Mercury uses OpenXR's 26-joint order.
    std::array<std::array<float,3>,26> joints{};
    std::array<float,26> radii{};
    int joint_count = 21;
    bool right = false;
};

struct HandScene {
    std::vector<HandSkeleton> hands;
    // true: joints are in the OpenXR LOCAL space (world-locked at capture
    // time). false: joints are in head/VIEW space and follow the display pose.
    bool local_space = true;
};

struct HeadPose {
    std::array<float,4> orientation{0,0,0,1};  // x, y, z, w
    std::array<float,3> position{};
};

// 3D hand joints and bones in a transparent stereo projection layer over
// OpenXR alpha-blended passthrough.
class HandViewerXr {
public:
    HandViewerXr();
    ~HandViewerXr();
    void initialize();
    bool pollEvents();
    bool active() const;
    // Headset (VIEW) pose in LOCAL at a CLOCK_MONOTONIC time. Safe to call from
    // a worker thread while the render loop runs. False when unavailable.
    bool headPoseAt(uint64_t monotonic_ns, HeadPose& pose) const;
    void render(const HandScene& scene);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
