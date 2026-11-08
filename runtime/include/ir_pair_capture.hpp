#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace FrameSpatial {

// Read-only QUERYBUF grab of XRService's OG01A1B pair (video9 right, video13 left).
// Never opens /dev/video*; duplicates XRService FDs via pidfd_getfd.
// Only long exposures are returned. XRService alternates long and short exposures
// (mean about 55 vs 4 of 255 in every E10A recording; short frames carry no dots).
// They are told apart by brightness against the neighbouring frame, because a dropped
// frame can shift a camera's sequence parity.
class IrPairCapture {
public:
    static constexpr int Width = 1056, Height = 1024;
    struct Pair {
        std::vector<uint8_t> left, right;  // video13, video9
        uint32_t leftSeq = 0, rightSeq = 0;
        int64_t leftTsNs = 0, rightTsNs = 0;  // CLOCK_MONOTONIC_RAW
        int64_t leftMonotonicNs = 0;          // leftTsNs on CLOCK_MONOTONIC
    };
    IrPairCapture();
    ~IrPairCapture();
    bool initialize();
    // The newest long-exposure pair not yet returned. With copyPixels false only the
    // sequences and timestamps are filled; fetchPixels can then copy that same pair.
    bool grab(Pair& out, int timeoutMs = 200, bool copyPixels = true);
    bool fetchPixels(Pair& pair);
    const std::string& lastError() const;
    // Fine-scale horizontal contrast of the central image. Dots raise it 9-112%
    // over a very steady unlit baseline (E10A), so it finds lit frames without
    // knowing when the emitter actually switched.
    static float dotScore(const std::vector<uint8_t>& image);
    // True when a frame is clearly brighter than the neighbouring alternate exposure.
    static bool isLongExposure(float mean, float neighbourMean) { return mean > 1.5f * neighbourMean + 0.5f; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace FrameSpatial
