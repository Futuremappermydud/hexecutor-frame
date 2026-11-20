#pragma once
// Exposure-identified stereo capture from vrcompositor's IMX616 DMA-BUF rings (E02b).
//
// The legacy VkYDetile path copies every slot and picks the highest-index slot
// whose hash changed since the previous read. At depth cadence (0.5-1.5 s)
// every slot has changed, so it returns a fixed slot of unknown age, and the
// left and right choices need not come from the same camera frame.
//
// This class instead runs a probe thread. About every 0.5 ms it copies a 16-row
// strip of every slot of both rings, detects the slot whose write begins, and
// tags that slot with the newest kernel V4L2 end-of-readout of the same eye
// that precedes the write start (writes start 2-10 ms after readout at 60 fps).
// This matching is from april-tag-xrlib; it is 1:1 at 60 fps.
//
// acquire() returns the newest completed left/right slots that carry the SAME
// V4L2 sequence. The sensors expose within ~0.2 ms of each other (E02). After
// the full copy the strips are re-probed, and the pair is rejected if either
// slot was rewritten during the copy. Every returned pair carries its
// sequence, per-eye end-of-readout, write starts and slots.
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace FrameSpatial {

struct TimedStereoPair {
    bool valid = false;
    uint32_t sequence = 0;                 // shared V4L2 sequence of both eyes
    uint32_t leftSequence = 0, rightSequence = 0;
    int64_t leftEndOfReadoutNs = 0, rightEndOfReadoutNs = 0; // CLOCK_MONOTONIC
    int64_t leftWriteStartNs = 0, rightWriteStartNs = 0;     // ring write start estimate
    int64_t copyBeginNs = 0, copyEndNs = 0;
    int leftSlot = -1, rightSlot = -1;
    uint64_t skippedSequences = 0;         // completed common sequences never returned
};

struct TimedStereoStats {
    uint64_t probes = 0, leftFrames = 0, rightFrames = 0;
    uint64_t leftUnmatched = 0, rightUnmatched = 0; // write start with no V4L2 EOF within 16 ms
    uint64_t tornCopies = 0, deliveredPairs = 0;
    double probeMsMean = 0, writeStartMinusEofMsMean = 0;
};

class TimedStereoCapture {
public:
    // Local (already pidfd_getfd-duplicated) DMA-BUF FDs of one active 4+4 pool.
    TimedStereoCapture(const std::vector<int>& leftFds, const std::vector<int>& rightFds, int width = 2464,
                       int height = 2464);
    ~TimedStereoCapture();
    TimedStereoCapture(const TimedStereoCapture&) = delete;
    TimedStereoCapture& operator=(const TimedStereoCapture&) = delete;

    // Waits for a verified pair whose sequence is newer than the last one returned.
    // Pixel pointers stay valid until the next acquire(). UV is copied only if requested.
    bool acquire(TimedStereoPair& out, int timeoutMs, bool includeUV);
    const uint8_t* leftY() const { return leftY_.data(); }
    const uint8_t* rightY() const { return rightY_.data(); }
    const uint8_t* leftUV() const { return uvValid_ ? leftUV_.data() : nullptr; }
    const uint8_t* rightUV() const { return uvValid_ ? rightUV_.data() : nullptr; }
    TimedStereoStats stats() const;
    std::string error() const;

private:
    struct Vk;
    struct Ring;
    void run();

    int width_, height_;
    std::unique_ptr<Vk> vk_;
    std::unique_ptr<Ring> left_, right_;
    std::thread thread_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false, want_ = false, wantUV_ = false, ready_ = false, uvValid_ = false;
    uint32_t lastDelivered_ = 0;
    TimedStereoPair result_;
    std::vector<uint8_t> leftY_, rightY_, leftUV_, rightUV_;
    TimedStereoStats stats_;
    double probeMsSum_ = 0, matchMsSum_ = 0;
    uint64_t matches_ = 0;
    std::string error_;
};

} // namespace FrameSpatial
