#pragma once
#include "hexecutor/executor.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <vector>
namespace hand_tracking {
struct Image {
    const uint8_t *rgb = nullptr;
    uint32_t width = 0, height = 0;
    size_t stride = 0;
};
struct Point {
    float x = 0, y = 0, z = 0;
};
// Pixel-space square crop, rotation clockwise in image coordinates (y down).
struct Roi {
    float x = 0, y = 0, size = 0, rotation = 0;
};
struct Hand {
    uint64_t id = 0;
    float presence = 0, right_probability = 0;
    // Image x/W, y/H, and relative z/W (MediaPipe topology).
    std::array<Point, 21> landmarks;
    // Model-relative coordinates in meters; rotated into the image axes.
    // These are not calibrated camera-space or world-space positions.
    std::array<Point, 21> model_world_landmarks;
    Roi next_roi;
};
struct Options {
    uint32_t max_hands = 2, detection_interval = 10;
    float detection_threshold = 0.5f, presence_threshold = 0.5f, suppression_iou = 0.3f;
    // Hands whose landmark bounds overlap above this are the same hand; the
    // lower-presence copy is dropped so a second hand can be reacquired.
    float duplicate_iou = 0.5f;
};
struct Frame {
    std::vector<Hand> hands;
    bool detector_ran = false;
};
using Infer = std::function<std::vector<hexecutor::Bytes>(size_t, const std::vector<float> &)>;
// Palm then landmark, with exact model input/output contracts checked before
// creating a device session. Paths point to separately provisioned assets.
std::vector<hexecutor::Program> load_models(const std::filesystem::path &directory);
std::vector<float> crop(const Image &, const Roi &, uint32_t size, bool zero_border);
class Tracker {
  public:
    explicit Tracker(hexecutor::Session &, Options = {});
    explicit Tracker(Infer, Options = {});
    Frame process(const Image &, uint64_t timestamp_ns);
    void reset();

  private:
    Infer infer_;
    Options options_;
    std::vector<Hand> tracks_;
    uint64_t next_id_ = 1, last_timestamp_ = 0, frame_ = 0;
    uint32_t width_ = 0, height_ = 0;
    bool have_timestamp_ = false;
    std::vector<Roi> detect(const Image &);
    bool landmarks(const Image &, const Roi &, uint64_t id, Hand &);
    bool duplicate(const Hand &, const std::vector<Hand> &) const;
};
} // namespace hand_tracking
