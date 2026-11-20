#include "hand_tracking/tracker.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace hand_tracking {
namespace {
constexpr float pi = 3.14159265358979323846f;
void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
float angle(float a) { return a - 2 * pi * std::floor((a + pi) / (2 * pi)); }
float value(const hexecutor::Bytes &b, size_t i) {
    float v;
    std::memcpy(&v, b.data() + i * 4, 4);
    require(std::isfinite(v), "non-finite hand model output");
    return v;
}
void outputs(const std::vector<hexecutor::Bytes> &out, std::initializer_list<size_t> elements) {
    require(out.size() == elements.size(), "incorrect hand model output count");
    size_t i = 0;
    for (auto n : elements)
        require(out[i++].size() == n * 4, "incorrect hand model output size");
}
struct Box {
    float x, y, w, h, score;
    std::array<Point, 7> keys;
};
float iou(float x, float y, float w, float h, float xx, float yy, float ww, float hh) {
    float area =
        std::max(0.0f, std::min(x + w / 2, xx + ww / 2) - std::max(x - w / 2, xx - ww / 2)) *
        std::max(0.0f, std::min(y + h / 2, yy + hh / 2) - std::max(y - h / 2, yy - hh / 2));
    float total = w * h + ww * hh - area;
    return total > 0 ? area / total : 0;
}
float iou(const Roi &a, const Roi &b) {
    return iou(a.x, a.y, a.size, a.size, b.x, b.y, b.size, b.size);
}
// Axis-aligned landmark bounds in pixels, as center/size.
std::array<float, 4> bounds(const Hand &hand, float width, float height) {
    float minx = std::numeric_limits<float>::max(), miny = minx, maxx = -minx, maxy = -minx;
    for (const auto &p : hand.landmarks) {
        minx = std::min(minx, p.x * width);
        maxx = std::max(maxx, p.x * width);
        miny = std::min(miny, p.y * height);
        maxy = std::max(maxy, p.y * height);
    }
    return {(minx + maxx) * 0.5f, (miny + maxy) * 0.5f, maxx - minx, maxy - miny};
}
Roi next_roi(const std::array<Point, 21> &lm, const Image &image) {
    // MediaPipe's partial-hand ROI excludes distal tips, which can bend into
    // the palm. Use wrist and the index/middle/ring proximal joint direction for rotation.
    const int indices[] = {0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18};
    Point wrist = lm[0];
    float dx = ((lm[5].x + lm[13].x) * 0.25f + lm[9].x * 0.5f - wrist.x) * image.width;
    float dy = ((lm[5].y + lm[13].y) * 0.25f + lm[9].y * 0.5f - wrist.y) * image.height;
    float r = angle(pi / 2 - std::atan2(-dy, dx)), cs = std::cos(r), sn = std::sin(r);
    float minx = std::numeric_limits<float>::max(), miny = minx, maxx = -minx, maxy = -minx;
    for (int i : indices) {
        float x = lm[i].x * image.width, y = lm[i].y * image.height;
        float rx = cs * x + sn * y, ry = -sn * x + cs * y;
        minx = std::min(minx, rx);
        maxx = std::max(maxx, rx);
        miny = std::min(miny, ry);
        maxy = std::max(maxy, ry);
    }
    float cx = (minx + maxx) * 0.5f, cy = (miny + maxy) * 0.5f - 0.1f * (maxy - miny);
    return {cs * cx - sn * cy, sn * cx + cs * cy, 2.0f * std::max(maxx - minx, maxy - miny), r};
}
} // namespace
std::vector<hexecutor::Program> load_models(const std::filesystem::path &directory) {
    auto palm = hexecutor::load_tensor_program(directory / "palm.hxfg", directory / "palm.bin");
    auto landmark =
        hexecutor::load_tensor_program(directory / "landmark.hxfg", directory / "landmark.bin");
    auto check = [](const hexecutor::Program &p, uint32_t side, std::initializer_list<size_t> out) {
        require(p.inputs.size() == 1 &&
                    p.inputs[0].shape == std::vector<size_t>({1, side, side, 3}),
                "incorrect hand model input contract");
        require(p.outputs.size() == out.size(), "incorrect hand model output contract");
        size_t i = 0;
        for (auto n : out)
            require(p.outputs[i++].bytes() == n * 4, "incorrect hand model output contract");
    };
    check(palm, 192, {2016 * 18, 2016});
    check(landmark, 224, {63, 1, 1, 63});
    return {std::move(palm), std::move(landmark)};
}
std::vector<float> crop(const Image &image, const Roi &roi, uint32_t size, bool zero_border) {
    require(image.rgb && image.width && image.height && image.width <= 16384 &&
                image.height <= 16384 && image.stride >= size_t(image.width) * 3,
            "invalid RGB image");
    require(size && size <= 1024 && std::isfinite(roi.x) && std::isfinite(roi.y) &&
                std::isfinite(roi.size) && roi.size > 0 && std::isfinite(roi.rotation),
            "invalid hand crop");
    std::vector<float> out(size_t(size) * size * 3);
    float cs = std::cos(roi.rotation), sn = std::sin(roi.rotation);
    for (uint32_t y = 0; y < size; y++)
        for (uint32_t x = 0; x < size; x++) {
            float lx = ((x + 0.5f) / size - 0.5f) * roi.size,
                  ly = ((y + 0.5f) / size - 0.5f) * roi.size;
            float sx = roi.x + cs * lx - sn * ly - 0.5f, sy = roi.y + sn * lx + cs * ly - 0.5f;
            sx = std::clamp(sx, -1.0f, float(image.width));
            sy = std::clamp(sy, -1.0f, float(image.height));
            int x0 = int(std::floor(sx)), y0 = int(std::floor(sy));
            float fx = sx - x0, fy = sy - y0;
            auto sample = [&](int xx, int yy, int c) {
                if (zero_border &&
                    (xx < 0 || yy < 0 || xx >= int(image.width) || yy >= int(image.height)))
                    return 0.0f;
                xx = std::clamp(xx, 0, int(image.width) - 1);
                yy = std::clamp(yy, 0, int(image.height) - 1);
                return image.rgb[size_t(yy) * image.stride + size_t(xx) * 3 + c] / 255.0f;
            };
            for (int c = 0; c < 3; c++) {
                float a = sample(x0, y0, c) * (1 - fx) + sample(x0 + 1, y0, c) * fx;
                float b = sample(x0, y0 + 1, c) * (1 - fx) + sample(x0 + 1, y0 + 1, c) * fx;
                out[(size_t(y) * size + x) * 3 + c] = a * (1 - fy) + b * fy;
            }
        }
    return out;
}
Tracker::Tracker(hexecutor::Session &s, Options o)
    : Tracker(
          [&s](size_t index, const std::vector<float> &x) {
              return s.run({{x.data(), x.size() * 4}}, index);
          },
          o) {}
Tracker::Tracker(Infer f, Options o) : infer_(std::move(f)), options_(o) {
    require(bool(infer_) && o.max_hands > 0 && o.max_hands <= 8 && o.detection_interval > 0 &&
                std::isfinite(o.detection_threshold) && o.detection_threshold > 0 &&
                o.detection_threshold < 1 && std::isfinite(o.presence_threshold) &&
                o.presence_threshold > 0 && o.presence_threshold < 1 &&
                std::isfinite(o.suppression_iou) && o.suppression_iou > 0 && o.suppression_iou < 1 &&
                std::isfinite(o.duplicate_iou) && o.duplicate_iou > 0 && o.duplicate_iou < 1,
            "invalid tracker options");
}
void Tracker::reset() {
    tracks_.clear();
    have_timestamp_ = false;
    frame_ = 0;
    width_ = height_ = 0;
}
std::vector<Roi> Tracker::detect(const Image &image) {
    float side = float(std::max(image.width, image.height));
    auto input = crop(image, {image.width * 0.5f, image.height * 0.5f, side, 0}, 192, true);
    auto out = infer_(0, input);
    outputs(out, {2016 * 18, 2016});
    std::vector<Box> boxes;
    size_t anchor = 0;
    for (int layer = 0; layer < 2; layer++) {
        int grid = layer ? 12 : 24, repeats = layer ? 6 : 2;
        for (int y = 0; y < grid; y++)
            for (int x = 0; x < grid; x++)
                for (int k = 0; k < repeats; k++, anchor++) {
                    float score =
                        1 / (1 + std::exp(-std::clamp(value(out[1], anchor), -100.0f, 100.0f)));
                    if (score < options_.detection_threshold)
                        continue;
                    float ax = (x + 0.5f) / grid, ay = (y + 0.5f) / grid;
                    const size_t at = anchor * 18;
                    Box box{(value(out[0], at) / 192 + ax) * side - (side - image.width) * 0.5f,
                            (value(out[0], at + 1) / 192 + ay) * side -
                                (side - image.height) * 0.5f,
                            value(out[0], at + 2) / 192 * side,
                            value(out[0], at + 3) / 192 * side,
                            score,
                            {}};
                    if (box.w <= 0 || box.h <= 0)
                        continue;
                    for (size_t i = 0; i < 7; i++)
                        box.keys[i] = {(value(out[0], at + 4 + i * 2) / 192 + ax) * side -
                                           (side - image.width) * 0.5f,
                                       (value(out[0], at + 5 + i * 2) / 192 + ay) * side -
                                           (side - image.height) * 0.5f,
                                       0};
                    boxes.push_back(box);
                }
    }
    std::sort(boxes.begin(), boxes.end(),
              [](const Box &a, const Box &b) { return a.score > b.score; });
    std::vector<Roi> rois;
    while (!boxes.empty() && rois.size() < options_.max_hands) {
        Box first = boxes.front(), mean{};
        float total = 0;
        std::vector<Box> remaining;
        for (const auto &b : boxes) {
            if (iou(first.x, first.y, first.w, first.h, b.x, b.y, b.w, b.h) >
                options_.suppression_iou) {
                total += b.score;
                mean.x += b.x * b.score;
                mean.y += b.y * b.score;
                mean.w += b.w * b.score;
                mean.h += b.h * b.score;
                for (size_t i = 0; i < 7; i++) {
                    mean.keys[i].x += b.keys[i].x * b.score;
                    mean.keys[i].y += b.keys[i].y * b.score;
                }
            } else
                remaining.push_back(b);
        }
        require(total > 0, "invalid palm suppression group");
        mean.x /= total;
        mean.y /= total;
        mean.w /= total;
        mean.h /= total;
        for (auto &k : mean.keys) {
            k.x /= total;
            k.y /= total;
        }
        float r = angle(pi / 2 - std::atan2(-(mean.keys[2].y - mean.keys[0].y),
                                            mean.keys[2].x - mean.keys[0].x));
        rois.push_back({mean.x + 0.5f * mean.h * std::sin(r), mean.y - 0.5f * mean.h * std::cos(r),
                        2.6f * std::max(mean.w, mean.h), r});
        boxes.swap(remaining);
    }
    return rois;
}
bool Tracker::landmarks(const Image &image, const Roi &roi, uint64_t id, Hand &hand) {
    auto out = infer_(1, crop(image, roi, 224, false));
    outputs(out, {63, 1, 1, 63});
    hand.id = id;
    hand.presence = value(out[1], 0);
    hand.right_probability = value(out[2], 0);
    require(hand.presence >= 0 && hand.presence <= 1 && hand.right_probability >= 0 &&
                hand.right_probability <= 1,
            "invalid hand probability");
    if (hand.presence < options_.presence_threshold)
        return false;
    float cs = std::cos(roi.rotation), sn = std::sin(roi.rotation);
    for (size_t i = 0; i < 21; i++) {
        float x = (value(out[0], i * 3) / 224 - 0.5f) * roi.size,
              y = (value(out[0], i * 3 + 1) / 224 - 0.5f) * roi.size;
        hand.landmarks[i] = {(roi.x + cs * x - sn * y) / image.width,
                             (roi.y + sn * x + cs * y) / image.height,
                             value(out[0], i * 3 + 2) / 224 / 0.4f * roi.size / image.width};
        float wx = value(out[3], i * 3), wy = value(out[3], i * 3 + 1);
        hand.model_world_landmarks[i] = {cs * wx - sn * wy, sn * wx + cs * wy,
                                         value(out[3], i * 3 + 2)};
    }
    hand.next_roi = next_roi(hand.landmarks, image);
    return std::isfinite(hand.next_roi.size) && hand.next_roi.size >= 1 &&
           hand.next_roi.size <= 8 * std::max(image.width, image.height);
}
bool Tracker::duplicate(const Hand &hand, const std::vector<Hand> &kept) const {
    const auto a = bounds(hand, float(width_), float(height_));
    for (const auto &other : kept) {
        const auto b = bounds(other, float(width_), float(height_));
        if (iou(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]) > options_.duplicate_iou)
            return true;
    }
    return false;
}
Frame Tracker::process(const Image &image, uint64_t timestamp_ns) {
    require(image.rgb && image.width && image.height && image.width <= 16384 &&
                image.height <= 16384 && image.stride >= size_t(image.width) * 3,
            "invalid RGB image");
    require(!have_timestamp_ || timestamp_ns > last_timestamp_,
            "hand timestamps must increase; reset before a new stream");
    if (image.width != width_ || image.height != height_) {
        tracks_.clear();
        frame_ = 0;
    }
    width_ = image.width;
    height_ = image.height;
    last_timestamp_ = timestamp_ns;
    have_timestamp_ = true;
    Frame result;
    bool lost = false;
    std::vector<Hand> tracked;
    for (const auto &track : tracks_) {
        Hand hand;
        if (landmarks(image, track.next_roi, track.id, hand))
            tracked.push_back(hand);
        else
            lost = true;
    }
    // Two tracks can converge on one hand. Without this, both slots stay
    // occupied and the detector never looks for the other hand again.
    std::stable_sort(tracked.begin(), tracked.end(),
                     [](const Hand &a, const Hand &b) { return a.presence > b.presence; });
    for (auto &hand : tracked) {
        if (duplicate(hand, result.hands))
            lost = true;
        else
            result.hands.push_back(hand);
    }
    if (result.hands.empty() || lost ||
        (result.hands.size() < options_.max_hands && frame_ % options_.detection_interval == 0)) {
        result.detector_ran = true;
        for (const auto &roi : detect(image)) {
            if (result.hands.size() >= options_.max_hands)
                break;
            bool known = false;
            for (const auto &h : result.hands)
                if (iou(roi, h.next_roi) > options_.suppression_iou)
                    known = true;
            if (known)
                continue;
            Hand hand;
            if (landmarks(image, roi, next_id_, hand) && !duplicate(hand, result.hands)) {
                next_id_++;
                result.hands.push_back(hand);
            }
        }
    }
    tracks_ = result.hands;
    frame_++;
    return result;
}
} // namespace hand_tracking
