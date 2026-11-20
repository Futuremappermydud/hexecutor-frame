#include "hand_tracking/tracker.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace hand_tracking;
namespace {
void check(bool ok) {
    if (!ok)
        throw std::runtime_error("tracker test failed");
}
hexecutor::Bytes bytes(const std::vector<float> &v) {
    hexecutor::Bytes b(v.size() * 4);
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}
} // namespace
int main() {
    std::vector<uint8_t> rgb(192 * 192 * 3, 255);
    Image image{rgb.data(), 192, 192, 192 * 3};
    auto x = crop(image, {96, 96, 192, 0}, 192, true);
    check(x.size() == 192 * 192 * 3);
    for (float f : x)
        check(f == 1);
    auto border = crop(image, {-100, -100, 10, 0}, 4, true);
    for (float f : border)
        check(f == 0);
    border = crop(image, {-100, -100, 10, 0}, 4, false);
    for (float f : border)
        check(f == 1);
    unsigned palms = 0, landmarks = 0;
    bool lose = false, no_palm = false;
    Infer infer = [&](size_t index, const std::vector<float> &input) {
        if (index == 0) {
            palms++;
            check(input.size() == 192 * 192 * 3);
            std::vector<float> boxes(2016 * 18), scores(2016, -100);
            size_t anchor = (12 * 24 + 12) * 2;
            if (!no_palm)
                scores[anchor] = 10;
            boxes[anchor * 18 + 2] = boxes[anchor * 18 + 3] = 50;
            boxes[anchor * 18 + 5] = 20;
            boxes[anchor * 18 + 9] = -20;
            return std::vector<hexecutor::Bytes>{bytes(boxes), bytes(scores)};
        }
        landmarks++;
        check(input.size() == 224 * 224 * 3);
        std::vector<float> lm(63), world(63);
        for (int i = 0; i < 21; i++) {
            lm[i * 3] = 80 + (i % 5) * 16;
            lm[i * 3 + 1] = 64 + (i / 5) * 24;
        }
        lm[0] = 112;
        lm[1] = 190;
        float presence = lose ? 0.0f : 0.99f;
        lose = false;
        return std::vector<hexecutor::Bytes>{bytes(lm), bytes({presence}), bytes({0.9f}),
                                             bytes(world)};
    };
    Options options;
    options.max_hands = 1;
    Tracker t(infer, options);
    auto a = t.process(image, 1);
    check(a.detector_ran && a.hands.size() == 1 && palms == 1 && landmarks == 1);
    auto id = a.hands[0].id;
    a = t.process(image, 2);
    check(!a.detector_ran && a.hands.size() == 1 && a.hands[0].id == id && palms == 1);
    lose = true;
    a = t.process(image, 3);
    check(a.detector_ran && a.hands.size() == 1 && a.hands[0].id != id && palms == 2);
    bool rejected = false;
    try {
        t.process(image, 3);
    } catch (const std::exception &) {
        rejected = true;
    }
    check(rejected);
    t.reset();
    no_palm = true;
    a = t.process(image, 1);
    check(a.detector_ran && a.hands.empty());
    // Two hands. The fake landmark model reports a small hand around the
    // bright pixels it sees, so its image-space result depends on the scene,
    // not on which ROI found it (as a real landmark model's does).
    std::vector<size_t> palm_anchors;
    Infer scene = [&](size_t index, const std::vector<float> &input) {
        if (index == 0) {
            std::vector<float> boxes(2016 * 18), scores(2016, -100);
            for (size_t anchor : palm_anchors) {
                scores[anchor] = 10;
                boxes[anchor * 18 + 2] = boxes[anchor * 18 + 3] = 50;
                boxes[anchor * 18 + 5] = 20;
                boxes[anchor * 18 + 9] = -20;
            }
            return std::vector<hexecutor::Bytes>{bytes(boxes), bytes(scores)};
        }
        float minx = 224, miny = 224, maxx = -1, maxy = -1;
        for (size_t y = 0; y < 224; y++)
            for (size_t x = 0; x < 224; x++)
                if (input[(y * 224 + x) * 3] > 0.5f) {
                    minx = std::min(minx, float(x));
                    maxx = std::max(maxx, float(x));
                    miny = std::min(miny, float(y));
                    maxy = std::max(maxy, float(y));
                }
        const bool n = maxx >= minx;
        std::vector<float> lm(63), world(63);
        for (int i = 0; i < 21; i++) {
            lm[i * 3] = minx + (i % 5) * 0.25f * (maxx - minx);
            lm[i * 3 + 1] = miny + (i / 5) * 0.25f * (maxy - miny);
        }
        return std::vector<hexecutor::Bytes>{bytes(lm), bytes({n ? 0.99f : 0.0f}), bytes({0.9f}),
                                             bytes(world)};
    };
    auto square = [](std::vector<uint8_t> &pixels, int x0, int y0) {
        for (int y = y0; y < y0 + 32; y++)
            for (int x = x0; x < x0 + 32; x++)
                for (int c = 0; c < 3; c++)
                    pixels[(size_t(y) * 192 + x) * 3 + c] = 255;
    };
    auto anchor = [](int column, int row) { return size_t(row * 24 + column) * 2; };
    Options two;
    two.max_hands = 2;
    // Two separate hands are both acquired and both keep their tracks.
    std::vector<uint8_t> apart(192 * 192 * 3, 0);
    square(apart, 12, 84);
    square(apart, 148, 84);
    palm_anchors = {anchor(3, 12), anchor(20, 12)};
    Tracker pair(scene, two);
    a = pair.process({apart.data(), 192, 192, 192 * 3}, 1);
    check(a.hands.size() == 2 && a.hands[0].id != a.hands[1].id);
    const auto first = a.hands[0].id, second = a.hands[1].id;
    a = pair.process({apart.data(), 192, 192, 192 * 3}, 2);
    check(!a.detector_ran && a.hands.size() == 2);
    check((a.hands[0].id == first && a.hands[1].id == second) ||
          (a.hands[0].id == second && a.hands[1].id == first));
    // Two palm detections on one hand produce one hand, not two copies that
    // would fill both slots and block detection of a second hand.
    std::vector<uint8_t> one(192 * 192 * 3, 0);
    square(one, 80, 84);
    palm_anchors = {anchor(9, 12), anchor(13, 12)};
    Tracker single(scene, two);
    a = single.process({one.data(), 192, 192, 192 * 3}, 1);
    check(a.detector_ran && a.hands.size() == 1);
    // The free slot keeps the periodic detector running for a second hand.
    for (uint64_t t = 2; t <= 10; t++)
        check(single.process({one.data(), 192, 192, 192 * 3}, t).hands.size() == 1);
    a = single.process({one.data(), 192, 192, 192 * 3}, 11);
    check(a.detector_ran && a.hands.size() == 1);
    std::cout << "RGB crops, detector decode, tracking reuse, loss/reacquisition, timestamp "
                 "reset, two-hand tracking and duplicate suppression passed\n";
}
