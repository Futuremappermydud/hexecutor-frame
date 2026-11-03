#include "stereo_matcher_experiment.hpp"
#include <iostream>
using namespace Spatial::StereoExperiment;
int main() {
    Config c; c.width = 40; c.height = 32; c.maxDisparity = 16;
    std::vector<uint8_t> a(1280, 100), b = a;
    auto blank = match(a, b, c, Method::Adaptive);
    for (float z : blank.depth) if (z != 0) return 1;
    // A target-only edge must stop the rightward arm, even if reference is flat.
    std::vector<uint64_t> ra(1280, 0), rb(1280, 0);
    auto v = detail::adaptiveCost(ra, rb, a, b, 20, 16, 5, false, c);
    if (v.first != 240 || v.second != 17) return 2;
    for (int x = 16; x < 40; ++x) b[16 * 40 + x] = 200;
    auto stopped = detail::adaptiveCost(ra, rb, a, b, 20, 16, 5, false, c);
    if (stopped.second != 13) return 3;
    // Maximum census disagreement stays at 240 after sample-count normalization.
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = b[i] = uint8_t(100 + (i % 40) % 8);
        rb[i] = (uint64_t(1) << 48) - 1;
    }
    v = detail::adaptiveCost(ra, rb, a, b, 20, 16, 8, false, c);
    if (v.first != 240) return 4;
    // Swapping reference/target and reversing direction preserves hypotheses.
    for (int d = 1; d < 12; ++d) {
        const auto left = detail::adaptiveCost(ra, rb, a, b, 20, 16, d, false, c);
        const auto right = detail::adaptiveCost(rb, ra, b, a, 20-d, 16, d, true, c);
        if (left != right) return 5;
    }
    c.supportRadius = 9;
    try { match(a, b, c, Method::Adaptive); return 6; }
    catch (const std::invalid_argument&) {}
    std::cout << "Adaptive support controls passed\n";
}
