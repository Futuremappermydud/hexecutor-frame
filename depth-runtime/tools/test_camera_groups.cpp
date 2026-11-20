// Camera DMA-BUF grouping: an older, unwritten allocation must not shift the
// live stereo pair (the live right camera would otherwise be read as left).
#include "camera_groups.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace FrameSpatial;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        std::vector<int> right, left;

        // One allocation: lower half right, upper half left (previous behaviour).
        require(splitCameraGroups({{376, 761}, {364, 758}, {378, 762}, {372, 759}, {374, 760},
                                   {382, 763}, {384, 764}, {386, 765}},
                                  4, right, left),
                "single group split");
        require(right == std::vector<int>({364, 372, 374, 376}), "single group right");
        require(left == std::vector<int>({378, 382, 384, 386}), "single group left");

        // Frame, 2026-11-09: a dead boot-time group (inodes 339-346) plus the
        // live group (758-765). Only the newest allocation is the stereo pair.
        std::vector<CameraBuffer> both{{333, 339}, {335, 340}, {337, 341}, {339, 342},
                                       {341, 343}, {343, 344}, {345, 345}, {347, 346},
                                       {364, 758}, {372, 759}, {374, 760}, {376, 761},
                                       {378, 762}, {382, 763}, {384, 764}, {386, 765}};
        require(splitCameraGroups(both, 4, right, left), "two groups split");
        require(right == std::vector<int>({364, 372, 374, 376}), "two groups right is live right");
        require(left == std::vector<int>({378, 382, 384, 386}), "two groups left is live left");

        // Descriptor numbers need not follow allocation order.
        require(splitCameraGroups({{319, 900}, {401, 901}, {402, 902}, {403, 903},
                                   {404, 904}, {405, 905}, {406, 906}, {407, 907},
                                   {500, 100}, {501, 101}},
                                  4, right, left),
                "inode order split");
        require(right == std::vector<int>({319, 401, 402, 403}), "inode order right");
        require(left == std::vector<int>({404, 405, 406, 407}), "inode order left");

        require(!splitCameraGroups({{3, 1}}, 4, right, left), "one buffer rejected");
        require(right.empty() && left.empty(), "rejected split leaves outputs empty");
    } catch (const std::exception& e) {
        std::cerr << "test_camera_groups: " << e.what() << '\n';
        return 1;
    }
    std::cout << "test_camera_groups: ok\n";
    return 0;
}
