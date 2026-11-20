// Offline re-rectification of a saved 2464x2464 luma pair with the current
// calibration source (set FRAME_STEREO_CALIBRATION to choose a history file).
#include "depth_pipeline.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
using namespace FrameSpatial;
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "Usage: " << argv[0] << " raw-left raw-right output-prefix\n"; return 2; }
    auto load = [](const char* p) { std::ifstream f(p, std::ios::binary); return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {}); };
    const auto l = load(argv[1]), r = load(argv[2]);
    if (l.size() != 2464u * 2464u || r.size() != l.size()) { std::cerr << "bad raw size\n"; return 1; }
    DepthPipeline rectifier;
    rectifier.rectify(l.data(), r.data());
    for (int eye = 0; eye < 2; ++eye) {
        const auto& px = eye ? rectifier.getRectifiedRight() : rectifier.getRectifiedLeft();
        std::ofstream(std::string(argv[3]) + (eye ? "-rect-right.u8" : "-rect-left.u8"), std::ios::binary)
            .write(reinterpret_cast<const char*>(px.data()), px.size());
    }
}
