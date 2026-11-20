// Read-only: prints sequence, score and pairing of the XRService IR stream for N
// seconds, using the same capture path as the live refine. Starts no emitter.
#include "ir_pair_capture.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>

int main(int argc, char** argv) {
    using namespace FrameSpatial;
    const double seconds = argc > 1 ? std::atof(argv[1]) : 5.0;
    IrPairCapture capture;
    if (!capture.initialize()) {
        std::cerr << capture.lastError() << std::endl;
        return 1;
    }
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    int64_t previous = 0;
    while (std::chrono::steady_clock::now() < end) {
        IrPairCapture::Pair pair;
        if (!capture.grab(pair, 200)) {
            std::cout << "no pair: " << capture.lastError() << std::endl;
            continue;
        }
        const double mean = std::accumulate(pair.left.begin(), pair.left.end(), 0.0) / double(pair.left.size());
        std::cout << "seq " << pair.leftSeq << "/" << pair.rightSeq << " dt " << (pair.leftTsNs - previous) * 1e-6
                  << " ms skew " << (pair.leftTsNs - pair.rightTsNs) * 1e-6 << " ms score "
                  << IrPairCapture::dotScore(pair.left) << " mean " << mean << std::endl;
        previous = pair.leftTsNs;
    }
    return 0;
}
