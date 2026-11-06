#pragma once
#include <cmath>
namespace FrameSpatial {
// Keep capture/history across short empty compositor submissions. These frames
// never receive masks, and preserving capture never renews a surface deadline.
class OcclusionCaptureGate {
    double lastDepthFrame_ = 0;
public:
    void reset() { lastDepthFrame_ = 0; }
    bool update(bool depthFrame, bool emptyAlphaFrame, double now) {
        if (!std::isfinite(now) || now <= 0) { reset(); return false; }
        if (depthFrame) { lastDepthFrame_ = now; return true; }
        if (emptyAlphaFrame && lastDepthFrame_ > 0 && now >= lastDepthFrame_ &&
            now - lastDepthFrame_ <= 0.25) return true;
        reset(); return false;
    }
};
}
