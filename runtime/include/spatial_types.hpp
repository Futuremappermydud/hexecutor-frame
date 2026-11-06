#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace FrameSpatial {

struct Point2D {
    float x;
    float y;
};

struct Point3D {
    float x;
    float y;
    float z;
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct HmdPose {
    float matrix[3][4];
    bool isValid;
};

struct StereoCaptureBuffer {
    const uint8_t* left;
    const uint8_t* right;
    size_t size;
    uint64_t timestampNs;
};

// Shared snapshot published by camera ingestion and consumed by the render loop.
struct RenderSnapshot {
    std::vector<Point3D> points;
    std::vector<uint8_t> rectL;
    std::vector<uint8_t> rectR;
    std::vector<float> depth;
    int rectW = 0;
    int rectH = 0;
    std::vector<uint32_t> previewL;
    std::vector<uint32_t> previewR;
    int previewW = 0;
    int previewH = 0;
    int sourceW = 0;
    int sourceH = 0;
    uint32_t frameIndex = 0;
    int leftSlot = -1;
    int rightSlot = -1;
    float captureHz = 0.0f;
    std::string status = "BOOT";
    bool camerasReady = false;
    bool gpuBusy = false;
};

} // namespace FrameSpatial
