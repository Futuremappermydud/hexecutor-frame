#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <csignal>
#include <cstring>
#include <openvr.h>

// Simple 5x7 ASCII bitmap font for on-screen HUD
static uint8_t FONT5x7[128][5];
static bool g_fontInitialized = false;

void initFont() {
    if (g_fontInitialized) return;
    std::memset(FONT5x7, 0, sizeof(FONT5x7));
    auto setG = [](char c, uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3, uint8_t b4) {
        FONT5x7[(int)c][0] = b0; FONT5x7[(int)c][1] = b1; FONT5x7[(int)c][2] = b2;
        FONT5x7[(int)c][3] = b3; FONT5x7[(int)c][4] = b4;
    };
    setG(' ', 0x00, 0x00, 0x00, 0x00, 0x00);
    setG('-', 0x08, 0x08, 0x08, 0x08, 0x08);
    setG('.', 0x00, 0x60, 0x60, 0x00, 0x00);
    setG(':', 0x00, 0x36, 0x36, 0x00, 0x00);
    setG('[', 0x00, 0x7F, 0x41, 0x41, 0x00);
    setG(']', 0x00, 0x41, 0x41, 0x7F, 0x00);
    setG('/', 0x20, 0x10, 0x08, 0x04, 0x02);
    setG('|', 0x00, 0x00, 0x7F, 0x00, 0x00);
    setG('0', 0x3E, 0x51, 0x49, 0x45, 0x3E);
    setG('1', 0x00, 0x42, 0x7F, 0x40, 0x00);
    setG('2', 0x42, 0x61, 0x51, 0x49, 0x46);
    setG('3', 0x21, 0x41, 0x45, 0x4B, 0x31);
    setG('4', 0x18, 0x14, 0x12, 0x7F, 0x10);
    setG('5', 0x27, 0x45, 0x45, 0x45, 0x39);
    setG('6', 0x3C, 0x4A, 0x49, 0x49, 0x30);
    setG('7', 0x01, 0x71, 0x09, 0x05, 0x03);
    setG('8', 0x36, 0x49, 0x49, 0x49, 0x36);
    setG('9', 0x06, 0x49, 0x49, 0x29, 0x1E);
    setG('A', 0x7C, 0x12, 0x11, 0x12, 0x7C);
    setG('B', 0x7F, 0x49, 0x49, 0x49, 0x36);
    setG('C', 0x3E, 0x41, 0x41, 0x41, 0x22);
    setG('D', 0x7F, 0x41, 0x41, 0x22, 0x1C);
    setG('E', 0x7F, 0x49, 0x49, 0x49, 0x41);
    setG('F', 0x7F, 0x09, 0x09, 0x09, 0x01);
    setG('G', 0x3E, 0x41, 0x49, 0x49, 0x7A);
    setG('H', 0x7F, 0x08, 0x08, 0x08, 0x7F);
    setG('I', 0x00, 0x41, 0x7F, 0x41, 0x00);
    setG('J', 0x20, 0x40, 0x41, 0x3F, 0x01);
    setG('K', 0x7F, 0x08, 0x14, 0x22, 0x41);
    setG('L', 0x7F, 0x40, 0x40, 0x40, 0x40);
    setG('M', 0x7F, 0x02, 0x0C, 0x02, 0x7F);
    setG('N', 0x7F, 0x04, 0x08, 0x10, 0x7F);
    setG('O', 0x3E, 0x41, 0x41, 0x41, 0x3E);
    setG('P', 0x7F, 0x09, 0x09, 0x09, 0x06);
    setG('Q', 0x3E, 0x41, 0x51, 0x21, 0x5E);
    setG('R', 0x7F, 0x09, 0x19, 0x29, 0x46);
    setG('S', 0x46, 0x49, 0x49, 0x49, 0x31);
    setG('T', 0x01, 0x01, 0x7F, 0x01, 0x01);
    setG('U', 0x3F, 0x40, 0x40, 0x40, 0x3F);
    setG('V', 0x07, 0x38, 0x40, 0x38, 0x07);
    setG('W', 0x7F, 0x20, 0x18, 0x20, 0x7F);
    setG('X', 0x63, 0x14, 0x08, 0x14, 0x63);
    setG('Y', 0x07, 0x08, 0x70, 0x08, 0x07);
    setG('Z', 0x61, 0x51, 0x49, 0x45, 0x43);
    setG('m', 0x7C, 0x04, 0x18, 0x04, 0x78);
    setG('s', 0x48, 0x54, 0x54, 0x54, 0x20);
    g_fontInitialized = true;
}

static volatile bool g_running = true;
void sigHandler(int) {
    g_running = false;
}

// Draw string on RGBA buffer
void drawText(uint32_t* buf, int bufW, int bufH, int startX, int startY, const std::string& str, uint32_t color) {
    int curX = startX;
    for (char c : str) {
        if (c < 0 || c > 127) c = ' ';
        const uint8_t* glyph = FONT5x7[(int)c];
        for (int col = 0; col < 5; col++) {
            uint8_t bits = glyph[col];
            for (int row = 0; row < 7; row++) {
                if (bits & (1 << row)) {
                    int px = curX + col;
                    int py = startY + row;
                    if (px >= 0 && px < bufW && py >= 0 && py < bufH) {
                        buf[py * bufW + px] = color;
                    }
                }
            }
        }
        curX += 7; // 5 + 2 spacing
    }
}

// Convert depth in meters [0.3 .. 3.5] to Turbo colormap RGBA
inline uint32_t depthToRGBA(float z, float alphaVal) {
    if (z <= 0.0f) return 0x00000000; // Transparent
    // Normalize z: near = 1.0 (warm), far = 0.0 (cool)
    float t = (3.5f - z) / (3.5f - 0.3f);
    t = std::max(0.0f, std::min(1.0f, t));

    // Turbo-like approximation
    float r = std::sin(t * 1.57f);
    float g = std::sin(t * 3.1415f);
    float b = std::cos(t * 1.57f);

    uint8_t ur = (uint8_t)(std::max(0.0f, std::min(1.0f, r)) * 255.0f);
    uint8_t ug = (uint8_t)(std::max(0.0f, std::min(1.0f, g)) * 255.0f);
    uint8_t ub = (uint8_t)(std::max(0.0f, std::min(1.0f, b)) * 255.0f);
    uint8_t ua = (uint8_t)(alphaVal * 255.0f);

    // OpenVR SetOverlayRaw RGBA byte order (little endian uint32_t: 0xAABBGGRR)
    return (uint32_t)ur | ((uint32_t)ug << 8) | ((uint32_t)ub << 16) | ((uint32_t)ua << 24);
}

int main() {
    initFont();
    std::signal(SIGINT, sigHandler);
    std::signal(SIGTERM, sigHandler);

    std::cout << "[SpatialRuntime] Initializing OpenVR Overlay subsystem..." << std::endl;
    vr::EVRInitError err = vr::VRInitError_None;
    vr::IVRSystem *vr_sys = vr::VR_Init(&err, vr::VRApplication_Overlay);
    if (err != vr::VRInitError_None) {
        std::cerr << "[SpatialRuntime] VR_Init failed: " << vr::VR_GetVRInitErrorAsEnglishDescription(err) << std::endl;
        return 1;
    }

    vr::IVROverlay *overlay = vr::VROverlay();
    if (!overlay) {
        std::cerr << "[SpatialRuntime] Failed to obtain IVROverlay interface!" << std::endl;
        vr::VR_Shutdown();
        return 1;
    }

    vr::VROverlayHandle_t ovHandle = vr::k_ulOverlayHandleInvalid;
    vr::EVROverlayError ovErr = overlay->CreateOverlay("valve.frame.spatial.depth", "Frame Spatial Depth Overlay", &ovHandle);
    if (ovErr != vr::VROverlayError_None) {
        std::cerr << "[SpatialRuntime] CreateOverlay failed: " << ovErr << std::endl;
        vr::VR_Shutdown();
        return 1;
    }

    // Configure Stereo Side-by-Side: Left half texture -> Left eye, Right half texture -> Right eye
    overlay->SetOverlayFlag(ovHandle, vr::VROverlayFlags_SideBySide_Parallel, true);

    // Overlay geometry: Quad locked 1.25 meters in front of user's HMD
    vr::HmdMatrix34_t hmdTransform = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, -1.25f
    };
    overlay->SetOverlayTransformTrackedDeviceRelative(ovHandle, vr::k_unTrackedDeviceIndex_Hmd, &hmdTransform);
    overlay->SetOverlayWidthInMeters(ovHandle, 1.8f);
    overlay->SetOverlayAlpha(ovHandle, 0.85f);
    overlay->ShowOverlay(ovHandle);

    std::cout << "[SpatialRuntime] Overlay registered & active. Opening video feed /dev/video99..." << std::endl;
    int vidFd = open("/dev/video99", O_RDONLY);
    if (vidFd < 0) {
        std::cerr << "[SpatialRuntime] Could not open /dev/video99!" << std::endl;
        overlay->DestroyOverlay(ovHandle);
        vr::VR_Shutdown();
        return 1;
    }

    const int RAW_W = 1920;
    const int RAW_H = 1080;
    std::vector<uint8_t> rawFrame(RAW_W * RAW_H * 3);

    // Processing resolution per eye: 320x360
    const int W = 320;
    const int H = 360;
    const int TOTAL_OVERLAY_W = W * 2; // 640x360 side-by-side
    const int TOTAL_OVERLAY_H = H;
    std::vector<uint32_t> overlayBuffer(TOTAL_OVERLAY_W * TOTAL_OVERLAY_H, 0);

    std::vector<uint8_t> leftGrey(W * H, 0);
    std::vector<uint8_t> rightGrey(W * H, 0);
    std::vector<float> depthLeft(W * H, 0.0f);
    std::vector<float> depthRight(W * H, 0.0f);

    const float sx = 960.0f / W;
    const float sy = 1080.0f / H;
    const int dyShift = 6;
    const int minD = 10;
    const int maxD = 200;
    const int winR = 2; // 5x5 stereo matching window
    const float focal = 280.0f;
    const float baseline = 0.063f; // 63mm

    auto lastFpsTime = std::chrono::high_resolution_clock::now();
    int frameCounter = 0;
    float currentFps = 0.0f;
    float closestObstacle = 99.0f;

    std::cout << "[SpatialRuntime] Entering real-time spatial processing loop (Press Ctrl+C to exit)..." << std::endl;

    while (g_running) {
        // Handle OpenVR events (e.g. Dashboard closed, quit requested)
        vr::VREvent_t vrEvent;
        while (overlay->PollNextOverlayEvent(ovHandle, &vrEvent, sizeof(vrEvent))) {
            if (vrEvent.eventType == vr::VREvent_Quit) {
                g_running = false;
                break;
            }
        }

        // Read 1 synchronized stereo frame from SteamVR loopback
        ssize_t bytesRead = read(vidFd, rawFrame.data(), rawFrame.size());
        if (bytesRead != (ssize_t)rawFrame.size()) {
            usleep(5000);
            continue;
        }

        // 1. Ingest & downsample stereo streams
        #pragma omp parallel for
        for (int y = 0; y < H; y++) {
            int srcY_L = (int)(y * sy);
            int srcY_R = (int)((y - dyShift) * sy);

            for (int x = 0; x < W; x++) {
                int srcX = (int)(x * sx);

                // Left eye = right half [960..1920)
                int idxL = (srcY_L * RAW_W + (srcX + 960)) * 3;
                uint8_t lr = rawFrame[idxL], lg = rawFrame[idxL+1], lb = rawFrame[idxL+2];
                leftGrey[y * W + x] = (uint8_t)(0.299f * lr + 0.587f * lg + 0.114f * lb);

                // Right eye = left half [0..960)
                if (srcY_R >= 0 && srcY_R < RAW_H) {
                    int idxR = (srcY_R * RAW_W + srcX) * 3;
                    uint8_t rr = rawFrame[idxR], rg = rawFrame[idxR+1], rb = rawFrame[idxR+2];
                    rightGrey[y * W + x] = (uint8_t)(0.299f * rr + 0.587f * rg + 0.114f * rb);
                } else {
                    rightGrey[y * W + x] = 0;
                }
            }
        }

        // 2. High-speed Stereo Disparity Matching
        std::fill(depthLeft.begin(), depthLeft.end(), 0.0f);
        std::fill(depthRight.begin(), depthRight.end(), 0.0f);
        closestObstacle = 99.0f;

        #pragma omp parallel for schedule(dynamic) reduction(min:closestObstacle)
        for (int y = winR; y < H - winR; y++) {
            for (int x = maxD + winR; x < W - winR; x++) {
                int bestCost = 1e9;
                int bestD = minD;
                int secondCost = 1e9;

                for (int d = minD; d < maxD; d++) {
                    int cost = 0;
                    for (int wy = -winR; wy <= winR; wy++) {
                        for (int wx = -winR; wx <= winR; wx++) {
                            int idxL = (y + wy) * W + (x + wx);
                            int idxR = (y + wy) * W + (x + wx - d);
                            cost += std::abs(leftGrey[idxL] - rightGrey[idxR]);
                        }
                    }
                    if (cost < bestCost) {
                        secondCost = bestCost;
                        bestCost = cost;
                        bestD = d;
                    } else if (cost < secondCost) {
                        secondCost = cost;
                    }
                }

                if (secondCost > 0 && (float)bestCost / secondCost < 0.93f) {
                    float z = (focal * baseline) / (float)bestD;
                    if (z >= 0.25f && z <= 5.0f) {
                        depthLeft[y * W + x] = z;
                        if (z < closestObstacle) closestObstacle = z;

                        // Reproject into Right eye for 3D stereoscopic parity
                        int rx = x - bestD;
                        int ry = y + dyShift;
                        if (rx >= 0 && rx < W && ry >= 0 && ry < H) {
                            depthRight[ry * W + rx] = z;
                        }
                    }
                }
            }
        }

        // 3. Clear overlay buffer
        std::fill(overlayBuffer.begin(), overlayBuffer.end(), 0x00000000);

        // 4. Render Depth Mesh & Neon Edges into Left & Right overlay panels
        #pragma omp parallel for
        for (int y = 1; y < H - 1; y++) {
            for (int x = 1; x < W - 1; x++) {
                // ---- LEFT EYE (x: 0..W-1 in overlayBuffer) ----
                float zL = depthLeft[y * W + x];
                if (zL > 0.0f) {
                    // Edge detection via depth gradient
                    float dz_dx = std::abs(depthLeft[y * W + (x + 1)] - depthLeft[y * W + (x - 1)]);
                    float dz_dy = std::abs(depthLeft[(y + 1) * W + x] - depthLeft[(y - 1) * W + x]);
                    bool isEdge = (dz_dx > 0.08f || dz_dy > 0.08f);

                    uint32_t colL;
                    if (isEdge) {
                        // High-contrast neon cyan edge outline (alpha 0.9)
                        colL = 0xE6FFFF00; // Cyan RGBA
                    } else {
                        // Semi-transparent surface depth heatmap (alpha 0.45)
                        colL = depthToRGBA(zL, 0.45f);
                    }
                    overlayBuffer[y * TOTAL_OVERLAY_W + x] = colL;
                }

                // ---- RIGHT EYE (x: W..TOTAL_OVERLAY_W-1 in overlayBuffer) ----
                float zR = depthRight[y * W + x];
                if (zR > 0.0f) {
                    float dz_dx = std::abs(depthRight[y * W + (x + 1)] - depthRight[y * W + (x - 1)]);
                    float dz_dy = std::abs(depthRight[(y + 1) * W + x] - depthRight[(y - 1) * W + x]);
                    bool isEdge = (dz_dx > 0.08f || dz_dy > 0.08f);

                    uint32_t colR;
                    if (isEdge) {
                        colR = 0xE6FFFF00; // Cyan RGBA
                    } else {
                        colR = depthToRGBA(zR, 0.45f);
                    }
                    overlayBuffer[y * TOTAL_OVERLAY_W + (W + x)] = colR;
                }
            }
        }

        // 5. Draw Head-Up-Display (HUD) Telemetry on both eyes
        frameCounter++;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsedSec = std::chrono::duration<double>(now - lastFpsTime).count();
        if (elapsedSec >= 1.0) {
            currentFps = (float)(frameCounter / elapsedSec);
            std::cout << "[SpatialRuntime] Live FPS: " << currentFps 
                      << " | Closest Surface: " << closestObstacle << "m" << std::endl;
            frameCounter = 0;
            lastFpsTime = now;
        }

        char hudText[64];
        if (closestObstacle < 10.0f) {
            snprintf(hudText, sizeof(hudText), "FPS: %0.1f | MIN DIST: %0.2fm", currentFps, closestObstacle);
        } else {
            snprintf(hudText, sizeof(hudText), "FPS: %0.1f | SCANNING ROOM...", currentFps);
        }

        uint32_t hudColor = (closestObstacle < 0.6f) ? 0xFF0055FF : 0xFF00FF55; // Orange warning if close, Green if clear
        // Left Eye HUD
        drawText(overlayBuffer.data(), TOTAL_OVERLAY_W, TOTAL_OVERLAY_H, 20, 20, hudText, hudColor);
        drawText(overlayBuffer.data(), TOTAL_OVERLAY_W, TOTAL_OVERLAY_H, 20, 32, "[FRAME SPATIAL RUNTIME - STEREO 3D]", 0xFF00E5FF);

        // Right Eye HUD
        drawText(overlayBuffer.data(), TOTAL_OVERLAY_W, TOTAL_OVERLAY_H, W + 20, 20, hudText, hudColor);
        drawText(overlayBuffer.data(), TOTAL_OVERLAY_W, TOTAL_OVERLAY_H, W + 20, 32, "[FRAME SPATIAL RUNTIME - STEREO 3D]", 0xFF00E5FF);

        // 6. Submit texture directly to SteamVR compositor
        overlay->SetOverlayRaw(ovHandle, overlayBuffer.data(), TOTAL_OVERLAY_W, TOTAL_OVERLAY_H, 4);
    }

    std::cout << "[SpatialRuntime] Shutting down overlay and releasing resources..." << std::endl;
    usleep(100000); // 100ms pause to ensure compositor finishes rendering last frame
    overlay->HideOverlay(ovHandle);
    usleep(50000);
    overlay->DestroyOverlay(ovHandle);
    vr::VR_Shutdown();
    close(vidFd);
    std::cout << "[SpatialRuntime] Clean exit." << std::endl;
    return 0;
}
