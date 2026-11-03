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

static uint8_t FONT5x7[128][5];
static bool g_fontInitialized = false;
void initFont() {
    if (g_fontInitialized) return;
    std::memset(FONT5x7, 0, sizeof(FONT5x7));
    auto setG = [](char c, uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3, uint8_t b4) {
        FONT5x7[(int)c][0] = b0; FONT5x7[(int)c][1] = b1; FONT5x7[(int)c][2] = b2;
        FONT5x7[(int)c][3] = b3; FONT5x7[(int)c][4] = b4;
    };
    setG(' ', 0x00, 0x00, 0x00, 0x00, 0x00); setG('-', 0x08, 0x08, 0x08, 0x08, 0x08);
    setG('.', 0x00, 0x60, 0x60, 0x00, 0x00); setG(':', 0x00, 0x36, 0x36, 0x00, 0x00);
    setG('/', 0x20, 0x10, 0x08, 0x04, 0x02); setG('|', 0x00, 0x00, 0x7F, 0x00, 0x00);
    setG('0', 0x3E, 0x51, 0x49, 0x45, 0x3E); setG('1', 0x00, 0x42, 0x7F, 0x40, 0x00);
    setG('2', 0x42, 0x61, 0x51, 0x49, 0x46); setG('3', 0x21, 0x41, 0x45, 0x4B, 0x31);
    setG('4', 0x18, 0x14, 0x12, 0x7F, 0x10); setG('5', 0x27, 0x45, 0x45, 0x45, 0x39);
    setG('6', 0x3C, 0x4A, 0x49, 0x49, 0x30); setG('7', 0x01, 0x71, 0x09, 0x05, 0x03);
    setG('8', 0x36, 0x49, 0x49, 0x49, 0x36); setG('9', 0x06, 0x49, 0x49, 0x29, 0x1E);
    setG('A', 0x7C, 0x12, 0x11, 0x12, 0x7C); setG('B', 0x7F, 0x49, 0x49, 0x49, 0x36);
    setG('C', 0x3E, 0x41, 0x41, 0x41, 0x22); setG('D', 0x7F, 0x41, 0x41, 0x22, 0x1C);
    setG('E', 0x7F, 0x49, 0x49, 0x49, 0x41); setG('F', 0x7F, 0x09, 0x09, 0x09, 0x01);
    setG('G', 0x3E, 0x41, 0x49, 0x49, 0x7A); setG('H', 0x7F, 0x08, 0x08, 0x08, 0x7F);
    setG('I', 0x00, 0x41, 0x7F, 0x41, 0x00); setG('J', 0x20, 0x40, 0x41, 0x3F, 0x01);
    setG('K', 0x7F, 0x08, 0x14, 0x22, 0x41); setG('L', 0x7F, 0x40, 0x40, 0x40, 0x40);
    setG('M', 0x7F, 0x02, 0x0C, 0x02, 0x7F); setG('N', 0x7F, 0x04, 0x08, 0x10, 0x7F);
    setG('O', 0x3E, 0x41, 0x41, 0x41, 0x3E); setG('P', 0x7F, 0x09, 0x09, 0x09, 0x06);
    setG('Q', 0x3E, 0x41, 0x51, 0x21, 0x5E); setG('R', 0x7F, 0x09, 0x19, 0x29, 0x46);
    setG('S', 0x46, 0x49, 0x49, 0x49, 0x31); setG('T', 0x01, 0x01, 0x7F, 0x01, 0x01);
    setG('U', 0x3F, 0x40, 0x40, 0x40, 0x3F); setG('V', 0x07, 0x38, 0x40, 0x38, 0x07);
    setG('W', 0x7F, 0x20, 0x18, 0x20, 0x7F); setG('X', 0x63, 0x14, 0x08, 0x14, 0x63);
    setG('Y', 0x07, 0x08, 0x70, 0x08, 0x07); setG('Z', 0x61, 0x51, 0x49, 0x45, 0x43);
    setG('[', 0x00, 0x7F, 0x41, 0x41, 0x00); setG(']', 0x00, 0x41, 0x41, 0x7F, 0x00);
    g_fontInitialized = true;
}

static volatile bool g_running = true;
void sigHandler(int) { g_running = false; }

struct Point3D { float x, y, z; };
struct Edge { int v0, v1; };

void drawLine(uint32_t* buf, int stride, int W, int H, int x0, int y0, int x1, int y1, uint32_t color) {
    int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1, sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    while (true) {
        if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) {
            buf[y0 * stride + x0] = color;
            if (x0 + 1 < W) buf[y0 * stride + (x0 + 1)] = color;
            if (y0 + 1 < H) buf[(y0 + 1) * stride + x0] = color;
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
    }
}

void drawPointMarker(uint32_t* buf, int stride, int W, int H, int cx, int cy, uint32_t color) {
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int px = cx + dx, py = cy + dy;
            if (px >= 0 && px < W && py >= 0 && py < H) buf[py * stride + px] = color;
        }
    }
}

void drawText(uint32_t* buf, int stride, int bufW, int bufH, int startX, int startY, const std::string& str, uint32_t color) {
    int curX = startX;
    for (char c : str) {
        if (c < 0 || c > 127) c = ' ';
        const uint8_t* glyph = FONT5x7[(int)c];
        for (int col = 0; col < 5; col++) {
            uint8_t bits = glyph[col];
            for (int row = 0; row < 7; row++) {
                if (bits & (1 << row)) {
                    int px = curX + col, py = startY + row;
                    if (px >= 0 && px < bufW && py >= 0 && py < bufH) buf[py * stride + px] = color;
                }
            }
        }
        curX += 7;
    }
}

int main() {
    initFont();
    std::signal(SIGINT, sigHandler);
    std::signal(SIGTERM, sigHandler);

    vr::EVRInitError err = vr::VRInitError_None;
    vr::IVRSystem *vr_sys = vr::VR_Init(&err, vr::VRApplication_Overlay);
    if (err != vr::VRInitError_None) return 1;

    vr::IVROverlay *overlay = vr::VROverlay();
    vr::VROverlayHandle_t ovHandle;
    overlay->CreateOverlay("frame.spatial.mesh3d.stereo", "Spatial Room Mesh", &ovHandle);
    
    // STEREO FIX: Set overlay to be Side-By-Side Parallel!
    overlay->SetOverlayFlag(ovHandle, vr::VROverlayFlags_SideBySide_Parallel, true);

    const float OVERLAY_DISTANCE = 1.25f;
    const float OVERLAY_WIDTH_METERS = 3.0f; // Wide FOV
    
    vr::HmdMatrix34_t hmdTransform = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, -OVERLAY_DISTANCE
    };
    overlay->SetOverlayTransformTrackedDeviceRelative(ovHandle, vr::k_unTrackedDeviceIndex_Hmd, &hmdTransform);
    overlay->SetOverlayWidthInMeters(ovHandle, OVERLAY_WIDTH_METERS);
    overlay->SetOverlayAlpha(ovHandle, 0.95f);
    overlay->ShowOverlay(ovHandle);

    std::vector<Point3D> landmarks, meshVerts;
    std::vector<Edge> meshEdges;
    {
        std::ifstream ply("/home/steamos/test_openvr/real_room_points.ply");
        std::string line; bool inHeader = true;
        while (std::getline(ply, line)) {
            if (line == "end_header") { inHeader = false; continue; }
            if (inHeader) continue;
            float x, y, z; int r, g, b;
            if (sscanf(line.c_str(), "%f %f %f %d %d %d", &x, &y, &z, &r, &g, &b) >= 3) landmarks.push_back({x, y, z});
        }
        std::ifstream obj("/home/steamos/test_openvr/real_room_mesh.obj");
        while (std::getline(obj, line)) {
            if (line.rfind("v ", 0) == 0) {
                float x, y, z; sscanf(line.c_str() + 2, "%f %f %f", &x, &y, &z);
                meshVerts.push_back({x, y, z});
            } else if (line.rfind("f ", 0) == 0) {
                int v0, v1, v2; sscanf(line.c_str() + 2, "%d %d %d", &v0, &v1, &v2);
                meshEdges.push_back({v0 - 1, v1 - 1}); meshEdges.push_back({v1 - 1, v2 - 1}); meshEdges.push_back({v2 - 1, v0 - 1});
            }
        }
    }

    const int EYE_W = 1280;
    const int EYE_H = 720;
    const int TEX_W = EYE_W * 2;
    const int TEX_H = EYE_H;
    std::vector<uint32_t> buffer(TEX_W * TEX_H, 0);

    const float OVERLAY_HEIGHT_METERS = OVERLAY_WIDTH_METERS * ((float)EYE_H / (float)EYE_W);
    
    vr::HmdMatrix34_t eL = vr_sys->GetEyeToHeadTransform(vr::Eye_Left);
    vr::HmdMatrix34_t eR = vr_sys->GetEyeToHeadTransform(vr::Eye_Right);
    float eyeOffsetX[2] = { eL.m[0][3], eR.m[0][3] }; // ~ -0.032, +0.032

    auto lastFpsTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float liveFps = 0.0f;

    while (g_running) {
        vr::VREvent_t vrEvent;
        while (overlay->PollNextOverlayEvent(ovHandle, &vrEvent, sizeof(vrEvent))) {
            if (vrEvent.eventType == vr::VREvent_Quit) g_running = false;
        }

        vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
        vr_sys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0.0f, poses, vr::k_unMaxTrackedDeviceCount);
        if (!poses[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid) { usleep(10000); continue; }

        const auto &hmdMat = poses[vr::k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking.m;
        float R[3][3] = { {hmdMat[0][0], hmdMat[1][0], hmdMat[2][0]}, {hmdMat[0][1], hmdMat[1][1], hmdMat[2][1]}, {hmdMat[0][2], hmdMat[1][2], hmdMat[2][2]} };
        float HmdPos[3] = {hmdMat[0][3], hmdMat[1][3], hmdMat[2][3]};

        std::fill(buffer.begin(), buffer.end(), 0x00000000);

        for (int eye = 0; eye < 2; eye++) {
            float ex = eyeOffsetX[eye];
            int xOffset = eye * EYE_W;

            std::vector<int> screenX(meshVerts.size(), -1);
            std::vector<int> screenY(meshVerts.size(), -1);
            std::vector<bool> vertValid(meshVerts.size(), false);

            for (size_t i = 0; i < meshVerts.size(); i++) {
                float dx = meshVerts[i].x - HmdPos[0], dy = meshVerts[i].y - HmdPos[1], dz = meshVerts[i].z - HmdPos[2];
                float hx = R[0][0] * dx + R[0][1] * dy + R[0][2] * dz;
                float hy = R[1][0] * dx + R[1][1] * dy + R[1][2] * dz;
                float hz = R[2][0] * dx + R[2][1] * dy + R[2][2] * dz;

                if (hz < -0.1f) {
                    float t = -OVERLAY_DISTANCE / hz; // ray intersect z = -OVERLAY_DISTANCE
                    float ix = ex + t * (hx - ex);
                    float iy = 0.0f + t * (hy - 0.0f);
                    
                    int sx = (int)(((ix / OVERLAY_WIDTH_METERS) + 0.5f) * EYE_W);
                    int sy = (int)((0.5f - (iy / OVERLAY_HEIGHT_METERS)) * EYE_H);
                    
                    screenX[i] = sx; screenY[i] = sy;
                    vertValid[i] = (sx >= -100 && sx < EYE_W + 100 && sy >= -100 && sy < EYE_H + 100);
                }
            }

            uint32_t wallColor = (eye == 0) ? 0xFFFFFF00 : 0xFFFFFF00;
            for (const auto& e : meshEdges) {
                if (vertValid[e.v0] && vertValid[e.v1]) {
                    drawLine(buffer.data() + xOffset, TEX_W, EYE_W, EYE_H, screenX[e.v0], screenY[e.v0], screenX[e.v1], screenY[e.v1], wallColor);
                }
            }

            for (const auto& pt : landmarks) {
                float dx = pt.x - HmdPos[0], dy = pt.y - HmdPos[1], dz = pt.z - HmdPos[2];
                float hx = R[0][0] * dx + R[0][1] * dy + R[0][2] * dz;
                float hy = R[1][0] * dx + R[1][1] * dy + R[1][2] * dz;
                float hz = R[2][0] * dx + R[2][1] * dy + R[2][2] * dz;

                if (hz < -0.1f) {
                    float t = -OVERLAY_DISTANCE / hz;
                    float ix = ex + t * (hx - ex);
                    float iy = 0.0f + t * (hy - 0.0f);
                    int sx = (int)(((ix / OVERLAY_WIDTH_METERS) + 0.5f) * EYE_W);
                    int sy = (int)((0.5f - (iy / OVERLAY_HEIGHT_METERS)) * EYE_H);

                    if (sx >= 2 && sx < EYE_W - 2 && sy >= 2 && sy < EYE_H - 2) {
                        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                        uint32_t ptColor = (dist < 1.0f) ? 0xFF00FF88 : 0xFF00E5FF;
                        drawPointMarker(buffer.data() + xOffset, TEX_W, EYE_W, EYE_H, sx, sy, ptColor);
                    }
                }
            }
            
            if (eye == 0) { // Only draw HUD on left eye
                char hud[128];
                snprintf(hud, sizeof(hud), "[STEREO 3D SPATIAL MESH] | FPS: %0.1f", liveFps);
                drawText(buffer.data() + xOffset, TEX_W, EYE_W, EYE_H, 40, 30, hud, 0xFF00FFFF);
            }
        }

        frameCount++;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - lastFpsTime).count();
        if (elapsed >= 0.5) { liveFps = (float)(frameCount / elapsed); frameCount = 0; lastFpsTime = now; }

        overlay->SetOverlayRaw(ovHandle, buffer.data(), TEX_W, TEX_H, 4);
        
        // WaitFrameSync to stop tearing (placed after SetOverlayRaw!)
        // overlay->WaitFrameSync(100); 
        usleep(11000); // 90 Hz
    }

    overlay->DestroyOverlay(ovHandle);
    vr::VR_Shutdown();
    return 0;
}
