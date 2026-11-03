// spatial_mesh_runtime.cpp — Live Stereo Depth to World-Locked Headset Mesh
#define _GNU_SOURCE
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <cmath>
#include <cstring>
#include <csignal>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <openvr.h>

#ifndef __NR_pidfd_open
#define __NR_pidfd_open 434
#endif
#ifndef __NR_pidfd_getfd
#define __NR_pidfd_getfd 438
#endif

static volatile bool g_running = true;
void sigHandler(int) { g_running = false; }

// --- Stereo Depth Constants ---
const int RAW_W = 2464;
const int RAW_H = 2464;
const int OUT_W = 616;
const int OUT_H = 616;
const int MAX_DISP = 64;

const float F_RECT = 380.0f;
const float CX_RECT = 308.0f;
const float CY_RECT = 308.0f;
const float BASELINE_M = 0.06396f; // 63.96 mm baseline
const float F_B = F_RECT * BASELINE_M;

// Left Camera Intrinsics & KB Distortion
const float FX_L = 1477.53f, FY_L = 1479.15f, CX_L = 1082.43f, CY_L = 1243.37f;
const float K1_L = -0.027497f, K2_L = -0.16945f, K3_L = 0.10018f, K4_L = -0.01506f;

// Right Camera Intrinsics & KB Distortion
const float FX_R = 1476.21f, FY_R = 1478.26f, CX_R = 1078.65f, CY_R = 1238.10f;
const float K1_R = -0.031552f, K2_R = -0.14942f, K3_R = 0.07644f, K4_R = -0.00628f;

struct Point3D {
    float x, y, z;
    uint8_t r, g, b;
};

struct Point2D { float x, y; };

static Point2D projectKB(float x, float y, float z, float fx, float fy, float cx, float cy, float k1, float k2, float k3, float k4) {
    float r_xy = sqrtf(x * x + y * y);
    if (r_xy < 1e-6f) return { cx, cy };
    float theta = atan2f(r_xy, z);
    float t2 = theta * theta;
    float t4 = t2 * t2;
    float t6 = t4 * t2;
    float t8 = t4 * t4;
    float r_dist = theta * (1.0f + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8);
    float scale = r_dist / r_xy;
    return { cx + fx * x * scale, cy + fy * y * scale };
}

static inline float sampleBilinear(const uint8_t *img, float px, float py) {
    if (px < 0.0f || px >= RAW_W - 1 || py < 0.0f || py >= RAW_H - 1) return 0.0f;
    int x0 = (int)px;
    int y0 = (int)py;
    float wx = px - (float)x0;
    float wy = py - (float)y0;
    float p00 = img[y0 * RAW_W + x0];
    float p10 = img[y0 * RAW_W + x0 + 1];
    float p01 = img[(y0 + 1) * RAW_W + x0];
    float p11 = img[(y0 + 1) * RAW_W + x0 + 1];
    return (p00 * (1.0f - wx) + p10 * wx) * (1.0f - wy) + (p01 * (1.0f - wx) + p11 * wx) * wy;
}

// Global thread-safe point cloud storage
std::vector<Point3D> g_worldPoints;
std::mutex g_pointsMutex;
std::atomic<bool> g_hasNewPoints(false);

// Simple 5x7 font
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
    for (int i=0; i<10; i++) setG('0'+i, 0x3E, 0x51, 0x49, 0x45, 0x3E);
    setG('0', 0x3E, 0x51, 0x49, 0x45, 0x3E); setG('1', 0x00, 0x42, 0x7F, 0x40, 0x00);
    setG('2', 0x42, 0x61, 0x51, 0x49, 0x46); setG('3', 0x21, 0x41, 0x45, 0x4B, 0x31);
    setG('4', 0x18, 0x14, 0x12, 0x7F, 0x10); setG('5', 0x27, 0x45, 0x45, 0x45, 0x39);
    setG('6', 0x3C, 0x4A, 0x49, 0x49, 0x30); setG('7', 0x01, 0x71, 0x09, 0x05, 0x03);
    setG('8', 0x36, 0x49, 0x49, 0x49, 0x36); setG('9', 0x06, 0x49, 0x49, 0x29, 0x1E);
    setG('A', 0x7C, 0x12, 0x11, 0x12, 0x7C); setG('B', 0x7F, 0x49, 0x49, 0x49, 0x36);
    setG('C', 0x3E, 0x41, 0x41, 0x41, 0x22); setG('D', 0x7F, 0x41, 0x41, 0x22, 0x1C);
    setG('E', 0x7F, 0x49, 0x49, 0x49, 0x41); setG('F', 0x7F, 0x09, 0x09, 0x09, 0x01);
    setG('G', 0x3E, 0x41, 0x49, 0x49, 0x7A); setG('H', 0x7F, 0x08, 0x08, 0x08, 0x7F);
    setG('I', 0x00, 0x41, 0x7F, 0x41, 0x00); setG('L', 0x7F, 0x40, 0x40, 0x40, 0x40);
    setG('M', 0x7F, 0x02, 0x0C, 0x02, 0x7F); setG('N', 0x7F, 0x04, 0x08, 0x10, 0x7F);
    setG('O', 0x3E, 0x41, 0x41, 0x41, 0x3E); setG('P', 0x7F, 0x09, 0x09, 0x09, 0x06);
    setG('R', 0x7F, 0x09, 0x19, 0x29, 0x46); setG('S', 0x46, 0x49, 0x49, 0x49, 0x31);
    setG('T', 0x01, 0x01, 0x7F, 0x01, 0x01); setG('U', 0x3F, 0x40, 0x40, 0x40, 0x3F);
    setG('V', 0x07, 0x38, 0x40, 0x38, 0x07); setG('W', 0x7F, 0x20, 0x18, 0x20, 0x7F);
    setG('X', 0x63, 0x14, 0x08, 0x14, 0x63); setG('Y', 0x07, 0x08, 0x70, 0x08, 0x07);
    g_fontInitialized = true;
}

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
        curX += 7;
    }
}

// --- Background Depth Thread ---
void depthWorkerThread(pid_t compositorPid, int rightFd, int leftFd, vr::IVRSystem *vrSys) {
    std::cout << "[DepthWorker] Starting depth worker for compositor PID=" << compositorPid << std::endl;

    int pfd = syscall(__NR_pidfd_open, compositorPid, 0);
    if (pfd < 0) {
        perror("[DepthWorker] pidfd_open failed");
        return;
    }

    int localRightFd = syscall(__NR_pidfd_getfd, pfd, rightFd, 0);
    int localLeftFd = syscall(__NR_pidfd_getfd, pfd, leftFd, 0);
    if (localRightFd < 0 || localLeftFd < 0) {
        perror("[DepthWorker] pidfd_getfd failed");
        close(pfd);
        return;
    }

    size_t dmaSize = 9564160;
    uint8_t *rightPtr = (uint8_t*)mmap(NULL, dmaSize, PROT_READ, MAP_SHARED, localRightFd, 0);
    uint8_t *leftPtr  = (uint8_t*)mmap(NULL, dmaSize, PROT_READ, MAP_SHARED, localLeftFd, 0);
    if (rightPtr == MAP_FAILED || leftPtr == MAP_FAILED) {
        perror("[DepthWorker] mmap failed");
        close(localRightFd); close(localLeftFd); close(pfd);
        return;
    }

    // Precompute Rectification Lookups
    std::vector<Point2D> lutL(OUT_W * OUT_H);
    std::vector<Point2D> lutR(OUT_W * OUT_H);
    for (int y = 0; y < OUT_H; y++) {
        for (int x = 0; x < OUT_W; x++) {
            float rx = ((float)x - CX_RECT) / F_RECT;
            float ry = ((float)y - CY_RECT) / F_RECT;
            lutL[y * OUT_W + x] = projectKB(rx, ry, 1.0f, FX_L, FY_L, CX_L, CY_L, K1_L, K2_L, K3_L, K4_L);
            lutR[y * OUT_W + x] = projectKB(-rx, -ry, 1.0f, FX_R, FY_R, CX_R, CY_R, K1_R, K2_R, K3_R, K4_R);
        }
    }

    std::vector<uint8_t> rectL(OUT_W * OUT_H);
    std::vector<uint8_t> rectR(OUT_W * OUT_H);
    std::vector<uint64_t> censusL(OUT_W * OUT_H, 0);
    std::vector<uint64_t> censusR(OUT_W * OUT_H, 0);
    std::vector<float> depthMap(OUT_W * OUT_H, 0.0f);

    while (g_running) {
        // 1. Grab current HMD Pose at frame capture time
        vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
        vrSys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0.0f, poses, vr::k_unMaxTrackedDeviceCount);

        vr::HmdMatrix34_t hmdMat;
        bool poseValid = poses[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid;
        if (poseValid) {
            hmdMat = poses[vr::k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking;
        }

        // 2. Rectify Left and Right
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < OUT_H; y++) {
            for (int x = 0; x < OUT_W; x++) {
                int idx = y * OUT_W + x;
                rectL[idx] = (uint8_t)sampleBilinear(leftPtr, lutL[idx].x, lutL[idx].y);
                rectR[idx] = (uint8_t)sampleBilinear(rightPtr, lutR[idx].x, lutR[idx].y);
            }
        }

        // 3. 7x7 Census Transform
        #pragma omp parallel for collapse(2)
        for (int y = 3; y < OUT_H - 3; y++) {
            for (int x = 3; x < OUT_W - 3; x++) {
                uint8_t cValL = rectL[y * OUT_W + x];
                uint8_t cValR = rectR[y * OUT_W + x];
                uint64_t codeL = 0, codeR = 0;
                int bit = 0;
                for (int dy = -3; dy <= 3; dy++) {
                    for (int dx = -3; dx <= 3; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        if (bit < 64) {
                            if (rectL[(y + dy) * OUT_W + (x + dx)] > cValL) codeL |= (1ULL << bit);
                            if (rectR[(y + dy) * OUT_W + (x + dx)] > cValR) codeR |= (1ULL << bit);
                            bit++;
                        }
                    }
                }
                censusL[y * OUT_W + x] = codeL;
                censusR[y * OUT_W + x] = codeR;
            }
        }

        // 4. Stereo Matching
        #pragma omp parallel for
        for (int y = 3; y < OUT_H - 3; y++) {
            for (int x = 3 + MAX_DISP; x < OUT_W - 3; x++) {
                uint64_t codeL = censusL[y * OUT_W + x];
                int bestDisp = 0;
                int bestCost = 9999;
                for (int d = 1; d < MAX_DISP; d++) {
                    uint64_t codeR = censusR[y * OUT_W + (x - d)];
                    int cost = __builtin_popcountll(codeL ^ codeR);
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestDisp = d;
                    }
                }
                if (bestCost < 16 && bestDisp > 1) {
                    depthMap[y * OUT_W + x] = F_B / (float)bestDisp;
                } else {
                    depthMap[y * OUT_W + x] = 0.0f;
                }
            }
        }

        // 5. Convert Valid Depth Points into World Point Cloud
        std::vector<Point3D> localCloud;
        localCloud.reserve(OUT_W * OUT_H / 4);

        // Step by 2 in X and Y for balanced density and high frame rate (~80,000 points)
        for (int y = 4; y < OUT_H - 4; y += 2) {
            for (int x = 4 + MAX_DISP; x < OUT_W - 4; x += 2) {
                float Z = depthMap[y * OUT_W + x];
                if (Z >= 0.25f && Z <= 5.0f) {
                    // Left Camera Space
                    float X_c = ((float)x - CX_RECT) * Z / F_RECT;
                    float Y_c = -((float)y - CY_RECT) * Z / F_RECT;
                    float Z_c = -Z; // OpenVR convention: -Z is forward

                    // Transform to HMD Head Space
                    // Camera is located at (-0.0528, -0.0186, -0.0368) relative to head center
                    float H_x = X_c - 0.0528f;
                    float H_y = Y_c - 0.0186f;
                    float H_z = Z_c - 0.0368f;

                    // Color based on depth: Close = Green, Mid = Cyan, Far = Magenta
                    uint8_t r = 0, g = 255, b = 255;
                    float norm = (Z - 0.3f) / 3.0f;
                    if (norm < 0.0f) norm = 0.0f;
                    if (norm > 1.0f) norm = 1.0f;
                    if (norm < 0.5f) {
                        r = (uint8_t)(norm * 2.0f * 255.0f);
                        g = 255;
                        b = (uint8_t)((1.0f - norm * 2.0f) * 255.0f);
                    } else {
                        r = 255;
                        g = (uint8_t)((1.0f - (norm - 0.5f) * 2.0f) * 255.0f);
                        b = (uint8_t)((norm - 0.5f) * 2.0f * 255.0f);
                    }

                    if (poseValid) {
                        // Transform to Absolute Tracking (World) Space: T_world = HmdPose * P_hmd
                        const auto &m = hmdMat.m;
                        float W_x = m[0][0] * H_x + m[0][1] * H_y + m[0][2] * H_z + m[0][3];
                        float W_y = m[1][0] * H_x + m[1][1] * H_y + m[1][2] * H_z + m[1][3];
                        float W_z = m[2][0] * H_x + m[2][1] * H_y + m[2][2] * H_z + m[2][3];
                        localCloud.push_back({ W_x, W_y, W_z, r, g, b });
                    } else {
                        // Fallback to head space
                        localCloud.push_back({ H_x, H_y, H_z, r, g, b });
                    }
                }
            }
        }

        // Commit new point cloud to render thread
        size_t cloudSize = localCloud.size();
        {
            std::lock_guard<std::mutex> lock(g_pointsMutex);
            g_worldPoints = std::move(localCloud);
            g_hasNewPoints.store(true);
        }
        std::cout << "[DepthWorker] Updated 3D point cloud with " << cloudSize << " metric points." << std::endl;

        usleep(20000); // ~40-50 FPS depth loop
    }

    munmap(rightPtr, dmaSize);
    munmap(leftPtr, dmaSize);
    close(localRightFd);
    close(localLeftFd);
    close(pfd);
    std::cout << "[DepthWorker] Exited cleanly." << std::endl;
}

int main() {
    initFont();
    std::signal(SIGINT, sigHandler);
    std::signal(SIGTERM, sigHandler);

    std::cout << "========================================================\n"
              << "       STEAM FRAME REAL-TIME SPATIAL ROOM MESH         \n"
              << "========================================================\n";

    // 1. Initialize OpenVR Overlay
    vr::EVRInitError err = vr::VRInitError_None;
    vr::IVRSystem *vrSys = vr::VR_Init(&err, vr::VRApplication_Overlay);
    if (err != vr::VRInitError_None) {
        std::cerr << "VR_Init failed: " << vr::VR_GetVRInitErrorAsEnglishDescription(err) << std::endl;
        return 1;
    }

    vr::IVROverlay *overlay = vr::VROverlay();
    vr::VROverlayHandle_t ovHandle = vr::k_ulOverlayHandleInvalid;
    vr::EVROverlayError ovErr = overlay->CreateOverlay("frame.spatial.room_mesh", "Spatial Room Mesh", &ovHandle);
    if (ovErr != vr::VROverlayError_None) {
        std::cerr << "CreateOverlay failed: " << ovErr << std::endl;
        vr::VR_Shutdown();
        return 1;
    }

    // Enable Side-by-Side Parallel 3D Stereo!
    overlay->SetOverlayFlag(ovHandle, vr::VROverlayFlags_SideBySide_Parallel, true);

    // Place overlay quad at 1.5m in front of HMD covering field of view
    vr::HmdMatrix34_t hmdTransform = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, -1.5f
    };
    overlay->SetOverlayTransformTrackedDeviceRelative(ovHandle, vr::k_unTrackedDeviceIndex_Hmd, &hmdTransform);
    overlay->SetOverlayWidthInMeters(ovHandle, 2.5f);
    overlay->SetOverlayAlpha(ovHandle, 0.95f);
    overlay->ShowOverlay(ovHandle);

    // Side-by-Side Framebuffer dimensions: 1280 x 720 (640 per eye)
    const int EYE_W = 640;
    const int EYE_H = 720;
    const int TOTAL_W = EYE_W * 2;
    const int TOTAL_H = EYE_H;
    std::vector<uint32_t> sbsBuffer(TOTAL_W * TOTAL_H, 0);

    // Get Eye Projections
    vr::HmdMatrix44_t projL = vrSys->GetProjectionMatrix(vr::Eye_Left, 0.1f, 30.0f);
    vr::HmdMatrix44_t projR = vrSys->GetProjectionMatrix(vr::Eye_Right, 0.1f, 30.0f);
    float projScaleXL = projL.m[0][0] * (EYE_W / 2.0f);
    float projScaleYL = projL.m[1][1] * (EYE_H / 2.0f);
    float projCenterXL = (EYE_W / 2.0f) * (1.0f - projL.m[0][2]);
    float projCenterYL = (EYE_H / 2.0f) * (1.0f + projL.m[1][2]);

    float projScaleXR = projR.m[0][0] * (EYE_W / 2.0f);
    float projScaleYR = projR.m[1][1] * (EYE_H / 2.0f);
    float projCenterXR = (EYE_W / 2.0f) * (1.0f - projR.m[0][2]);
    float projCenterYR = (EYE_H / 2.0f) * (1.0f + projR.m[1][2]);

    // Eye to Head transforms (IPD offset)
    vr::HmdMatrix34_t eyeToHeadL = vrSys->GetEyeToHeadTransform(vr::Eye_Left);
    vr::HmdMatrix34_t eyeToHeadR = vrSys->GetEyeToHeadTransform(vr::Eye_Right);

    // 2. Discover Compositor PID and Camera FDs
    FILE *pgrep = popen("pgrep vrcompositor", "r");
    pid_t compPid = 0;
    if (pgrep) {
        if (fscanf(pgrep, "%d", &compPid) != 1) compPid = 0;
        pclose(pgrep);
    }
    if (compPid == 0) {
        std::cerr << "vrcompositor process not found!" << std::endl;
        vr::VR_Shutdown();
        return 1;
    }
    std::cout << "Found vrcompositor PID: " << compPid << std::endl;

    // FDs: Right=216, Left=223
    int rightFd = 216;
    int leftFd = 223;

    // Start Background Depth Worker Thread
    std::thread worker(depthWorkerThread, compPid, rightFd, leftFd, vrSys);

    // 3. Real-Time Stereoscopic Render Loop (60-90 FPS)
    std::vector<Point3D> renderCloud;
    auto lastFpsTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float liveFps = 0.0f;

    std::cout << "[SpatialMesh] Main render loop active. Hologram live in headset!" << std::endl;

    while (g_running) {
        vr::VREvent_t vrEvent;
        while (overlay->PollNextOverlayEvent(ovHandle, &vrEvent, sizeof(vrEvent))) {
            if (vrEvent.eventType == vr::VREvent_Quit) {
                g_running = false;
                break;
            }
        }

        // Copy latest point cloud if available
        if (g_hasNewPoints.load()) {
            std::lock_guard<std::mutex> lock(g_pointsMutex);
            renderCloud = g_worldPoints;
            g_hasNewPoints.store(false);
        }

        // Get live HMD pose
        vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
        vrSys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0.0f, poses, vr::k_unMaxTrackedDeviceCount);

        if (!poses[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid) {
            usleep(10000);
            continue;
        }

        const auto &hmdMat = poses[vr::k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking.m;
        // World to HMD rotation & position
        float R[3][3] = {
            {hmdMat[0][0], hmdMat[1][0], hmdMat[2][0]},
            {hmdMat[0][1], hmdMat[1][1], hmdMat[2][1]},
            {hmdMat[0][2], hmdMat[1][2], hmdMat[2][2]}
        };
        float HmdPos[3] = {hmdMat[0][3], hmdMat[1][3], hmdMat[2][3]};

        // Clear side-by-side buffer (100% transparent alpha: 0x00000000)
        std::fill(sbsBuffer.begin(), sbsBuffer.end(), 0x00000000);

        // Render to Both Eyes (Left: [0, EYE_W), Right: [EYE_W, TOTAL_W))
        int visibleCountL = 0;
        int visibleCountR = 0;

        for (const auto &pt : renderCloud) {
            float dx = pt.x - HmdPos[0];
            float dy = pt.y - HmdPos[1];
            float dz = pt.z - HmdPos[2];

            // In HMD coordinates
            float hx = R[0][0] * dx + R[0][1] * dy + R[0][2] * dz;
            float hy = R[1][0] * dx + R[1][1] * dy + R[1][2] * dz;
            float hz = R[2][0] * dx + R[2][1] * dy + R[2][2] * dz;

            if (hz < -0.15f) {
                float invZ = -1.0f / hz;
                uint32_t color = 0xFF000000 | (pt.b << 16) | (pt.g << 8) | pt.r;

                // Left Eye (EyeToHead offset: X is around -0.032m)
                float hx_L = hx - eyeToHeadL.m[0][3];
                int sx_L = (int)(projCenterXL + hx_L * projScaleXL * invZ);
                int sy_L = (int)(projCenterYL - hy * projScaleYL * invZ);

                if (sx_L >= 1 && sx_L < EYE_W - 1 && sy_L >= 1 && sy_L < EYE_H - 1) {
                    sbsBuffer[sy_L * TOTAL_W + sx_L] = color;
                    sbsBuffer[sy_L * TOTAL_W + (sx_L + 1)] = color; // 2px splat
                    visibleCountL++;
                }

                // Right Eye (EyeToHead offset: X is around +0.032m)
                float hx_R = hx - eyeToHeadR.m[0][3];
                int sx_R = (int)(projCenterXR + hx_R * projScaleXR * invZ) + EYE_W; // offset by EYE_W
                int sy_R = (int)(projCenterYR - hy * projScaleYR * invZ);

                if (sx_R >= EYE_W + 1 && sx_R < TOTAL_W - 1 && sy_R >= 1 && sy_R < EYE_H - 1) {
                    sbsBuffer[sy_R * TOTAL_W + sx_R] = color;
                    sbsBuffer[sy_R * TOTAL_W + (sx_R + 1)] = color; // 2px splat
                    visibleCountR++;
                }
            }
        }

        // Telemetry HUD overlay on both eyes
        frameCount++;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - lastFpsTime).count();
        if (elapsed >= 0.5) {
            liveFps = (float)(frameCount / elapsed);
            frameCount = 0;
            lastFpsTime = now;
            std::cout << "[Render] FPS: " << liveFps << " | Total 3D Pts: " << renderCloud.size() 
                      << " | Visible in FOV: " << visibleCountL << std::endl;
        }

        char hudText[128];
        snprintf(hudText, sizeof(hudText), "FRAME SPATIAL MESH | FPS: %0.1f | PTS: %zu | VIS: %d",
                 liveFps, renderCloud.size(), visibleCountL);

        // Draw HUD to Left Eye and Right Eye
        drawText(sbsBuffer.data(), TOTAL_W, TOTAL_H, 20, 20, hudText, 0xFF00FF00); // Neon Green
        drawText(sbsBuffer.data(), TOTAL_W, TOTAL_H, 20 + EYE_W, 20, hudText, 0xFF00FF00);

        // Push SBS Frame to OpenVR Compositor
        overlay->SetOverlayRaw(ovHandle, sbsBuffer.data(), TOTAL_W, TOTAL_H, 4);

        usleep(11000); // ~90 Hz display refresh
    }

    std::cout << "[SpatialMesh] Shutting down..." << std::endl;
    worker.join();

    overlay->HideOverlay(ovHandle);
    overlay->DestroyOverlay(ovHandle);
    vr::VR_Shutdown();
    std::cout << "[SpatialMesh] Clean exit." << std::endl;
    return 0;
}
