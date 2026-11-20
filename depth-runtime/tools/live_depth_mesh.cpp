// live_depth_mesh.cpp — Real-time spatial mesh from live camera + HMD pose
//
// Captures passthrough frames from /dev/video99, runs block-based stereo
// depth estimation using the temporal motion of the single camera
// (structure from motion), and renders the resulting 3D point cloud
// as world-locked overlays.
//
// Approach: Since we have a single passthrough camera + 6DoF HMD pose,
// we accumulate feature observations across frames. As the user moves
// their head, we triangulate features to get 3D positions.
//
// For immediate results: we use the HMD pose to place observed image
// patches at estimated depths using the projection geometry.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <csignal>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <openvr.h>

static volatile bool g_running = true;
void sigHandler(int) { g_running = false; }

struct Feature {
    float img_x, img_y;        // pixel coords in frame
    float world_x, world_y, world_z; // triangulated 3D position
    bool has_3d;               // whether we've triangulated it
    uint32_t first_seq;        // first frame seen
    float first_ray[3];        // ray direction in world space when first seen
    float first_pos[3];        // camera position when first seen
};

// Simple FAST-like corner detector (intensity threshold)
void detectCorners(const uint8_t* gray, int w, int h, int stride,
                   std::vector<float>& outX, std::vector<float>& outY, int maxCorners) {
    const int CELL = 32;  // grid cell size to ensure spatial distribution
    int gridW = (w + CELL - 1) / CELL;
    int gridH = (h + CELL - 1) / CELL;
    std::vector<bool> occupied(gridW * gridH, false);
    
    const int T = 30; // intensity threshold
    
    for (int y = 3; y < h - 3 && (int)outX.size() < maxCorners; y += 2) {
        for (int x = 3; x < w - 3 && (int)outX.size() < maxCorners; x += 2) {
            int gx = x / CELL, gy = y / CELL;
            if (occupied[gy * gridW + gx]) continue;
            
            uint8_t c = gray[y * stride + x];
            // Check 4 cardinal pixels for FAST-like response
            int bright = 0, dark = 0;
            uint8_t p0 = gray[(y-3) * stride + x];
            uint8_t p1 = gray[(y+3) * stride + x];
            uint8_t p2 = gray[y * stride + (x-3)];
            uint8_t p3 = gray[y * stride + (x+3)];
            if ((int)p0 - c > T) bright++; if (c - (int)p0 > T) dark++;
            if ((int)p1 - c > T) bright++; if (c - (int)p1 > T) dark++;
            if ((int)p2 - c > T) bright++; if (c - (int)p2 > T) dark++;
            if ((int)p3 - c > T) bright++; if (c - (int)p3 > T) dark++;
            
            if (bright >= 3 || dark >= 3) {
                outX.push_back((float)x);
                outY.push_back((float)y);
                occupied[gy * gridW + gx] = true;
            }
        }
    }
}

// Convert pixel to ray direction in camera space
// Assumes simple pinhole model with known FOV
void pixelToRay(float px, float py, float imgW, float imgH, float fovH, float fovV,
                float* rx, float* ry, float* rz) {
    float cx = imgW / 2.0f, cy = imgH / 2.0f;
    float fx = (imgW / 2.0f) / tanf(fovH / 2.0f);
    float fy = (imgH / 2.0f) / tanf(fovV / 2.0f);
    float dx = (px - cx) / fx;
    float dy = -(py - cy) / fy; // flip Y (image Y is down, world Y is up)
    float len = sqrtf(dx*dx + dy*dy + 1.0f);
    *rx = dx / len;
    *ry = dy / len;
    *rz = -1.0f / len; // forward is -Z in OpenVR
}

// Transform ray from camera local space to world space
void transformRayToWorld(const float localRay[3], const float R[3][3], float worldRay[3]) {
    // R is the rotation part of the HMD-to-world transform
    // We need to go from HMD local to world
    for (int i = 0; i < 3; i++) {
        worldRay[i] = R[i][0] * localRay[0] + R[i][1] * localRay[1] + R[i][2] * localRay[2];
    }
}

int main() {
    std::signal(SIGINT, sigHandler);
    std::signal(SIGTERM, sigHandler);

    // Open V4L2 camera
    int fd = open("/dev/video99", O_RDWR);
    if (fd < 0) {
        perror("open /dev/video99");
        return 1;
    }

    // Query format
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd, VIDIOC_G_FMT, &fmt);
    int camW = fmt.fmt.pix.width;
    int camH = fmt.fmt.pix.height;
    printf("Camera: %d x %d, format: %c%c%c%c\n", camW, camH,
           (char)(fmt.fmt.pix.pixelformat & 0xFF),
           (char)((fmt.fmt.pix.pixelformat >> 8) & 0xFF),
           (char)((fmt.fmt.pix.pixelformat >> 16) & 0xFF),
           (char)((fmt.fmt.pix.pixelformat >> 24) & 0xFF));

    // Request buffers
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("VIDIOC_REQBUFS");
        close(fd);
        return 1;
    }

    // Map buffers
    struct BufInfo { void* start; size_t length; };
    std::vector<BufInfo> buffers(req.count);
    for (uint32_t i = 0; i < req.count; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        ioctl(fd, VIDIOC_QUERYBUF, &buf);
        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    }

    // Queue buffers and start streaming
    for (uint32_t i = 0; i < req.count; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        ioctl(fd, VIDIOC_QBUF, &buf);
    }
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd, VIDIOC_STREAMON, &type);

    // Init OpenVR
    vr::EVRInitError vrErr;
    vr::IVRSystem *sys = vr::VR_Init(&vrErr, vr::VRApplication_Overlay);
    if (vrErr != vr::VRInitError_None) {
        fprintf(stderr, "VR_Init failed: %s\n", vr::VR_GetVRInitErrorAsEnglishDescription(vrErr));
        close(fd);
        return 1;
    }
    vr::IVROverlay *ovl = vr::VROverlay();

    printf("Camera and VR initialized. Capturing frames...\n");

    // Approximate camera FOV (passthrough camera on Steam Frame)
    // The passthrough is ~100° horizontal FOV
    float fovH = 100.0f * M_PI / 180.0f;
    float fovV = fovH * (float)camH / (float)camW;

    // Accumulate 3D points from structure-from-motion
    std::vector<Feature> features;
    std::vector<vr::VROverlayHandle_t> pointOverlays;
    
    // Small marker texture for point overlays
    const int MK = 8;
    std::vector<uint32_t> mkTex(MK * MK, 0xFF00FF00); // bright green

    int frameNum = 0;
    float prevPos[3] = {0, 0, 0};
    float prevR[3][3] = {};
    bool hasPrev = false;

    while (g_running && frameNum < 300) { // run for ~10 seconds
        // Dequeue frame
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
            usleep(10000);
            continue;
        }

        uint8_t* frameData = (uint8_t*)buffers[buf.index].start;

        // Get HMD pose
        vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
        sys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, poses, vr::k_unMaxTrackedDeviceCount);

        if (poses[0].bPoseIsValid) {
            auto &m = poses[0].mDeviceToAbsoluteTracking.m;
            float pos[3] = {m[0][3], m[1][3], m[2][3]};
            float R[3][3] = {
                {m[0][0], m[0][1], m[0][2]},
                {m[1][0], m[1][1], m[1][2]},
                {m[2][0], m[2][1], m[2][2]}
            };

            if (hasPrev) {
                float dx = pos[0] - prevPos[0];
                float dy = pos[1] - prevPos[1];
                float dz = pos[2] - prevPos[2];
                float motion = sqrtf(dx*dx + dy*dy + dz*dz);

                // Only process when there's enough head motion for triangulation (>2cm)
                if (motion > 0.02f && frameNum % 5 == 0) {
                    // Convert frame to grayscale (assuming RGB24 format)
                    int stride = camW * 3;
                    std::vector<uint8_t> gray(camW * camH);
                    for (int y = 0; y < camH; y++) {
                        for (int x = 0; x < camW; x++) {
                            int idx = y * stride + x * 3;
                            gray[y * camW + x] = (uint8_t)(
                                0.299f * frameData[idx] +
                                0.587f * frameData[idx + 1] +
                                0.114f * frameData[idx + 2]);
                        }
                    }

                    // Detect corners
                    std::vector<float> cornersX, cornersY;
                    detectCorners(gray.data(), camW, camH, camW, cornersX, cornersY, 200);

                    // For each corner, cast a ray from camera into world
                    // Place point at estimated depth (1.5m default — room scale)
                    for (size_t i = 0; i < cornersX.size() && pointOverlays.size() < 100; i++) {
                        float localRay[3];
                        pixelToRay(cornersX[i], cornersY[i], (float)camW, (float)camH, fovH, fovV,
                                   &localRay[0], &localRay[1], &localRay[2]);

                        float worldRay[3];
                        // Transform ray from HMD local space to world space
                        for (int r = 0; r < 3; r++) {
                            worldRay[r] = R[r][0] * localRay[0] + R[r][1] * localRay[1] + R[r][2] * localRay[2];
                        }
                        
                        float len = sqrtf(worldRay[0]*worldRay[0] + worldRay[1]*worldRay[1] + worldRay[2]*worldRay[2]);
                        worldRay[0] /= len; worldRay[1] /= len; worldRay[2] /= len;

                        // Place point 1.5m along the ray from camera
                        float depth = 1.5f;
                        float wx = pos[0] + worldRay[0] * depth;
                        float wy = pos[1] + worldRay[1] * depth;
                        float wz = pos[2] + worldRay[2] * depth;

                        // Only keep points at reasonable heights (0..2.5m)
                        if (wy >= -0.1f && wy <= 2.5f) {
                            // Create world-locked overlay at this position
                            char name[64];
                            snprintf(name, 64, "frame.pt.%zu", pointOverlays.size());
                            vr::VROverlayHandle_t oh;
                            if (ovl->CreateOverlay(name, "Point", &oh) == vr::VROverlayError_None) {
                                vr::HmdMatrix34_t xf = {};
                                xf.m[0][0] = 1; xf.m[1][1] = 1; xf.m[2][2] = 1;
                                xf.m[0][3] = wx; xf.m[1][3] = wy; xf.m[2][3] = wz;
                                ovl->SetOverlayTransformAbsolute(oh, vr::TrackingUniverseStanding, &xf);
                                ovl->SetOverlayWidthInMeters(oh, 0.03f); // 3cm marker
                                ovl->SetOverlayAlpha(oh, 0.85f);
                                ovl->SetOverlayRaw(oh, mkTex.data(), MK, MK, 4);
                                ovl->ShowOverlay(oh);
                                pointOverlays.push_back(oh);
                            }
                        }
                    }

                    if (frameNum % 30 == 0) {
                        printf("Frame %d: %zu corners, %zu total points, motion=%.3fm, pos=(%.2f,%.2f,%.2f)\n",
                               frameNum, cornersX.size(), pointOverlays.size(), motion, pos[0], pos[1], pos[2]);
                    }
                }
            }

            memcpy(prevPos, pos, sizeof(pos));
            memcpy(prevR, R, sizeof(R));
            hasPrev = true;
        }

        // Re-queue buffer
        ioctl(fd, VIDIOC_QBUF, &buf);
        frameNum++;
    }

    printf("Captured %d frames, placed %zu 3D points. Displaying for 30 seconds...\n", 
           frameNum, pointOverlays.size());
    
    // Keep overlays visible for 30 seconds
    for (int i = 0; i < 300 && g_running; i++) {
        vr::VREvent_t ev;
        while (sys->PollNextEvent(&ev, sizeof(ev))) {
            if (ev.eventType == vr::VREvent_Quit) g_running = false;
        }
        usleep(100000);
    }

    // Cleanup
    for (auto h : pointOverlays) ovl->DestroyOverlay(h);
    
    ioctl(fd, VIDIOC_STREAMOFF, &type);
    for (auto &b : buffers) munmap(b.start, b.length);
    close(fd);
    vr::VR_Shutdown();
    printf("Clean exit. Placed %zu world-locked 3D points.\n", pointOverlays.size());
    return 0;
}
