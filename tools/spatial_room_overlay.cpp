// spatial_room_overlay.cpp — Real room mesh as world-locked AR overlays
//
// Loads the SLAM room boundary from real_room_mesh.obj and creates
// world-locked overlay quads for each wall segment. Also tries to
// auto-calibrate SLAM-to-OpenVR coordinate offset using HMD position.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <csignal>
#include <openvr.h>

static volatile bool g_running = true;
void sigHandler(int) { g_running = false; }

struct Vec3 { float x, y, z; };

int main() {
    std::signal(SIGINT, sigHandler);
    std::signal(SIGTERM, sigHandler);

    vr::EVRInitError err;
    vr::IVRSystem *sys = vr::VR_Init(&err, vr::VRApplication_Overlay);
    if (err != vr::VRInitError_None) {
        fprintf(stderr, "VR_Init failed: %s\n", vr::VR_GetVRInitErrorAsEnglishDescription(err));
        return 1;
    }

    vr::IVROverlay *ovl = vr::VROverlay();
    std::vector<vr::VROverlayHandle_t> handles;

    // ═══ Load SLAM room boundary from OBJ ═══
    std::vector<Vec3> allVerts;
    {
        std::ifstream obj("/home/steamos/test_openvr/real_room_mesh.obj");
        std::string line;
        while (std::getline(obj, line)) {
            if (line.rfind("v ", 0) == 0) {
                float x, y, z;
                sscanf(line.c_str() + 2, "%f %f %f", &x, &y, &z);
                allVerts.push_back({x, y, z});
            }
        }
    }
    printf("Loaded %zu vertices from OBJ\n", allVerts.size());

    // The OBJ has 40 vertices: first 20 are floor (Y=-0.4), next 20 are ceiling (Y=1.8)
    int nBoundary = (int)allVerts.size() / 2;
    if (nBoundary < 3) {
        fprintf(stderr, "Not enough boundary vertices\n");
        vr::VR_Shutdown();
        return 1;
    }

    // ═══ Compute SLAM→OpenVR Y offset ═══
    // SLAM floor is at Y=-0.4, OpenVR floor is at Y=0
    // Simple offset: add 0.4 to all SLAM Y coordinates
    float slamFloorY = allVerts[0].y; // -0.4
    float openvrFloorY = 0.0f;
    float yOffset = openvrFloorY - slamFloorY; // +0.4

    // For XZ: compare HMD position. The SLAM origin might differ from OpenVR origin.
    // We'll try raw first (no XZ offset) and see if it aligns.
    float xOffset = 0.0f, zOffset = 0.0f;

    // Get current HMD position for reference
    vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
    sys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, poses, vr::k_unMaxTrackedDeviceCount);
    float hmdX = 0, hmdY = 0, hmdZ = 0;
    if (poses[0].bPoseIsValid) {
        auto &m = poses[0].mDeviceToAbsoluteTracking.m;
        hmdX = m[0][3]; hmdY = m[1][3]; hmdZ = m[2][3];
    }

    printf("HMD (OpenVR Standing): (%.3f, %.3f, %.3f)\n", hmdX, hmdY, hmdZ);
    printf("SLAM floor Y: %.3f → offset: +%.3f\n", slamFloorY, yOffset);
    printf("Boundary vertices: %d\n", nBoundary);

    // Print boundary polygon center for comparison
    float bcx = 0, bcz = 0;
    for (int i = 0; i < nBoundary; i++) {
        bcx += allVerts[i].x;
        bcz += allVerts[i].z;
    }
    bcx /= nBoundary; bcz /= nBoundary;
    printf("SLAM boundary center (XZ): (%.3f, %.3f)\n", bcx, bcz);

    // ═══ Wall texture — neon cyan grid ═══
    const int TW = 64, TH = 64;
    std::vector<uint32_t> wallTex(TW * TH, 0x18FFFF00); // very faint cyan
    // Bright border
    for (int x = 0; x < TW; x++) { wallTex[x] = 0xB0FFFF00; wallTex[(TH-1)*TW+x] = 0xB0FFFF00; }
    for (int y = 0; y < TH; y++) { wallTex[y*TW] = 0xB0FFFF00; wallTex[y*TW+TW-1] = 0xB0FFFF00; }
    // Grid
    for (int g = 8; g < TW; g += 8) for (int y = 0; y < TH; y++) wallTex[y*TW+g] = 0x40FFFF00;
    for (int g = 8; g < TH; g += 8) for (int x = 0; x < TW; x++) wallTex[g*TW+x] = 0x40FFFF00;

    // ═══ Create wall overlays from room boundary ═══
    float ceilY = allVerts[nBoundary].y + yOffset; // ceiling in OpenVR coords
    float floorY = allVerts[0].y + yOffset;        // should be ~0
    float wallHeight = ceilY - floorY;

    printf("Wall height: %.2fm (floor=%.2f, ceiling=%.2f in OpenVR)\n", wallHeight, floorY, ceilY);

    for (int i = 0; i < nBoundary; i++) {
        int j = (i + 1) % nBoundary;

        // Floor vertices i and j, transformed to OpenVR coords
        float x0 = allVerts[i].x + xOffset, z0 = allVerts[i].z + zOffset;
        float x1 = allVerts[j].x + xOffset, z1 = allVerts[j].z + zOffset;

        float dx = x1 - x0, dz = z1 - z0;
        float w = sqrtf(dx * dx + dz * dz);
        if (w < 0.01f) continue;

        // Wall center
        float cx = (x0 + x1) * 0.5f;
        float cy = floorY + wallHeight * 0.5f;
        float cz = (z0 + z1) * 0.5f;

        // Right = along wall bottom edge, Up = Y, Normal = inward (90° CW in XZ)
        float rx = dx / w, rz = dz / w;
        float nx = -rz, nz = rx;

        vr::HmdMatrix34_t xf = {};
        xf.m[0][0] = rx;  xf.m[0][1] = 0; xf.m[0][2] = nx; xf.m[0][3] = cx;
        xf.m[1][0] = 0;   xf.m[1][1] = 1; xf.m[1][2] = 0;  xf.m[1][3] = cy;
        xf.m[2][0] = rz;  xf.m[2][1] = 0; xf.m[2][2] = nz; xf.m[2][3] = cz;

        char name[64];
        snprintf(name, 64, "frame.rwall.%d", i);
        vr::VROverlayHandle_t oh;
        if (ovl->CreateOverlay(name, "Room Wall", &oh) == vr::VROverlayError_None) {
            ovl->SetOverlayTransformAbsolute(oh, vr::TrackingUniverseStanding, &xf);
            ovl->SetOverlayWidthInMeters(oh, w);
            ovl->SetOverlayTexelAspect(oh, w / wallHeight);
            ovl->SetOverlayAlpha(oh, 0.55f);
            ovl->SetOverlayRaw(oh, wallTex.data(), TW, TH, 4);
            ovl->ShowOverlay(oh);
            handles.push_back(oh);
        }
    }
    printf("Created %zu room wall overlays\n", handles.size());

    // ═══ Floor overlay ═══
    {
        // Compute bounding box of boundary for floor size
        float minX = 1e9, maxX = -1e9, minZ = 1e9, maxZ = -1e9;
        for (int i = 0; i < nBoundary; i++) {
            float x = allVerts[i].x + xOffset, z = allVerts[i].z + zOffset;
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        }
        float fw = maxX - minX, fh = maxZ - minZ;
        float fcx = (minX + maxX) * 0.5f, fcz = (minZ + maxZ) * 0.5f;

        const int FW = 128, FH = 128;
        std::vector<uint32_t> ftex(FW * FH, 0x1000FF00);
        for (int y = 0; y < FH; y++)
            for (int x = 0; x < FW; x++)
                if (x % 16 == 0 || y % 16 == 0) ftex[y*FW+x] = 0x3000FF00;
        for (int x = 0; x < FW; x++) { ftex[x] = 0x6000FF00; ftex[(FH-1)*FW+x] = 0x6000FF00; }
        for (int y = 0; y < FH; y++) { ftex[y*FW] = 0x6000FF00; ftex[y*FW+FW-1] = 0x6000FF00; }

        vr::HmdMatrix34_t fx = {};
        fx.m[0][0] = 1;
        fx.m[1][2] = 1;
        fx.m[2][1] = -1;
        fx.m[0][3] = fcx;
        fx.m[1][3] = floorY + 0.005f;
        fx.m[2][3] = fcz;

        vr::VROverlayHandle_t floorOH;
        if (ovl->CreateOverlay("frame.rfloor", "Room Floor", &floorOH) == vr::VROverlayError_None) {
            ovl->SetOverlayTransformAbsolute(floorOH, vr::TrackingUniverseStanding, &fx);
            ovl->SetOverlayWidthInMeters(floorOH, fw);
            ovl->SetOverlayTexelAspect(floorOH, fw / fh);
            ovl->SetOverlayAlpha(floorOH, 0.3f);
            ovl->SetOverlayRaw(floorOH, ftex.data(), FW, FH, 4);
            ovl->ShowOverlay(floorOH);
            handles.push_back(floorOH);
            printf("Floor: %.2f x %.2f m centered at (%.2f, %.2f)\n", fw, fh, fcx, fcz);
        }
    }

    // ═══ Ceiling overlay ═══
    {
        float minX = 1e9, maxX = -1e9, minZ = 1e9, maxZ = -1e9;
        for (int i = 0; i < nBoundary; i++) {
            float x = allVerts[i].x + xOffset, z = allVerts[i].z + zOffset;
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        }
        float fw = maxX - minX, fh = maxZ - minZ;
        float fcx = (minX + maxX) * 0.5f, fcz = (minZ + maxZ) * 0.5f;

        const int CW = 128, CH = 128;
        std::vector<uint32_t> ctex(CW * CH, 0x0800FFFF); // very faint yellow
        for (int y = 0; y < CH; y++)
            for (int x = 0; x < CW; x++)
                if (x % 16 == 0 || y % 16 == 0) ctex[y*CW+x] = 0x2000FFFF;

        vr::HmdMatrix34_t cx = {};
        cx.m[0][0] = 1;
        cx.m[1][2] = -1;  // normal faces DOWN
        cx.m[2][1] = 1;
        cx.m[0][3] = fcx;
        cx.m[1][3] = ceilY - 0.005f;
        cx.m[2][3] = fcz;

        vr::VROverlayHandle_t ch;
        if (ovl->CreateOverlay("frame.rceil", "Room Ceiling", &ch) == vr::VROverlayError_None) {
            ovl->SetOverlayTransformAbsolute(ch, vr::TrackingUniverseStanding, &cx);
            ovl->SetOverlayWidthInMeters(ch, fw);
            ovl->SetOverlayTexelAspect(ch, fw / fh);
            ovl->SetOverlayAlpha(ch, 0.2f);
            ovl->SetOverlayRaw(ch, ctex.data(), CW, CH, 4);
            ovl->ShowOverlay(ch);
            handles.push_back(ch);
        }
    }

    // ═══ Red origin marker for calibration reference ═══
    {
        const int MW = 32, MH = 32;
        std::vector<uint32_t> mtex(MW * MH, 0xFF0000FF);
        for (int x = 0; x < MW; x++) { mtex[x] = 0xFFFFFFFF; mtex[(MH-1)*MW+x] = 0xFFFFFFFF; }
        for (int y = 0; y < MH; y++) { mtex[y*MW] = 0xFFFFFFFF; mtex[y*MW+MW-1] = 0xFFFFFFFF; }

        vr::HmdMatrix34_t mx = {};
        mx.m[0][0] = 1; mx.m[1][1] = 1; mx.m[2][2] = 1;
        mx.m[1][3] = 1.0f;

        vr::VROverlayHandle_t mh;
        if (ovl->CreateOverlay("frame.origin.mk", "Origin", &mh) == vr::VROverlayError_None) {
            ovl->SetOverlayTransformAbsolute(mh, vr::TrackingUniverseStanding, &mx);
            ovl->SetOverlayWidthInMeters(mh, 0.15f);
            ovl->SetOverlayAlpha(mh, 0.9f);
            ovl->SetOverlayRaw(mh, mtex.data(), MW, MH, 4);
            ovl->ShowOverlay(mh);
            handles.push_back(mh);
        }
    }

    printf("═══ %zu total overlays. Running... ═══\n", handles.size());
    fflush(stdout);

    while (g_running) {
        vr::VREvent_t ev;
        while (sys->PollNextEvent(&ev, sizeof(ev))) {
            if (ev.eventType == vr::VREvent_Quit) g_running = false;
        }
        usleep(100000);
    }

    for (auto h : handles) ovl->DestroyOverlay(h);
    vr::VR_Shutdown();
    printf("Clean exit.\n");
    return 0;
}
