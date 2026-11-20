// spatial_world_overlay.cpp — World-locked AR overlay using OpenVR chaperone data
// 
// Key insight: instead of projecting 3D points onto a head-locked 2D quad (which
// has no real depth), we place overlay quads AT the actual wall positions in 3D
// tracking space. The SteamVR compositor handles stereo rendering, parallax, and
// depth automatically. No manual projection math. No flickering (static textures).
//
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <csignal>
#include <openvr.h>

static volatile bool g_running = true;
void sigHandler(int) { g_running = false; }

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
    vr::IVRChaperone *chap = vr::VRChaperone();
    vr::IVRChaperoneSetup *chapSetup = vr::VRChaperoneSetup();

    std::vector<vr::VROverlayHandle_t> handles;

    // ═══════════════════════════════════════════════════════════════════
    // DIAGNOSTICS — print everything we know about the tracking space
    // ═══════════════════════════════════════════════════════════════════
    printf("═══ STEAM FRAME SPATIAL DIAGNOSTICS ═══\n");

    vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
    sys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, poses, vr::k_unMaxTrackedDeviceCount);
    if (poses[0].bPoseIsValid) {
        auto &m = poses[0].mDeviceToAbsoluteTracking.m;
        printf("HMD position (Standing): (%.3f, %.3f, %.3f)\n", m[0][3], m[1][3], m[2][3]);
    }

    float paX = 0, paZ = 0;
    vr::HmdQuad_t playRect;
    bool hasPlay = chap ? chap->GetPlayAreaRect(&playRect) : false;
    if (chap) chap->GetPlayAreaSize(&paX, &paZ);
    printf("Play area: %s, size: %.2f x %.2f m\n", hasPlay ? "YES" : "NO", paX, paZ);
    if (hasPlay) {
        for (int i = 0; i < 4; i++)
            printf("  corner[%d] = (%.3f, %.3f, %.3f)\n", i,
                playRect.vCorners[i].v[0], playRect.vCorners[i].v[1], playRect.vCorners[i].v[2]);
    }

    uint32_t qCount = 0;
    if (chapSetup) chapSetup->GetLiveCollisionBoundsInfo(nullptr, &qCount);
    printf("Collision bound quads: %u\n", qCount);

    std::vector<vr::HmdQuad_t> quads(qCount);
    if (qCount > 0) chapSetup->GetLiveCollisionBoundsInfo(quads.data(), &qCount);

    // ═══════════════════════════════════════════════════════════════════
    // WALL TEXTURE — neon wireframe grid
    // ═══════════════════════════════════════════════════════════════════
    const int TW = 64, TH = 64;
    std::vector<uint32_t> wallTex(TW * TH, 0x20FFFF00); // faint cyan fill (ABGR format)
    // Bright border
    for (int x = 0; x < TW; x++) {
        wallTex[x] = 0xC0FFFF00;
        wallTex[(TH - 1) * TW + x] = 0xC0FFFF00;
    }
    for (int y = 0; y < TH; y++) {
        wallTex[y * TW] = 0xC0FFFF00;
        wallTex[y * TW + TW - 1] = 0xC0FFFF00;
    }
    // Interior grid every 8 pixels
    for (int g = 8; g < TW; g += 8)
        for (int y = 0; y < TH; y++) wallTex[y * TW + g] = 0x50FFFF00;
    for (int g = 8; g < TH; g += 8)
        for (int x = 0; x < TW; x++) wallTex[g * TW + x] = 0x50FFFF00;

    // ═══════════════════════════════════════════════════════════════════
    // WALL OVERLAYS — one per chaperone collision quad, world-locked
    // ═══════════════════════════════════════════════════════════════════
    for (uint32_t i = 0; i < qCount && i < 64; i++) {
        const auto &q = quads[i];
        printf("Wall %u: ", i);
        for (int c = 0; c < 4; c++)
            printf("(%.2f,%.2f,%.2f) ", q.vCorners[c].v[0], q.vCorners[c].v[1], q.vCorners[c].v[2]);
        printf("\n");

        // Chaperone quads: corners ordered BL(0), TL(1), TR(2), BR(3)
        // Bottom edge (horizontal): corner[0] → corner[3]
        float bx = q.vCorners[3].v[0] - q.vCorners[0].v[0];
        float by = q.vCorners[3].v[1] - q.vCorners[0].v[1];
        float bz = q.vCorners[3].v[2] - q.vCorners[0].v[2];
        float w = sqrtf(bx * bx + by * by + bz * bz);
        if (w < 0.01f) continue;

        // Left edge (vertical): corner[0] → corner[1]
        float sx = q.vCorners[1].v[0] - q.vCorners[0].v[0];
        float sy = q.vCorners[1].v[1] - q.vCorners[0].v[1];
        float sz = q.vCorners[1].v[2] - q.vCorners[0].v[2];
        float h = sqrtf(sx * sx + sy * sy + sz * sz);
        if (h < 0.01f) continue;

        // Center of the quad
        float cx = (q.vCorners[0].v[0] + q.vCorners[2].v[0]) * 0.5f;
        float cy = (q.vCorners[0].v[1] + q.vCorners[2].v[1]) * 0.5f;
        float cz = (q.vCorners[0].v[2] + q.vCorners[2].v[2]) * 0.5f;

        // Overlay local axes in world space:
        //   local X (right) = along bottom edge
        //   local Y (up)    = along left edge
        //   local Z (normal)= right × up
        float rx = bx / w, ry = by / w, rz = bz / w;
        float ux = sx / h, uy = sy / h, uz = sz / h;
        float fx = ry * uz - rz * uy;
        float fy = rz * ux - rx * uz;
        float fz = rx * uy - ry * ux;

        // HmdMatrix34_t columns: [localX | localY | localZ | translation]
        vr::HmdMatrix34_t xf = {};
        xf.m[0][0] = rx; xf.m[0][1] = ux; xf.m[0][2] = fx; xf.m[0][3] = cx;
        xf.m[1][0] = ry; xf.m[1][1] = uy; xf.m[1][2] = fy; xf.m[1][3] = cy;
        xf.m[2][0] = rz; xf.m[2][1] = uz; xf.m[2][2] = fz; xf.m[2][3] = cz;

        char name[64];
        snprintf(name, 64, "frame.wall.%u", i);
        vr::VROverlayHandle_t oh;
        if (ovl->CreateOverlay(name, "Wall", &oh) == vr::VROverlayError_None) {
            ovl->SetOverlayTransformAbsolute(oh, vr::TrackingUniverseStanding, &xf);
            ovl->SetOverlayWidthInMeters(oh, w);
            // Fix aspect ratio: overlay height = width * (texH/texW) by default (= w).
            // We want height = h, so texelAspect = w / h.
            ovl->SetOverlayTexelAspect(oh, w / h);
            ovl->SetOverlayAlpha(oh, 0.5f);
            ovl->SetOverlayRaw(oh, wallTex.data(), TW, TH, 4);
            ovl->ShowOverlay(oh);
            handles.push_back(oh);
            printf("  → wall overlay: %.2fm wide x %.2fm tall at (%.2f,%.2f,%.2f)\n", w, h, cx, cy, cz);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // FLOOR GRID — from play area rectangle
    // ═══════════════════════════════════════════════════════════════════
    if (hasPlay && paX > 0 && paZ > 0) {
        const int FW = 128, FH = 128;
        std::vector<uint32_t> ftex(FW * FH, 0x1500FF00); // faint green
        for (int y = 0; y < FH; y++)
            for (int x = 0; x < FW; x++)
                if (x % 16 == 0 || y % 16 == 0) ftex[y * FW + x] = 0x4000FF00;
        for (int x = 0; x < FW; x++) { ftex[x] = 0x8000FF00; ftex[(FH - 1) * FW + x] = 0x8000FF00; }
        for (int y = 0; y < FH; y++) { ftex[y * FW] = 0x8000FF00; ftex[y * FW + FW - 1] = 0x8000FF00; }

        // Floor: local X → world X, local Y → world -Z, local Z (normal) → world +Y
        vr::HmdMatrix34_t fx = {};
        fx.m[0][0] = 1;              // local X → world X
        fx.m[1][2] = 1;              // local Z → world Y (normal faces up)
        fx.m[2][1] = -1;             // local Y → world -Z
        fx.m[1][3] = 0.005f;         // 5mm above floor

        vr::VROverlayHandle_t fh;
        if (ovl->CreateOverlay("frame.floor.grid", "Floor Grid", &fh) == vr::VROverlayError_None) {
            ovl->SetOverlayTransformAbsolute(fh, vr::TrackingUniverseStanding, &fx);
            ovl->SetOverlayWidthInMeters(fh, paX);
            ovl->SetOverlayTexelAspect(fh, paX / paZ);
            ovl->SetOverlayAlpha(fh, 0.35f);
            ovl->SetOverlayRaw(fh, ftex.data(), FW, FH, 4);
            ovl->ShowOverlay(fh);
            handles.push_back(fh);
            printf("Floor grid: %.2f x %.2f m\n", paX, paZ);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // ORIGIN MARKER — red square at tracking origin (1m height)
    // Verifies world-locking is working correctly
    // ═══════════════════════════════════════════════════════════════════
    {
        const int MW = 32, MH = 32;
        std::vector<uint32_t> mtex(MW * MH, 0xFF0000FF); // solid red (ABGR)
        // White border
        for (int x = 0; x < MW; x++) { mtex[x] = 0xFFFFFFFF; mtex[(MH - 1) * MW + x] = 0xFFFFFFFF; }
        for (int y = 0; y < MH; y++) { mtex[y * MW] = 0xFFFFFFFF; mtex[y * MW + MW - 1] = 0xFFFFFFFF; }

        vr::HmdMatrix34_t mx = {};
        mx.m[0][0] = 1; mx.m[1][1] = 1; mx.m[2][2] = 1;
        mx.m[1][3] = 1.0f; // 1m above floor

        vr::VROverlayHandle_t mh;
        if (ovl->CreateOverlay("frame.origin.marker", "Origin", &mh) == vr::VROverlayError_None) {
            ovl->SetOverlayTransformAbsolute(mh, vr::TrackingUniverseStanding, &mx);
            ovl->SetOverlayWidthInMeters(mh, 0.3f);
            ovl->SetOverlayAlpha(mh, 0.8f);
            ovl->SetOverlayRaw(mh, mtex.data(), MW, MH, 4);
            ovl->ShowOverlay(mh);
            handles.push_back(mh);
            printf("Origin marker: 30cm red square at (0, 1.0, 0)\n");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // If chaperone had zero data, create walls from play area corners
    // ═══════════════════════════════════════════════════════════════════
    if (qCount == 0 && hasPlay) {
        printf("No collision bounds — generating walls from play area corners\n");
        float wallHeight = 2.5f; // assume 2.5m ceiling

        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            float x0 = playRect.vCorners[i].v[0], z0 = playRect.vCorners[i].v[2];
            float x1 = playRect.vCorners[j].v[0], z1 = playRect.vCorners[j].v[2];

            float dx = x1 - x0, dz = z1 - z0;
            float w = sqrtf(dx * dx + dz * dz);
            if (w < 0.01f) continue;

            float cx = (x0 + x1) * 0.5f;
            float cz = (z0 + z1) * 0.5f;
            float cy = wallHeight * 0.5f;

            // Right = along wall, Up = Y, Normal = inward
            float rx = dx / w, rz = dz / w;
            float nx = -rz, nz = rx; // 90° rotation in XZ plane → inward normal

            vr::HmdMatrix34_t xf = {};
            xf.m[0][0] = rx;  xf.m[0][1] = 0; xf.m[0][2] = nx; xf.m[0][3] = cx;
            xf.m[1][0] = 0;   xf.m[1][1] = 1; xf.m[1][2] = 0;  xf.m[1][3] = cy;
            xf.m[2][0] = rz;  xf.m[2][1] = 0; xf.m[2][2] = nz; xf.m[2][3] = cz;

            char name[64];
            snprintf(name, 64, "frame.pawall.%d", i);
            vr::VROverlayHandle_t oh;
            if (ovl->CreateOverlay(name, "PA Wall", &oh) == vr::VROverlayError_None) {
                ovl->SetOverlayTransformAbsolute(oh, vr::TrackingUniverseStanding, &xf);
                ovl->SetOverlayWidthInMeters(oh, w);
                ovl->SetOverlayTexelAspect(oh, w / wallHeight);
                ovl->SetOverlayAlpha(oh, 0.5f);
                ovl->SetOverlayRaw(oh, wallTex.data(), TW, TH, 4);
                ovl->ShowOverlay(oh);
                handles.push_back(oh);
                printf("  → PA wall %d: %.2fm x %.2fm at (%.2f,%.2f,%.2f)\n", i, w, wallHeight, cx, cy, cz);
            }
        }
    }

    printf("═══ %zu overlays active. Waiting... ═══\n", handles.size());
    fflush(stdout);

    // Main loop — minimal CPU since overlays are static and world-locked
    while (g_running) {
        vr::VREvent_t ev;
        while (sys->PollNextEvent(&ev, sizeof(ev))) {
            if (ev.eventType == vr::VREvent_Quit) g_running = false;
        }
        usleep(100000); // 10 Hz is plenty for event polling
    }

    for (auto h : handles) ovl->DestroyOverlay(h);
    vr::VR_Shutdown();
    printf("Clean exit.\n");
    return 0;
}
