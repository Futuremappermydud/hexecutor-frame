// Controller surface annotation with a headset HUD and world-locked mark dots.
// Trigger traces the selected class; grip cycles classes. Raw controller pose is
// retained so a calibrated tip/model hull can be applied offline.
#include <openvr.h>
#include <unistd.h>
#include <array>
#include <cstdint>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

static long long stamp(clockid_t clock) {
    timespec t{}; clock_gettime(clock, &t);
    return int64_t(t.tv_sec) * 1000000000ll + t.tv_nsec;
}

static const char* names[]{"WALL", "TABLE", "FLOOR", "CEILING", "EDGE", "OPENING"};
static volatile sig_atomic_t running = 1;
static void stopSignal(int) { running = 0; }
static const std::array<std::array<float, 3>, 6> colors{{
    {0.3f, 0.75f, 1.f}, {1.f, 0.65f, 0.2f}, {0.3f, 1.f, 0.45f},
    {0.8f, 0.55f, 1.f}, {1.f, 0.25f, 0.3f}, {1.f, 1.f, 0.3f}
}};

static const uint8_t* glyph(char c) {
    static const uint8_t empty[5]{};
    static const struct { char c; uint8_t p[5]; } font[]{
        {'A',{0x7C,0x12,0x11,0x12,0x7C}}, {'B',{0x7F,0x49,0x49,0x49,0x36}},
        {'C',{0x3E,0x41,0x41,0x41,0x22}}, {'D',{0x7F,0x41,0x41,0x22,0x1C}},
        {'E',{0x7F,0x49,0x49,0x49,0x41}}, {'F',{0x7F,0x09,0x09,0x09,0x01}},
        {'G',{0x3E,0x41,0x49,0x49,0x7A}}, {'I',{0x00,0x41,0x7F,0x41,0x00}},
        {'L',{0x7F,0x40,0x40,0x40,0x40}}, {'N',{0x7F,0x04,0x08,0x10,0x7F}},
        {'O',{0x3E,0x41,0x41,0x41,0x3E}}, {'P',{0x7F,0x09,0x09,0x09,0x06}},
        {'R',{0x7F,0x09,0x19,0x29,0x46}}, {'T',{0x01,0x01,0x7F,0x01,0x01}},
        {'S',{0x46,0x49,0x49,0x49,0x31}},
        {'W',{0x7F,0x20,0x18,0x20,0x7F}},
        {'0',{0x3E,0x51,0x49,0x45,0x3E}}, {'1',{0x00,0x42,0x7F,0x40,0x00}},
        {'2',{0x42,0x61,0x51,0x49,0x46}}, {'3',{0x21,0x41,0x45,0x4B,0x31}},
        {'4',{0x18,0x14,0x12,0x7F,0x10}}, {'5',{0x27,0x45,0x45,0x45,0x39}},
        {'6',{0x3C,0x4A,0x49,0x49,0x30}}, {'7',{0x01,0x71,0x09,0x05,0x03}},
        {'8',{0x36,0x49,0x49,0x49,0x36}}, {'9',{0x06,0x49,0x49,0x29,0x1E}},
    };
    for (const auto& item : font) if (item.c == c) return item.p;
    return empty;
}

static void text(std::vector<uint32_t>& pixels, int width, int x, int y, const std::string& message) {
    for (char c : message) {
        const auto* g = glyph(c);
        for (int col = 0; col < 5; ++col)
            for (int row = 0; row < 7; ++row)
                if (g[col] & (1 << row)) pixels[(y + row) * width + x + col] = 0xffffffffu;
        x += 7;
    }
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 6 && argc != 7) {
        std::cerr << "usage: mark_surfaces OUT.csv DURATION_MS [TIP_X TIP_Y TIP_Z [STOP_FILE]]\n";
        return 2;
    }
    const int duration = std::atoi(argv[2]);
    if (duration < 0) return 2;
    const std::array<float, 3> tip = argc >= 6 ? std::array<float, 3>{std::atof(argv[3]), std::atof(argv[4]), std::atof(argv[5])}
                                                : std::array<float, 3>{0.f, 0.f, 0.f};
    const std::string stopFile = argc == 7 ? argv[6] : "";
    std::signal(SIGTERM, stopSignal);
    std::signal(SIGINT, stopSignal);
    std::cerr << "marker starting, output=" << argv[1] << std::endl;
    vr::EVRInitError error = vr::VRInitError_None;
    vr::IVRSystem* system = vr::VR_Init(&error, vr::VRApplication_Overlay);
    if (error != vr::VRInitError_None) {
        std::cerr << "VR_Init: " << vr::VR_GetVRInitErrorAsEnglishDescription(error) << '\n';
        return 1;
    }
    auto* overlay = vr::VROverlay();
    if (!overlay) { vr::VR_Shutdown(); return 1; }
    std::cerr << "marker OpenVR initialized" << std::endl;
    auto* input = vr::VRInput();
    auto* settings = vr::VRSettings();
    const bool priorGlobal = settings->GetBool("steamvr", "globalActionSetPriority", nullptr);
    if (!priorGlobal) settings->SetBool("steamvr", "globalActionSetPriority", true);
    const std::string manifest = (std::filesystem::absolute(argv[0]).parent_path().parent_path() /
                                  "actions/room_marks.json").string();
    const auto manifestError = input->SetActionManifestPath(manifest.c_str());
    vr::VRActionSetHandle_t actionSet{};
    vr::VRActionHandle_t actions[4]{};
    const char* actionNames[]{"left_trigger", "left_grip", "right_trigger", "right_grip"};
    bool actionReady = manifestError == vr::VRInputError_None &&
                       input->GetActionSetHandle("/actions/marks", &actionSet) == vr::VRInputError_None;
    for (int n = 0; n < 4 && actionReady; ++n)
        actionReady = input->GetActionHandle((std::string("/actions/marks/in/") + actionNames[n]).c_str(),
                                              &actions[n]) == vr::VRInputError_None;
    std::cerr << "marker SteamVR actions: " << (actionReady ? "ready" : "unavailable")
              << " manifest_error=" << int(manifestError) << " global=" << int(priorGlobal) << std::endl;
    const std::string key = "frame.dataset.labels." + std::to_string(getpid());
    vr::VROverlayHandle_t hud = vr::k_ulOverlayHandleInvalid;
    if (overlay->CreateOverlay(key.c_str(), "Surface annotation", &hud) != vr::VROverlayError_None) {
        vr::VR_Shutdown(); return 1;
    }
    vr::HmdMatrix34_t hudTransform{};
    for (int i = 0; i < 3; ++i) hudTransform.m[i][i] = 1.f;
    hudTransform.m[0][3] = -0.28f; hudTransform.m[1][3] = -0.18f; hudTransform.m[2][3] = -1.0f;
    overlay->SetOverlayTransformTrackedDeviceRelative(hud, vr::k_unTrackedDeviceIndex_Hmd, &hudTransform);
    overlay->SetOverlayWidthInMeters(hud, 0.42f);
    overlay->SetOverlayAlpha(hud, 0.85f);
    overlay->ShowOverlay(hud);
    std::ofstream csv(argv[1]);
    csv << "mono_ns,raw_ns,device,role,label,stroke,valid,result,tip_x,tip_y,tip_z,"
           "m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23\n";
    if (!csv) return 1;
    csv.flush();
    std::cerr << "marker CSV opened" << std::endl;
    int selected = 0, points = 0, strokes[vr::k_unMaxTrackedDeviceCount]{};
    bool trigger[vr::k_unMaxTrackedDeviceCount]{}, grip[vr::k_unMaxTrackedDeviceCount]{};
    long long stopHeldAt[vr::k_unMaxTrackedDeviceCount]{};
    std::vector<vr::VROverlayHandle_t> markers;
    unsigned markerIndex = 0;
    std::array<float, 3> lastMark{NAN, NAN, NAN};
    const auto rgbDir = std::filesystem::path(argv[1]).parent_path() / "rgb";
    int frameCount = 0;
    long long nextFrameCount = 0, nextDiagnostic = 0;
    const auto end = stamp(CLOCK_MONOTONIC) + int64_t(duration) * 1000000ll;
    while (running && (duration == 0 || stamp(CLOCK_MONOTONIC) < end)) {
        const long long loopNow = stamp(CLOCK_MONOTONIC);
        if (loopNow >= nextFrameCount) {
            frameCount = 0;
            std::error_code ec;
            for (const auto& entry : std::filesystem::directory_iterator(rgbDir, ec))
                if (entry.path().extension() == ".json" && entry.path().filename().string().rfind("frame-", 0) == 0)
                    ++frameCount;
            nextFrameCount = loopNow + 500000000ll;
        }
        const auto color = colors[selected];
        std::vector<uint32_t> hudPixels(256 * 32, 0x9a151b20u);
        for (int y = 4; y < 27; ++y) for (int x = 4; x < 14; ++x)
            hudPixels[y * 256 + x] = 0xff000000u | (int(color[2]*255)<<16) |
                                     (int(color[1]*255)<<8) | int(color[0]*255);
        text(hudPixels, 256, 24, 6, names[selected]);
        text(hudPixels, 256, 24, 17, std::to_string(points));
        text(hudPixels, 256, 110, 17, "F" + std::to_string(frameCount));
        overlay->SetOverlayRaw(hud, hudPixels.data(), 256, 32, 4);
        vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
        system->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0.f, poses,
                                                vr::k_unMaxTrackedDeviceCount);
        bool actionDown[4]{}, actionActive[4]{};
        if (actionReady) {
            vr::VRActiveActionSet_t activeSet{};
            activeSet.ulActionSet = actionSet;
            activeSet.nPriority = vr::k_nActionSetOverlayGlobalPriorityMin + 0x100;
            if (input->UpdateActionState(&activeSet, sizeof activeSet, 1) == vr::VRInputError_None)
                for (int n = 0; n < 4; ++n) {
                    vr::InputDigitalActionData_t data{};
                    if (input->GetDigitalActionData(actions[n], &data, sizeof data,
                                                    vr::k_ulInvalidInputValueHandle) == vr::VRInputError_None) {
                        actionActive[n] = data.bActive;
                        actionDown[n] = data.bActive && data.bState;
                    }
                }
        }
        int controllers = 0, states = 0;
        for (vr::TrackedDeviceIndex_t i = 0; i < vr::k_unMaxTrackedDeviceCount; ++i) {
            if (system->GetTrackedDeviceClass(i) != vr::TrackedDeviceClass_Controller) continue;
            ++controllers;
            const long long a = stamp(CLOCK_MONOTONIC), ar = stamp(CLOCK_MONOTONIC_RAW);
            vr::VRControllerState_t state{};
            const bool legacyState = system->GetControllerState(i, &state, sizeof(state));
            if (legacyState) ++states;
            const auto& pose = poses[i];
            const long long b = stamp(CLOCK_MONOTONIC), br = stamp(CLOCK_MONOTONIC_RAW);
            const auto role = system->GetControllerRoleForTrackedDeviceIndex(i);
            const int base = role == vr::TrackedControllerRole_LeftHand ? 0 :
                             role == vr::TrackedControllerRole_RightHand ? 2 : -1;
            const bool pressedGrip = base >= 0 && actionReady ? actionDown[base + 1] :
                                     legacyState && (state.ulButtonPressed & vr::ButtonMaskFromId(vr::k_EButton_Grip));
            const bool pressedTrigger = base >= 0 && actionReady ? actionDown[base] :
                                        legacyState && (state.ulButtonPressed & vr::ButtonMaskFromId(vr::k_EButton_SteamVR_Trigger));
            if (pressedGrip && !grip[i] && !pressedTrigger) {
                selected = (selected + 1) % 6; std::cout << "label " << names[selected] << '\n';
            }
            grip[i] = pressedGrip;
            if (pressedGrip && pressedTrigger && !stopFile.empty()) {
                if (!stopHeldAt[i]) stopHeldAt[i] = a;
                if (a - stopHeldAt[i] >= 2000000000ll) {
                    std::ofstream(stopFile) << "controller_stop\n";
                    running = 0;
                    break;
                }
            } else stopHeldAt[i] = 0;
            if (pressedTrigger && !trigger[i]) { ++strokes[i]; lastMark = {NAN,NAN,NAN}; }
            trigger[i] = pressedTrigger;
            if (!pressedTrigger) continue;
            const auto& m = pose.mDeviceToAbsoluteTracking.m;
            const float x = m[0][0]*tip[0] + m[0][1]*tip[1] + m[0][2]*tip[2] + m[0][3];
            const float y = m[1][0]*tip[0] + m[1][1]*tip[1] + m[1][2]*tip[2] + m[1][3];
            const float z = m[2][0]*tip[0] + m[2][1]*tip[1] + m[2][2]*tip[2] + m[2][3];
            csv << (a+b)/2 << ',' << (ar+br)/2 << ',' << i << ','
                << int(system->GetControllerRoleForTrackedDeviceIndex(i)) << ',' << names[selected] << ','
                << strokes[i] << ',' << (pose.bPoseIsValid ? 1 : 0) << ',' << int(pose.eTrackingResult)
                << ',' << x << ',' << y << ',' << z;
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 4; ++col) csv << ',' << m[row][col];
            csv << '\n'; ++points;
            if (!pose.bPoseIsValid || (std::isfinite(lastMark[0]) &&
                std::hypot(std::hypot(x-lastMark[0],y-lastMark[1]),z-lastMark[2]) < 0.04f)) continue;
            lastMark = {x,y,z};
            vr::VROverlayHandle_t marker;
            if (markers.size() < 96) {
                const std::string markerKey = key + "." + std::to_string(markers.size());
                if (overlay->CreateOverlay(markerKey.c_str(), "Surface mark", &marker) != vr::VROverlayError_None) continue;
                uint32_t pixel = 0xffffffffu;
                overlay->SetOverlayRaw(marker, &pixel, 1, 1, 4);
                overlay->SetOverlayWidthInMeters(marker, 0.025f);
                overlay->SetOverlayAlpha(marker, 0.9f);
                markers.push_back(marker);
            } else marker = markers[markerIndex++ % markers.size()];
            vr::HmdMatrix34_t transform{};
            const auto& hmd = poses[vr::k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking.m;
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col) transform.m[row][col] = hmd[row][col];
            transform.m[0][3] = x; transform.m[1][3] = y; transform.m[2][3] = z;
            overlay->SetOverlayTransformAbsolute(marker, vr::TrackingUniverseStanding, &transform);
            overlay->SetOverlayColor(marker, color[0], color[1], color[2]);
            overlay->ShowOverlay(marker);
        }
        if (loopNow >= nextDiagnostic) {
            std::cerr << "marker input: controllers=" << controllers << " states=" << states
                      << " actions_active=" << int(actionActive[0]) << int(actionActive[1])
                      << int(actionActive[2]) << int(actionActive[3])
                      << " points=" << points << " frames=" << frameCount << std::endl;
            nextDiagnostic = loopNow + 5000000000ll;
        }
        usleep(15000);
    }
    csv.close();
    for (auto marker : markers) overlay->DestroyOverlay(marker);
    overlay->DestroyOverlay(hud);
    vr::VR_Shutdown();
    std::cout << "marked points=" << points << '\n';
    return csv ? 0 : 1;
}
