// HMD pose log on the IR frames' clock. Samples SteamVR's current (zero-prediction) standing-universe
// HMD pose at ~1 kHz and stamps each with CLOCK_MONOTONIC (ir_record's V4L2 timestamps) and
// CLOCK_MONOTONIC_RAW (its copy times).
// Usage: pose_log OUT.csv DURATION_MS
// Build: g++ -O2 -std=c++17 -I/opt/steamvr/tools/hellovr_vulkan_linux/src/openvr/headers pose_log.cpp
//        -L/opt/steamvr/bin/linuxarm64 -lopenvr_api -Wl,-rpath=/opt/steamvr/bin/linuxarm64 -o pose_log
#include <openvr.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <csignal>

static volatile sig_atomic_t running = 1;
static void stopSignal(int) { running = 0; }

static long long ns(clockid_t c) {
  timespec t{};
  clock_gettime(c, &t);
  return t.tv_sec * 1000000000ll + t.tv_nsec;
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: pose_log OUT.csv DURATION_MS\n"); return 2; }
  std::signal(SIGTERM, stopSignal);
  std::signal(SIGINT, stopSignal);
  vr::EVRInitError err = vr::VRInitError_None;
  vr::IVRSystem *sys = vr::VR_Init(&err, vr::VRApplication_Background);
  if (err != vr::VRInitError_None) {
    fprintf(stderr, "VR_Init failed: %s\n", vr::VR_GetVRInitErrorAsEnglishDescription(err));
    return 1;
  }
  FILE *f = fopen(argv[1], "w");
  fprintf(f, "mono_ns,raw_ns,valid,result,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23\n");
  const long long duration = atoll(argv[2]);
  if (duration < 0) return 2;
  const long long end = ns(CLOCK_MONOTONIC) + duration * 1000000ll;
  unsigned n = 0, invalid = 0;
  while (running && (duration == 0 || ns(CLOCK_MONOTONIC) < end)) {
    vr::TrackedDevicePose_t p[vr::k_unMaxTrackedDeviceCount];
    const long long a = ns(CLOCK_MONOTONIC), ar = ns(CLOCK_MONOTONIC_RAW);
    sys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0.0f, p, vr::k_unMaxTrackedDeviceCount);
    const long long b = ns(CLOCK_MONOTONIC), br = ns(CLOCK_MONOTONIC_RAW);
    const auto &h = p[vr::k_unTrackedDeviceIndex_Hmd];
    const auto &m = h.mDeviceToAbsoluteTracking.m;
    fprintf(f, "%lld,%lld,%d,%d", (a + b) / 2, (ar + br) / 2, h.bPoseIsValid ? 1 : 0, int(h.eTrackingResult));
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 4; c++) fprintf(f, ",%.7f", m[r][c]);
    fputc('\n', f);
    n++;
    if (!h.bPoseIsValid) invalid++;
    usleep(1000);
  }
  fclose(f);
  vr::VR_Shutdown();
  printf("%u poses, %u invalid, %s\n", n, invalid, argv[1]);
}
