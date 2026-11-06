#include <stdio.h>
#include <openvr.h>

int main() {
    vr::EVRInitError err = vr::VRInitError_None;
    vr::IVRSystem *sys = vr::VR_Init(&err, vr::VRApplication_Background);
    if (err != vr::VRInitError_None) {
        printf("VR_Init failed: %d\n", err);
        return 1;
    }
    printf("VR_Init OK!\n");

    const char *interfaces[] = {
        "IVRCameraPassthroughInternal_001",
        "IVRCameraPassthroughInternal_002",
        "IVRCameraPassthrough_001",
        "IVRTrackedCameraInternal_001",
        "IVRTrackedCameraInternal_XXX",
        "IVRCompositorSystemInternal_001",
        "IVRHeadsetView_001",
        NULL
    };

    for (int i = 0; interfaces[i]; i++) {
        vr::EVRInitError ifErr = vr::VRInitError_None;
        void *iface = vr::VR_GetGenericInterface(interfaces[i], &ifErr);
        printf("Interface %s: %p (err=%d)\n", interfaces[i], iface, ifErr);
    }

    vr::VR_Shutdown();
    return 0;
}
