#include <stdio.h>
#include <unistd.h>
#include <openvr.h>

int main() {
    vr::EVRInitError err = vr::VRInitError_None;
    vr::IVRSystem *vr_sys = vr::VR_Init(&err, vr::VRApplication_Overlay);
    if (err != vr::VRInitError_None) {
        printf("VR_Init failed: %s\n", vr::VR_GetVRInitErrorAsEnglishDescription(err));
        return 1;
    }
    printf("VR_Init succeeded with VRApplication_Overlay!\n");
    
    vr::IVROverlay *overlay = vr::VROverlay();
    if (!overlay) {
        printf("Failed to get IVROverlay interface!\n");
        vr::VR_Shutdown();
        return 1;
    }
    
    vr::VROverlayHandle_t handle;
    vr::EVROverlayError ovErr = overlay->CreateOverlay("frame.spatial.test", "Spatial Test", &handle);
    if (ovErr != vr::VROverlayError_None) {
        printf("CreateOverlay failed: %d\n", ovErr);
        vr::VR_Shutdown();
        return 1;
    }
    printf("CreateOverlay succeeded! Handle: %llu\n", (unsigned long long)handle);
    
    overlay->DestroyOverlay(handle);
    vr::VR_Shutdown();
    printf("Done!\n");
    return 0;
}
