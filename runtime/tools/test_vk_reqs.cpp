#include <stdio.h>
#include <openvr.h>

int main() {
    vr::EVRInitError err = vr::VRInitError_None;
    vr::IVRSystem *sys = vr::VR_Init(&err, vr::VRApplication_Background);
    if (err != vr::VRInitError_None) {
        printf("VR_Init failed: %d\n", err);
        return 1;
    }
    
    vr::IVRCompositor *comp = vr::VRCompositor();
    if (comp) {
        char buf[2048] = {0};
        uint32_t len = comp->GetVulkanInstanceExtensionsRequired(buf, sizeof(buf));
        printf("Required Instance Extensions (%u bytes): %s\n", len, buf);
    }
    
    vr::VR_Shutdown();
    return 0;
}
