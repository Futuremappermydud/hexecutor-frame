#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <openvr.h>

int main() {
    vr::EVRInitError err = vr::VRInitError_None;
    vr::IVRSystem *sys = vr::VR_Init(&err, vr::VRApplication_Background);
    if (err != vr::VRInitError_None) return 1;

    vr::EVRInitError ifErr = vr::VRInitError_None;
    void *iface = vr::VR_GetGenericInterface("IVRCameraPassthroughInternal_001", &ifErr);
    if (!iface) return 1;

    void **vtable = *(void***)iface;
    Dl_info binfo;
    dladdr(vtable[0], &binfo);
    uintptr_t fbase = (uintptr_t)binfo.dli_fbase;
    printf("vrclient.so base = 0x%lx\n", (unsigned long)fbase);

    for (int i = 0; i < 20; i++) {
        if (!vtable[i]) break;
        uintptr_t offset = (uintptr_t)vtable[i] - fbase;
        printf("  method[%2d] = offset 0x%lx\n", i, (unsigned long)offset);
    }

    vr::VR_Shutdown();
    return 0;
}
