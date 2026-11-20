#include <stdio.h>
#include <openvr.h>
int main() {
    vr::EVRInitError err = vr::VRInitError_None;
    vr::IVRSystem *sys = vr::VR_Init(&err, vr::VRApplication_Background);
    vr::HmdMatrix34_t eL = sys->GetEyeToHeadTransform(vr::Eye_Left);
    vr::HmdMatrix34_t eR = sys->GetEyeToHeadTransform(vr::Eye_Right);
    printf("Left: %f %f %f\n", eL.m[0][3], eL.m[1][3], eL.m[2][3]);
    printf("Right: %f %f %f\n", eR.m[0][3], eR.m[1][3], eR.m[2][3]);
    vr::VR_Shutdown();
    return 0;
}
