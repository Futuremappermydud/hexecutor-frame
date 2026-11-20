// camera_probe.cpp — Discover what IVRTrackedCamera gives us on Steam Frame
//
// Queries camera capabilities, frame layout (mono/stereo), intrinsics,
// and captures a test frame to verify the pipeline.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <openvr.h>

int main() {
    vr::EVRInitError err;
    vr::IVRSystem *sys = vr::VR_Init(&err, vr::VRApplication_Background);
    if (err != vr::VRInitError_None) {
        fprintf(stderr, "VR_Init failed: %s\n", vr::VR_GetVRInitErrorAsEnglishDescription(err));
        return 1;
    }

    vr::IVRTrackedCamera *cam = vr::VRTrackedCamera();
    if (!cam) {
        fprintf(stderr, "IVRTrackedCamera not available\n");
        vr::VR_Shutdown();
        return 1;
    }

    // Check if HMD has a camera
    bool hasCamera = false;
    vr::EVRTrackedCameraError camErr = cam->HasCamera(vr::k_unTrackedDeviceIndex_Hmd, &hasCamera);
    printf("HasCamera: %s (err=%d)\n", hasCamera ? "YES" : "NO", camErr);
    if (!hasCamera) {
        vr::VR_Shutdown();
        return 1;
    }

    // Query frame sizes for each frame type
    const char* typeNames[] = {"Distorted", "Undistorted", "MaximumUndistorted"};
    vr::EVRTrackedCameraFrameType types[] = {
        vr::VRTrackedCameraFrameType_Distorted,
        vr::VRTrackedCameraFrameType_Undistorted,
        vr::VRTrackedCameraFrameType_MaximumUndistorted
    };

    for (int t = 0; t < 3; t++) {
        uint32_t w = 0, h = 0, bufSize = 0;
        camErr = cam->GetCameraFrameSize(vr::k_unTrackedDeviceIndex_Hmd, types[t], &w, &h, &bufSize);
        printf("\nFrame type %s: err=%d\n", typeNames[t], camErr);
        if (camErr == vr::VRTrackedCameraError_None) {
            printf("  Size: %u x %u, buffer: %u bytes\n", w, h, bufSize);
            printf("  Bytes per pixel: %.1f\n", bufSize > 0 ? (float)bufSize / (w * h) : 0);
        }
    }

    // Query camera intrinsics for each eye
    const char* eyeNames[] = {"Left", "Right"};
    for (int e = 0; e < 2; e++) {
        vr::HmdVector2_t focalLength, principalPoint;
        camErr = cam->GetCameraIntrinsics(
            vr::k_unTrackedDeviceIndex_Hmd,
            0, // camera index
            types[1], // Undistorted
            &focalLength, &principalPoint);
        printf("\nIntrinsics (camera %d, %s): err=%d\n", 0, eyeNames[e], camErr);
        if (camErr == vr::VRTrackedCameraError_None) {
            printf("  Focal length: (%.2f, %.2f)\n", focalLength.v[0], focalLength.v[1]);
            printf("  Principal point: (%.2f, %.2f)\n", principalPoint.v[0], principalPoint.v[1]);
        }
    }

    // Query camera projection for each eye
    for (int e = 0; e < 2; e++) {
        vr::HmdMatrix44_t proj;
        camErr = cam->GetCameraProjection(
            vr::k_unTrackedDeviceIndex_Hmd,
            e, // camera index
            types[1],
            0.01f, 100.0f,
            &proj);
        printf("\nProjection (camera %d): err=%d\n", e, camErr);
        if (camErr == vr::VRTrackedCameraError_None) {
            printf("  [%.4f %.4f %.4f %.4f]\n", proj.m[0][0], proj.m[0][1], proj.m[0][2], proj.m[0][3]);
            printf("  [%.4f %.4f %.4f %.4f]\n", proj.m[1][0], proj.m[1][1], proj.m[1][2], proj.m[1][3]);
            printf("  [%.4f %.4f %.4f %.4f]\n", proj.m[2][0], proj.m[2][1], proj.m[2][2], proj.m[2][3]);
            printf("  [%.4f %.4f %.4f %.4f]\n", proj.m[3][0], proj.m[3][1], proj.m[3][2], proj.m[3][3]);
        }
    }

    // Check depth-related device properties
    printf("\n=== Device Properties ===\n");
    vr::ETrackedPropertyError propErr;
    
    bool supportsDepth = sys->GetBoolTrackedDeviceProperty(
        vr::k_unTrackedDeviceIndex_Hmd, 
        (vr::ETrackedDeviceProperty)2079, // Prop_SupportsRoomViewDepthProjection_Bool
        &propErr);
    printf("SupportsRoomViewDepthProjection: %s (propErr=%d)\n", supportsDepth ? "YES" : "NO", propErr);

    bool supportsRoomViewDirect = sys->GetBoolTrackedDeviceProperty(
        vr::k_unTrackedDeviceIndex_Hmd,
        (vr::ETrackedDeviceProperty)2105, // Prop_Hmd_SupportsRoomViewDirect_Bool
        &propErr);
    printf("SupportsRoomViewDirect: %s (propErr=%d)\n", supportsRoomViewDirect ? "YES" : "NO", propErr);

    int32_t numCams = sys->GetInt32TrackedDeviceProperty(
        vr::k_unTrackedDeviceIndex_Hmd,
        vr::Prop_NumCameras_Int32,
        &propErr);
    printf("NumCameras: %d (propErr=%d)\n", numCams, propErr);

    // Try to acquire video streaming and grab a frame
    printf("\n=== Attempting to acquire video stream ===\n");
    vr::TrackedCameraHandle_t cameraHandle = 0;
    camErr = cam->AcquireVideoStreamingService(vr::k_unTrackedDeviceIndex_Hmd, &cameraHandle);
    printf("AcquireVideoStreamingService: err=%d, handle=%lu\n", camErr, (unsigned long)cameraHandle);

    if (camErr == vr::VRTrackedCameraError_None && cameraHandle != 0) {
        // Wait a moment for the stream to start
        usleep(500000);

        // Try to grab a frame (Undistorted)
        uint32_t w = 0, h = 0, bufSize = 0;
        cam->GetCameraFrameSize(vr::k_unTrackedDeviceIndex_Hmd,
            vr::VRTrackedCameraFrameType_Undistorted, &w, &h, &bufSize);

        if (bufSize > 0) {
            std::vector<uint8_t> frameBuf(bufSize);
            vr::CameraVideoStreamFrameHeader_t header;
            memset(&header, 0, sizeof(header));

            camErr = cam->GetVideoStreamFrameBuffer(
                cameraHandle,
                vr::VRTrackedCameraFrameType_Undistorted,
                frameBuf.data(), bufSize,
                &header, sizeof(header));

            printf("GetVideoStreamFrameBuffer: err=%d\n", camErr);
            if (camErr == vr::VRTrackedCameraError_None) {
                printf("  Frame: %u x %u\n", header.nWidth, header.nHeight);
                printf("  Bytes per pixel: %u\n", header.nBytesPerPixel);
                printf("  Frame sequence: %u\n", header.nFrameSequence);
                printf("  Pose valid: %s\n", header.trackedDevicePose.bPoseIsValid ? "YES" : "NO");
                
                if (header.trackedDevicePose.bPoseIsValid) {
                    auto &m = header.trackedDevicePose.mDeviceToAbsoluteTracking.m;
                    printf("  Camera pose: (%.3f, %.3f, %.3f)\n", m[0][3], m[1][3], m[2][3]);
                }

                // Save raw frame for inspection
                char fname[256];
                snprintf(fname, sizeof(fname), "/home/steamos/test_openvr/camera_frame_%ux%u_%dbpp.raw",
                    header.nWidth, header.nHeight, header.nBytesPerPixel);
                FILE *f = fopen(fname, "wb");
                if (f) {
                    fwrite(frameBuf.data(), 1, bufSize, f);
                    fclose(f);
                    printf("  Saved raw frame to %s\n", fname);
                }

                // Try another frame a moment later to check frame rate
                usleep(50000);
                vr::CameraVideoStreamFrameHeader_t header2;
                camErr = cam->GetVideoStreamFrameBuffer(
                    cameraHandle,
                    vr::VRTrackedCameraFrameType_Undistorted,
                    frameBuf.data(), bufSize,
                    &header2, sizeof(header2));
                if (camErr == vr::VRTrackedCameraError_None) {
                    printf("  Frame sequence delta: %u → %u (diff=%d)\n",
                        header.nFrameSequence, header2.nFrameSequence,
                        (int)header2.nFrameSequence - (int)header.nFrameSequence);
                }
            }
        }

        cam->ReleaseVideoStreamingService(cameraHandle);
        printf("Released video stream.\n");
    }

    vr::VR_Shutdown();
    printf("Done.\n");
    return 0;
}
