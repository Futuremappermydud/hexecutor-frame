/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef HEXECUTOR_XRSERVICE_BROKER_SHIM_H
#define HEXECUTOR_XRSERVICE_BROKER_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Set or override the Unix domain socket path used by the shim.
// Defaults to the HEXECUTORD_SOCKET environment variable, or "/tmp/hexecutord.sock".
void broker_shim_set_socket(const char *socket_path);

// Check if the broker daemon is reachable and responding. Returns 1 if connected, 0 otherwise.
int broker_shim_is_connected(void);

// Dispatches Stereo FeatureNN (controller keypoint tracking) to the CDSP broker.
// input_u8: packed u8 stereo crops (128x128 per eye, 2 eyes per controller)
// output_records: output buffer receiving 0x450 (1104) bytes per controller view
// views: total number of eye views (2 for 1 controller, 4 for 2 controllers)
// width, height: crop dimensions (must be 128x128)
// Returns 0 on success, or a negative error code on failure.
int broker_shim_dispatch_stereofeaturenn(const void *input_u8, void *output_records,
                                        uint32_t views, uint32_t width, uint32_t height);

// Dispatches Gazenet eye image evaluation to the CDSP broker.
// input_planar_f32: two planar 128x128 float images (left + right eye)
// output_64_f32: buffer receiving exactly 64 output floats
// Returns 0 on success, or a negative error code on failure.
int broker_shim_dispatch_gazenet(const float *input_planar_f32, float *output_64_f32);

// Dispatches Gazenet float vector evaluation to the CDSP broker.
// input_f32: input float vector (typically 4,800 floats)
// in_floats: number of floats in input_f32
// output_f32: buffer receiving output floats
// max_out_floats: capacity of output_f32
// out_floats: written with the actual count of output floats returned
// Returns 0 on success, or a negative error code on failure.
int broker_shim_dispatch_gazenet_vector(const float *input_f32, uint32_t in_floats,
                                       float *output_f32, uint32_t max_out_floats,
                                       uint32_t *out_floats);

// Drop-in LD_PRELOAD symbol interception for XRService's dispatchStereoFeatureNNOnCDSP
int dispatchStereoFeatureNNOnCDSP(const void *input_u8, void *output_records,
                                  uint32_t views, uint32_t width, uint32_t height);

#ifdef __cplusplus
}
#endif

#endif // HEXECUTOR_XRSERVICE_BROKER_SHIM_H
