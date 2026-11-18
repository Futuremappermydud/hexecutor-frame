/* Float32 execution forms of the QNN FP16 convolution-family leaves.
 *
 * The engine gathers each operand once into float32 NHWC scratch, prepares
 * weights once per operation (scale folded in, clamp bounds expanded), runs
 * these loops split by output rows over `parts` workers, and scatters the
 * result back to FP16. Results stay within the HMX tolerance of the FP16
 * reference handlers in qnn16_hvx.h (only the scale/weight rounding order
 * differs). Every function processes rows part, part + parts, ... */
#ifndef QNN16_F32_H
#define QNN16_F32_H
#include "qnn16_hvx.h"
#include <stdint.h>

/* Prepared per-output-channel epilogue: y = min(max(acc + bias, lo), hi). */
typedef struct {
    float *bias, *lo, *hi;
} Qnn16Epilogue;

/* FP16 tensor <-> float32 logical NHWC for output rows [part::parts]. */
int qnn16_gather_f32(float *logical, const Qnn16Tensor *tensor, uint32_t part, uint32_t parts);
int qnn16_scatter_f32(Qnn16Tensor *tensor, const float *logical, uint32_t part, uint32_t parts);

/* Weight preparation (single thread). `w` receives float weights with the
 * scale folded in; `e` receives bias/lo/hi for `channels` outputs. */
void qnn16_prepare_epilogue(Qnn16Epilogue e, const uint8_t *fi, uint32_t tables, uint32_t channels);
void qnn16_prepare_dw3x3(float *w, const uint16_t *weights, const uint8_t *fi, uint32_t tables,
                         uint32_t channels);          /* w[c][9] */
void qnn16_prepare_dw3x3_s2(float *w, const uint16_t *weights, const uint8_t *fi, uint32_t tables,
                            uint32_t channels, uint32_t compact); /* w[c][9]; compact: Q16_DW3X3_S2 p1 */
void qnn16_prepare_dw1x1(float *w, const uint16_t *weights, const uint8_t *fi, uint32_t tables,
                         uint32_t channels);          /* w[c][2] */
void qnn16_prepare_pw_swish(float *w, Qnn16Epilogue e, const uint16_t *weights, const uint8_t *fi, uint32_t ic, uint32_t logical_ic, uint32_t oc);
void qnn16_pw_swish_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oc, const float *w, Qnn16Epilogue e, uint32_t part, uint32_t parts);
void qnn16_prepare_pw_tanh(float *w, Qnn16Epilogue e, const uint16_t *weights, const uint8_t *fi, uint32_t ic, uint32_t logical_ic, uint32_t oc, float unit_slope);
void qnn16_pw_tanh_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oc, const float *w, Qnn16Epilogue e, uint32_t part, uint32_t parts);
void qnn16_prepare_conv3x3_s2(float *w, const uint16_t *weights, const uint8_t *fi, uint32_t tables,
                              uint32_t ic, uint32_t oc);
void qnn16_conv3x3_s2_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oh, uint32_t ow,
                          uint32_t oc, const float *w, Qnn16Epilogue e, uint32_t part, uint32_t parts);
void qnn16_prepare_conv3x3(float *w, const uint16_t *weights, const uint8_t *fi, uint32_t tables,
                           uint32_t ic, uint32_t logical_ic, uint32_t oc); /* w[co][9][logical_ic] */
void qnn16_prepare_pw(float *w, const uint16_t *weights, const uint8_t *fi, uint32_t tables,
                      uint32_t ic, uint32_t logical_ic, uint32_t oc);      /* w[co][logical_ic] */

void qnn16_dw3x3_f32(float *y, const float *x, Qnn16Shape xs, const float *w, Qnn16Epilogue e,
                     uint32_t part, uint32_t parts);
void qnn16_dw3x3_s2_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oh, uint32_t ow, uint32_t channels,
                        const float *w, Qnn16Epilogue e, uint32_t part, uint32_t parts);
void qnn16_dw1x1_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oc, const float *w,
                     Qnn16Epilogue e, uint32_t part, uint32_t parts);
void qnn16_conv3x3_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oc, const float *w,
                       Qnn16Epilogue e, uint32_t part, uint32_t parts);
void qnn16_pw_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oc, const float *w,
                  Qnn16Epilogue e, uint32_t part, uint32_t parts);
void qnn16_s2d_f32(float *y, const float *x, Qnn16Shape xs, uint32_t oh, uint32_t ow, uint32_t part,
                   uint32_t parts);
#endif
