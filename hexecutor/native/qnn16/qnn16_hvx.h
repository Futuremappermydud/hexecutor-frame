/* Scalar reference forms of the QNN FP16 non-matrix leaves.
 *
 * These helpers use logical contiguous NHWC buffers. They are intended as
 * correctness primitives for lowering/run-list code; crouton block-table
 * gathering/scattering belongs to the caller. Offsets and extents are in
 * elements, not bytes.
 */
#ifndef QNN16_HVX_H
#define QNN16_HVX_H
#include <stdint.h>

typedef struct { uint32_t b, h, w, c; } Qnn16Shape;

/* Runtime view matching graph.json tensor.shape/storage. For blocked storage,
 * block_offsets contains the selected tensor blocks in [b][h/8][w/4][c/32]
 * order, as byte offsets from base; UINT32_MAX denotes an absent block.
 * full includes graph padding; padding locates the logical tensor in full.
 * A direct tensor uses direct=1 and contiguous row-major NHWC storage.
 */
typedef struct {
    uint8_t *base;
    const uint32_t *block_offsets;
    uint32_t block_count;
    uint32_t base_bytes;
    uint32_t direct_offset;
    Qnn16Shape dims, full, padding;
    uint32_t direct;
} Qnn16Tensor;

/* Convert between graph-described VTCM tensors and logical contiguous NHWC. */
int qnn16_gather_f16(uint16_t *logical, const Qnn16Tensor *tensor);
int qnn16_scatter_f16(Qnn16Tensor *tensor, const uint16_t *logical);
/* ForceFormat_Flat_c2f and ForceFormat_Crouton_f2c: move only logical lanes
 * between graph-described pool-1 tensors without an intermediate tensor. */
int qnn16_tensor_copy_format(Qnn16Tensor *out, const Qnn16Tensor *in);
/* Cast a pool-1 half tensor (flat or crouton) into compact logical float32. */
int qnn16_tensor_f16_to_f32(float *out, const Qnn16Tensor *in);
/* ConvLayer.opt.inconv_im2col_cast: single-channel float32 padded slice to
 * 3x3/stride-2 FP16 crouton output, preserving QNN's tap order. */
/* Rows part, part + parts, ... of the same operation. */
int qnn16_tensor_im2col_cast_rows(Qnn16Tensor *out, const float *in, Qnn16Shape in_shape, uint32_t part,
                                  uint32_t parts);
int qnn16_tensor_im2col_cast(Qnn16Tensor *out, const float *in,
                              Qnn16Shape in_shape);
/* Descriptor-backed execution helpers. Scratch has dims.b*dims.h*dims.w*dims.c
 * FP16 elements; binary multiply requires a second equally sized scratch. */
int qnn16_tensor_softmax(Qnn16Tensor *out, const Qnn16Tensor *in);
int qnn16_tensor_resize_nearest(Qnn16Tensor *out, const Qnn16Tensor *in, uint32_t oy, uint32_t ox);
int qnn16_tensor_resize(Qnn16Tensor *out, const Qnn16Tensor *in, float sy, float oy, float sx, float ox);
int qnn16_tensor_decimate(Qnn16Tensor *out, const Qnn16Tensor *in, uint32_t sy, uint32_t sx);
int qnn16_tensor_add_swish(Qnn16Tensor *out, const Qnn16Tensor *in);
int qnn16_tensor_swish(Qnn16Tensor *out, const Qnn16Tensor *in);
int qnn16_tensor_sigmoid(Qnn16Tensor *out, const Qnn16Tensor *in, uint16_t *scratch);
int qnn16_tensor_minmax(Qnn16Tensor *out, const Qnn16Tensor *in, uint16_t *scratch,
                        float lo, float hi);
int qnn16_tensor_mul(Qnn16Tensor *out, const Qnn16Tensor *a, const Qnn16Tensor *b,
                     uint16_t *scratch_a, uint16_t *scratch_b);

/* Exact binary16 conversions (round to nearest even), without library calls. */
float qnn16_half_to_float(uint16_t h);
uint16_t qnn16_float_to_half(float f);
void qnn16_copy(void *dst, const void *src, uint32_t bytes);
void qnn16_zero(void *dst, uint32_t bytes);
void qnn16_f32_to_f16(uint16_t *dst, const float *src, uint32_t count);
void qnn16_f16_to_f32(float *dst, const uint16_t *src, uint32_t count);
void qnn16_mul_f16(uint16_t *dst, const uint16_t *a, const uint16_t *b, uint32_t n);
void qnn16_minmax_f16(uint16_t *dst, const uint16_t *src, uint32_t n, float lo, float hi);
void qnn16_sigmoid_f16(uint16_t *dst, const uint16_t *src, uint32_t n);
/* QNN t=544 pairwise depthwise 1x1 convolution over logical NHWC. Each output
 * combines c and c+32. Weights and Fi retain their packed Xpe/mxmem2 formats. */
void qnn16_depthwise1x1_f16(uint16_t *dst, const uint16_t *src, Qnn16Shape shape,
                            uint32_t out_channels, const uint16_t *weights,
                            const uint8_t *fi, uint32_t tables);
/* 1x1 Xpe convolution for packed input-channel counts that are not a
 * multiple of 32: each 32-output group holds ceil(ic/2) channel pairs, so the
 * last K tile is partial and the HMX whole-tile leaf does not apply. */
void qnn16_pointwise_f16(uint16_t *dst, const uint16_t *src, Qnn16Shape shape,
                         uint32_t input_channels, uint32_t out_channels,
                         const uint16_t *weights, const uint8_t *fi, uint32_t tables);
/* QNN t=514 depthwise 3x3 over a logical input that already holds its
 * one-pixel halo (output is h-2 x w-2). Xpe weights are [group][tap
 * ky*3+2-kx][16 channel pairs][even, 0, 0, odd]: 576 halves per group. */
void qnn16_depthwise3x3_f16(uint16_t *dst, const uint16_t *src, Qnn16Shape shape,
                            const uint16_t *weights, const uint8_t *fi, uint32_t tables);
/* ConvLayer_SpaceToDepth_stride2: one 32-channel-aligned logical input to
 * [oh][ow][4C] with channel ((c/32)*4 + q)*32 + c%32 taken from row
 * 2y + q/2, column 2x - q%2, zero outside the input. */
int qnn16_space_to_depth_stride2_f16(uint16_t *dst, const uint16_t *src, Qnn16Shape shape,
                                     uint32_t out_h, uint32_t out_w);
/* ConvLayer.fp16.s2 t=514: depthwise 3x3/stride 2 over the space-to-depth
 * input above (shape is that input, 4C channels; out_h + 1 <= h and
 * out_w + 1 <= w). Xwe holds nine diagonal 32x32 tiles per channel group,
 * group-major. Returns 0, or -1 for an invalid descriptor. */
int qnn16_depthwise3x3_stride2_f16(uint16_t *dst, const uint16_t *src, Qnn16Shape shape,
                                   uint32_t out_h, uint32_t out_w, const uint16_t *weights,
                                   const uint8_t *fi, uint32_t tables);
/* Full 3x3 Xpe/Xwe stride-1 convolution (output h-2 x w-2). input_channels is
 * the packed weight extent and may exceed the logical NHWC channel count;
 * missing logical channels are zero. Output channels need not be a multiple
 * of 32. */
void qnn16_conv3x3_f16(uint16_t *dst, const uint16_t *src, Qnn16Shape shape,
                       uint32_t input_channels, uint32_t out_channels,
                       const uint16_t *weights, const uint8_t *fi, uint32_t tables);
void qnn16_concat_f16(uint16_t *dst, const uint16_t *const *src, const Qnn16Shape *shapes,
                      uint32_t count, uint32_t axis);
void qnn16_slice_f16(uint16_t *dst, const uint16_t *src, Qnn16Shape shape,
                     uint32_t axis, uint32_t start, uint32_t length);
void qnn16_transpose_hw_f16(uint16_t *dst, const uint16_t *src, Qnn16Shape shape);
/* General logical 4D transpose: perm[i] is the source axis for output axis i. */
void qnn16_transpose_f16(uint16_t *dst, const uint16_t *src, Qnn16Shape shape,
                         const uint32_t perm[4]);
void qnn16_channel_shuffle_f16(uint16_t *dst, const uint16_t *src, Qnn16Shape shape,
                               uint32_t groups);
void qnn16_avgpool2x2_f16(uint16_t *dst, const uint16_t *src, Qnn16Shape shape,
                          uint32_t stride, uint32_t ceil_mode);
/* Nonoverlapping valid average pool used by Mercury's 3x3 and 6x6 heads. */
void qnn16_avgpool_valid_f16(uint16_t *dst, const uint16_t *src, Qnn16Shape shape,
                             uint32_t kernel, uint32_t stride);
/* x2s_opt: 2x depth-to-space in DCR order; shape is input shape. */
void qnn16_x2s_f16(uint16_t *dst, const uint16_t *src, Qnn16Shape shape);

/* *InputSlicePad / *InputSlice: dst[b][y][x] = src[b][y + y0][x + x0], zero
 * outside src. The origin is the slice start minus the leading pad from the
 * operation's s4 parameters (Mercury bands use y0 = 16k - 1, x0 = -1). */
int qnn16_window_f32(float *dst, Qnn16Shape out, const float *src, Qnn16Shape in,
                     int32_t y0, int32_t x0);
/* *OutputSlice: copy a float32 tile into the external output at (y0, x0). */
int qnn16_place_f32(float *dst, Qnn16Shape out, const float *src, Qnn16Shape tile,
                    uint32_t y0, uint32_t x0);
/* @BlockZap: zero rows >= row_start and columns >= col_start of each FP16
 * crouton block. qnn16_zap_region decodes a zap table code: 0xc0 | k clears
 * the last k rows; 0x50, 0x4c and 0x54 clear columns 3, 2-3 and 1-3. Other
 * codes return -1. The lane mask did not change the result in any traced zap. */
int qnn16_zap_region(uint32_t code, uint32_t *row_start, uint32_t *col_start);
int qnn16_zap_blocks(uint8_t *const *blocks, uint32_t count, uint32_t row_start,
                     uint32_t col_start);

#endif
