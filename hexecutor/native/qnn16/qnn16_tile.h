/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* HVX forms of the depthwise and space-to-depth leaves that work directly on
 * QNN's FP16 crouton tiles in VTCM (no logical scratch, no float32 staging).
 *
 * One 128-byte vector is one tile row's pixel pair: 32 channels x 2 pixels,
 * halves interleaved [c][x & 1]. Products accumulate in qf32; scale and bias
 * apply in qf32 before one conversion to FP16 and the optional clamp, as the
 * float path does. Only output lanes inside the logical tensor are stored.
 *
 * Every function processes its share of (channel group, output row) work for
 * part `part` of `parts` and touches only VTCM, never L1-cached memory. */
#ifndef QNN16_TILE_H
#define QNN16_TILE_H
#include "qnn16_hvx.h"
#include <stdint.h>

typedef struct {
    const Qnn16Tensor *x, *out;
    const uint16_t *weights; /* VTCM */
    const uint8_t *fi;       /* VTCM */
    uint32_t tables;
    uint32_t packed_ic; /* conv3x3 / pw only */
    uint32_t compact;   /* dw3x3_s2: diagonal-only weights (Q16_DW3X3_S2 p1) */
} Qnn16TileJob;

/* Whether the tile form supports these operands (32-channel groups without
 * channel padding, blocked storage, non-negative column offset). */
int qnn16_tile_dw3x3_ok(const Qnn16Tensor *x, const Qnn16Tensor *out);
int qnn16_tile_dw3x3_s2_ok(const Qnn16Tensor *x, const Qnn16Tensor *out);
int qnn16_tile_dw1x1_ok(const Qnn16Tensor *x, const Qnn16Tensor *out);
int qnn16_tile_s2d_ok(const Qnn16Tensor *x, const Qnn16Tensor *out);
/* Dense 3x3 (t=0 Xwe heads); `packed_ic` is the weights' packed input channels.
 * Input and output channel counts need not be multiples of 32. */
int qnn16_tile_conv3x3_ok(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t packed_ic);

void qnn16_tile_dw3x3(const Qnn16TileJob *, uint32_t part, uint32_t parts);
void qnn16_tile_dw3x3_s2(const Qnn16TileJob *, uint32_t part, uint32_t parts);
void qnn16_tile_dw1x1(const Qnn16TileJob *, uint32_t part, uint32_t parts);
void qnn16_tile_s2d(const Qnn16TileJob *, uint32_t part, uint32_t parts);
void qnn16_tile_conv3x3(const Qnn16TileJob *, uint32_t part, uint32_t parts);
/* Pointwise with packed Xpe ([co/32][ceil(packed_ic/2) pairs][32 co][2]). */
int qnn16_tile_pw_ok(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t packed_ic);
void qnn16_tile_pw(const Qnn16TileJob *, uint32_t part, uint32_t parts);
/* HMX 3x3 staging. Output tile t = (batch, row block, column block) of `out`
 * needs, per input channel group and weight tile q (QNN's Xpe/Xwe tap order),
 * the 8 x 4 input window that tap reads. An aligned window is the input block
 * itself; any other is copied, shifted, into a scratch block. a[(t - first) *
 * kg * 9 + cig * 9 + q] receives the tile address and scratch holds
 * tiles * kg * 9 blocks. Workers take tiles part::parts. */
typedef struct {
    const Qnn16Tensor *x, *out;
    const uint8_t **a;
    uint8_t *const *scratch;
    uint32_t first, tiles, kg, stride2;
} Qnn16StageJob;
int qnn16_tile_stage3x3_ok(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t stride2, uint32_t kg);
void qnn16_tile_stage3x3(const Qnn16StageJob *, uint32_t part, uint32_t parts);
/* @BlockZap: zero rows [row_start, 8) x columns [col_start, 4) of each block. */
int qnn16_tile_zap(uint8_t *const *blocks, uint32_t count, uint32_t row_start, uint32_t col_start);
/* Tile forms of small elementwise and data-movement ops, rows part::parts.
 * Each returns 1 when it ran, 0 when the operands are unsupported; a 0 from
 * any part means the caller must run the generic path for the whole op. */
int qnn16_tile_minmax(const Qnn16Tensor *x, const Qnn16Tensor *out, float lo, float hi, uint32_t part, uint32_t parts);
int qnn16_tile_mul_scalar(const Qnn16Tensor *x, const Qnn16Tensor *scalar, const Qnn16Tensor *out, uint32_t part,
                          uint32_t parts);
int qnn16_tile_x2s(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts);
int qnn16_tile_transpose_hw(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts);
int qnn16_tile_transpose_wc(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts);
int qnn16_tile_format(const Qnn16Tensor *x, const Qnn16Tensor *out);   /* tiles <-> direct FP16 */
int qnn16_tile_cast_f32(const Qnn16Tensor *x, const Qnn16Tensor *out); /* tiles -> direct float32 */
/* IM2COL (3x3 stride 2, one float32 channel -> 9 FP16 channels), rows part::parts. */
int qnn16_tile_im2col_ok(const Qnn16Tensor *x, const Qnn16Tensor *out);
void qnn16_tile_im2col(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts);
int qnn16_tile_shuffle(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t groups, uint32_t part, uint32_t parts);
int qnn16_tile_to_compact(const Qnn16Tensor *x, const Qnn16Tensor *out); /* RESHAPE to compact direct */
int qnn16_tile_avgpool(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t kernel, uint32_t stride);
/* GroupNorm pieces and SiLU. Welford writes float32 [mean C][sum of squared
 * deviations C] and assumes reciprocal-count weights (Q16_WELFORD_PARTIAL p0). */
int qnn16_tile_welford(const Qnn16Tensor *x, float *out);
int qnn16_tile_instance_norm(const Qnn16Tensor *x, const Qnn16Tensor *out, const float *moments, const uint16_t *scale,
                             const float *bias, uint32_t part, uint32_t parts);
int qnn16_tile_swish(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts);
int qnn16_tile_add_swish(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts);
/* ---- Further tile forms, one source file each (qnn16_tile_<name>.c). Every
 * function returns 1 when it handled its share (rows part::parts) and 0 when
 * the operands are unsupported; that decision must not depend on `part`. ---- */
/* Bilinear resize, v = {y scale, y offset, x scale, x offset} (source = out * scale + offset, clamped). */
int qnn16_tile_resize2x(const Qnn16Tensor *x, const Qnn16Tensor *out, const float v[4], uint32_t part, uint32_t parts);
/* out(y, x) = in(oy + y / 2, ox + x / 2). */
int qnn16_tile_resize_nearest(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t oy, uint32_t ox, uint32_t part,
                              uint32_t parts);
/* out(y, x) = in(y * sy, x * sx). */
int qnn16_tile_decimate(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t sy, uint32_t sx, uint32_t part,
                        uint32_t parts);
/* General transpose, perm[i] = source axis of output axis i; tried before the hw/wc forms. */
int qnn16_tile_transpose(const Qnn16Tensor *x, const Qnn16Tensor *out, const uint32_t perm[4], uint32_t part,
                         uint32_t parts);
/* out = a * b with b the same shape as a or broadcast along any axis; tried after mul_scalar. */
int qnn16_tile_mul(const Qnn16Tensor *a, const Qnn16Tensor *b, const Qnn16Tensor *out, uint32_t part, uint32_t parts);
/* Softmax over channels per pixel. */
int qnn16_tile_softmax(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts);
/* Sum over every axis whose output extent is 1 (serial). */
int qnn16_tile_reduce_sum(const Qnn16Tensor *x, const Qnn16Tensor *out);
/* Mean over every row (out [1,1,W,C] from x [1,H,W,C]), split by pixel pair over the parts. */
int qnn16_tile_reduce_mean_h(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts);
/* L2NORM over axis 1 (H) of [b,h,w,c] tiles: x / sqrt(max(sum of squares over h, epsilon)); units part::parts. */
int qnn16_tile_l2norm(const Qnn16Tensor *x, const Qnn16Tensor *out, float epsilon, uint32_t part, uint32_t parts);
/* float32 direct -> FP16 (direct or tiles), exact round to nearest even. */
int qnn16_tile_cast_f16(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts);
/* FORMAT forms tried before qnn16_tile_format. */
int qnn16_tile_format_fast(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t part, uint32_t parts);
/* dst[r][0..oc) = src[r][start..start + oc) over `rows` rows of xc float32, both in VTCM (serial). */
int qnn16_tile_slice_f32(float *dst, const float *src, uint32_t rows, uint32_t xc, uint32_t oc, uint32_t start);
/* Q16_WELFORD_MERGE of `count` partials ([mean c][m2 c] float32 each, in VTCM) into out (serial). */
int qnn16_tile_welford_merge(float *out, const float *const *partials, uint32_t count, const float *factor,
                             const float *ratio, uint32_t c);
/* Faster qnn16_tile_stage3x3 for the tiles of part::parts; 0 to use the plain form for them. */
int qnn16_tile_stage3x3_fast(const Qnn16StageJob *, uint32_t part, uint32_t parts);
/* PW1X1 on the HMX leaf (qnn16_tile_pw_hmx.c). _ok sets kg (input groups) and ng (output groups) and returns 0
 * for layouts the form does not handle; the other pieces are driven by exec.c, each split over part::parts.
 * Needs ng + kg * ng scratch tiles (weights, then one temporary block per output block of a row chunk). */
/* Q16_CONCAT along H of `count` tile tensors with the output's batch, width and channels (qnn16_tile_concat.c).
 * _ok is 0 when a layout is unsupported or an input shares a VTCM block with the output. */
#define Q16_TILE_VTCM_BLOCKS 4096u
int qnn16_tile_concat_h_ok(const Qnn16Tensor *in, uint32_t count, const Qnn16Tensor *out);
void qnn16_tile_concat_h(const Qnn16Tensor *in, uint32_t count, const Qnn16Tensor *out, uint32_t part, uint32_t parts);
/* Q16_PW_SWISH on a one-pixel tile tensor (qnn16_tile_pw_swish.c); output groups part::parts. The weights and
 * the five-table Fi must be 128-byte aligned in VTCM; `tables` is unused. */
int qnn16_tile_pw_swish_ok(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t packed_ic);
void qnn16_tile_pw_swish(const Qnn16TileJob *, uint32_t part, uint32_t parts);
int qnn16_tile_pw_hmx_ok(const Qnn16Tensor *x, const Qnn16Tensor *y, uint32_t packed_ic, uint32_t *kg, uint32_t *ng);
void qnn16_tile_pw_hmx_tails(const Qnn16Tensor *x, uint32_t kg, uint32_t part, uint32_t parts);
void qnn16_tile_pw_hmx_weights(const uint16_t *w, const Qnn16Tensor *y, uint32_t packed_ic, uint32_t kg, uint32_t ng,
                               uint8_t *const *tiles, uint32_t part, uint32_t parts);
void qnn16_tile_pw_hmx_rows(const Qnn16Tensor *x, uint32_t kg, uint32_t first, uint32_t rows, const uint8_t **a);
void qnn16_tile_pw_hmx_store(const Qnn16Tensor *y, uint32_t ng, uint8_t *const *temps, uint32_t first, uint32_t rows,
                             uint32_t part, uint32_t parts);
#endif
