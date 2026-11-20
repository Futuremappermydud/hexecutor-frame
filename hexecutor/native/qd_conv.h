/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_CONV_H
#define QD_CONV_H
#include <stdint.h>
/* Persistent convolution protocol (payload qd_conv.c behind qd_loader.S).
 * Command at arena offset 0, per-worker QdStatus at QD_STATUS_OFFSET, barrier
 * counter at QDC_BARRIER_OFFSET; all accessed through the uncached view.
 * X: u8 [h][g][wp][4] (4 input channels per pixel word, zero padded).
 * W: s8 [kh][kw][g][n][4], i.e. one int32 word per (tap, group, out channel).
 * C: s32 [oh][n][owp], C = sum x*w over kh*kw*4g, exact in int32. */
#define QDC_MAGIC 0x32564451u /* QDV2 */
#define QDC_VERSION 6u
#define QDC_NATIVE_CM_VERSION 8u /* native HMX crouton load/store, same numerical conversion as v7 */
#define QDC_NATIVE_VERSION 7u /* opt-in native HMX quantization; measured numerical differences */
#define QDC_OP_CONV 1u
#define QDC_OP_EXIT 2u
#define QDC_BARRIER_OFFSET 96u /* three sequence-tagged worker arrival words */
#define QDC_CLOCK_OFFSET 256u /* clock report and saved registers below 1 KB */
#define QDC_NO_CLOCK 1u
#define QDC_QUANTIZED8 2u
#define QDC_USE_HMX 4u
#define QDC_U16 8u /* x: low then high u8 plane of a u16 activation; C: u16 [oh][n/2][owp/32][2][32] */
#define QDC_WEIGHTS_RESIDENT 16u /* HMX: weights reserved[2] are still in VTCM from the previous command */
#define QDC_X32 32u /* HMX: x packed [h][g/8][wp][32] (g a multiple of 8) instead of [h][g][wp][4] */
#define QDC_CROUTON8 64u /* HMX output: [h/8][w/8][n/32][8][8][32].
 * reserved[0] bits 8..15: logical h; bits 16..31: logical w, both multiples
 * of 8. For flattened 1x1 commands, h*w <= owp; otherwise h=oh, w<=owp.
 * Only h*w*n bytes are written; the legacy padded C allocation is unchanged. */
#define QDC_INPUT_VTCM 128u /* resident module only: X and its zero pad already occupy VTCM offset 0 */
#define QDC_HMX_PREPARE_OFFSET 80u
#define QDC_HMX_FINISH_OFFSET 176u
#define QDC_HMX_ERROR_OFFSET 188u
typedef struct { int32_t correction; uint32_t multiplier, preshift; } QdQuant;
/* u16 epilogue: p = acc_hi + ch + ((acc_lo + cl) >> 8), clamped to [0, pmax];
 * out = min(65535, (p*multiplier + round) >> post). */
typedef struct { int32_t ch, cl, pmax; uint32_t multiplier, post, round, reserved[2]; } QdQuant16;
#define QDC_TIMING_OFFSET 2048u /* per worker: copy, wait, compute, store cycles */
#define QDC_VTCM_BASE 0x31000000u /* scratch the stock FLOPS wake also clears */
#define QDC_VTCM_BYTES 0x2d0000u
#define QDC_L2TCM_BASE 0x30000000u
#define QDC_L2TCM_BYTES 0x100000u
#define QDC_COPY_CHUNK 1024u
#define QDC_PB 2u /* pixel vectors per unit */
#define QDC_NB 8u /* output channels per unit */
typedef struct {
    uint32_t magic, version, sequence, op;
    uint32_t h, wp, g, kh, kw, n, oh, owp;
    uint32_t x_offset, w_offset, c_offset, arena_bytes;
    uint32_t idle_mcycles, reserved[3];
} QdConv;
/* QdStatus.reserved: written as each worker leaves the persistent loop. */
#define QDC_EXITED 0x54495845u /* 'EXIT' */
enum { QDC_OK=0, QDC_BAD_HEADER=1, QDC_BAD_SHAPE=2, QDC_BAD_RANGE=3, QDC_IDLE_EXIT=4, QDC_BARRIER_TIMEOUT=5, QDC_CLOCK_ERROR=6 };
#endif
