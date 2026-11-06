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
#define QDC_VERSION 2u
#define QDC_OP_CONV 1u
#define QDC_OP_EXIT 2u
#define QDC_BARRIER_OFFSET 64u
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
enum { QDC_OK=0, QDC_BAD_HEADER=1, QDC_BAD_SHAPE=2, QDC_BAD_RANGE=3, QDC_IDLE_EXIT=4 };
#endif
