/* Native handlers for QAIRT 2.50 FP16 HTP run lists (Mercury hand tracking).
 *
 * Operands keep QNN's VTCM formats, so no repacking happens at run time:
 *   activations  FP16 crouton blocks, 2 KiB = h8 x w4 x d32 stored as
 *                [16 pixel pairs][32 channels][2 pixels] (an HMX activation tile),
 *                addressed through per-tensor block tables;
 *   Xpe weights  [co/32][ci/32] tiles of [16 ci pairs][32 co][2 ci];
 *   Fi tables    one 256-byte HMX scale/bias table per 32 output channels
 *                (words 0-31 FP16 scale in the low half, 32-63 FP16 bias in the
 *                high half), loaded unchanged with mxmem2.
 * Every HMX load covers one tile-aligned 2 KiB tile, so none crosses a 4 MiB
 * VTCM TLB page. Callers must drain the matrix stream (qnn16_hmx_finish)
 * before HVX or scalar code reads results. */
#ifndef QNN16_H
#define QNN16_H
#include <stdint.h>

typedef struct {
    const uint8_t *const *a; /* m * k activation block pointers, [m][k] */
    const uint8_t *w;        /* n * k weight tiles, [n][k] */
    const uint8_t *fi;       /* n scale/bias tables */
    uint8_t *const *c;       /* m * n output block pointers, [m][n] */
    uint32_t m, k, n;
    uint32_t tables;         /* scale/bias tables per output group (fi holds n * tables) */
    const uint32_t *acc;     /* cvt.hf = acc(acc[i]) parameter for table i */
} Qnn16Conv1x1;

/* ConvLayer.fp16.s1.tcm, 1x1 Xpe with one activation K tile: t == 0, or the
 * small-input t == 3 variant with partially populated logical input channels.
 * One table per group is linear; two apply clipped activation.
 * Every tile is addressed through its block-table pointer; no contiguity
 * is assumed. Results are within HMX accumulation rounding of QNN's x86
 * simulator (see hexecutor/tests/test_qnn16_conv_sim.c). */
void qnn16_conv1x1(const Qnn16Conv1x1 *p);
/* Same, with p->w a table of n * k weight-tile pointers ([n][k]) instead of contiguous tiles. */
void qnn16_conv1x1_wt(const Qnn16Conv1x1 *p);

/* Clear the accumulator before the first matrix operation of a session. */
void qnn16_hmx_reset(void);

/* Ordered sentinel store into a dedicated 2 KiB VTCM tile, then sync. */
void qnn16_hmx_finish(void *completion_tile);

#endif
