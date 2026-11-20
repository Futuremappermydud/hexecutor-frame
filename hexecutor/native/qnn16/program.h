/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Q16P v1: a QAIRT 2.50 FP16 HTP graph lowered to a serial handler program.
 *
 * Offline lowering (hexecutor/offline/qnn16_program.py) records QNN's own
 * pool-1 (VTCM) placements, so tensors keep their QNN offsets inside the
 * 8 MiB VTCM window and no layout planning happens on the device. Pool-2
 * constants travel in a separate immutable pool and reach VTCM through
 * Q16_COPY, the program's form of ConvLayer.opt.weights_to_vtcm.
 *
 * Little-endian. No pointers, firmware addresses, model names or embedded
 * weights. External I/O is compact float32 NHWC in a host exchange area. */
#ifndef HEXECUTOR_QNN16_PROGRAM_H
#define HEXECUTOR_QNN16_PROGRAM_H
#include <stdint.h>
#include "qnn16.h"
#include "qnn16_hvx.h"
#ifdef __cplusplus
extern "C" {
#endif

#define Q16P_MAGIC 0x50363151u /* "Q16P" */
#define Q16P_VERSION 1u
#define Q16P_ARENA_VERSION 2u
#define Q16P_VTCM_BYTES 0x800000u
/* The last VTCM tile is never placed by QNN; it holds the HMX completion store. */
#define Q16P_COMPLETION_OFFSET (Q16P_VTCM_BYTES - 2048u)
#define Q16P_MAX_TENSORS 4096u
#define Q16P_MAX_OPS 4096u
#define Q16P_MAX_WORDS (1u << 20)
#define Q16P_MAX_IO 16u
#define Q16P_NONE 0xffffffffu

enum {
    Q16_COPY = 1,  /* out: direct tensor; p0 constant offset, p1 bytes */
    Q16_ZERO,      /* out */
    Q16_CONV1X1,   /* x, y weights, z Fi, out; p0 M, p1 K, p2 N, p3 tables, p4 word index of
                    * M*K input then M*N output VTCM block offsets */
    Q16_DW1X1,     /* x, y, z, out; p0 tables (t=544) */
    Q16_DW3X3,     /* x, y, z, out; p0 tables (t=514) */
    Q16_CONV3X3,   /* x, y, z, out; p0 tables, p1 packed input channels, p2 output channels; p3/p4 word
                    * index and count of dead VTCM blocks for HMX tile staging (0: none) */
    Q16_S2D,       /* x, out */
    Q16_DW3X3_S2,  /* x (space-to-depth), y, z, out; p0 tables; p1 1: y holds only the diagonal,
                    * one 128-byte vector per (group, tap ky*3+kx); 0: QNN's Xwe tiles */
    Q16_IM2COL,    /* x float32 direct, out */
    Q16_INPUT,     /* out float32 direct; p0 slot, p1 y0, p2 x0 (int32) */
    Q16_OUTPUT,    /* x float32 direct or FP16; p0 slot, p1 y0, p2 x0 */
    Q16_FORMAT,    /* x, out */
    Q16_CAST_F32,  /* x FP16, out float32 direct */
    Q16_CAST_F16,  /* x float32 direct, out FP16 */
    Q16_MUL,       /* x, y (same shape or one element), out */
    Q16_MINMAX,    /* x, out; p0 low, p1 high (float32 bits) */
    Q16_SIGMOID,   /* x, out */
    Q16_RESHAPE,   /* x, out: same element count in NHWC order */
    Q16_CONCAT,    /* out; p0 axis, p1 word index of input tensor ids, p2 count */
    Q16_SLICE,     /* x, out; p0 axis, p1 start */
    Q16_SHUFFLE,   /* x, out; p0 groups */
    Q16_TRANSPOSE, /* x, out; p0..p3 source axis of each output axis */
    Q16_AVGPOOL,   /* x, out; p0 kernel, p1 stride (valid) */
    Q16_X2S,       /* x, out: 2x depth-to-space */
    Q16_ZAP,       /* p0 word index of VTCM block offsets, p1 count, p2 row start, p3 column start */
    Q16_PW1X1,     /* x, y, z, out; p0 tables, p1 packed input channels (not a multiple of 32) */
    Q16_SWISH,     /* x, out: x * sigmoid(x) */
    Q16_WELFORD_PARTIAL, /* x FP16, y float32 reciprocal counts, out float32 mean/M2; p0 1: y was checked
                          * offline to be 1 / (i + 1) for pixel i, so the closed form applies */
    Q16_CONV3X3_S2, /* x space-to-depth, y packed weights, z Fi, out; p0 tables, p1 input channels;
                     * p3/p4 staging blocks as Q16_CONV3X3 */
    Q16_WELFORD_MERGE, /* x/y float32 coefficients, p0 word index of p1 partial tensor ids */
    Q16_MOVE, /* x, out: copy equal-sized contiguous tensors */
    Q16_ADD_SWISH, /* two 32-channel halves of x are summed then SiLU */
    Q16_GROUP_MOMENTS, /* x partial, y/z merge coefficients, p0 gamma tensor, p1 inverse count, p2 epsilon, p3 group */
    Q16_DECIMATE, /* x, out; p0/p1 row/column strides */
    Q16_SLICE_F32, /* x, out; p0 channel start, contiguous NHWC float32 */
    Q16_INSTANCE_NORM, /* x, y moments, z interleaved half scale, p0 float32 bias */
    Q16_RESIZE_BILINEAR, /* x, out; p0..p3 float bits: y scale/offset, x scale/offset */
    Q16_SOFTMAX, /* x, out; softmax over channels */
    Q16_PW_SWISH, /* x, packed pointwise weights, five-table Fi, out; p0 input channels */
    Q16_REDUCE_SUM, /* x, out; sum dimensions reduced to one */
    Q16_RESIZE_NEAREST, /* x, out; p0/p1 source row/column offset at 2x */
    Q16_L2NORM, /* x, out; p0 axis, p1 epsilon (float32 bits): x / sqrt(max(sum of squares along the axis, epsilon)) */
    Q16_REDUCE_MEAN, /* x, out; as Q16_REDUCE_SUM, divided by the number of elements reduced */
    Q16_PW_TANH, /* x, packed pointwise weights, six-table Fi, out; p0 input channels, p1 float32 bits of the
                  * table-3 slope that stands for a convolution scale of 1 */
    Q16_AFFINE, /* x FP16, out FP16 or contiguous float32, same dims; p0 scale, p1 bias (float32 bits) */
    Q16_KIND_END
};
enum { Q16_F16 = 0, Q16_F32 = 1, Q16_RAW = 2 };
enum {
    Q16_OK = 0,
    Q16_HEADER,
    Q16_RANGE,
    Q16_SHAPE,
    Q16_OPERATOR,
    Q16_SCRATCH,
    Q16_CANCELLED
};

typedef struct {
    uint32_t magic, version, bytes, tensors, ops, words, inputs, outputs;
    uint32_t tensor_offset, op_offset, word_offset, io_offset;
    uint32_t constant_bytes, scratch_bytes, exchange_bytes, vtcm_bytes;
    uint8_t constant_sha256[32];
    uint32_t reserved[8]; /* v2: reserved[0] is persistent arena bytes within scratch */
} Q16Header; /* 128 bytes */

/* direct=1: contiguous NHWC at VTCM `offset`, `bytes` long; v2 direct=2:
 * contiguous in the persistent prefix of scratch at `offset`. Otherwise FP16
 * crouton blocks: words[block_first .. +block_count) hold VTCM offsets in
 * [b][h/8][w/4][c/32] order of `full`, Q16P_NONE for an absent block. */
typedef struct {
    uint32_t direct, dtype, offset, bytes, block_first, block_count;
    Qnn16Shape dims, full, padding;
    uint32_t reserved[2];
} Q16Tensor; /* 80 bytes */

/* External float32 tensor: compact NHWC at `offset` in the exchange area. */
typedef struct {
    Qnn16Shape dims;
    uint32_t offset, bytes, reserved[2];
} Q16Io; /* 32 bytes */

typedef struct {
    uint32_t kind, x, y, z, out, p[11];
} Q16Op; /* 64 bytes */

typedef struct {
    const Q16Header *header;
    const Q16Tensor *tensors;
    const Q16Op *ops;
    const uint32_t *words;
    const Q16Io *io; /* inputs, then outputs */
} Q16Program;

/* Structural validation shared by the host preflight and the DSP. Checks
 * every range against the image, VTCM, constant pool, exchange area and
 * scratch area, and every operation against its handler's contract. */
uint32_t q16_program_open(Q16Program *, const void *image, uint32_t image_bytes,
                          uint32_t constant_bytes, uint32_t *bad_op);
uint32_t q16_tensor_elements(const Q16Tensor *);
/* Logical scratch bytes an operation needs (validated against scratch_bytes). */
uint32_t q16_op_scratch(const Q16Program *, const Q16Op *);

/* Execution context. vtcm/exchange/scratch/constants are mapped addresses. */
typedef struct Q16Exec Q16Exec;
typedef void (*Q16Conv1x1Fn)(const Qnn16Conv1x1 *);
typedef void (*Q16FinishFn)(void *completion_tile);
/* Runs task(ctx, part, parts) for every part and joins; NULL runs one part. */
typedef void (*Q16Task)(void *ctx, uint32_t part, uint32_t parts);
typedef void (*Q16ParallelFn)(Q16Task task, void *ctx);
struct Q16Exec {
    const Q16Program *program;
    uint8_t *vtcm;
    uint8_t *exchange;
    uint8_t *scratch;
    const uint8_t *constants;
    Q16Conv1x1Fn conv1x1; /* HMX leaf on the DSP, a reference on hosts */
    Q16FinishFn finish;   /* drains the matrix unit before other readers; may be NULL */
    Q16ParallelFn parallel;
    uint32_t matrix_ops;
    uint32_t flags; /* Q16_EXEC_* */
};
/* Use the float32 gather/compute/scatter path even where an HVX tile form exists. */
#define Q16_EXEC_NO_TILE 1u
/* Run one operation. Returns Q16_OK or an error code. */
uint32_t q16_exec_op(Q16Exec *, uint32_t op);
/* Run the consecutive Q16_COPY operations starting at `op` (at most 64) as one
 * job split over the workers, or serially if destinations overlap; *ran
 * receives how many ran. Equivalent to q16_exec_op on each in order. */
uint32_t q16_exec_copies(Q16Exec *, uint32_t op, uint32_t *ran);
/* Runs the operation at `op`, or the run of consecutive operations from it that
 * q16_exec_copies or the batched tile forms take as one job (see exec.c).
 * *ran receives how many operations ran; the result equals q16_exec_op on each in order. */
uint32_t q16_exec_run(Q16Exec *, uint32_t op, uint32_t *ran);
/* Portable reference for Q16_CONV1X1 (float accumulation; reference.c, hosts only). */
void q16_conv1x1_reference(const Qnn16Conv1x1 *);

#ifdef __cplusplus
}
#endif
#endif
