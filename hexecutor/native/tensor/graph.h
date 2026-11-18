#ifndef HEXECUTOR_TENSOR_GRAPH_H
#define HEXECUTOR_TENSOR_GRAPH_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* HXFG v1: little-endian, static batch-one float32 NHWC tensors, channels
 * padded to 32 floats (compact RGB is allowed at convolution inputs). Offsets are relative to the
 * private arena/constants. No model names, firmware addresses, pointers or embedded weights. */
#define HXFG_MAGIC 0x47465848u
#define HXFG_VERSION 1u
#define HXFG_MAX_TENSORS 1024u
#define HXFG_MAX_OPS 1024u
#define HXFG_MAX_IO 16u
#define HXFG_MAX_GRAPHS 8u
#define HXFG_CONFIG_MAGIC 0x53465848u
#define HXFG_RESULT_MAGIC 0x52465848u
#define HXFG_NONE 0xffffffffu
/* Immutable constants live at relative private DDR [32 MiB, 96 MiB): the VTCM backup starts at
 * 96 MiB and nothing of ours is mapped between 64 and 96 MiB. One graph's pool may use all 64. */
#define HXFG_MAX_CONSTANT_BYTES 0x4000000u
#define HXFG_CONSTANT_REGION_END 0x6000000u

enum {
    HX_CONV = 1,
    HX_DEPTHWISE,
    HX_ADD,
    HX_PRELU,
    HX_MAXPOOL,
    HX_PAD,
    HX_RESIZE,
    HX_RESHAPE,
    HX_CONCAT,
    HX_MEAN,
    HX_LOGISTIC,
    HX_ACT,
    HX_BINARY,
    HX_AVGPOOL,
    HX_SOFTMAX, /* softmax over channels per pixel, or axis p[0] with optional mask y */
    HX_SLICE,   /* reserved: p[0..2] = h, w, channel start; output is the sub-box */
    HX_ZEROINSERT, /* reserved: zero-insert upsample, p[0]=sh p[1]=sw (ConvTranspose) */
    HX_TRANSPOSE,  /* permute (h,w,c) axes, p[0..2] = source axis of each output axis */
    HX_REDUCE,     /* unary reduction over a subset of (h,w,c); see HX_RED_* below */
    HX_NORM,       /* normalisation; see HX_NORM_* below (p[0..4]) */
    HX_MATMUL,     /* 21: runtime matrix product (batched/multi-head, see below) */
    HX_SDPA = 27,  /* 27: fused scaled dot-product attention (Q, K, V) */
    HX_ATTN_SOFTMAX = 28 /* 28: attention softmax with key-axis and causal/additive mask */
};
/* HX_ACT p[0] functions. */
enum {
    HX_ACT_RELU = 0,
    HX_ACT_RELU6,
    HX_ACT_HARDSWISH,
    HX_ACT_HARDSIGMOID, /* clamp(alpha*x + beta, 0, 1) */
    HX_ACT_SILU,
    HX_ACT_TANH,
    HX_ACT_LEAKY, /* x < 0 ? alpha*x : x */
    HX_ACT_GELU,  /* exact erf form */
    HX_ACT_EXP,
    HX_ACT_SQRT,
    HX_ACT_ABS,
    HX_ACT_NEG,
    HX_ACT_ERF, /* standalone erf(x), same polynomial as GELU */
    HX_ACT_COUNT
};
/* HX_BINARY p[0] operations. */
enum { HX_BIN_ADD = 0, HX_BIN_SUB, HX_BIN_MUL, HX_BIN_COUNT };
/* HX_NORM p[0] modes: CHANNEL normalises each pixel over its real channels (p[1] = 0); GROUP
 * splits the channels into p[1] equal groups, each normalised over all pixels. */
enum { HX_NORM_CHANNEL = 0, HX_NORM_GROUP, HX_NORM_COUNT };
/* HX_REDUCE p[0] operations; p[1] is a non-zero HWC bitmask (bit0 h, bit1 w, bit2 c). */
enum { HX_RED_MEAN = 0, HX_RED_SUM, HX_RED_MAX, HX_RED_MIN, HX_RED_COUNT };
enum {
    HX_OK = 0,
    HX_HEADER,
    HX_RANGE,
    HX_SHAPE,
    HX_OPERATOR,
    HX_ALIAS,
    HX_UNINITIALIZED,
    HX_CANCELLED
};
typedef struct {
    uint32_t magic, version, bytes, tensors, ops, inputs, outputs, arena_bytes;
    uint32_t constant_bytes, tensor_offset, op_offset, io_offset;
    uint8_t constant_sha256[32];
    uint32_t reserved[44];
} HxHeader;
typedef struct {
    uint32_t offset, h, w, c, stride, reserved[3];
} HxTensor;
/* Conv/depthwise p: kh,kw,sh,sw,pad_top,pad_left,weights,bias,activation.
 * p[9]/p[10] optionally declare prepacked fp16 HMX weights and K tile count.
 * Activation: 0 none, 1 ReLU, 2 ReLU[-1,1], 3 ReLU6.
 * Convolution weights [kh,kw,input C,output stride]; depthwise [kh,kw,stride].
 * Add/concat use y. PReLU p[0] is the padded channel slope constant offset.
 * Pool p: kh,kw,sh,sw,pad_top,pad_left. Pad p: top,left,channels_before.
 * Reshape p[0]: 0 HWC-order copy (equal elements); 1 NCHW flatten, source HxWxC to 1x1xN;
 * 2 NCHW unflatten, source 1x1xN to HxWxC (n = c*H*W + h*W + w). Transpose p[0..2]: source axis
 * of each output axis. Resize p: align_corners,half_pixel_centers; p[2] 1 is nearest (source index floor(out*in/out)
 * per axis, clamped; then p[0] and p[1] are zero). Concat p[0]: axis in HWC (0..2).
 * Act (kind HX_ACT): p[0] function, p[1]/p[2] float32 bit patterns alpha/beta (leaky uses alpha,
 * hardsigmoid alpha and beta; others zero). Same shape in and out.
 * Binary (HX_BINARY): p[0] operation, p[1] mode (0: y is a tensor, 1: y is NONE and p[2] is a
 * constant per-channel float vector of stride floats), p[3] reversed (computes y op x; with the
 * constant, c op x). Each of y's h, w, c is the output's or 1 (NumPy broadcast); x and the output
 * have the same shape. Avg pool p: pool fields as max pool, p[6] count_include_pad.
 * z is reserved; all unused fields must be zero (y/z use HXFG_NONE). Conv/depthwise only: z is
 * HXFG_NONE (dilation 1,1) or (dh << 8) | dw with 1 <= dh, dw <= 8 and at least one above 1; the
 * effective kernel extent is (k - 1) * d + 1 and tap (ky, kx) reads input row oy * sh + ky * dh -
 * pad_top, column ox * sw + kx * dw - pad_left.
 * Matmul (HX_MATMUL, 21): Matrix product extending to batched and multi-head layouts:
 * x = A [B, M, K], y = B [B, K, N] (or transposed), out = Out [B, M, N].
 * p[0] transB: 0 means B is [K, N], 1 means B is [N, K] (e.g. Q.K^T).
 * p[1] float32 bit pattern of scalar factor applied to result (0 selects 1.0f).
 * p[2] batch count B when tensors have w = 1 (if 0 or 1, inferred or single batch).
 * p[3] transA: 0 means A is [M, K], 1 means A is [K, M].
 * All three tensors have K <= 4096. Accumulation is float32.
 * Softmax (HX_SOFTMAX 15 / HX_ATTN_SOFTMAX 28):
 * x in, out same shape. y optional additive mask tensor (or HXFG_NONE).
 * p[0] axis: 0 H, 1 W, 2 C (default 2/channel axis, which is key axis for [B, H, Lq, Lk]).
 * p[1] causal mask flag (1: causal, masks out k > q; 0: none).
 * p[2] sequence length L for causal query indexing if w = 1 (0 selects default).
 * SDPA (HX_SDPA, 27): Fused scaled dot-product attention: Out = Softmax(Q.K^T * scale + mask) * V.
 * x = Q [B, Lq, D], y = K [B, Lk, D], z = V [B, Lk, Dv], out = Out [B, Lq, Dv].
 * p[0] optional additive mask tensor ID (or HXFG_NONE).
 * p[1] float32 bits of scale (0 selects 1.0 / sqrt(D)).
 * p[2] causal mask flag (1: causal, masks out k > q; 0: none).
 * p[3] batch/head count B when tensors have w = 1. */
typedef struct {
    uint32_t kind, x, y, z, out, p[11];
} HxOp;
/* Dilation of a conv/depthwise op: 1 unless z encodes (dh << 8) | dw. Other kinds use 1. */
static inline uint32_t hx_dilation_h(const HxOp *o) {
    return o->z == HXFG_NONE ? 1u : (o->z >> 8) & 0xffu;
}
static inline uint32_t hx_dilation_w(const HxOp *o) {
    return o->z == HXFG_NONE ? 1u : o->z & 0xffu;
}
typedef struct {
    uint32_t image, image_bytes, constants, constant_bytes, arena, arena_bytes;
} HxGraphConfig;
typedef struct {
    uint32_t magic, version, code_bytes, graphs;
    HxGraphConfig graph[HXFG_MAX_GRAPHS];
} HxSessionConfig;
struct HxFp16Plan;
typedef struct {
    const HxHeader *header;
    const HxTensor *tensors;
    const HxOp *ops;
    const uint32_t *io;
    const uint8_t *constants;
    uint8_t *arena;
    const struct HxFp16Plan *fp16; /* DSP-private; NULL selects FP32 storage */
} HxGraph;
uint32_t hx_tensor_bytes(const HxTensor *t);
uint32_t hx_tensor_elements(const HxTensor *t);
uint32_t hx_graph_open(HxGraph *, const void *image, uint32_t image_bytes, const void *constants,
                       uint32_t constant_bytes, void *arena, uint32_t arena_bytes,
                       uint32_t *bad_op);
/* Parts write disjoint output pixels/channels. Caller joins all parts after
 * every op. No allocation, host callback, or interpreter in the live path. */
void hx_execute_op(const HxGraph *, uint32_t op, uint32_t part, uint32_t parts);
/* Scalar activation shared by every backend so semantics cannot diverge. */
float hx_activation(uint32_t function, uint32_t alpha_bits, uint32_t beta_bits, float x);
void hx_pack(float *dst, const float *src, const HxTensor *);
void hx_unpack(float *dst, const float *src, const HxTensor *);
int hx_session_config_valid(const HxSessionConfig *);
#ifdef __cplusplus
}
#endif
#endif
