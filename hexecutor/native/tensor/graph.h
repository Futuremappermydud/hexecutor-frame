/* SPDX-License-Identifier: LGPL-3.0-or-later */
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
#define HXFG_MAX_CONSTANT_BYTES 0x20000000u /* 512 MiB per graph */
#define HXFG_CONSTANT_REGION_END 0x6000000u
/* Large pools (LFM-class models) live above the first 128 MiB, where nothing of ours is placed; the
 * engine must map that far (QDO_SLOTS) and the lease must cover it. */
#define HXFG_HIGH_CONSTANT_START 0x8000000u
#define HXFG_HIGH_CONSTANT_END 0x60000000u

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
    HX_MATMUL,     /* 2026-11-17: runtime matrix product, see the HX_MATMUL note below */
    HX_RMSNORM,    /* 22: RMSNorm, p[0]=eps float bits, p[1]=gamma constant, p[2]=beta constant or HXFG_NONE */
    HX_ROPE,       /* 23: Rotary position embedding, p[0]=head_dim, p[1]=heads, p[2]=pos/table, p[3]=mode */
    HX_SHORTCONV,  /* 24: Causal short conv with persistent state y, p[0]=k taps, p[1]=w, p[2]=bias, p[3]=act */
    HX_LINEAR_Q,   /* 25: Quantized linear matvec (Q4_0 / INT8), p[0]=fmt, p[1]=w, p[2]=bias, p[3]=K, p[4]=N, p[5]=act */
    HX_KV_WRITE,   /* 26: KV cache write/append at pos: x [1,Lin,C], out [1,Lmax,C], y pos tensor or p[0] */
    HX_SDPA = 27,  /* 27: fused scaled dot-product attention (Q, K, V) */
    HX_ATTN_SOFTMAX = 28, /* 28: attention softmax with key-axis and causal/additive mask */
    HX_GATHER_ELEMENTS = 29, /* 29: GatherElements: x data, y indices, out; axis p[0] */
    HX_GATHER_ND = 30,       /* 30: GatherND: x data, y indices, out; batch_dims p[0] */
    HX_SCATTER_ND = 31,      /* 31: ScatterND: x data, y indices, z updates, out; p[0] in-place/copy mode */
    HX_GATHER = 32, /* token/embedding table lookup; x indices, y table or p[1] const, out */
    HX_WHERE = 33,  /* select x ? y : z; x condition, y/z tensors or scalars */
    HX_ARGMAX = 34, /* argmax over channel axis; p[0] select_last, p[1] dtype */
    HX_TOPK = 35,   /* top-k over channel axis; p[0] k, p[1] largest, p[2] sorted; z indices */
    HX_TILE = 36,   /* expand/tile repeating input over output */
    HX_CAST = 37,   /* element type conversion (float32, int32, bool) */
    HX_ARGMIN = 38  /* argmin over channel axis; p[0] select_last, p[1] dtype */
};
#define HX_TENSOR_PERSISTENT 1u

/* Tensor element types (HxTensor.dtype). */
enum {
    HX_DTYPE_FLOAT32 = 0,
    HX_DTYPE_INT32 = 1,
    HX_DTYPE_BOOL = 2,
    HX_DTYPE_COUNT = 3
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
    HX_ACT_LOG,
    HX_ACT_RECIPROCAL,
    HX_ACT_FLOOR,
    HX_ACT_CEIL,
    HX_ACT_SIN,
    HX_ACT_COS,
    HX_ACT_NOT,
    HX_ACT_SIGN,
    HX_ACT_ROUND,
    HX_ACT_ELU,     /* x < 0 ? alpha*(exp(x)-1) : x */
    HX_ACT_SELU,    /* fixed ONNX defaults alpha 1.67326319, gamma 1.05070102 */
    HX_ACT_CELU,    /* max(0,x) + min(0, alpha*(exp(x/alpha)-1)) */
    HX_ACT_SOFTPLUS,
    HX_ACT_MISH,
    HX_ACT_SOFTSIGN,
    HX_ACT_LOGSIGMOID,
    HX_ACT_SINH,
    HX_ACT_COSH,
    HX_ACT_COUNT
};
/* HX_BINARY p[0] operations. */
enum {
    HX_BIN_ADD = 0,
    HX_BIN_SUB,
    HX_BIN_MUL,
    HX_BIN_DIV,
    HX_BIN_MIN,
    HX_BIN_MAX,
    HX_BIN_POW,
    HX_BIN_EQUAL,
    HX_BIN_GREATER,
    HX_BIN_LESS,
    HX_BIN_GE,
    HX_BIN_LE,
    HX_BIN_AND,
    HX_BIN_OR,
    HX_BIN_COUNT
};
#define HX_BIN_GREATER_EQUAL HX_BIN_GE
#define HX_BIN_LESS_EQUAL HX_BIN_LE
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
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#endif
typedef struct {
    uint32_t offset, h, w, c, stride;
    union {
        uint32_t reserved[3];
        struct {
            uint32_t flags;    /* bit 0: HX_TENSOR_PERSISTENT */
            uint32_t state_id; /* name hash or state id */
            uint32_t dtype;    /* HX_DTYPE_* (0: float32, 1: int32, 2: bool) */
        };
    };
} HxTensor;
#ifdef __clang__
#pragma clang diagnostic pop
#endif
/* Conv/depthwise p: kh,kw,sh,sw,pad_top,pad_left,weights,bias,activation.
 * p[9]/p[10] optionally declare prepacked fp16 HMX weights and K tile count.
 * Activation: 0 none, 1 ReLU, 2 ReLU[-1,1], 3 ReLU6.
 * Convolution weights [kh,kw,input C,output stride]; depthwise [kh,kw,stride].
 * Add/concat use y. PReLU p[0] is the padded channel slope constant offset.
 * Pool p: kh,kw,sh,sw,pad_top,pad_left. Pad p: top,left,channels_before.
 * Reshape p[0]: 0 HWC-order copy (equal elements); 1 NCHW flatten, source HxWxC to 1x1xN;
 * 2 NCHW unflatten, source 1x1xN to HxWxC (n = c*H*W + h*W + w). Transpose p[0..2]: source axis
 * of each output axis. Resize p: p[0] align_corners, p[1] half_pixel_centers; p[2] nearest (0 bilinear,
 * 1 nearest); p[3] coordinate_transformation_mode (0 asymmetric, 1 align_corners, 2 half_pixel,
 * 3 pytorch_half_pixel); p[4] nearest_mode (0 floor, 1 round_prefer_floor, 2 round_prefer_ceil, 3 ceil);
 * p[5] cubic (1 bicubic Catmull-Rom). Concat p[0]: axis in HWC (0..2).
 * Act (kind HX_ACT): p[0] function, p[1]/p[2] float32 bit patterns alpha/beta (leaky uses alpha,
 * hardsigmoid alpha and beta; others zero). Same shape in and out.
 * Binary (HX_BINARY): p[0] operation, p[1] mode (0: y is a tensor; 1: y is NONE and p[2] is a
 * constant per-channel float vector of stride floats; 2: y is NONE and p[2] is a full-size constant
 * tensor of shape [b->h, b->w, b->stride]), p[3] reversed (computes y op x; with the constant, c op x).
 * Each of y's h, w, c is the output's or 1 (NumPy broadcast); x and the output have the same shape.
 * Avg pool p: pool fields as max pool, p[6] count_include_pad.
 * GatherElements (HX_GATHER_ELEMENTS, 29): x data, y indices (same rank as x/out), out.
 * p[0] axis (0: H, 1: W, 2: C), p[1] index_dtype (0: int32, 1: float32),
 * p[2] mode (0: both tensors; 1: x in constant pool at p[3], y is indices; 2: y in constant pool at p[3], x is data).
 * GatherND (HX_GATHER_ND, 30): x data, y indices, out.
 * p[0] batch_dims (default 0), p[1] index_dtype (0: int32, 1: float32),
 * p[2] mode (0: both tensors; 1: x in constant pool at p[3], y is indices; 2: y in constant pool at p[3], x is data),
 * p[3] constant offset, p[4] tuple_len (last dim of indices, 1..3), p[5] slice_elements (size of gathered slice).
 * ScatterND (HX_SCATTER_ND, 31): x data, y indices, z updates, out.
 * p[0] mode (0: out distinct from x, full copy then update; 1: in-place row update on persistent x==out),
 * p[1] index_dtype (0: int32, 1: float32), p[2] tuple_len (1..3), p[3] slice_elements,
 * p[4] reduction (0: assignment, 1: add, 2: mul, 3: max, 4: min),
 * p[5] const_mask (bit 0: y in constant pool at p[6]; bit 1: z in constant pool at p[7]).
 * Gather (HX_GATHER, 32): x indices or data, y table or NONE, out.
 * p[0] mode (0: x indices, y table tensor; 1: x indices tensor, table in constant pool at p[1];
 *            2: x data tensor, indices in constant pool at p[8], p[9] count).
 * p[1] constant table offset, p[2] vocab_size, p[3] index_dtype (0 int32, 1 float32),
 * p[4] axis (0: outer/vocab, 1: spatial/W, 2: channel/C; or arbitrary axis),
 * p[5] inner_dim (elements after gathered axis; 0 selects default),
 * p[6] outer_dim (elements before gathered axis; 0 selects default),
 * p[7] table_stride (stride of table rows; 0 selects table/output stride),
 * p[8] constant indices offset, p[9] constant indices count.
 * Where (HX_WHERE, 33): x condition, y X, z Y, out. p[0] X mode (0 tensor y, 1 scalar bits in p[1],
 * 2 per-channel const in p[1], 3 full-size const in p[1]), p[2] Y mode (0 tensor z, 1 scalar bits in p[3],
 * 2 per-channel const in p[3], 3 full-size const in p[3]). Integer/bool and float supported.
 * ArgMax/ArgMin (HX_ARGMAX 34 / HX_ARGMIN 38): x in, out [h, w, 1]. p[0] select_last, p[1] output_dtype (0 float32, 1 int32).
 * TopK (HX_TOPK, 35): x in, out values [h, w, K], z optional indices [h, w, K] or NONE. p[0] K,
 * p[1] largest (1 largest, 0 smallest), p[2] sorted (1 sorted, 0 unsorted), p[3] index_dtype (0 float, 1 int).
 * Tile (HX_TILE, 36): x in, out. Tiles/expands x into out where out's h, w, c are integer multiples of x's.
 * Cast (HX_CAST, 37): x in, out. p[0] mode: 0 float->int32, 1 int32->float, 2 bool/int->float, 3 float->bool,
 * 4 int32->bool, 5 bool->int32, 6 identity (int32->int32, float->float).
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
 * p[3] (HX_ATTN_SOFTMAX, axis 2 only) float32 bits of a temperature: the scores are multiplied by it in float,
 *       before the mask, so scores stored scaled down to fit FP16 can be restored (0 selects 1.0f).
 * SDPA (HX_SDPA, 27): Fused scaled dot-product attention: Out = Softmax(Q.K^T * scale + mask) * V.
 * x = Q [B, Lq, D], y = K [B, Lk, D], z = V [B, Lk, Dv], out = Out [B, Lq, Dv].
 * p[0] optional additive mask tensor ID (or HXFG_NONE).
 * p[1] float32 bits of scale (0 selects 1.0 / sqrt(D)).
 * p[2] causal mask flag (1: causal, masks out k > q; 0: none).
 * p[3] batch/head count B when tensors have w = 1.
 * RMSNorm (HX_RMSNORM, 22): x in, out same shape. p[0] eps float bits, p[1] gamma constant offset,
 * p[2] optional beta constant offset or HXFG_NONE.
 * RoPE (HX_ROPE, 23): x in, out same shape. p[0] head_dim, p[1] num_heads, p[2] pos or table offset,
 * p[3] mode (0 NeoX interleaved, 1 split). y optional scalar position tensor.
 * ShortConv (HX_SHORTCONV, 24): x token [1, 1, C], y persistent state tensor [1, k-1, C], z optional
 * gate token [1, 1, C] or HXFG_NONE, out [1, 1, C]. p[0] k taps, p[1] weight offset [k, C],
 * p[2] optional bias offset, p[3] activation. State y is shifted and updated in-place.
 * QuantLinear (HX_LINEAR_Q, 25): x [1, 1, K], out [1, 1, N]. p[0] format (0 Q4_0, 1 Q8_0),
 * p[1] weights offset, p[2] bias offset or HXFG_NONE, p[3] K, p[4] N, p[5] activation.
 * KVWrite (HX_KV_WRITE, 26): x token slice [1, Lin, C], out persistent KV cache [1, Lmax, C] or
 * [Lmax, 1, C]. y optional runtime pos scalar [1, 1, 1] (or HXFG_NONE with static pos in p[0]).
 * Writes x into out at pos along sequence axis. */
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
    uint32_t state, state_bytes; /* Persistent state region in private DDR lease */
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
    uint8_t *state;                /* DSP/Host persistent state; NULL uses arena */
    uint32_t state_bytes;
} HxGraph;
uint32_t hx_tensor_bytes(const HxTensor *t);
/* LINEAR_Q fmt 0/1 rows -> grouped fmt 2/3 in place (same size); k % 64 == 0, n % 32 == 0, scratch k * 34 bytes. */
void hx_lq_group_layout(uint8_t *w, uint32_t fmt, uint32_t k, uint32_t n, uint8_t *scratch);
uint32_t hx_tensor_elements(const HxTensor *t);
void hx_graph_set_state(HxGraph *g, void *state, uint32_t state_bytes);
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
