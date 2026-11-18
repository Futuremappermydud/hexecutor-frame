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
    HX_SOFTMAX, /* reserved 2026-11-17: softmax over channels per pixel; no p fields */
    HX_SLICE    /* reserved: p[0..2] = h, w, channel start; output is the sub-box */,
    HX_ZEROINSERT, /* reserved: zero-insert upsample, p[0]=sh p[1]=sw (ConvTranspose) */
    HX_TRANSPOSE,  /* permute (h,w,c) axes, p[0..2] = source axis of each output axis */
    HX_REDUCE,     /* unary reduction over a subset of (h,w,c); see HX_RED_* below */
    HX_NORM,       /* normalisation; see HX_NORM_* below (p[0..4]) */
    /* Kinds 21-31 reserved for coordinator merges (kinds 22-26 uncommitted LLM ops). */
    HX_GATHER = 32, /* token/embedding table lookup; x indices, y table or p[1] const, out */
    HX_WHERE = 33,  /* select x ? y : z; x condition, y/z tensors or scalars */
    HX_ARGMAX = 34, /* argmax over channel axis; p[0] select_last, p[1] dtype */
    HX_TOPK = 35,   /* top-k over channel axis; p[0] k, p[1] largest, p[2] sorted; z indices */
    HX_TILE = 36,   /* expand/tile repeating input over output */
    HX_CAST = 37    /* element type conversion (float32, int32, bool) */
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
    HX_BIN_COUNT
};
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
 * Gather (HX_GATHER, 32): x indices, y table tensor (mode 0) or NONE with constant table offset in p[1]
 * (mode 1), out. p[0] mode, p[1] constant offset, p[2] vocab_size, p[3] index_dtype (0 int32, 1 float32).
 * Where (HX_WHERE, 33): x condition, y X, z Y, out. p[0] X mode (0 tensor y, 1 scalar float bits in p[1],
 * 2 per-channel const in p[1], 3 full-size const in p[1]), p[2] Y mode (0 tensor z, 1 scalar bits in p[3],
 * 2 per-channel const in p[3], 3 full-size const in p[3]).
 * ArgMax (HX_ARGMAX, 34): x in, out [h, w, 1]. p[0] select_last, p[1] output_dtype (0 float32, 1 int32).
 * TopK (HX_TOPK, 35): x in, out values [h, w, K], z optional indices [h, w, K] or NONE. p[0] K,
 * p[1] largest (1 largest, 0 smallest), p[2] sorted (1 sorted, 0 unsorted), p[3] index_dtype (0 float, 1 int).
 * Tile (HX_TILE, 36): x in, out. Tiles/expands x into out where out's h, w, c are integer multiples of x's.
 * Cast (HX_CAST, 37): x in, out. p[0] mode: 0 float->int32, 1 int32->float, 2 bool/int->float, 3 float->bool.
 * z is reserved; all unused fields must be zero (y/z use HXFG_NONE). Conv/depthwise only: z is
 * HXFG_NONE (dilation 1,1) or (dh << 8) | dw with 1 <= dh, dw <= 8 and at least one above 1; the
 * effective kernel extent is (k - 1) * d + 1 and tap (ky, kx) reads input row oy * sh + ky * dh -
 * pad_top, column ox * sw + kx * dw - pad_left. */
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
