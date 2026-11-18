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
    HX_AVGPOOL
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
    HX_ACT_COUNT
};
/* HX_BINARY p[0] operations. */
enum { HX_BIN_ADD = 0, HX_BIN_SUB, HX_BIN_MUL, HX_BIN_COUNT };
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
 * Resize p: align_corners,half_pixel_centers. Concat p[0]: axis in HWC (0..2).
 * Act (kind HX_ACT): p[0] function, p[1]/p[2] float32 bit patterns alpha/beta (leaky uses alpha,
 * hardsigmoid alpha and beta; others zero). Same shape in and out.
 * Binary (HX_BINARY): p[0] operation, p[1] mode (0: y is a tensor, 1: y is NONE and p[2] is a
 * constant per-channel float vector of stride floats), p[3] reversed (computes y op x; with the
 * constant, c op x). Each of y's h, w, c is the output's or 1 (NumPy broadcast); x and the output
 * have the same shape. Avg pool p: pool fields as max pool, p[6] count_include_pad.
 * z is reserved; all unused fields must be zero (y/z use HXFG_NONE). */
typedef struct {
    uint32_t kind, x, y, z, out, p[11];
} HxOp;
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
