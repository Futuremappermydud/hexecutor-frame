/* Whole-token Qwen3-0.6B forward pass on the Frame CDSP (HVX, 3 workers).
 *
 * Weights, RoPE table and KV cache live in the cdsp_mem buffer (VA = CDSP
 * IOVA 0xa0000000, mapped here with 16 MB JTLB entries). The host writes one
 * command per batch of up to 8 tokens; the workers run every layer and the
 * output head, then return logits and the greedy token. See dsp_qwen.py for
 * the image layout and the command protocol.
 *
 * Matrices use a decode-oriented Q4_0 layout, per group of 32 output rows:
 *   codes:  K/32 blocks x 4 vectors; vector j, lane r, byte i holds
 *           q[r][8j+i] | q[r][8j+4+i] << 4 (unsigned codes 0..15)
 *   scales: K/64 vectors; halfword 2r = d[r][2p], 2r+1 = d[r][2p+1] (fp16)
 * Activations are quantized per 32-block to int8 (Q8_0 style), so each
 * block is 8 `vrmpy(ub codes, int8 activation word)` per 32 rows.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#define ARENA 0xf1800000u          /* uncached view of the 8 MB arena */
#define BIG 0xa0000000u            /* cdsp_mem, VA = IOVA */
#define BIG_SLOTS 48u              /* 768 MB in 16 MB pages */
#define FIRST_SLOT 64u
#define L2 0x30000000u             /* 1 MB L2-TCM scratch */
#define VTCM 0x31000000u

#define MAGIC 0x334e5751u          /* 'QWN3' */
#define IMAGE_MAGIC 0x50534451u    /* 'QDSP' */
#define EXIT_MAGIC 0x32564451u
#define MAX_TOK 32u
#define SMALL_TOK 8u               /* batches up to this use L2-TCM buffers */
#define DIM 1024u
#define QKV 4096u
#define FFN 3072u
#define VOCAB 151936u
#define LAYERS 28u

/* Arena layout */
#define CMD_OFF 0u
#define STATUS_OFF 128u            /* 3 x {done_seq, 0, 0, exit marker} */
#define OUT_OFF 2048u              /* thread-0 report words */
#define EMB_OFF 0x20000u           /* B raw Q4_0 embedding rows (576 B) */
#define LOGITS_OFF 0x100000u       /* fp32 logits of the last token (flags & 2) */
#define ARGMAX_OFF 0x5000u         /* per-token greedy ids (flags & 8) */
#define MAX_VERIFY 6u              /* tokens with logits per command (flags & 8) */

/* Activation buffers (bytes from their base). Batches of up to SMALL_TOK
 * tokens use L2-TCM (fast scalar access); larger prefill batches use VTCM
 * at VTCM_ACT, with 1 MB weight-ring slots instead of 2 MB. */
#define X_OFF      0x00000u        /* residual [B][1024] f32 */
#define XN_OFF     0x08000u        /* normed   [B][1024] f32 */
#define QKV_OFF    0x10000u        /* [B][4096] f32 */
#define ATT_OFF    0x30000u        /* [B][2048] f32 */
#define GU_OFF     0x40000u        /* [B][6144] f32; also o/down output */
#define H_OFF      0x70000u        /* [B][3072] f32 */
#define QX_OFF     0x88000u        /* quantized activations, QRow[B] */
#define VTCM_ACT   (VTCM + 0x400000u)
#define BIG_SCALE  4u              /* VTCM offsets = L2 offsets x 4 (32 rows) */
/* Fixed L2-TCM layout */
#define NORM_OFF   0x90000u        /* this layer's norms: attn, q, k, ffn (9 KB) */
#define ROPE_OFF   0x93000u        /* [32][64][cos, sin] for this command */
#define EMBR_OFF   0x98000u        /* raw embedding rows (32 x 576 B) */
#define OUTN_OFF   0x9d000u        /* output norm */
#define LOCAL_OFF  0x9e000u        /* per-thread argmax */
#define SCORE_OFF  0xb0000u        /* 3 x 2 x 2048 f32 attention scores */

typedef struct {
    uint32_t magic, version, layers, ctx;
    uint32_t rope, out_norm, out, kv, kv_layer_stride;
    uint32_t small, small_bytes;   /* scalar-read tensors (RoPE, norms) */
    uint32_t reserved[5];
    struct { uint32_t qkv, o, gateup, down, attn_norm, q_norm, k_norm, ffn_norm; } layer[LAYERS];
} Image;

/* Per token: int8 codes [K], per 64-block pair hf(dx) words, 8*sum(q) per block. */
/* One quantized activation row: int8 codes, per 64-block pair hf(dx)
 * words, 8 * sum(q) per 32-block. */
typedef struct {
    int8_t q[FFN];
    uint32_t dxp[FFN / 64];
    int32_t s8[FFN / 32];
} QRow;

/* Freestanding: the compiler's runtime division helpers, shift-subtract. */
uint32_t __hexagon_udivsi3(uint32_t n, uint32_t d) {
    uint32_t q = 0, r = 0;
    for (int i = 31; i >= 0; i--) {
        r = (r << 1) | ((n >> i) & 1u);
        if (r >= d) { r -= d; q |= 1u << i; }
    }
    return q;
}

uint32_t __hexagon_umodsi3(uint32_t n, uint32_t d) {
    return n - __hexagon_udivsi3(n, d) * d;
}

static inline uint64_t cycles(void) {
    uint64_t c; __asm__ volatile("%0 = upcycle" : "=r"(c)); return c;
}

static inline void fence(void) { __asm__ volatile("barrier" ::: "memory"); }

/* Barrier for the 3 workers in L2-TCM: each worker publishes how many
 * barriers it has reached; nobody touches DDR, which matters while the DMA
 * engine saturates the DSP's DDR port. Returns nonzero on timeout/error. */
#define SYNC_OFF 0x9f000u                  /* flags[3], error, local gens[3] */
static int sync3(void) {
    volatile uint32_t *f = (volatile uint32_t *)(0x30000000u + SYNC_OFF);
    uint32_t tid;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    uint64_t t0 = cycles();
    uint32_t gen = ++f[4u + tid];
    fence();
    f[tid] = gen;
    uint64_t end = t0 + 4000000000ull;
    while (f[0] < gen || f[1] < gen || f[2] < gen) {
        if (f[3] || cycles() > end) { f[3] = 1; break; }
    }
    ((volatile uint32_t *)(ARENA + OUT_OFF + 192u))[tid] += (uint32_t)((cycles() - t0) >> 4);
    return (int)f[3];
}

static uint16_t float_to_half(float f) {
    union { uint32_t u; float f; } v;
    v.f = f;
    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 255u) - 112;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        uint32_t x = (mant | 0x800000u) >> (uint32_t)(1 - exp);
        uint32_t rem = x & 0x1fffu;
        x = (x >> 13) + (rem > 0x1000u || (rem == 0x1000u && ((x >> 13) & 1u)));
        return (uint16_t)(sign | x);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t rem = mant & 0x1fffu;
    uint32_t body = ((uint32_t)exp << 10) | (mant >> 13);
    body += rem > 0x1000u || (rem == 0x1000u && (body & 1u));
    return (uint16_t)(sign | body);
}

static float half_to_float(uint16_t h) {
    union { uint32_t u; float f; } v;
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16, exp = (h >> 10) & 31u, mant = h & 1023u;
    if (!exp) { float x = (float)mant * 0x1p-24f; return sign ? -x : x; }
    v.u = sign | ((exp + 112u) << 23) | (mant << 13);
    return v.f;
}

static float rsqrt(float value) {
    union { uint32_t bits; float number; } u;
    u.number = value;
    u.bits = 0x5f3759dfu - (u.bits >> 1);
    float y = u.number, half = 0.5f * value;
    y = y * (1.5f - half * y * y);
    y = y * (1.5f - half * y * y);
    y = y * (1.5f - half * y * y);
    return y;
}

/* Scalar reference for vexpf (bench/qwen_ops_test.c). */
static __attribute__((unused)) float exp_approx(float value) {
    if (value < -80.0f) return 0.0f;
    if (value > 80.0f) value = 80.0f;
    float scaled = value * 1.4426950408889634f;
    int32_t nearest = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    float r = scaled - (float)nearest;
    float polynomial = 1.0f + r * (0.69314718056f + r * (0.24022650695f +
        r * (0.05550410866f + r * (0.00961812911f + r * (0.00133335581f +
        r * 0.00015403530f)))));
    union { uint32_t bits; float number; } scale;
    scale.bits = (uint32_t)(nearest + 127) << 23;
    return scale.number * polynomial;
}

/* Rows 0, 1, ... count-1 of x get RMSNorm with weights w (width wide).
 * Scalar reference for rmsnorm_hvx (bench/qwen_ops_test.c). */
static __attribute__((unused)) void rmsnorm(float *out, const float *x, const float *w, uint32_t width, uint32_t count) {
    for (uint32_t r = 0; r < count; r++) {
        const float *row = x + r * width;
        float sum = 0.0f;
        for (uint32_t i = 0; i < width; i++) sum += row[i] * row[i];
        float f = rsqrt(sum / (float)width + 1e-6f);
        for (uint32_t i = 0; i < width; i++) out[r * width + i] = row[i] * f * w[i];
    }
}

/* Quantize rows [t0, t1) of x (stride `stride`, width k) into qx.
 * Scalar reference for quantize_hvx (bench/qwen_ops_test.c). */
static __attribute__((unused)) void quantize(QRow *qx, const float *x, uint32_t stride, uint32_t k, uint32_t t0, uint32_t t1) {
    for (uint32_t t = t0; t < t1; t += 3u) {
        const float *row = x + t * stride;
        uint16_t dh[2] = {0, 0};
        for (uint32_t b = 0; b < k / 32u; b++) {
            float m = 0.0f;
            for (uint32_t i = 0; i < 32u; i++) {
                float a = row[b * 32u + i];
                a = a < 0.0f ? -a : a;
                if (a > m) m = a;
            }
            uint16_t d16 = float_to_half(m / 127.0f);
            float d = half_to_float(d16);
            float inv = d > 0.0f ? 1.0f / d : 0.0f;
            int32_t sum = 0;
            for (uint32_t i = 0; i < 32u; i++) {
                float v = row[b * 32u + i] * inv;
                int32_t q = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                qx[t].q[b * 32u + i] = (int8_t)q;
                sum += q;
            }
            qx[t].s8[b] = 8 * sum;
            dh[b & 1u] = d16;
            if (b & 1u) qx[t].dxp[b / 2u] = (uint32_t)dh[0] | ((uint32_t)dh[1] << 16);
        }
    }
}

static inline HVX_Vector int_to_sf(HVX_Vector v) {
    const HVX_Vector magic = Q6_V_vsplat_R(0x4b400000);
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(Q6_Vw_vadd_VwVw(v, magic), magic));
}

/* HVX versions of the per-element phases. Arithmetic goes through qf32 (sf
 * vcvt does not work on this firmware), so values can differ from the scalar
 * reference functions above in the last bits; bench/qwen_ops_test.c compares
 * them on the DSP. Rows, heads and weights are 128-byte aligned; QRow is not
 * a multiple of 128 bytes, so its codes are stored unaligned. */
static inline HVX_Vector vsplatf(float f) {
    union { float f; uint32_t u; } v; v.f = f; return Q6_V_vsplat_R(v.u);
}
static inline float vlane0f(HVX_Vector v) {
    union { uint32_t u; float f; } r; r.u = Q6_R_vextract_VR(v, 0); return r.f;
}
static inline HVX_Vector vmulf(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}
static inline HVX_Vector vaddf(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
static inline HVX_Vector vsubf(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
}
/* Float bits <-> signed ints with the same ordering (involution). */
static inline HVX_Vector vord(HVX_Vector v) {
    return Q6_V_vxor_VV(v, Q6_V_vand_VV(Q6_Vw_vasr_VwR(v, 31), Q6_V_vsplat_R(0x7fffffff)));
}
/* Horizontal reductions; the result is splat to every lane. */
static inline HVX_Vector vhsumf(HVX_Vector v) {
    for (int s = 64; s >= 4; s >>= 1) v = vaddf(v, Q6_V_vror_VR(v, s));
    return v;
}
static inline HVX_Vector vhsumw(HVX_Vector v) {
    for (int s = 64; s >= 4; s >>= 1) v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, s));
    return v;
}
static inline HVX_Vector vhmaxw(HVX_Vector v) {
    for (int s = 64; s >= 4; s >>= 1) v = Q6_Vw_vmax_VwVw(v, Q6_V_vror_VR(v, s));
    return v;
}
static inline HVX_Vector vhminw(HVX_Vector v) {
    for (int s = 64; s >= 4; s >>= 1) v = Q6_Vw_vmin_VwVw(v, Q6_V_vror_VR(v, s));
    return v;
}

static void rmsnorm_hvx(float *out, const float *x, const float *w, uint32_t width) {
    const HVX_Vector *xv = (const HVX_Vector *)x, *wv = (const HVX_Vector *)w;
    HVX_Vector *ov = (HVX_Vector *)out;
    HVX_Vector acc = Q6_V_vzero();
    for (uint32_t i = 0; i < width / 32u; i++)
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(xv[i], xv[i]));
    float sum = vlane0f(vhsumf(Q6_Vsf_equals_Vqf32(acc)));
    HVX_Vector f = vsplatf(rsqrt(sum / (float)width + 1e-6f));
    for (uint32_t i = 0; i < width / 32u; i++) ov[i] = vmulf(vmulf(xv[i], f), wv[i]);
}

static void add_hvx(float *x, const float *add, uint32_t width) {
    HVX_Vector *xv = (HVX_Vector *)x;
    const HVX_Vector *av = (const HVX_Vector *)add;
    for (uint32_t i = 0; i < width / 32u; i++) xv[i] = vaddf(xv[i], av[i]);
}

/* As quantize(): per 32-block absmax scale d = fp16(max/127) (scalar, same
 * expressions), codes round-half-away(x / d) clamped to +-127. One vector is
 * one block; |v| + 0.5 is truncated from its exponent and mantissa bits. */
static inline HVX_Vector quant_block(HVX_Vector v, float inv) {
    const HVX_Vector zero = Q6_V_vzero();
    HVX_Vector p = vmulf(v, vsplatf(inv));
    HVX_Vector a = vaddf(Q6_V_vand_VV(p, Q6_V_vsplat_R(0x7fffffff)), vsplatf(0.5f));
    HVX_Vector mant = Q6_V_vor_VV(Q6_V_vand_VV(a, Q6_V_vsplat_R(0x7fffff)), Q6_V_vsplat_R(0x800000));
    HVX_Vector shift = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(150), Q6_Vw_vasr_VwR(a, 23));
    shift = Q6_Vw_vmin_VwVw(shift, Q6_V_vsplat_R(31));
    HVX_Vector t = Q6_Vw_vasr_VwVw(mant, shift);
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VwVw(zero, p);
    HVX_Vector q = Q6_V_vmux_QVV(neg, Q6_Vw_vsub_VwVw(zero, t), t);
    q = Q6_Vw_vmin_VwVw(q, Q6_V_vsplat_R(127));
    return Q6_Vw_vmax_VwVw(q, Q6_V_vsplat_R(-127));
}

/* Blocks 4g .. 4g+3 of one row: one packed 128-byte code store, two dxp
 * words, four s8 sums, so groups never share a written word. */
static void quant_group(QRow *qr, const HVX_Vector *row, uint32_t g) {
    uint16_t dh[2] = {0, 0};
    HVX_Vector q[4];
    for (uint32_t j = 0; j < 4u; j++) {
        uint32_t b = 4u * g + j;
        HVX_Vector v = row[b];
        union { uint32_t u; float f; } m;
        m.u = Q6_R_vextract_VR(vhmaxw(Q6_V_vand_VV(v, Q6_V_vsplat_R(0x7fffffff))), 0);
        uint16_t d16 = float_to_half(m.f / 127.0f);
        float d = half_to_float(d16);
        float inv = d > 0.0f ? 1.0f / d : 0.0f;
        q[j] = quant_block(v, inv);
        qr->s8[b] = 8 * (int32_t)Q6_R_vextract_VR(vhsumw(q[j]), 0);
        dh[j & 1u] = d16;
        if (j & 1u) qr->dxp[b / 2u] = (uint32_t)dh[0] | ((uint32_t)dh[1] << 16);
    }
    HVX_Vector h01 = Q6_Vh_vpack_VwVw_sat(q[1], q[0]);
    HVX_Vector h23 = Q6_Vh_vpack_VwVw_sat(q[3], q[2]);
    *(HVX_UVector *)(qr->q + g * 128u) = Q6_Vb_vpack_VhVh_sat(h23, h01);
}

/* Rows [0, ntok) of x (stride `stride`, width k); the three workers take
 * interleaved 4-block groups of every row, so one decode row is split too. */
static void quantize_split(QRow *qx, const float *x, uint32_t stride, uint32_t k, uint32_t ntok, uint32_t tid) {
    for (uint32_t t = 0; t < ntok; t++)
        for (uint32_t g = tid; g < k / 128u; g += 3u)
            quant_group(&qx[t], (const HVX_Vector *)(x + t * stride), g);
}

/* e^x with exp_approx's reduction and polynomial; x is clamped to [-87, 80]
 * (exp_approx returns 0 below -80; both are negligible next to 1 in SiLU). */
static inline HVX_Vector vexpf(HVX_Vector x) {
    x = vord(Q6_Vw_vmax_VwVw(vord(x), vord(vsplatf(-87.0f))));
    x = vord(Q6_Vw_vmin_VwVw(vord(x), vord(vsplatf(80.0f))));
    HVX_Vector scaled = vmulf(x, vsplatf(1.4426950408889634f));
    HVX_Vector magic = vsplatf(12582912.0f);                  /* 1.5 * 2^23 */
    HVX_Vector rounded = vaddf(scaled, magic);                /* nearest integer in the low bits */
    HVX_Vector n = Q6_Vw_vsub_VwVw(rounded, magic);
    HVX_Vector r = vsubf(scaled, vsubf(rounded, magic));
    HVX_Vector p = vsplatf(0.00015403530f);
    p = vaddf(vmulf(p, r), vsplatf(0.00133335581f));
    p = vaddf(vmulf(p, r), vsplatf(0.00961812911f));
    p = vaddf(vmulf(p, r), vsplatf(0.05550410866f));
    p = vaddf(vmulf(p, r), vsplatf(0.24022650695f));
    p = vaddf(vmulf(p, r), vsplatf(0.69314718056f));
    p = vaddf(vmulf(p, r), vsplatf(1.0f));
    HVX_Vector scale = Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(n, Q6_V_vsplat_R(127)), 23);
    return vmulf(scale, p);
}

/* 1 / d for d >= 1: bit-trick estimate and three Newton steps. */
static inline HVX_Vector vrecipf(HVX_Vector d) {
    HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7ef311c7), d);
    HVX_Vector two = vsplatf(2.0f);
    for (int i = 0; i < 3; i++) y = vmulf(y, vsubf(two, vmulf(d, y)));
    return y;
}

/* h[t][j] = silu(gate[t][j]) * up[t][j] for vectors [a, b) of the batch. */
static void silu_hvx(float *h, const float *gu, uint32_t a, uint32_t b) {
    const HVX_Vector one = vsplatf(1.0f);
    const HVX_Vector sign = Q6_V_vsplat_R(0x80000000);
    for (uint32_t i = a; i < b; i++) {
        uint32_t t = i / (FFN / 32u), j = i % (FFN / 32u);
        HVX_Vector g = ((const HVX_Vector *)(gu + t * 2u * FFN))[j];
        HVX_Vector u = ((const HVX_Vector *)(gu + t * 2u * FFN + FFN))[j];
        HVX_Vector e = vexpf(Q6_V_vxor_VV(g, sign));
        HVX_Vector s = vmulf(g, vrecipf(vaddf(one, e)));
        ((HVX_Vector *)(h + t * FFN))[j] = vmulf(s, u);
    }
}

/* Per-head RMSNorm and NeoX RoPE (cos/sin interleaved per pair) of one head. */
static void rope_head_hvx(float *head, const float *w, const float *cs) {
    rmsnorm_hvx(head, head, w, 128u);
    HVX_Vector *hv = (HVX_Vector *)head;
    const HVX_Vector *cv = (const HVX_Vector *)cs;
    for (uint32_t g = 0; g < 2u; g++) {
        HVX_VectorPair pair = Q6_W_vdeal_VVR(cv[2u * g + 1u], cv[2u * g], -4);
        HVX_Vector c = Q6_V_lo_W(pair), s = Q6_V_hi_W(pair);
        HVX_Vector a = hv[g], b = hv[2u + g];
        hv[g] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, c), Q6_Vqf32_vmpy_VsfVsf(b, s)));
        hv[2u + g] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, s), Q6_Vqf32_vmpy_VsfVsf(b, c)));
    }
}

/* First index of the largest value in vectors [a, b) of row; *value gets it. */
static uint32_t argmax_hvx(const float *row, uint32_t a, uint32_t b, float *value) {
    static const uint32_t lanes[32] __attribute__((aligned(128))) = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    const HVX_Vector *rv = (const HVX_Vector *)row;
    HVX_Vector best = Q6_V_vsplat_R(0x80000000);
    for (uint32_t i = a; i < b; i++) best = Q6_Vw_vmax_VwVw(best, vord(rv[i]));
    best = vhmaxw(best);
    HVX_Vector idx = Q6_Vw_vadd_VwVw(*(const HVX_Vector *)lanes, Q6_V_vsplat_R(a * 32u));
    HVX_Vector first = Q6_V_vsplat_R(0x7fffffff), step = Q6_V_vsplat_R(32);
    for (uint32_t i = a; i < b; i++) {
        first = Q6_Vw_vmin_VwVw(first, Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(vord(rv[i]), best), idx,
                                                     Q6_V_vsplat_R(0x7fffffff)));
        idx = Q6_Vw_vadd_VwVw(idx, step);
    }
    *value = vlane0f(vord(best));
    return Q6_R_vextract_VR(vhminw(first), 0);
}

/* out[t][g*32 .. +32] for tokens [t0, t0+NT) and row groups [g0, g1).
 * NT is a compile-time constant at each call site, so every accumulator and
 * unpacked code vector stays in registers (the worker stack is in DDR). */
static inline __attribute__((always_inline)) void mv_rows(
        const uint8_t *m, uint32_t k, const QRow *qx, uint32_t t0, const uint32_t NT,
        float *out, uint32_t out_stride, uint32_t g0, uint32_t g1) {
    const HVX_Vector mask = Q6_Vb_vsplat_R(0x0f);
    const uint32_t group_bytes = k * 18u, blocks = k / 32u;
    for (uint32_t g = g0; g < g1; g++) {
        const HVX_Vector *codes = (const HVX_Vector *)(m + g * group_bytes);
        const HVX_Vector *scales = (const HVX_Vector *)(m + g * group_bytes + k * 16u);
        HVX_Vector sum[4];
#pragma clang loop unroll(full)
        for (uint32_t t = 0; t < NT; t++) sum[t] = Q6_V_vzero();
        for (uint32_t p = 0; p < blocks / 2u; p++) {
            HVX_Vector sc = scales[p];
#pragma clang loop unroll(full)
            for (uint32_t half = 0; half < 2u; half++) {
                uint32_t b = p * 2u + half;
                HVX_Vector v0 = codes[b * 4u], v1 = codes[b * 4u + 1u];
                HVX_Vector v2 = codes[b * 4u + 2u], v3 = codes[b * 4u + 3u];
                HVX_Vector l0 = Q6_V_vand_VV(v0, mask), h0 = Q6_Vub_vlsr_VubR(v0, 4);
                HVX_Vector l1 = Q6_V_vand_VV(v1, mask), h1 = Q6_Vub_vlsr_VubR(v1, 4);
                HVX_Vector l2 = Q6_V_vand_VV(v2, mask), h2 = Q6_Vub_vlsr_VubR(v2, 4);
                HVX_Vector l3 = Q6_V_vand_VV(v3, mask), h3 = Q6_Vub_vlsr_VubR(v3, 4);
#pragma clang loop unroll(full)
                for (uint32_t t = 0; t < NT; t++) {
                    const uint32_t *xw = (const uint32_t *)&qx[t0 + t].q[b * 32u];
                    HVX_Vector acc = Q6_V_vsplat_R(-qx[t0 + t].s8[b]);
                    acc = Q6_Vw_vrmpyacc_VwVubRb(acc, l0, xw[0]);
                    acc = Q6_Vw_vrmpyacc_VwVubRb(acc, h0, xw[1]);
                    acc = Q6_Vw_vrmpyacc_VwVubRb(acc, l1, xw[2]);
                    acc = Q6_Vw_vrmpyacc_VwVubRb(acc, h1, xw[3]);
                    acc = Q6_Vw_vrmpyacc_VwVubRb(acc, l2, xw[4]);
                    acc = Q6_Vw_vrmpyacc_VwVubRb(acc, h2, xw[5]);
                    acc = Q6_Vw_vrmpyacc_VwVubRb(acc, l3, xw[6]);
                    acc = Q6_Vw_vrmpyacc_VwVubRb(acc, h3, xw[7]);
                    HVX_VectorPair s = Q6_Wqf32_vmpy_VhfVhf(sc, Q6_V_vsplat_R(qx[t0 + t].dxp[p]));
                    HVX_Vector scale = Q6_Vsf_equals_Vqf32(half ? Q6_V_hi_W(s) : Q6_V_lo_W(s));
                    sum[t] = Q6_Vqf32_vadd_Vqf32Vqf32(sum[t], Q6_Vqf32_vmpy_VsfVsf(int_to_sf(acc), scale));
                }
            }
        }
#pragma clang loop unroll(full)
        for (uint32_t t = 0; t < NT; t++)
            *(HVX_Vector *)(out + (t0 + t) * out_stride + g * 32u) = Q6_Vsf_equals_Vqf32(sum[t]);
    }
}

static void __attribute__((noinline)) mv1(const uint8_t *m, uint32_t k, const QRow *qx, uint32_t t0,
                                          float *out, uint32_t stride, uint32_t g0, uint32_t g1) {
    mv_rows(m, k, qx, t0, 1u, out, stride, g0, g1);
}

static void __attribute__((noinline)) mv2(const uint8_t *m, uint32_t k, const QRow *qx, uint32_t t0,
                                          float *out, uint32_t stride, uint32_t g0, uint32_t g1) {
    mv_rows(m, k, qx, t0, 2u, out, stride, g0, g1);
}

static void __attribute__((noinline)) mv4(const uint8_t *m, uint32_t k, const QRow *qx, uint32_t t0,
                                          float *out, uint32_t stride, uint32_t g0, uint32_t g1) {
    mv_rows(m, k, qx, t0, 4u, out, stride, g0, g1);
}

static void split(uint32_t n, uint32_t tid, uint32_t *a, uint32_t *b) {
    *a = n * tid / 3u;
    *b = n * (tid + 1u) / 3u;
}

static void mv_any(const uint8_t *m, uint32_t k, const QRow *qx, uint32_t ntok,
                   float *out, uint32_t stride, uint32_t g0, uint32_t g1) {
    uint32_t t = 0;
    if (g0 >= g1) return;
    for (; t + 4u <= ntok; t += 4u) mv4(m, k, qx, t, out, stride, g0, g1);
    if (t + 2u <= ntok) { mv2(m, k, qx, t, out, stride, g0, g1); t += 2u; }
    if (t < ntok) mv1(m, k, qx, t, out, stride, g0, g1);
}

/* Weight stream. Every matrix of the command is cut into chunks of at most
 * 2 MB; each chunk is split in two halves fetched by the user-DMA engines of
 * workers 0 and 1 (mode 1) or both by worker 0 (mode 2) into a 3-slot VTCM
 * ring that runs ahead of the compute, also across the scalar phases. All
 * three workers then compute from VTCM, which costs almost nothing next to
 * the DDR traffic. One engine moves ~1.25 GB/s here, two ~2.2 GB/s, and the
 * HVX-load path ~1.2 GB/s in total (mode 0, used if an engine is busy).
 * Engines are never dmpause'd while running: that blocks forever. */
#define RING (VTCM + 0x100000u)            /* 3 slots of 2 MB (1 MB for big batches) */
#define SLOTS 3u
#define DESC_RING (ARENA + 0x7f0000u)      /* per owner 63 x 32 B + dummy, uncached */
#define STREAM_OFF 0xa0000u                /* Stream in L2-TCM (~25 KB) */
#define MAX_CHUNKS 512u
#define MATS (LAYERS * 4u + 1u)

typedef struct {
    uint32_t mode, n, tag, slot;
    uint32_t engine_ok[2], saved_dm2[2], started[2];
    uint32_t issued[2], tail[2], ring[2];
    uint32_t cons[3];
    uint32_t desc[SLOTS][2];
    volatile uint32_t ready[SLOTS][2];
    uint32_t mat_first[MATS + 1u];
    uint32_t t_issue[MAX_CHUNKS][2], t_done[MAX_CHUNKS][2], t_wait[MAX_CHUNKS];
    struct { uint32_t src, bytes, half, first_group; } ch[MAX_CHUNKS];
} Stream;

static uint32_t dm_read(uint32_t index) {
    uint32_t value;
    __asm__ volatile("r1 = %1\n .word 0xa801c0a0\n %0 = r0" : "=r"(value) : "r"(index) : "r0", "r1", "memory");
    return value;
}

static void dm_write(uint32_t index, uint32_t value) {
    __asm__ volatile("r1 = %0\n r0 = %1\n .word 0xa801c0c0" :: "r"(index), "r"(value) : "r0", "r1", "memory");
}

/* Worker 0: chunk table for the matrices of this command, in compute order. */
static void stream_plan(Stream *st, const Image *img, uint32_t with_head) {
    uint32_t n = 0;
    for (uint32_t mat = 0; mat < MATS; mat++) {
        st->mat_first[mat] = n;
        uint32_t off, k, rows;
        if (mat == LAYERS * 4u) {
            if (!with_head) continue;
            off = img->out; k = DIM; rows = VOCAB;
        } else {
            uint32_t layer = mat / 4u, which = mat % 4u;
            off = which == 0 ? img->layer[layer].qkv : which == 1 ? img->layer[layer].o :
                  which == 2 ? img->layer[layer].gateup : img->layer[layer].down;
            k = which == 1 ? 2048u : which == 3 ? FFN : DIM;
            rows = which == 0 ? QKV : which == 2 ? 2u * FFN : DIM;
        }
        uint32_t gb = k * 18u, groups = rows / 32u, per = st->slot / gb;
        uint32_t count = (groups + per - 1u) / per, each = (groups + count - 1u) / count;
        for (uint32_t g = 0; g < groups && n < MAX_CHUNKS; g += each) {
            uint32_t take = groups - g < each ? groups - g : each;
            st->ch[n].src = BIG + off + g * gb;
            st->ch[n].bytes = take * gb;
            st->ch[n].half = (take / 2u) * gb;
            st->ch[n].first_group = g;
            n++;
        }
    }
    st->mat_first[MATS] = n;
    st->n = n;
    st->issued[0] = st->issued[1] = 0;
}

/* Owners at command start: check and configure their own engine. */
static void stream_engine_check(Stream *st, uint32_t tid) {
    uint32_t state;
    __asm__ volatile("%0 = dmpoll" : "=r"(state) :: "memory");
    if ((state & 3u) == 1u) { st->engine_ok[tid] = 0; return; }   /* still running: leave it */
    if (state & 3u) __asm__ volatile("%0 = dmpause" : "=r"(state) :: "memory");
    st->engine_ok[tid] = 1;
}

static void stream_start(Stream *st, uint32_t tid) {
    st->saved_dm2[tid] = dm_read(2);
    st->started[tid] = 1;
    dm_write(2, 0x2e3u);                           /* as Qualcomm's H2 booter */
    uint32_t dummy = DESC_RING + tid * 0x1000u + 63u * 32u;
    for (uint32_t i = 0; i < 8u; i++) ((volatile uint32_t *)dummy)[i] = 0;
    st->tail[tid] = dummy;
    st->ring[tid] = 0;
}

static void stream_stop(Stream *st, uint32_t tid) {
    dm_write(2, st->saved_dm2[tid]);
}

static uint32_t desc_issue(Stream *st, uint32_t owner, uint32_t src, uint32_t dst, uint32_t bytes) {
    uint32_t addr = DESC_RING + owner * 0x1000u + (st->ring[owner] % 63u) * 32u;
    volatile uint32_t *d = (volatile uint32_t *)addr;
    st->ring[owner]++;
    d[0] = 0;
    d[1] = (bytes & 0xffffffu) | (1u << 24) | (1u << 28) | (1u << 29);  /* 2D, bypass, !done */
    d[2] = src; d[3] = dst;
    d[4] = 9u;                                      /* 2D, 24-bit fields */
    d[5] = (bytes & 0xffffffu) | (1u << 24);        /* row_size, nrows_lo = 1 */
    d[6] = (bytes & 0xffffffu) << 8;                /* nrows_hi = 0, src_stride */
    d[7] = 0;
    fence();
    uint32_t tail = st->tail[owner];
    __asm__ volatile("release(%0):at" :: "r"(addr) : "memory");
    __asm__ volatile("dmlink(%0, %1)" :: "r"(tail), "r"(addr) : "memory");
    st->tail[owner] = addr;
    return addr;
}

static inline uint32_t half_owner(const Stream *st, uint32_t h) {
    return st->mode == 1u ? h : 0u;
}

/* Issue every half this worker owns that has a free slot. */
static void stream_issue(Stream *st, uint32_t tid) {
    for (uint32_t h = 0; h < 2u; h++) {
        if (half_owner(st, h) != tid) continue;
        while (st->issued[h] < st->n && st->issued[h] < st->cons[tid] + SLOTS) {
            uint32_t c = st->issued[h], slot = c % SLOTS;
            uint32_t lo = h ? st->ch[c].half : 0, hi = h ? st->ch[c].bytes : st->ch[c].half;
            st->desc[slot][h] = hi > lo ? desc_issue(st, tid, st->ch[c].src + lo,
                                                     RING + slot * st->slot + lo, hi - lo) : 0;
            st->t_issue[c][h] = (uint32_t)(cycles() >> 6);
            st->issued[h] = c + 1u;
        }
    }
}

/* Owners poll their descriptor (uncached DDR, so with backoff) and publish a
 * per-command tag in L2-TCM; everyone else spins on TCM. */
static int stream_wait(Stream *st, uint32_t c, uint32_t tid) {
    uint32_t slot = c % SLOTS, tag = st->tag + c + 1u;
    uint64_t end = cycles() + 2000000000ull;
    if (tid == 0) st->t_wait[c] = (uint32_t)(cycles() >> 6);
    volatile uint32_t *err = (volatile uint32_t *)(L2 + SYNC_OFF) + 3;
    for (uint32_t h = 0; h < 2u; h++) {
        if (half_owner(st, h) == tid) {
            uint32_t desc = st->desc[slot][h];
            if (desc)
                while (!(((volatile uint32_t *)desc)[1] >> 31)) {
                    if (cycles() > end || *err) { *err = 4; return 1; }
                    uint64_t until = cycles() + 1500u;
                    while (cycles() < until) { }
                }
            st->ready[slot][h] = tag;
            st->t_done[c][h] = (uint32_t)(cycles() >> 6);
        }
    }
    for (uint32_t h = 0; h < 2u; h++)
        while (st->ready[slot][h] != tag)
            if (cycles() > end || *err) { *err = 4; return 1; }
    return 0;
}

/* All three workers: n outputs over k inputs, matrix `mat` of the stream. */
static int matvec3(Stream *st, uint32_t mat, uint32_t off, uint32_t k, uint32_t n, const QRow *qx,
                   uint32_t ntok, float *out, uint32_t stride, uint32_t tid) {
    uint32_t a, b;
    if (!st->mode) {
        split(n / 32u, tid, &a, &b);
        mv_any((const uint8_t *)(BIG + off), k, qx, ntok, out, stride, a, b);
        return 0;
    }
    const uint32_t gb = k * 18u;
    volatile uint32_t *prof = (volatile uint32_t *)(ARENA + OUT_OFF + 128u) + tid * 4u;
    for (uint32_t c = st->mat_first[mat]; c < st->mat_first[mat + 1u]; c++) {
        uint64_t t0 = cycles();
        if (stream_wait(st, c, tid)) return 1;
        uint64_t t1 = cycles();
        uint32_t g0 = st->ch[c].first_group, groups = st->ch[c].bytes / gb;
        split(groups, tid, &a, &b);
        const uint8_t *base = (const uint8_t *)(RING + (c % SLOTS) * st->slot) - g0 * gb;
        mv_any(base, k, qx, ntok, out, stride, g0 + a, g0 + b);
        prof[1] += (uint32_t)((t1 - t0) >> 4);
        prof[2] += (uint32_t)((cycles() - t1) >> 4);
        if (sync3()) return 1;
        st->cons[tid] = c + 1u;
        stream_issue(st, tid);
    }
    return 0;
}

/* HVX copy of `bytes` (multiple of 128) between 128-byte aligned buffers. */
static void copy128(void *dst, const void *src, uint32_t bytes) {
    HVX_Vector *d = (HVX_Vector *)dst;
    const HVX_Vector *s = (const HVX_Vector *)src;
    for (uint32_t i = 0; i < bytes / 128u; i++) d[i] = s[i];
}

static void dequant_embedding(float *x, const uint8_t *raw, uint32_t ntok) {
    for (uint32_t t = 0; t < ntok; t++)
        for (uint32_t b = 0; b < DIM / 32u; b++) {
            const uint8_t *q = raw + (t * (DIM / 32u) + b) * 18u;
            float d = half_to_float((uint16_t)(q[0] | (q[1] << 8)));
            for (uint32_t i = 0; i < 16u; i++) {
                x[t * DIM + b * 32u + i] = d * (float)((int32_t)(q[2u + i] & 15u) - 8);
                x[t * DIM + b * 32u + i + 16u] = d * (float)((int32_t)(q[2u + i] >> 4) - 8);
            }
        }
}

/* KV cache per layer, fp16: K transposed as [kvh][ctx/64][128 dims][64
 * positions] (ctx * 2 KB), then V as [ctx][kvh][128] (ctx * 2 KB). Only HVX
 * touches it, so there is no L1 coherence to manage. Products accumulate in
 * qf32; hf x hf multiplies yield even lanes in the low vector, odd in high. */
#define KV_TMP_OFF 0xa8000u                /* per-thread 512 B staging */

static inline HVX_VectorPred halfword_lane(uint32_t lane) {
    static const uint16_t lanes[64] __attribute__((aligned(128))) = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
        32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63};
    return Q6_Q_vcmp_eq_VhVh(*(const HVX_Vector *)lanes, Q6_Vh_vsplat_R(lane));
}

/* Per-head RMSNorm and NeoX RoPE for tokens [0, ntok). */
static void qk_prepare(const Image *img, uint32_t layer, float *qkv, uint32_t ntok, uint32_t pos, uint32_t tid) {
    const float *qn = (const float *)(L2 + NORM_OFF + 4096u);
    const float *kn = (const float *)(L2 + NORM_OFF + 4608u);
    const float *rope = (const float *)(L2 + ROPE_OFF);
    (void)img; (void)layer; (void)pos;
    for (uint32_t item = tid; item < ntok * 24u; item += 3u) {
        uint32_t t = item / 24u, h = item % 24u;
        rope_head_hvx(qkv + t * QKV + h * 128u, h < 16u ? qn : kn, rope + t * 128u);
    }
}

/* Append K/V of tokens [0, ntok). One worker owns each KV head: tokens of a
 * batch share K^T vectors, and concurrent byte-masked stores to one vector
 * from different workers can lose updates. */
static void kv_append(const Image *img, uint32_t layer, const float *qkv, uint32_t ntok, uint32_t pos, uint32_t tid) {
    uint8_t *kv = (uint8_t *)(BIG + img->kv + layer * img->kv_layer_stride);
    uint32_t ctx = img->ctx;
    uint16_t *tmp = (uint16_t *)(L2 + KV_TMP_OFF + tid * 512u);
    for (uint32_t h = tid; h < 8u; h += 3u)
        for (uint32_t t = 0; t < ntok; t++) {
            uint32_t s = pos + t;
            const float *k = qkv + t * QKV + 2048u + h * 128u;
            HVX_VectorPred lane = halfword_lane(s & 63u);
            HVX_Vector *dst = (HVX_Vector *)(kv + h * ctx * 256u + (s / 64u) * 16384u);
            for (uint32_t d = 0; d < 128u; d++)
                Q6_vmem_QRIV(lane, dst + d, Q6_Vh_vsplat_R(float_to_half(k[d])));
            const float *v = qkv + t * QKV + 3072u + h * 128u;
            for (uint32_t i = 0; i < 128u; i++) tmp[i] = float_to_half(v[i]);
            copy128(kv + ctx * 2048u + s * 2048u + h * 256u, tmp, 256u);
        }
}

/* In place: s[i] = exp(s[i] / sqrt(128) - max) for i < len (later lanes 0);
 * returns 1 / sum. */
static float softmax_hvx(float *s, uint32_t len) {
    static const uint32_t lanes[32] __attribute__((aligned(128))) = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    HVX_Vector *sv = (HVX_Vector *)s;
    const HVX_Vector lane = *(const HVX_Vector *)lanes, scale = vsplatf(0.08838834764831845f);
    const HVX_Vector low = vsplatf(-1.0e30f), zero = Q6_V_vzero();
    uint32_t nv = (len + 31u) / 32u;
    HVX_Vector best = vord(low);
    for (uint32_t i = 0; i < nv; i++) {
        HVX_VectorPred live = Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(len - 32u * i), lane);
        sv[i] = Q6_V_vmux_QVV(live, vmulf(sv[i], scale), low);
        best = Q6_Vw_vmax_VwVw(best, vord(sv[i]));
    }
    HVX_Vector m = vord(vhmaxw(best)), sum = zero;
    for (uint32_t i = 0; i < nv; i++) {
        HVX_VectorPred live = Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(len - 32u * i), lane);
        HVX_Vector e = Q6_V_vmux_QVV(live, vexpf(vsubf(sv[i], m)), zero);
        sv[i] = e;
        sum = Q6_Vqf32_vadd_Vqf32Vsf(sum, e);
    }
    return 1.0f / vlane0f(vhsumf(Q6_Vsf_equals_Vqf32(sum)));
}

static void attention(const Image *img, uint32_t layer, const float *qkv, float *att,
                      uint32_t ntok, uint32_t pos, uint32_t tid) {
    const uint8_t *kv = (const uint8_t *)(BIG + img->kv + layer * img->kv_layer_stride);
    uint32_t ctx = img->ctx;
    float *s0 = (float *)(L2 + SCORE_OFF) + tid * 4096u, *s1 = s0 + 2048u;
    /* K^T and V live in DDR, where these loops' dependent loads cost ~6x
     * their TCM time; each 64-position block is first copied (pipelined) to
     * this worker's 32 KB of VTCM, unused during attention (logits are
     * written there only in the head phase, the DMA ring starts at +1 MB). */
    HVX_Vector *kst = (HVX_Vector *)(VTCM + tid * 0x8000u), *vst = kst + 128u;
    /* Worker 0's time per step (q to fp16, Q.K, softmax, V), in out[26..29]. */
    volatile uint32_t *sub = (volatile uint32_t *)(ARENA + OUT_OFF) + 26u;
    uint64_t c0, c1;
    for (uint32_t item = tid; item < ntok * 8u; item += 3u) {
        c0 = cycles();
        uint32_t t = item / 8u, kvh = item % 8u, len = pos + t + 1u;
        const float *qf = qkv + t * QKV + kvh * 256u;
        uint16_t *qh = (uint16_t *)(L2 + KV_TMP_OFF + tid * 512u);
        for (uint32_t i = 0; i < 256u; i++) qh[i] = float_to_half(qf[i]);
        if (!tid) { c1 = cycles(); sub[0] += (uint32_t)((c1 - c0) >> 4); c0 = c1; }
        const uint16_t *q0 = qh, *q1 = qh + 128u;
        const HVX_Vector *kt = (const HVX_Vector *)(kv + kvh * ctx * 256u);
        for (uint32_t tb = 0; tb < (len + 63u) / 64u; tb++) {
            HVX_Vector e0 = Q6_V_vzero(), o0 = e0, e1 = e0, o1 = e0;
            copy128(kst, kt + tb * 128u, 16384u);
            const HVX_Vector *col = kst;
            for (uint32_t d = 0; d < 128u; d++) {
                HVX_Vector k = col[d];
                HVX_VectorPair p0 = Q6_Wqf32_vmpy_VhfVhf(k, Q6_Vh_vsplat_R(q0[d]));
                HVX_VectorPair p1 = Q6_Wqf32_vmpy_VhfVhf(k, Q6_Vh_vsplat_R(q1[d]));
                e0 = Q6_Vqf32_vadd_Vqf32Vqf32(e0, Q6_V_lo_W(p0));
                o0 = Q6_Vqf32_vadd_Vqf32Vqf32(o0, Q6_V_hi_W(p0));
                e1 = Q6_Vqf32_vadd_Vqf32Vqf32(e1, Q6_V_lo_W(p1));
                o1 = Q6_Vqf32_vadd_Vqf32Vqf32(o1, Q6_V_hi_W(p1));
            }
            HVX_VectorPair w0 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(o0), Q6_Vsf_equals_Vqf32(e0), -4);
            HVX_VectorPair w1 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(o1), Q6_Vsf_equals_Vqf32(e1), -4);
            ((HVX_Vector *)s0)[tb * 2u] = Q6_V_lo_W(w0); ((HVX_Vector *)s0)[tb * 2u + 1u] = Q6_V_hi_W(w0);
            ((HVX_Vector *)s1)[tb * 2u] = Q6_V_lo_W(w1); ((HVX_Vector *)s1)[tb * 2u + 1u] = Q6_V_hi_W(w1);
        }
        if (!tid) { c1 = cycles(); sub[1] += (uint32_t)((c1 - c0) >> 4); c0 = c1; }
        float i0 = softmax_hvx(s0, len), i1 = softmax_hvx(s1, len);
        if (!tid) { c1 = cycles(); sub[2] += (uint32_t)((c1 - c0) >> 4); c0 = c1; }
        /* acc[head][vector][even/odd dims] */
        HVX_Vector a00 = Q6_V_vzero(), a01 = a00, a02 = a00, a03 = a00;
        HVX_Vector a10 = a00, a11 = a00, a12 = a00, a13 = a00;
        const HVX_Vector *v = (const HVX_Vector *)(kv + ctx * 2048u + kvh * 256u);
        for (uint32_t s = 0; s < len; s++) {
            if (!(s & 63u)) {
                uint32_t n = len - s < 64u ? len - s : 64u;
                for (uint32_t j = 0; j < n; j++) { vst[2u * j] = v[(s + j) * 16u]; vst[2u * j + 1u] = v[(s + j) * 16u + 1u]; }
            }
            HVX_Vector w0 = Q6_Vh_vsplat_R(float_to_half(s0[s] * i0));
            HVX_Vector w1 = Q6_Vh_vsplat_R(float_to_half(s1[s] * i1));
            const HVX_Vector *row = vst + 2u * (s & 63u);
            HVX_Vector r0 = row[0], r1 = row[1];
            HVX_VectorPair p;
            p = Q6_Wqf32_vmpy_VhfVhf(r0, w0);
            a00 = Q6_Vqf32_vadd_Vqf32Vqf32(a00, Q6_V_lo_W(p)); a01 = Q6_Vqf32_vadd_Vqf32Vqf32(a01, Q6_V_hi_W(p));
            p = Q6_Wqf32_vmpy_VhfVhf(r1, w0);
            a02 = Q6_Vqf32_vadd_Vqf32Vqf32(a02, Q6_V_lo_W(p)); a03 = Q6_Vqf32_vadd_Vqf32Vqf32(a03, Q6_V_hi_W(p));
            p = Q6_Wqf32_vmpy_VhfVhf(r0, w1);
            a10 = Q6_Vqf32_vadd_Vqf32Vqf32(a10, Q6_V_lo_W(p)); a11 = Q6_Vqf32_vadd_Vqf32Vqf32(a11, Q6_V_hi_W(p));
            p = Q6_Wqf32_vmpy_VhfVhf(r1, w1);
            a12 = Q6_Vqf32_vadd_Vqf32Vqf32(a12, Q6_V_lo_W(p)); a13 = Q6_Vqf32_vadd_Vqf32Vqf32(a13, Q6_V_hi_W(p));
        }
        if (!tid) sub[3] += (uint32_t)((cycles() - c0) >> 4);
        HVX_Vector *o = (HVX_Vector *)(att + t * 2048u + kvh * 256u);
        HVX_VectorPair w;
        w = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(a01), Q6_Vsf_equals_Vqf32(a00), -4);
        o[0] = Q6_V_lo_W(w); o[1] = Q6_V_hi_W(w);
        w = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(a03), Q6_Vsf_equals_Vqf32(a02), -4);
        o[2] = Q6_V_lo_W(w); o[3] = Q6_V_hi_W(w);
        w = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(a11), Q6_Vsf_equals_Vqf32(a10), -4);
        o[4] = Q6_V_lo_W(w); o[5] = Q6_V_hi_W(w);
        w = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(a13), Q6_Vsf_equals_Vqf32(a12), -4);
        o[6] = Q6_V_lo_W(w); o[7] = Q6_V_hi_W(w);
    }
}

static int install_tlb(void) {
    /* Drop stale entries for our VA range in other slots first: two
     * matching entries for one VA is a TLB multi-hit fault. */
    for (uint32_t index = 12u; index < 128u; index++) {
        uint64_t have;
        __asm__ volatile("%0 = tlbr(%1)" : "=r"(have) : "r"(index) : "memory");
        uint32_t hi = (uint32_t)(have >> 32), va = (hi & 0xfffffu) << 12;
        if (!(hi >> 31) || va < BIG || va >= BIG + BIG_SLOTS * 0x1000000u) continue;
        if (index == FIRST_SLOT + (va - BIG) / 0x1000000u) continue;
        uint64_t zero = 0;
        __asm__ volatile("tlbw(%0,%1)\n isync" :: "r"(zero), "r"(index) : "memory");
    }
    for (uint32_t i = 0; i < BIG_SLOTS; i++) {
        uint32_t pn = (BIG >> 12) + i * 0x1000u;
        uint64_t want = ((uint64_t)(0xc0000000u | pn) << 32) | (0xf7000000u | (pn << 1) | 0x40u);
        uint64_t have;
        uint32_t index = FIRST_SLOT + i;
        __asm__ volatile("%0 = tlbr(%1)" : "=r"(have) : "r"(index) : "memory");
        if (have == want) continue;
        if (have >> 63) return 1;
        __asm__ volatile("tlbw(%0,%1)\n isync" :: "r"(want), "r"(index) : "memory");
    }
    return 0;
}

static void remove_tlb(void) {
    uint64_t zero = 0;
    for (uint32_t i = 0; i < BIG_SLOTS; i++) {
        uint32_t index = FIRST_SLOT + i;
        __asm__ volatile("tlbw(%0,%1)\n isync" :: "r"(zero), "r"(index) : "memory");
    }
}

/* Q6 AXI-master clock (turing_cc+0x254). Valve's firmware leaves it on XO
 * with the Turing core PLLs off, which caps DDR reads near 1.2 GB/s. QuRT's
 * 750 MHz point: core2 PLL 1500 MHz main output, RCG source 5 / 2, SVS_L1
 * (below Valve's NOM rail votes). Measured with bench/axim_tune.c. Thread 0
 * raises it per command (no-op once raised) and restores it on exit; the
 * saved registers live in the arena because the FLOPS wake clears L2-TCM. */
#define AXIM_CC ((volatile uint32_t *)0x32008254u)  /* CMD; CFG at [1] */
#define CORE2_PLL ((volatile uint32_t *)0x32002000u)
#define AXIM_SAVE ((volatile uint32_t *)(ARENA + 0x7e0000u))  /* raised, cfg, pll[16] */
#define AXIM_RAISED 0x41584931u                   /* 'AXI1' */
#define AXIM_CFG_750 0x503u                       /* source 5, divide by 2 */
#ifndef AXIM_PLL_L                                /* core2 PLL 1500 MHz -> 750 MHz; */
#define AXIM_PLL_L 78u                            /* L=104, alpha 0x2aaa: 2000 -> 1000 */
#define AXIM_PLL_ALPHA 0x2000u                    /* MHz at NOM, Valve's vote limit */
#endif
#define AXIM_NO_RAISE 16u                         /* command flag: stay on XO */

static int axim_wait(volatile uint32_t *reg, uint32_t mask, uint32_t want) {
    uint64_t end = cycles() + 200000000u;
    while ((*reg & mask) != want) if (cycles() > end) return 1;
    return 0;
}

static int axim_select(uint32_t cfg) {
    AXIM_CC[1] = cfg;
    AXIM_CC[0] |= 1u;
    int ret = axim_wait(&AXIM_CC[0], 1u, 0);
    __asm__ volatile("syncht" ::: "memory");
    return ret;
}

static uint32_t axim_raise(void) {
    volatile uint32_t *pll = CORE2_PLL, *save = AXIM_SAVE;
    if (save[0] == AXIM_RAISED) return 0;
    save[1] = AXIM_CC[1];
    for (uint32_t i = 0; i < 16u; i++) save[2u + i] = pll[i];
    if ((save[2] & 5u) || save[3] != 0) return 2;   /* PLL already in use */
    pll[0] = 0;
    pll[4] = 0x44440000u | AXIM_PLL_L;
    pll[5] = AXIM_PLL_ALPHA;
    pll[6] = 0x00008401u;
    pll[1] = 0;
    pll[0] = 0x00800000u;
    pll[0] = 0x00800004u;
    pll[1] = 1;
    if (axim_wait(&pll[0], 0x80000000u, 0x80000000u)) { pll[1] = 0; pll[0] = save[2]; return 3; }
    pll[0] = 5;
    __asm__ volatile("syncht" ::: "memory");
    save[0] = AXIM_RAISED;
    if (axim_select((save[1] & ~0x71fu) | AXIM_CFG_750)) return 4;
    return 0;
}

static void axim_restore(void) {
    volatile uint32_t *pll = CORE2_PLL, *save = AXIM_SAVE;
    if (save[0] != AXIM_RAISED) return;
    axim_select(save[1]);
    pll[0] = 0; pll[1] = 0;
    pll[4] = save[6]; pll[5] = save[7]; pll[6] = save[8];
    pll[1] = save[3]; pll[0] = save[2];
    __asm__ volatile("syncht" ::: "memory");
    save[0] = 0;
}

static void invalidate(uint32_t address, uint32_t bytes) {
    for (uint32_t p = address & ~31u; p < address + bytes; p += 32u)
        __asm__ volatile("dcinva(%0)" :: "r"(p) : "memory");
    fence();
}

static uint64_t mark;
static void phase(volatile uint32_t *out, uint32_t tid, uint32_t which) {
    if (tid) return;
    uint64_t now = cycles();
    out[16u + which] += (uint32_t)((now - mark) >> 4);
    mark = now;
}

/* Rows tid, tid+3, ... of the batch: residual add then RMSNorm; after a
 * barrier every worker quantizes its share of all rows. */
static int norm_rows(float *x, const float *add, float *xn, const float *w, QRow *qx,
                     uint32_t ntok, uint32_t tid) {
    for (uint32_t t = tid; t < ntok; t += 3u) {
        if (add) add_hvx(x + t * DIM, add + t * DIM, DIM);
        rmsnorm_hvx(xn + t * DIM, x + t * DIM, w, DIM);
    }
    if (sync3()) return 1;
    quantize_split(qx, xn, DIM, DIM, ntok, tid);
    return 0;
}

/* One command: ntok tokens at positions pos..pos+ntok-1. */
static uint32_t run(volatile uint32_t *cmd, volatile uint32_t *out, uint32_t tid) {
    const Image *img = (const Image *)BIG;
    uint32_t ntok = cmd[4], pos = cmd[5], flags = cmd[6];
    uint32_t act = ntok > SMALL_TOK ? VTCM_ACT : L2, scale = ntok > SMALL_TOK ? BIG_SCALE : 1u;
    float *x = (float *)(act + X_OFF * scale), *xn = (float *)(act + XN_OFF * scale);
    float *qkv = (float *)(act + QKV_OFF * scale), *att = (float *)(act + ATT_OFF * scale);
    float *gu = (float *)(act + GU_OFF * scale), *h = (float *)(act + H_OFF * scale);
    /* The matvec inner loop reads activation words with scalar loads, so the
     * quantized rows always live in L2-TCM (offset 0 is free in big mode). */
    QRow *qx = (QRow *)(ntok > SMALL_TOK ? L2 : L2 + QX_OFF);
    Stream *st = (Stream *)(L2 + STREAM_OFF);
    const float *norms = (const float *)(L2 + NORM_OFF);
    uint32_t a, b;

    if (tid == 0) {
        for (uint32_t i = 16u; i < 52u; i++) out[i] = 0;
        mark = cycles();
        invalidate(BIG, 4096u);
        if (img->magic != IMAGE_MAGIC || img->layers != LAYERS || !ntok || ntok > MAX_TOK ||
            pos + ntok > img->ctx || ((flags & 8u) && ntok > MAX_VERIFY))
            ((volatile uint32_t *)(L2 + SYNC_OFF))[3] = 2;
        else {
            st->mode = cmd[7] <= 2u ? cmd[7] : 0u;
            st->tag = cmd[2] << 10;
            /* Verification (all-token logits) keeps [B][VOCAB] logits at
             * VTCM + 4 MB, so it also uses the smaller ring slots. */
            st->slot = ntok > SMALL_TOK || (flags & 8u) ? 0x100000u : 0x200000u;
            st->cons[0] = st->cons[1] = st->cons[2] = 0;
            stream_plan(st, img, flags & 1u);
            copy128((void *)(L2 + ROPE_OFF), (const void *)(BIG + img->rope + pos * 512u), ntok * 512u);
            copy128((void *)(L2 + EMBR_OFF), (const void *)(ARENA + EMB_OFF), (ntok * 576u + 127u) & ~127u);
            dequant_embedding(x, (const uint8_t *)(L2 + EMBR_OFF), ntok);
            copy128((void *)(L2 + NORM_OFF), (const void *)(BIG + img->layer[0].attn_norm), 9216u);
        }
    }
    if (tid < 2u && st->mode) stream_engine_check(st, tid);
    if (sync3()) return 30;
    /* Every worker derives the same mode from the engines' health. */
    if (st->mode == 1u && !st->engine_ok[1]) st->mode = 2u;
    if (st->mode && !st->engine_ok[0]) st->mode = 0;
    if (sync3()) return 30;
    if (st->mode && tid <= (st->mode == 1u ? 1u : 0u)) { stream_start(st, tid); stream_issue(st, tid); }
    if (tid == 0) out[5] = st->mode;
    phase(out, tid, 0);
    for (uint32_t layer = 0; layer < LAYERS; layer++) {
        /* Layer 0's norms were staged with the embedding; later layers' at
         * the end of the previous layer. */
        if (norm_rows(x, 0, xn, norms, qx, ntok, tid) || sync3()) return 31;
        phase(out, tid, 1);
        if (matvec3(st, layer * 4u, img->layer[layer].qkv, DIM, QKV, qx, ntok, qkv, QKV, tid)) return 50;
        if (sync3()) return 32;
        phase(out, tid, 2);
        qk_prepare(img, layer, qkv, ntok, pos, tid);
        if (sync3()) return 33;
        kv_append(img, layer, qkv, ntok, pos, tid);
        if (sync3()) return 33;
        phase(out, tid, 3);
        attention(img, layer, qkv, att, ntok, pos, tid);
        if (sync3()) return 34;
        phase(out, tid, 4);
        quantize_split(qx, att, 2048u, 2048u, ntok, tid);
        if (sync3()) return 35;
        phase(out, tid, 1);
        if (matvec3(st, layer * 4u + 1u, img->layer[layer].o, 2048u, DIM, qx, ntok, gu, DIM, tid)) return 51;
        if (sync3()) return 36;
        phase(out, tid, 5);
        if (norm_rows(x, gu, xn, norms + 1280u, qx, ntok, tid) || sync3()) return 37;
        phase(out, tid, 1);
        if (matvec3(st, layer * 4u + 2u, img->layer[layer].gateup, DIM, 2u * FFN, qx, ntok, gu, 2u * FFN, tid)) return 52;
        if (sync3()) return 38;
        phase(out, tid, 6);
        split(ntok * (FFN / 32u), tid, &a, &b);
        silu_hvx(h, gu, a, b);
        if (sync3()) return 39;
        phase(out, tid, 7);
        quantize_split(qx, h, FFN, FFN, ntok, tid);
        if (sync3()) return 40;
        phase(out, tid, 1);
        if (matvec3(st, layer * 4u + 3u, img->layer[layer].down, FFN, DIM, qx, ntok, gu, DIM, tid)) return 53;
        if (sync3()) return 41;
        phase(out, tid, 8);
        for (uint32_t t = tid; t < ntok; t += 3u)
            add_hvx(x + t * DIM, gu + t * DIM, DIM);
        if (tid == 0 && layer + 1u < LAYERS)
            copy128((void *)(L2 + NORM_OFF), (const void *)(BIG + img->layer[layer + 1u].attn_norm), 9216u);
        if (sync3()) return 44;
    }
    if (!(flags & 1u)) return 0;
    /* Logits for the last token, or for every token when verifying drafts. */
    uint32_t first = flags & 8u ? 0 : ntok - 1u, rows = ntok - first;
    if (tid == 0) copy128((void *)(L2 + OUTN_OFF), (const void *)(BIG + img->out_norm), 4096u);
    if (sync3()) return 42;
    for (uint32_t t = tid; t < rows; t += 3u)
        rmsnorm_hvx(xn + t * DIM, x + (first + t) * DIM, (const float *)(L2 + OUTN_OFF), DIM);
    if (sync3()) return 42;
    quantize_split(qx, xn, DIM, DIM, rows, tid);
    if (sync3()) return 42;
    phase(out, tid, 1);
    float *logits = (float *)(flags & 8u ? VTCM + 0x400000u : VTCM);
    if (matvec3(st, LAYERS * 4u, img->out, DIM, VOCAB, qx, rows, logits, VOCAB, tid)) return 54;
    split(VOCAB / 32u, tid, &a, &b);
    if (flags & 2u)
        for (uint32_t i = a; i < b; i++)
            ((HVX_Vector *)(ARENA + LOGITS_OFF))[i] = ((const HVX_Vector *)(logits + (rows - 1u) * VOCAB))[i];
    volatile uint32_t *local = (volatile uint32_t *)(L2 + LOCAL_OFF);
    for (uint32_t t = 0; t < rows; t++) {
        float best;
        uint32_t arg = argmax_hvx(logits + t * VOCAB, a, b, &best);
        union { float f; uint32_t u; } bits;
        bits.f = best;
        local[(t * 3u + tid) * 2u] = arg; local[(t * 3u + tid) * 2u + 1u] = bits.u;
    }
    if (sync3()) return 43;
    phase(out, tid, 9);
    if (tid == 0)
        for (uint32_t t = 0; t < rows; t++) {
            uint32_t win = 0;
            for (uint32_t i = 1; i < 3u; i++) {
                union { uint32_t u; float f; } c, w;
                c.u = local[(t * 3u + i) * 2u + 1u]; w.u = local[(t * 3u + win) * 2u + 1u];
                if (c.f > w.f) win = i;
            }
            ((volatile uint32_t *)(ARENA + ARGMAX_OFF))[t] = local[(t * 3u + win) * 2u];
            out[4] = local[(t * 3u + win) * 2u];
        }
    return 0;
}

#ifndef QWEN_OPS_TEST  /* bench/qwen_ops_test.c includes this file with its own entry */
__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    if (tid >= 3) return;
    volatile uint32_t *cmd = (volatile uint32_t *)(ARENA + CMD_OFF);
    volatile uint32_t *st = (volatile uint32_t *)(ARENA + STATUS_OFF + 16u * tid);
    volatile uint32_t *out = (volatile uint32_t *)(ARENA + OUT_OFF);
    uint32_t last_seq = 0;
    for (;;) {
        if (last_seq)
            while (cmd[2] == last_seq) { }
        uint32_t seq = cmd[2];
        if (cmd[9] != ~seq) { if (!last_seq) break; continue; }
        last_seq = seq;
        if (cmd[0] == EXIT_MAGIC && cmd[3] == 2u) {
            if (tid == 0) { remove_tlb(); axim_restore(); }
            st[3] = 0x54495845u; st[0] = seq;
            fence();
            return;
        }
        if (cmd[0] != MAGIC) break;
        uint64_t t0 = cycles();
        uint32_t status = 0;
        if (tid == 0) {
            volatile uint32_t *f = (volatile uint32_t *)(L2 + SYNC_OFF);
            for (uint32_t i = 0; i < 7u; i++) f[i] = 0;
            if (install_tlb()) f[3] = 3;
            if (cmd[6] & AXIM_NO_RAISE) { axim_restore(); out[7] = 0; }
            else out[7] = axim_raise();
            out[6] = AXIM_CC[1];
            out[0] = 1; out[1] = 0;
            fence();
            out[15] = seq;
            fence();
        }
        /* Workers 1 and 2 must not start before thread 0 reset the barrier. */
        while (out[15] != seq) { }
        status = run(cmd, out, tid);
        {
            Stream *stream = (Stream *)(L2 + STREAM_OFF);
            if (tid < 2u && stream->started[tid]) { stream_stop(stream, tid); stream->started[tid] = 0; }
            if (tid == 0 && (cmd[6] & 4u)) {
                volatile uint32_t *dump = (volatile uint32_t *)(ARENA + 0x60000u);
                dump[0] = stream->n;
                for (uint32_t i = 0; i < stream->n; i++) {
                    dump[1u + i * 5u] = stream->t_issue[i][0]; dump[2u + i * 5u] = stream->t_issue[i][1];
                    dump[3u + i * 5u] = stream->t_done[i][0]; dump[4u + i * 5u] = stream->t_done[i][1];
                    dump[5u + i * 5u] = stream->t_wait[i];
                }
            }
        }
        if (tid == 0) {
            uint64_t dt = cycles() - t0;
            out[1] = status ? status : ((volatile uint32_t *)(L2 + SYNC_OFF))[3];
            out[2] = (uint32_t)dt; out[3] = (uint32_t)(dt >> 32);
            out[0] = 6;
        }
        fence();
        st[3] = 0; st[0] = seq;
        fence();
    }
    if (tid == 0) axim_restore();
    st[3] = 0x54495845u; st[0] = cmd[2];
    fence();
}
#endif
