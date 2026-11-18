/* HVX tile form (qnn16_tile.h): pointwise convolution fused with SiLU (Q16_PW_SWISH) on one pixel.
 *
 * The fully connected layers of a network arrive as 1x1 convolutions on a [1,1,1,C] tile tensor. Xpe holds, per
 * 32-output group, one 128-byte vector per input-channel pair: halves [co lane][ci & 1]. Multiplying that vector
 * by the pair (x[2k], x[2k + 1]) replicated in every word gives the two products of each output lane as the low
 * and high qf32 words, so one output group is a sum over the pairs. The five-table Fi keeps the scale in the low
 * halves of words 0-31 and the bias in the high halves of words 32-63 of its last table (as
 * qnn16_prepare_pw_swish reads them). SiLU is swish_qf. */
#include "qnn16_tile_priv.h"

static int one_pixel(const Qnn16Tensor *t) {
    return blocked_any(t) && t->dims.b == 1u && t->dims.h == 1u && t->dims.w == 1u && !t->padding.b &&
           !t->padding.h && !t->padding.w;
}

int qnn16_tile_pw_swish_ok(const Qnn16Tensor *x, const Qnn16Tensor *out, uint32_t packed_ic) {
    return one_pixel(x) && one_pixel(out) && packed_ic && !(packed_ic & 1u) && x->dims.c <= packed_ic &&
           packed_ic <= x->full.c;
}

void qnn16_tile_pw_swish(const Qnn16TileJob *j, uint32_t part, uint32_t parts) {
    const Qnn16Tensor *x = j->x, *out = j->out;
    const Map mx = map_of(x), mo = map_of(out);
    const uint32_t pairs = j->packed_ic / 2u, ng = (out->dims.c + 31u) / 32u, kg = (j->packed_ic + 31u) / 32u;
    const HVX_Vector one = Q6_V_vsplat_R(0x3c003c00), zero = Q6_V_vzero();
    const SwishK k = swish_k();
    for (uint32_t g = part; g < ng; g += parts) {
        const HVX_Vector *w = (const HVX_Vector *)((const uint8_t *)j->weights + (uint64_t)g * pairs * 128u);
        HVX_Vector acc0 = zero, acc1 = zero;
        for (uint32_t ig = 0; ig < kg; ig++) {
            /* The pixel's 32 channels of this group are the even lanes; copy them out once. */
            uint16_t h[64] __attribute__((aligned(128)));
            *(HVX_Vector *)h = ld(&mx, 0, 0, 0, ig);
            const uint32_t live = x->dims.c > ig * 32u ? x->dims.c - ig * 32u : 0u;
            const uint32_t n = pairs - ig * 16u < 16u ? pairs - ig * 16u : 16u;
            for (uint32_t i = 0; i < n; i++) {
                const uint32_t a = 2u * i < live ? h[4u * i] : 0u, b = 2u * i + 1u < live ? h[4u * i + 2u] : 0u;
                const HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(w[ig * 16u + i], Q6_V_vsplat_R((int)(a | b << 16)));
                acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_V_lo_W(p));
                acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_V_hi_W(p));
            }
        }
        const HVX_Vector *fi = (const HVX_Vector *)(j->fi + ((uint64_t)g * 5u + 4u) * 256u);
        const HVX_Vector scale = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(fi[0], one));
        const HVX_Vector bias = Q6_V_hi_W(Q6_Wqf32_vmpy_VhfVhf(fi[1], one));
        HVX_Vector v = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(acc0, acc1), scale);
        v = swish_qf(Q6_Vqf32_vadd_Vqf32Vqf32(v, bias), &k);
        /* Even lane of channel c = output channel c of the group. */
        store_live(vec(&mo, 0, 0, 0, g), Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(zero, v)), 1u, channel_mask(out, g));
    }
}
