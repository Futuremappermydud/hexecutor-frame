/* Probe HVX integer/qfloat instructions used by the Q4 decode kernel.
 * Inputs at arena+0x4000 (4 vectors), outputs at arena+0x5000 (16 vectors). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#define ARENA_UNCACHED 0xf1800000u

__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid; __asm__ volatile("%0 = htid" : "=r"(tid));
    if (tid >= 3) return;
    volatile uint32_t *cmd = (volatile uint32_t *)ARENA_UNCACHED;
    uint32_t seq = cmd[1];
    if (tid == 0 && cmd[0] == 0x46515250u) {
        const HVX_Vector *in = (const HVX_Vector *)(ARENA_UNCACHED + 0x4000u);
        HVX_Vector *out = (HVX_Vector *)(ARENA_UNCACHED + 0x5000u);
        HVX_Vector a = in[0], b = in[1], c = in[2], d = in[3];
        uint32_t word = cmd[2];
        /* 0: vrmpy ub x scalar b */
        out[0] = Q6_Vw_vrmpy_VubRb(a, word);
        /* 1: int32 -> sf */
        out[1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(Q6_Vw_vadd_VwVw(b, Q6_V_vsplat_R(0x4b000000)), Q6_V_vsplat_R(0x4b000000)));
        /* 2,3: hf x hf -> qf32 pair, converted to sf */
        HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(c, Q6_V_vsplat_R(cmd[3]));
        out[2] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
        out[3] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
        /* 4: sf x sf -> qf32 -> sf */
        out[4] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(d, d));
        /* 5: qf32 accumulate */
        HVX_Vector q = Q6_Vqf32_vmpy_VsfVsf(d, d);
        q = Q6_Vqf32_vadd_Vqf32Vqf32(q, Q6_Vqf32_vmpy_VsfVsf(d, d));
        out[5] = Q6_Vsf_equals_Vqf32(q);
        /* 6: sf add via qf32 */
        out[6] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(d, d));
        /* 7: sf -> int32 (truncating) through qf32 */
        HVX_VectorPair cv = Q6_Wsf_vcvt_Vhf(c);
        out[7] = Q6_V_lo_W(cv);
        out[10] = Q6_V_hi_W(cv);
        /* 8: hf vcvt from h */
        out[8] = Q6_Vhf_vcvt_Vh(b);
        /* 9: max sf */
        out[9] = Q6_Vsf_vmax_VsfVsf(d, Q6_V_vzero());
        __asm__ volatile("barrier" ::: "memory");
        cmd[8] = 0x600d;
    }
    *(volatile uint32_t *)(ARENA_UNCACHED + 128 + tid * 16) = seq;
    __asm__ volatile("barrier" ::: "memory");
}
