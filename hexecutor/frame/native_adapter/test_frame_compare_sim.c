/* ISA check for the restoration comparison, without touching Frame hardware. */
#include <stdio.h>
#include "adapter_frame_probe.c"

static uint8_t compare_a[4096] __attribute__((aligned(128)));
static uint8_t compare_b[4096] __attribute__((aligned(128)));

int main(void) {
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" :: "r"(ssr) : "memory");
    const uint32_t blocks[] = {0, 128, sizeof compare_a - 128};
    fill128(compare_a, sizeof compare_a, 0);
    fill128(compare_b, sizeof compare_b, 0);
    if (!same128(compare_a, compare_b, sizeof compare_a)) return 1;
    for (uint32_t b = 0; b < 3; ++b) for (uint32_t lane = 0; lane < 32; ++lane) {
        fill128(compare_b, sizeof compare_b, 0);
        HVX_VectorPred low = Q6_Q_vsetq_R(lane * 4);
        HVX_VectorPred high = lane == 31 ? Q6_Q_not_Q(Q6_Q_vsetq_R(0))
                                        : Q6_Q_vsetq_R((lane + 1) * 4);
        HVX_VectorPred selected = Q6_Q_and_QQn(high, low);
        *(HVX_Vector *)(compare_b + blocks[b]) =
            Q6_V_vmux_QVV(selected, Q6_V_vsplat_R(1), Q6_V_vzero());
        if (same128(compare_a, compare_b, sizeof compare_a)) return 2;
    }
    puts("PASS restoration comparison: equal buffers and 96 independent lane corruptions");
    return 0;
}
