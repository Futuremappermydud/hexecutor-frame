/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
typedef unsigned u32;

void callback(void) {
    u32 tid;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    volatile u32 *status = (volatile u32 *)0xf0b12100;
    const HVX_Vector *a = (const HVX_Vector *)0x87b12400;
    const HVX_Vector *b = (const HVX_Vector *)0x87b12500;
    HVX_Vector *c = (HVX_Vector *)(0x87b12800 + tid*0x100);
    /* Host uses the uncached DMA mapping; discard any stale DSP cache lines. */
    for (int i=0;i<128;i+=32) {
        __asm__ volatile("dcinva(%0)" :: "r"((u32)a+i) : "memory");
        __asm__ volatile("dcinva(%0)" :: "r"((u32)b+i) : "memory");
        __asm__ volatile("dcinva(%0)" :: "r"((u32)c+i) : "memory");
    }
    *c = Q6_Vw_vadd_VwVw(*a,*b);
    for (int i=0;i<128;i+=32)
        __asm__ volatile("dccleana(%0)" :: "r"((u32)c+i) : "memory");
    __asm__ volatile("barrier" ::: "memory");
    status[tid*4] = 0xc0de0000 | tid;
    status[tid*4+1] = 63+tid;
    status[tid*4+3] = 128;
    __asm__ volatile("barrier" ::: "memory");
    status[tid*4+2] = 1;
}
