/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Read-only probe #2 for the L2 work (2026-11-01): per-thread stack pointer,
 * all 128 TLB entries (tlbr), and the number of non-zero words in each 16 KB
 * slice of the 1 MB L2-TCM at callback entry. Writes only to the arena. */
#include <stdint.h>
#include "../native/qd_protocol.h"
#include "../native/qd_conv.h"
#define ARENA 0xf1800000u
static uint64_t tlbr(uint32_t i){uint64_t e;__asm__ volatile("%0=tlbr(%1)":"=r"(e):"r"(i));return e;}
__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;__asm__ volatile("%0=htid":"=r"(tid));if(tid>=3)return;
    volatile uint32_t *a=(volatile uint32_t *)ARENA;
    uint32_t seq=a[2];
    volatile uint32_t *out=(volatile uint32_t *)(ARENA+2048);
    uint32_t sp;__asm__ volatile("%0=r29":"=r"(sp));
    out[8+tid]=sp;
    if(tid==0) {
        out[0]=0x50414d32; /* "2MAP" */
        for(uint32_t i=0;i<128;i++){uint64_t e=tlbr(i);out[64+2*i]=(uint32_t)e;out[65+2*i]=(uint32_t)(e>>32);}
        for(uint32_t s=0;s<64;s++) {
            const volatile uint32_t *p=(const volatile uint32_t *)(0x30000000u+s*16384u);
            uint32_t n=0,h=0;for(uint32_t w=0;w<4096;w++){uint32_t v=p[w];n+=v!=0;h=(h^v)*16777619u+w;}
            out[320+s]=n;out[384+s]=h;
        }
    }
    volatile QdStatus *st=(volatile QdStatus *)(ARENA+QD_STATUS_OFFSET)+tid;
    st->error=0;st->rows=0;st->reserved=QDC_EXITED;
    __asm__ volatile("barrier":::"memory");st->done_sequence=seq;
}
