/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Read-only: reports every JTLB entry (low and high word) so free indices for a larger private window can be
 * chosen. Writes nothing to the TLB. Result: ARENA+2048 magic, entries at ARENA+8192 (two words each). */
#include "../native/qd_protocol.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "../native/qd_conv.h"
#define ARENA 0xf1800000u
__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;__asm__ volatile("%0=htid":"=r"(tid));if(tid>=3)return;
    volatile uint32_t *a=(volatile uint32_t *)ARENA;
    uint32_t seq=a[2];
    volatile uint32_t *out=(volatile uint32_t *)(ARENA+2048);
    if(tid==0) {
        volatile uint32_t *table=(volatile uint32_t *)(ARENA+8192);
        for(uint32_t i=0;i<128;i++) {
            uint64_t e;__asm__ volatile("%0=tlbr(%1)":"=r"(e):"r"(i):"memory");
            table[2*i]=(uint32_t)e;table[2*i+1]=(uint32_t)(e>>32);
        }
        out[0]=0x4254314a;out[1]=0;
    }
    volatile QdStatus *st=(volatile QdStatus *)(ARENA+QD_STATUS_OFFSET)+tid;
    st->error=0;st->rows=0;st->reserved=QDC_EXITED;
    __asm__ volatile("barrier":::"memory");st->done_sequence=seq;
}
