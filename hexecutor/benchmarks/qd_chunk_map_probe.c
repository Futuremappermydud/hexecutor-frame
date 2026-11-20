/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "../native/qd_protocol.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "../native/qd_conv.h"
#include "../native/qd_owned_memory.h"
#define ARENA 0xf1800000u
static uint64_t cycles(void){uint64_t c;__asm__ volatile("%0=upcycle":"=r"(c));return c;}
__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;__asm__ volatile("%0=htid":"=r"(tid));if(tid>=3)return;
    volatile uint32_t *a=(volatile uint32_t *)ARENA;
    uint32_t seq=a[2],nonce=a[4];
    QdoMap *s=(QdoMap *)(ARENA+QDO_STATE);
    volatile uint32_t *out=(volatile uint32_t *)(ARENA+2048);
    if(tid==0) {
        out[0]=0x31544d51;out[1]=qdo_map(s);out[2]=0;out[3]=0;
        if(!out[1]) {
            for(uint32_t i=0;i<QDO_SLOTS && i<24;i++) {
                uint64_t entry=qdo_tlbr(QDO_IDX(s,i));
                out[51+2*i]=(uint32_t)entry;out[52+2*i]=(uint32_t)(entry>>32);
            }
            /* Read-only chunk map: word 0 of the first page of every 2 MB chunk. */
            /* The table moved to ARENA+8192 (out[4] = its word index in `out`) so windows beyond
             * 128 MiB cannot run into the mapper state at ARENA+3072; the legacy table at
             * out[100..] is still filled when it fits below that state. */
            volatile uint32_t *table=(volatile uint32_t *)(ARENA+8192);
            for(uint32_t ch=0;ch<QDO_BYTES/0x200000u;ch++) {
                uint32_t v=*(volatile uint32_t *)(QDO_BASE+ch*0x200000u);
                table[ch]=v;
                if(QDO_SLOTS<=8)out[100+ch]=v;
            }
            out[3]=QDO_BYTES/0x200000u;out[4]=2048;
            out[6]=qdo_unmap(s);
            if(out[6])out[1]=out[6];
        }
    }
    volatile QdStatus *st=(volatile QdStatus *)(ARENA+QD_STATUS_OFFSET)+tid;
    st->error=0;st->rows=0;st->reserved=QDC_EXITED;
    __asm__ volatile("barrier":::"memory");st->done_sequence=seq;
}
