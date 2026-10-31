#include "../qd_protocol.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "../qd_conv.h"
#include "../qd_owned_memory.h"
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
            for(uint32_t i=0;i<QDO_SLOTS;i++) {
                uint64_t entry=qdo_tlbr(QDO_FIRST+i);
                out[51+2*i]=(uint32_t)entry;out[52+2*i]=(uint32_t)(entry>>32);
            }
            uint64_t start=cycles();
            for(uint32_t off=0;off<QDO_BYTES;off+=4096) {
                volatile uint32_t *p=(volatile uint32_t *)(QDO_BASE+off);
                uint32_t expected=nonce^off;
                /* Complete a read-only proof of every page before writing to
                 * any mapped address. Fresh nonces reject stale old leases. */
                uint32_t indexes[3]={0,1,1023};
                uint32_t values[3]={expected,~expected,expected^0xb519274du};
                for(uint32_t i=0;i<3;i++) {
                    uint32_t got=p[indexes[i]];
                    if(got!=values[i]) {
                        out[1]=5;out[2]=off+indexes[i]*4;
                        out[48]=got;out[49]=values[i];out[50]=nonce;break;
                    }
                }
                if(out[1])break;
                out[3]++;
            }
            __asm__ volatile("barrier":::"memory");
            uint64_t elapsed=cycles()-start;out[4]=(uint32_t)elapsed;out[5]=(uint32_t)(elapsed>>32);
            /* Test scalar/vector visibility independently on every mapped page.
             * No firmware addresses: all data is inside our own allocation. */
            if(!out[1])for(uint32_t off=0;off<QDO_BYTES;off+=4096)
                *(volatile uint32_t *)(QDO_BASE+off+8)=~(nonce^off);
            if(!out[1])for(uint32_t page=0;page<QDO_SLOTS;page++) {
                uint32_t addr=QDO_BASE+page*0x1000000u+128;
                volatile uint32_t *scalar=(volatile uint32_t *)addr;
                volatile HVX_Vector *vec=(volatile HVX_Vector *)addr;
                __asm__ volatile("dcinva(%0)\n barrier"::"r"(addr):"memory");
                *vec=Q6_V_vsplat_R(1);__asm__ volatile("barrier":::"memory");
                out[8+page*5]=scalar[0];scalar[0]=2;__asm__ volatile("barrier":::"memory");
                out[9+page*5]=(uint32_t)(*(volatile HVX_Vector *)vec)[0];
                __asm__ volatile("dccleana(%0)\n barrier"::"r"(addr):"memory");
                out[10+page*5]=(uint32_t)(*(volatile HVX_Vector *)vec)[0];
                *vec=Q6_V_vsplat_R(3);__asm__ volatile("barrier":::"memory");
                out[11+page*5]=scalar[0];
                __asm__ volatile("dcinva(%0)\n barrier"::"r"(addr):"memory");
                out[12+page*5]=scalar[0];
            }
            out[6]=qdo_unmap(s);
            if(out[6])out[1]=out[6];
        }
    }
    volatile QdStatus *st=(volatile QdStatus *)(ARENA+QD_STATUS_OFFSET)+tid;
    st->error=0;st->rows=0;st->reserved=QDC_EXITED;
    __asm__ volatile("barrier":::"memory");st->done_sequence=seq;
}
