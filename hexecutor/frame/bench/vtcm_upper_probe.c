/* Inventory upper VTCM during an exclusive callback, before granting any use.
 * Mode 0 is read-only. Mode 1 saves the complete range, checks unique page
 * sentinels, and restores/verifies EVERY byte before publishing exit.
 * The backup is in the already owned stock arena, not firmware memory. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "../qd_protocol.h"
#include "../qd_conv.h"
static uint64_t cycles(void){uint64_t c;__asm__ volatile("%0=upcycle":"=r"(c));return c;}
#include "../qd_axim.h"
#define ARENA 0xf1800000u
#define MAGIC 0x31545551u
#define LOW 0x2d0000u
#define BYTES (0x800000u-LOW)
#define BACKUP 0x10000u
static int same(const volatile HVX_Vector *a,const volatile HVX_Vector *b,uint32_t bytes) {
    HVX_Vector diff=Q6_V_vzero();
    for(uint32_t i=0;i<bytes/128;i++)diff=Q6_V_vor_VV(diff,Q6_V_vxor_VV(a[i],b[i]));
    uint32_t words[32] __attribute__((aligned(128)));*(HVX_Vector *)words=diff;
    uint32_t any=0;for(uint32_t i=0;i<32;i++)any|=words[i];return !any;
}
static HVX_Vector pattern(uint32_t page,uint32_t nonce) {
    uint32_t words[32] __attribute__((aligned(128)));
    for(uint32_t i=0;i<32;i++)words[i]=nonce^(page*0x9e3779b9u)^(i*0x7f4a7c15u);
    return *(HVX_Vector *)words;
}
__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;__asm__ volatile("%0=htid":"=r"(tid));if(tid>=3)return;
    volatile uint32_t *cmd=(volatile uint32_t *)ARENA,*out=(volatile uint32_t *)(ARENA+2048);
    uint32_t seq=cmd[2];
    if(!tid) {
        for(uint32_t i=0;i<32;i++)out[i]=0;
        out[0]=MAGIC;out[2]=1;
        uint64_t end=cycles()+2000000000u;
        while(*(volatile uint32_t *)(ARENA+144)!=seq || *(volatile uint32_t *)(ARENA+160)!=seq)
            if(cycles()>end){out[1]=1;goto done;}
        if(cmd[0]!=MAGIC || cmd[1]!=1 || cmd[3]>1 || !cmd[4] || cmd[4]>32){out[1]=2;goto done;}
        uint64_t e8,e9;
        __asm__ volatile("%0=tlbr(%1)":"=r"(e8):"r"(8));
        __asm__ volatile("%0=tlbr(%1)":"=r"(e9):"r"(9));
        out[16]=e8;out[17]=e8>>32;out[18]=e9;out[19]=e9>>32;
        if(e8!=0xc0031000f7062020ull || e9!=0xc0031400f7062820ull){out[1]=7;goto done;}
        volatile uint32_t *clocks=(volatile uint32_t *)(ARENA+QDC_CLOCK_OFFSET);
        out[1]=clock_raise(clocks);if(out[1])goto restore_clock;
        volatile HVX_Vector *vtcm=(volatile HVX_Vector *)(QDC_VTCM_BASE+LOW);
        volatile HVX_Vector *backup=(volatile HVX_Vector *)(ARENA+BACKUP);
        uint64_t started=cycles();
        /* Write mode must reuse the host-validated read-only snapshot from
         * the immediately preceding command in this same exclusive lease. */
        if(!cmd[3])for(uint32_t i=0;i<BYTES/128;i++)backup[i]=vtcm[i];
        __asm__ volatile("syncht":::"memory");
        out[4]=(uint32_t)(cycles()-started);out[2]=2;
        for(uint32_t pass=0;pass<cmd[4];pass++) {
            if(!same(vtcm,backup,BYTES)){out[1]=3;out[5]=pass;goto restore_clock;}
            out[6]++;
            end=cycles()+18432000u;while(cycles()<end)__asm__ volatile("pause(#32)");
        }
        out[2]=3;
        if(cmd[3]) {
            started=cycles();
            /* All pages stay tagged until all are checked: detects aliasing. */
            for(uint32_t page=0;page<BYTES/4096;page++)vtcm[page*32]=pattern(page,cmd[5]);
            __asm__ volatile("syncht":::"memory");
            for(uint32_t page=0;page<BYTES/4096;page++) {
                HVX_Vector want=pattern(page,cmd[5]);
                if(!same(vtcm+page*32,&want,128)){out[7]++;out[8]=page;}
            }
            out[9]=(uint32_t)(cycles()-started);out[2]=4;
            started=cycles();
            for(uint32_t i=0;i<BYTES/128;i++)vtcm[i]=backup[i];
            __asm__ volatile("syncht":::"memory");
            out[10]=(uint32_t)(cycles()-started);
            if(!same(vtcm,backup,BYTES)){out[1]=5;out[2]=5;goto restore_clock;}
            out[11]=1;
            if(out[7])out[1]=4;
        }
        out[2]=6;out[12]=BYTES;out[13]=BYTES/4096;
restore_clock:
        out[14]=clock_restore(clocks);if(out[14])out[1]=6;
    }
done:
    __asm__ volatile("barrier":::"memory");
    volatile QdStatus *st=(volatile QdStatus *)(ARENA+QD_STATUS_OFFSET)+tid;
    st->error=tid?0:out[1];st->rows=0;st->reserved=QDC_EXITED;
    __asm__ volatile("barrier":::"memory");st->done_sequence=seq;
}
