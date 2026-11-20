/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Hardware L2 size sweep. One requested SYSCFG.L2CFG value per callback.
 * Thread 0 runs on the firmware DDR stack, saves all 1 MiB TCM to uncached
 * arena, accesses only private DDR while cache is on, then disables cache and
 * restores all TCM bytes. Workers 1/2 do not touch TCM. */
#include <stdint.h>
#include "../native/qd_protocol.h"
#define ARENA 0xf1800000u
#define OUT ((volatile uint32_t *)(ARENA+2048u))
#define SAVE ((volatile uint32_t *)(ARENA+0x100000u))
#define TCM ((volatile uint32_t *)0x30000000u)
#define TEST ((volatile uint32_t *)0x88b00000u) /* arena +0x300000, C7 */
#define WORDS (0x100000u/4u)
static uint64_t cycle(void){uint64_t v;__asm__ volatile("%0=upcycle":"=r"(v));return v;}
static uint32_t syscfg(void){uint32_t v;__asm__ volatile("%0=syscfg":"=r"(v));return v;}
static uint32_t ccr(void){uint32_t v;__asm__ volatile("%0=ccr":"=r"(v));return v;}
static void setcfg(uint32_t v){__asm__ volatile("isync\n\tsyncht\n\tsyscfg=%0\n\tisync"::"r"(v):"memory");}
static uint32_t pass(uint32_t words){uint32_t sum=0;for(uint32_t i=0;i<words;i++)sum+=TEST[i];return sum;}
__attribute__((section(".text.entry"))) void payload(void){
    uint32_t tid;__asm__ volatile("%0=htid":"=r"(tid));if(tid>=3)return;
    volatile uint32_t *a=(volatile uint32_t *)ARENA;uint32_t seq=a[2],err=0;
    if(tid==0){
        uint32_t request=a[16],old=syscfg(),oldccr=ccr();
        OUT[0]=0x5057324cu;OUT[1]=request;OUT[2]=old;OUT[3]=oldccr;
        if(request>5u||old!=0x0a80a057u||oldccr!=0x150049u)err=1;
        if(!err){
            for(uint32_t i=0;i<WORDS;i++)SAVE[i]=TCM[i];
            uint32_t cleared=oldccr&~(15u<<16);
            if(request){
                __asm__ volatile("ccr=%0\n\tisync"::"r"(cleared):"memory");
                setcfg(old&~(7u<<16));
                __asm__ volatile("l2kill\n\tsyncht":::"memory");
                setcfg((old&~(7u<<16))|(request<<16));
                __asm__ volatile("ccr=%0\n\tisync"::"r"(oldccr):"memory");
            }
            OUT[4]=syscfg();
            const uint32_t sizes[6]={2048u,16384u,65536u,131072u,196608u,262144u};
            for(uint32_t k=0;k<6;k++){
                uint64_t t0=cycle();uint32_t s0=pass(sizes[k]);uint64_t t1=cycle();
                uint32_t s1=pass(sizes[k]);uint64_t t2=cycle();
                OUT[8+4*k]=(uint32_t)(t1-t0);OUT[9+4*k]=(uint32_t)(t2-t1);
                OUT[10+4*k]=s0;OUT[11+4*k]=s1;
            }
            if(request){
                __asm__ volatile("ccr=%0\n\tisync"::"r"(cleared):"memory");
                __asm__ volatile("syncht\n\tl2gcleaninv\n\tsyncht":::"memory");
                setcfg(old&~(7u<<16));
            }
            OUT[5]=syscfg();
            for(uint32_t s=0;s<64;s++)OUT[32+s]=0;
            uint32_t changed=0;
            for(uint32_t i=0;i<WORDS;i++)if(TCM[i]!=SAVE[i]){changed++;OUT[32+i/4096u]++;}
            OUT[6]=changed;
            for(uint32_t i=0;i<WORDS;i++)TCM[i]=SAVE[i];
            __asm__ volatile("ccr=%0\n\tisync"::"r"(oldccr):"memory");
            OUT[7]=ccr();
            if(OUT[4]!=((old&~(7u<<16))|(request<<16))||OUT[5]!=old||OUT[7]!=oldccr)err=2;
        }
        OUT[96]=err;
    }
    volatile QdStatus *st=(volatile QdStatus *)(ARENA+QD_STATUS_OFFSET)+tid;
    st->error=err;st->rows=0;st->reserved=0x45584954u;
    __asm__ volatile("barrier":::"memory");st->done_sequence=seq;
}
