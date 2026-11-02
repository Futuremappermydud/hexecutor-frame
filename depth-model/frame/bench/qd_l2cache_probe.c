/* Bounded L2 cache carve-out probe. Run with all three callback workers idle.
 * Thread 0 uses its firmware DDR stack. The full 1 MiB of L2-TCM is saved
 * through the uncached arena before changing SYSCFG, then restored afterwards.
 * This is a hardware probe, not a production cache manager. */
#include <stdint.h>
#include "../qd_protocol.h"
#include "../qd_conv.h"
#define ARENA 0xf1800000u
#define OUT ((volatile uint32_t *)(ARENA+2048u))
#define SAVE ((volatile uint32_t *)(ARENA+0x100000u))
#define TCM ((volatile uint32_t *)0x30000000u)
#define WORDS (0x100000u/4u)
/* Cacheable alias of arena +0x40000; f1840000 is its uncached alias. */
#define TEST ((volatile uint32_t *)0x88840000u)

static uint64_t cycle(void) { uint64_t v; __asm__ volatile("%0=upcycle":"=r"(v)); return v; }
static uint32_t syscfg(void) { uint32_t v; __asm__ volatile("%0=syscfg":"=r"(v)); return v; }
static uint32_t ccr(void) { uint32_t v; __asm__ volatile("%0=ccr":"=r"(v)); return v; }
static void setcfg(uint32_t v) {
    __asm__ volatile("isync\n\tsyncht\n\tsyscfg=%0\n\tisync"::"r"(v):"memory");
}
static uint32_t pass(void) {
    uint32_t sum=0;
    for(uint32_t i=0;i<8192u;i+=16u)
        sum+=TEST[i]+TEST[i+1]+TEST[i+2]+TEST[i+3];
    return sum;
}
__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid; __asm__ volatile("%0=htid":"=r"(tid)); if(tid>=3)return;
    volatile uint32_t *a=(volatile uint32_t *)ARENA;
    uint32_t seq=a[2], error=0;
    if(tid==0) {
        OUT[0]=0x4341324cu; /* L2AC */
        uint32_t old=syscfg(), oldccr=ccr();
        OUT[1]=old;OUT[2]=oldccr;
        if(((old>>16)&7u)!=0u)error=1;
        if(!error) {
            for(uint32_t i=0;i<WORDS;i++)SAVE[i]=TCM[i];
            uint32_t cleared=oldccr&~(15u<<16);
            __asm__ volatile("ccr=%0\n\tisync"::"r"(cleared):"memory");
            /* SDK standalone CRT sequence, with a 64 KiB request (field 1). */
            setcfg(old&~(7u<<16));
            __asm__ volatile("l2kill\n\tsyncht":::"memory");
            setcfg((old&~(7u<<16))|(1u<<16));
            OUT[3]=syscfg();
            __asm__ volatile("ccr=%0\n\tisync"::"r"(oldccr):"memory");
            uint64_t t0=cycle();uint32_t s0=pass();uint64_t t1=cycle();
            uint32_t s1=pass();uint64_t t2=cycle();
            OUT[4]=(uint32_t)(t1-t0);OUT[5]=(uint32_t)(t2-t1);
            OUT[6]=s0;OUT[7]=s1;
            __asm__ volatile("ccr=%0\n\tisync"::"r"(cleared):"memory");
            __asm__ volatile("syncht\n\tl2gcleaninv\n\tsyncht":::"memory");
            setcfg(old&~(7u<<16));
            OUT[8]=syscfg();
            uint32_t changed=0,first=WORDS;
            for(uint32_t i=0;i<64;i++)OUT[32+i]=0;
            for(uint32_t i=0;i<WORDS;i++)if(TCM[i]!=SAVE[i]) {
                changed++;OUT[32+i/4096u]++;if(first==WORDS)first=i;
            }
            OUT[9]=changed;OUT[10]=first;
            for(uint32_t i=0;i<WORDS;i++)TCM[i]=SAVE[i];
            __asm__ volatile("ccr=%0\n\tisync"::"r"(oldccr):"memory");
            OUT[11]=ccr();
        }
        OUT[12]=error;
    }
    volatile QdStatus *st=(volatile QdStatus *)(ARENA+QD_STATUS_OFFSET)+tid;
    st->error=error;st->rows=0;st->reserved=0x45584954u;
    __asm__ volatile("barrier":::"memory");st->done_sequence=seq;
}
