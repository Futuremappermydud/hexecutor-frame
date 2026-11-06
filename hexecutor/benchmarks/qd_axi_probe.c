#include <stdint.h>
#include "../native/qd_protocol.h"
#include "../native/qd_conv.h"
static uint64_t cycles(void){uint64_t c;__asm__ volatile("%0=upcycle":"=r"(c));return c;}
#include "../native/qd_axim.h"
typedef uint32_t Vec __attribute__((vector_size(128)));
#define A 0xf1800000u
static uint32_t vector_read(uint32_t *sink) {
    const volatile Vec *p=(const volatile Vec *)(A+0x100000);
    Vec a={0},b={0},c={0},d={0};uint64_t t=cycles();
    for(uint32_t i=0;i<8192;i+=4){a^=p[i];b^=p[i+1];c^=p[i+2];d^=p[i+3];}
    *sink^=(a^b^c^d)[0];return (uint32_t)(cycles()-t);
}
static uint32_t scalar_read(uint32_t *sink) {
    const volatile uint32_t *p=(const volatile uint32_t *)(A+0x100000);
    uint32_t i=0;uint64_t t=cycles();
    for(uint32_t j=0;j<4096;j++)i=p[i]&4095;
    *sink^=i;return (uint32_t)(cycles()-t);
}
__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;__asm__ volatile("%0=htid":"=r"(tid));if(tid>=3)return;
    volatile uint32_t *cmd=(volatile uint32_t *)A,*out=(volatile uint32_t *)(A+2048);
    volatile uint32_t *state=(volatile uint32_t *)(A+QDC_CLOCK_OFFSET);
    uint32_t seq=cmd[2];
    if(!tid) {
        uint32_t sink=0;out[0]=QDC_AXIM_MHZ;out[1]=clock_raise(state);
        if(!out[1]) {
            out[2]=QDC_CORE2_PLL[4];out[3]=QDC_CORE2_PLL[5];out[4]=QDC_AXIM_CC[1];
            for(uint32_t r=0;r<10;r++){out[8+2*r]=vector_read(&sink);out[9+2*r]=scalar_read(&sink);}
        }
        out[5]=clock_restore(state);out[6]=state[0];out[7]=sink;
    }
    volatile QdStatus *st=(volatile QdStatus *)(A+QD_STATUS_OFFSET)+tid;
    st->error=st->rows=0;st->reserved=QDC_EXITED;
    __asm__ volatile("barrier":::"memory");st->done_sequence=seq;
}
