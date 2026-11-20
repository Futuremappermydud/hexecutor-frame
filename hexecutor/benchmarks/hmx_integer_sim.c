/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Integer HMX arithmetic/layout probe, for Qualcomm's ISA simulator only. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
static void dot(void *a,void *b,void *fi,void *out,uint32_t convert) {
    __asm__ volatile("mxclracc\n bias = mxmem2(%2)\n"
                     "{ activation.ub = mxmem(%0,%4)\n weight.b = mxmem(%1,%5) }\n"
                     "cvt.ub = acc(%6)\n mxmem(%3,%7) = cvt\n"
                     ::"r"(a),"r"(b),"r"(fi),"r"(out),"r"(2047),"r"(1023),"r"(convert),"r"(0):"memory");
    for(uint32_t i=0;i<128;i++)__asm__ volatile("nop");
}
int main(void) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    printf("vtcm=%08x tcm=%08x coproc=%08x\n",(unsigned)base,(unsigned)__rdcfg(__tcm_size),(unsigned)__rdcfg(__coproc_type));
    for(uint32_t i=0;i<16384;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    unsigned char *a=(unsigned char *)base,*b=a+2048,*out=a+4096;
    int32_t *fi=(int32_t *)(base+3072);
    for(uint32_t scale=0x3000;scale<=0x6800;scale+=0x400)for(uint32_t conv=0;conv<4;conv++) {
        memset(a,1,2048);memset(b,1,1024);memset(fi,0,256);memset(out,0xee,4096);
        for(int i=0;i<32;i++)fi[i]=0x400000|scale;
        dot(a,b,fi,out,conv);
        printf("s=%04x c=%u:",(unsigned)scale,(unsigned)conv);
        for(int i=0;i<8;i++)printf(" %u",(unsigned)out[i]);
        unsigned sum=0;for(int i=0;i<4096;i++)sum+=out[i];printf(" sum=%u\n",sum);
    }
    return 0;
}
