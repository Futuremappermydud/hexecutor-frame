/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Does cvt.ub=acc(sel) ever wrap instead of saturate? A sweep of x through bias:
 * one input lane * weight 0 (acc 0) so x = bias; store :cm so out[p*32+n]. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <hexagon_standalone.h>
int main(void) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x100000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint8_t *A=(uint8_t *)base,*B=A+0x1000,*O=A+0x4000;int32_t *fi=(int32_t *)(A+0x8000);
    memset(A,0,2048);memset(B,0,2048);
    static const uint32_t sels[]={0,1,2,3,0x1003,0x2003,0x3003};
    static const uint32_t hs[]={0x400000|0x5000,0x400000|0x4000,0x400000|0x6000,0x7000|0x400000,0x5000,0x4000,0x7000};
    static const int32_t xs[]={0,1,15,16,255,256,4080,4096,4097,65535,65536,70000,-1,-100,1<<20,(1<<20)+300,0x7fffff};
    for(unsigned hi=0;hi<sizeof hs/sizeof *hs;hi++)for(unsigned s=0;s<sizeof sels/sizeof *sels;s++) {
        for(int n=0;n<32;n++){for(int b=0;b<4;b++){fi[b*64+n]=hs[hi];fi[b*64+32+n]=n<17?xs[n]:0;}}
        memset(O,0xee,4096);
        uint32_t sel=sels[s];
        __asm__ volatile(
         "mxclracc\n"
         "bias=mxmem2(%[f1])\n bias=mxmem2(%[f2])\n bias=mxmem2(%[f3])\n bias=mxmem2(%[f0])\n"
         "{ activation.ub=mxmem(%[a],%[ar]):deep:cm\n weight.b=mxmem(%[b],%[br]) }\n"
         "cvt.ub=acc(%[s])\n"
         "mxmem(%[o],%[z]):cm=cvt\n"
         ::[f1]"r"((uint32_t)(fi+64)|1),[f2]"r"((uint32_t)(fi+128)|2),[f3]"r"((uint32_t)(fi+192)|3),[f0]"r"(fi),
           [a]"r"(A),[ar]"r"(2047),[b]"r"(B),[br]"r"(1023),[s]"r"(sel),[o]"r"(O),[z]"r"(0):"memory");
        for(int i=0;i<200;i++)__asm__ volatile("nop");
        printf("h=%07lx sel=%04lx:",(unsigned long)hs[hi],(unsigned long)sel);
        for(int n=0;n<17;n++)printf(" %u",O[n]);printf("\n");
    }
    return 0;
}
