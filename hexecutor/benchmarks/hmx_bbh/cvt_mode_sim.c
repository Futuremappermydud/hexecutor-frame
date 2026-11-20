/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* What does cvt.ub=acc(Rs) do with Rs low bits and bits 12/13, on native 8-bit
 * crouton accumulators (QNN hmx_v75_convbbh1x1_stride1)? */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <hexagon_standalone.h>
static void dump(const char *n,uint8_t *o){
    printf("%s:",n);int cnt=0;for(int i=0;i<2048;i++)cnt+=o[i]!=0xee;printf(" written=%d\n",cnt);
    for(int y=0;y<2;y++){printf("  y%d:",y);for(int x=0;x<8;x++)printf(" %3u/%3u",o[(y*8+x)*32],o[(y*8+x)*32+1]);printf("\n");}
    printf("  raw:");for(int i=0;i<24;i++)printf(" %u",o[i]);printf("\n");
}
int main(void) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x100000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint8_t *A=(uint8_t *)base,*B=A+0x1000,*O=A+0x4000;int32_t *fi=(int32_t *)(A+0x8000);
    /* A: pixel p=y*8+x, input lane 0 = 1+p, lane 4 = 1+p (second byte group for channel 1) */
    memset(A,0,2048);for(int p=0;p<64;p++){A[p*32]=(uint8_t)(1+p);}
    /* B: weight 100 from lane 0 to every output channel n (bytes ((k/4)*32+n)*4+k%4, k=0) */
    memset(B,0,2048);for(int n=0;n<32;n++)B[(0*32+n)*4]=100;
    /* scale exp so x = acc*2^-? : banks differ */
    static const uint32_t sels[]={0,1,2,3,0x1000,0x1003,0x2000,0x2003,0x3000,0x3003,0x1002,0x2002,0x3002};
    for(uint32_t h=0x5000;h<=0x5000;h+=0x800)
    for(unsigned s=0;s<sizeof sels/sizeof *sels;s++) {
        /* bank i: scale h (e=20,M=1024 => pre 0), bias 1000*i */
        for(int b=0;b<4;b++)for(int n=0;n<32;n++){fi[b*64+n]=0x400000|h;fi[b*64+32+n]=b*7;}
        memset(O,0xee,0x4000);
        uint32_t sel=sels[s];
        __asm__ volatile(
         "mxclracc\n"
         "bias=mxmem2(%[f1])\n bias=mxmem2(%[f2])\n bias=mxmem2(%[f3])\n bias=mxmem2(%[f0])\n"
         "{ activation.ub=mxmem(%[a],%[ar]):deep:cm\n weight.b=mxmem(%[b],%[br]) }\n"
         "cvt.ub=acc(%[s])\n"
         "mxmem(%[o],%[z])=cvt\n"
         ::[f1]"r"((uint32_t)(fi+64)|1),[f2]"r"((uint32_t)(fi+128)|2),[f3]"r"((uint32_t)(fi+192)|3),[f0]"r"(fi),
           [a]"r"(A),[ar]"r"(2047),[b]"r"(B),[br]"r"(1023),[s]"r"(sel),[o]"r"(O),[z]"r"(0):"memory");
        for(int i=0;i<200;i++)__asm__ volatile("nop");
        char nm[64];sprintf(nm,"h=%04lx sel=%04lx",(unsigned long)h,(unsigned long)sel);dump(nm,O);
    }
    return 0;
}
