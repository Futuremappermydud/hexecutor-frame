/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Two conversions per native :cm tile: what survives the first cvt?
 * Bank 1 hi=floor(x/256)+128 clipped; bank 0 lo = x mod 256 (sel bit 1). */
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
    for(int p=0;p<64;p++)for(int c=0;c<32;c++)A[p*32+c]=(uint8_t)(1+p*3+c);
    for(int n=0;n<32;n++)B[(n/4*32+n)*4+n%4]=100; /* output n reads input lane n */
    for(int n=0;n<32;n++){fi[n]=0x6000;fi[32+n]=0;fi[64+n]=0x4000;fi[64+32+n]=1<<15;}
    static const uint32_t s1[]={0x1000,0x1002,0x1003,0x1001,0x1000,0x1000};
    static const uint32_t s2[]={2,2,2,2,3,0};
    for(unsigned t=0;t<sizeof s1/sizeof *s1;t++){
        memset(O,0xee,0x3000);
        __asm__ volatile(
         "mxclracc\n bias=mxmem2(%[f1])\n bias=mxmem2(%[f0])\n"
         "{ activation.ub=mxmem(%[a],%[ar]):deep:cm\n weight.b=mxmem(%[b],%[br]) }\n"
         "cvt.ub=acc(%[s1])\n mxmem(%[o1],%[z]):cm=cvt\n"
         "cvt.ub=acc(%[s2])\n mxmem(%[o2],%[z]):cm=cvt\n"
         "cvt.ub=acc(%[s2])\n mxmem(%[o3],%[z]):cm=cvt\n"
         ::[f1]"r"((uint32_t)(fi+64)|1),[f0]"r"(fi),[a]"r"(A),[ar]"r"(2047),[b]"r"(B),[br]"r"(1023),
           [s1]"r"(s1[t]),[s2]"r"(s2[t]),[o1]"r"(O),[o2]"r"(O+0x1000),[o3]"r"(O+0x2000),[z]"r"(0):"memory");
        for(int i=0;i<300;i++)__asm__ volatile("nop");
        long badh=0,badl=0,bad3=0;
        for(int p=0;p<64;p++)for(int n=0;n<32;n++){
            int x=(1+p*3+n)*100,xa=A[p*32+n]*100;
            int h=(xa>>8)+128,l=xa&255;
            if(O[p*32+n]!=h)badh++;if(O[0x1000+p*32+n]!=l)badl++;if(O[0x2000+p*32+n]!=l)bad3++;
        }
        printf("s1=%04lx s2=%04lx: hi bad %ld lo bad %ld third==lo bad %ld | sample hi %u lo %u third %u (want %d %d)\n",(unsigned long)s1[t],(unsigned long)s2[t],badh,badl,bad3,
          O[32*5+3],O[0x1000+32*5+3],O[0x2000+32*5+3],(A[32*5+3]*100>>8)+128,A[32*5+3]*100&255);
    }
    return 0;
}
