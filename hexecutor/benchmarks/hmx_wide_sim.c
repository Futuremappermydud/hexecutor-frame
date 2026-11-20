/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Determine the integer HMX wide-output contract using only the ISA simulator. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
static void dot(void *a,void *b,void *fi,void *out,uint32_t op) {
    __asm__ volatile("mxclracc\n bias=mxmem2(%2)\n"
                     "{ activation.ub=mxmem(%0,%3)\n weight.b=mxmem(%1,%4) }\n"
                     ::"r"(a),"r"(b),"r"(fi),"r"(2047),"r"(1023):"memory");
    if(op==0)__asm__ volatile("cvt.uh=acc(%1):2x1\n mxmem(%0,%1)=cvt"::"r"(out),"r"(0):"memory");
    if(op==1)__asm__ volatile("cvt.uh=acc(%1):2x1\n mxmem(%0,%1):2x2=cvt"::"r"(out),"r"(0):"memory");
    if(op==2)__asm__ volatile("mxmem(%0,%1):after:sat.uh=acc:2x1"::"r"(out),"r"(0):"memory");
    if(op==3)__asm__ volatile("mxmem(%0,%1):before:sat.uh=acc:2x1"::"r"(out),"r"(0):"memory");
    if(op==4)__asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1"::"r"(out),"r"(0):"memory");
    for(int i=0;i<256;i++)__asm__ volatile("nop");
}
int main(void) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<32768;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    unsigned char *a=(unsigned char *)base,*b=a+2048;int32_t *fi=(int32_t *)(base+3072);uint16_t *out=(uint16_t *)(base+4096);
    for(uint32_t bias=0;bias<3;bias++)for(uint32_t h=0x4000;h<=0x6000;h+=0x800)for(uint32_t op=0;op<5;op++) {
        memset(a,bias?0:1,2048);memset(b,bias==1?0:1,1024);memset(fi,0,256);memset(out,0xee,8192);
        if(bias==2)for(unsigned m=0;m<64;m++)for(unsigned k=0;k<32;k++)a[(m/4*32+k)*4+m%4]=(unsigned char)(m+1);
        for(int n=0;n<32;n++){fi[n]=0x400000|h;fi[32+n]=bias==1?n+1:0;}
        dot(a,b,fi,out,op);
        unsigned count=0;for(unsigned i=0;i<4096;i++)if(out[i]!=0xeeee)count++;
        printf("bias=%u h=%04x op=%u count=%u:",(unsigned)bias,(unsigned)h,(unsigned)op,count);
        for(int i=0;i<16;i++)printf(" %u",(unsigned)out[i]);printf(" second=%u",(unsigned)out[1024]);
        if(bias==2 && h==0x4000 && op==0){printf(" rows:");for(unsigned m=0;m<32;m++)printf(" %u",(unsigned)out[(m/2*32)*2+m%2]);}
        printf("\n");
    }
    return 0;
}
