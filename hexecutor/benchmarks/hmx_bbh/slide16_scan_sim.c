/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Scan Rs/Rt low bits of the 16-bit activation load: which taps read what. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <hexagon_standalone.h>
extern void qd_hmx_finish(void *,void *);
static uint8_t *done;static int32_t *zero;
static void fin(void){*(volatile uint32_t *)done=0xffffffff;qd_hmx_finish(zero,done);uint32_t t=0;while(*(volatile uint32_t *)done&&t++<100000)__asm__ volatile("nop");if(*(volatile uint32_t*)done)puts("NO COMPLETION");}
int main(int argc,char **argv) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x100000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint8_t *blk=(uint8_t *)base;
    uint8_t *w=(uint8_t *)base+0x20000,*out=(uint8_t *)base+0x30000;
    int32_t *fi=(int32_t *)(base+0x38000);zero=(int32_t *)(base+0x38800);done=(uint8_t *)base+0x39000;
    for(int b=0;b<4;b++){uint8_t *p=blk+b*0x4000;memset(p,0,2048);
        for(int y=0;y<8;y++)for(int x=0;x<4;x++){int pp=x>>1,j=x&1;uint8_t *q=p+(y*2+pp)*128;
            q[0*4+2*j+1]=(uint8_t)(b*64+y*8+x);q[1*4+2*j+1]=1;}}
    memset(w,0,32*1024);
    for(int t=0;t<16;t++){w[t*1024+((0/4)*32+t)*4+0]=1;w[t*1024+((1/4)*32+16+t)*4+1]=1;}
    for(int c=0;c<32;c++){fi[c]=0x404000;fi[32+c]=0;zero[c]=0x404000;zero[32+c]=0;}
    static const uint32_t rts[]={0x700,0x71f,0x73f,0x77f,0x7ff,0x3f,0x1f,0x0};
    uint8_t *A0=blk,*B0=blk+0x4000;
    const int T=12;uint32_t wr=T*1024-1;
    for(int dil=0;dil<2;dil++)for(unsigned ri=0;ri<sizeof rts/sizeof *rts;ri++)for(uint32_t rs=0;rs<0x800;rs+=0x40) {
        uint32_t a=(uint32_t)A0|rs,t=(uint32_t)(B0-A0)|rts[ri];
        memset(out,0xee,0x800);
        __asm__ volatile("mxclracc\n bias=mxmem2(%0)\n"::"r"(fi):"memory");
        if(dil)__asm__ volatile("{ activation.ub=mxmem(%0,%1)\n weight.b=mxmem(%2,%3):dilate }\n"::"r"(a),"r"(t),"r"(w),"r"(wr):"memory");
        else __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n weight.b=mxmem(%2,%3) }\n"::"r"(a),"r"(t),"r"(w),"r"(wr):"memory");
        __asm__ volatile("cvt.uh=acc(%1):2x1\n mxmem(%0,%1)=cvt\n"::"r"(out),"r"(0):"memory");
        fin();
        uint16_t *o=(uint16_t *)out;
        printf("d%d rt=%03lx rs=%03lx:",dil,(unsigned long)rts[ri],(unsigned long)rs);
        for(int k=0;k<T;k++){uint16_t a=o[k*2],b=o[k*2+1],c=o[64+k*2],d=o[128+k*2],ma=o[(16+k)*2];
            if(ma==0&&a==0&&b==0)printf(" .");else printf(" [%03o %03o %03o %03o]",a,b,c,d);}
        printf("\n");
    }
    return 0;
}
