/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Decode HMX non-deep activation windows: tagged blocks, one-hot weights.
 * Block b pixel (y,x) channel 0 = b*64+y*8+x (+1 so 0 = "zero/none" marker? no: use +0 and channel1=1 marker). */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <hexagon_standalone.h>
extern void qd_hmx_finish(void *,void *);
static uint8_t *done;static int32_t *zero;
static void fin(void){*(volatile uint32_t *)done=0xffffffff;qd_hmx_finish(zero,done);uint32_t t=0;while(*(volatile uint32_t *)done&&t++<100000)__asm__ volatile("nop");if(*(volatile uint32_t*)done)puts("NO COMPLETION");}
#define RUN(NAME,MODE) static void run_##NAME(void *a,uint32_t rt,void *b,uint32_t wrt,void *fi,void *out){ \
    __asm__ volatile("mxclracc\n bias=mxmem2(%2)\n { activation.ub=mxmem(%0,%1)" #MODE "\n weight.b=mxmem(%3,%4) }\n cvt.ub=acc(%6)\n mxmem(%5,%6):cm=cvt\n" \
    ::"r"(a),"r"(rt),"r"(fi),"r"(b),"r"(wrt),"r"(out),"r"(0):"memory");}
RUN(cm,:cm)
RUN(above_cm,:above:cm)
RUN(single_cm,:single:cm)
RUN(dilate_cm,:dilate:cm)
static void run_two(void *a0,uint32_t rt0,void *a,uint32_t rt,void *bz,void *b,uint32_t wrt,void *fi,void *out,int m2){
    if(m2)__asm__ volatile("mxclracc\n bias=mxmem2(%4)\n { activation.ub=mxmem(%0,%1):cm\n weight.b=mxmem(%5,%7) }\n { activation.ub=mxmem(%2,%3):cm\n weight.b=mxmem(%6,%7) }\n cvt.ub=acc(%9)\n mxmem(%8,%9):cm=cvt\n"
    ::"r"(a0),"r"(rt0),"r"(a),"r"(rt),"r"(fi),"r"(bz),"r"(b),"r"(wrt),"r"(out),"r"(0):"memory");
    else __asm__ volatile("mxclracc\n bias=mxmem2(%4)\n { activation.ub=mxmem(%0,%1):above:cm\n weight.b=mxmem(%5,%7) }\n { activation.ub=mxmem(%2,%3):cm\n weight.b=mxmem(%6,%7) }\n cvt.ub=acc(%9)\n mxmem(%8,%9):cm=cvt\n"
    ::"r"(a0),"r"(rt0),"r"(a),"r"(rt),"r"(fi),"r"(bz),"r"(b),"r"(wrt),"r"(out),"r"(0):"memory");
}
int main(int argc,char **argv) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x100000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint8_t *blk=(uint8_t *)base; /* 4 blocks of 2048 at base + b*0x4000 (spaced) */
    uint8_t *w=(uint8_t *)base+0x20000,*out=(uint8_t *)base+0x30000;
    int32_t *fi=(int32_t *)(base+0x38000);zero=(int32_t *)(base+0x38800);done=(uint8_t *)base+0x39000;
    for(int b=0;b<4;b++){uint8_t *p=blk+b*0x4000;memset(p,0,2048);for(int y=0;y<8;y++)for(int x=0;x<8;x++){p[(y*8+x)*32+0]=(uint8_t)(b*64+y*8+x);p[(y*8+x)*32+1]=1;}}
    memset(w,0,16*1024);
    /* tile t: k=0 -> n=t ; also tile t k=1 (marker 1) -> n=16+t so we see whether the tap read anything */
    for(int t=0;t<16;t++){w[t*1024+((0/4)*32+t)*4+0]=1;w[t*1024+((1/4)*32+16+t)*4+1]=1;}
    for(int c=0;c<32;c++){fi[c]=0x406000;fi[32+c]=0;zero[c]=0x404000;zero[32+c]=0;}
    int mode=atoi(argv[1]);uint32_t rs_off=strtoul(argv[2],0,0),rt_low=strtoul(argv[3],0,0),T=atoi(argv[4]);
    int bsel=argc>5?atoi(argv[5]):1; /* delta to block bsel */
    uint8_t *A=blk,*B=blk+bsel*0x4000;
    uint32_t rs=(uint32_t)A|rs_off,rt=(uint32_t)(B-A)|rt_low;
    memset(out,0xee,4096);
    if(mode==0)run_cm((void*)rs,rt,w,T*1024-1,fi,out);
    if(mode==1)run_above_cm((void*)rs,rt,w,T*1024-1,fi,out);
    if(mode==2)run_single_cm((void*)rs,rt,w,T*1024-1,fi,out);
    if(mode>=4){uint8_t *L=blk+2*0x4000,*bz=w+0x8000;memset(bz,0,16*1024);
        uint32_t rs0=(uint32_t)L|rs_off,rt0=(uint32_t)(blk+3*0x4000-L)|rt_low;
        run_two((void*)rs0,rt0,(void*)rs,rt,bz,w,T*1024-1,fi,out,mode==4);}
    if(mode==3)run_dilate_cm((void*)rs,rt,w,T*1024-1,fi,out);
    fin();
    printf("mode=%d rs_off=0x%lx rt_low=0x%lx T=%lu\n",mode,(unsigned long)rs_off,(unsigned long)rt_low,(unsigned long)T);
    for(uint32_t t=0;t<T;t++){
        printf("tap %lu:\n",(unsigned long)t);
        for(int y=0;y<8;y++){printf("  ");for(int x=0;x<8;x++){uint8_t v=out[(y*8+x)*32+t],m=out[(y*8+x)*32+16+t];
            if(!m)printf("  --  ");else printf(" %d%d%d ",v/64,(v/8)%8,v%8),printf(" ");}printf("\n");}
    }
    return 0;
}
