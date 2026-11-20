/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <hexagon_standalone.h>
extern void qd_hmx_finish(void *,void *);
static uint8_t *done;static int32_t *zero;
static void fin(void){*(volatile uint32_t *)done=0xffffffff;qd_hmx_finish(zero,done);uint32_t t=0;while(*(volatile uint32_t *)done&&t++<100000)__asm__ volatile("nop");if(*(volatile uint32_t*)done)puts("NO COMPLETION");}
static void show(const char *n,uint8_t *out,int T){
    printf("%s\n",n);
    for(int t=0;t<T;t++){printf(" tap %d:",t);for(int y=0;y<2;y++){printf(" |");for(int x=0;x<8;x++){uint8_t v=out[(y*8+x)*32+t],m=out[(y*8+x)*32+16+t];if(!m)printf("  --");else printf(" %d%d%d",v/64,(v/8)%8,v%8);}}printf("\n");}
}
int main(int argc,char **argv) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x100000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint8_t *blk=(uint8_t *)base;
    uint8_t *w=(uint8_t *)base+0x20000,*out=(uint8_t *)base+0x30000;
    int32_t *fi=(int32_t *)(base+0x38000);zero=(int32_t *)(base+0x38800);done=(uint8_t *)base+0x39000;
    for(int b=0;b<4;b++){uint8_t *p=blk+b*0x4000;memset(p,0,2048);for(int y=0;y<8;y++)for(int x=0;x<8;x++){p[(y*8+x)*32+0]=(uint8_t)(b*64+y*8+x);p[(y*8+x)*32+1]=1;}}
    memset(w,0,16*1024);
    for(int t=0;t<16;t++){w[t*1024+((0/4)*32+t)*4+0]=1;w[t*1024+((1/4)*32+16+t)*4+1]=1;}
    for(int c=0;c<32;c++){fi[c]=0x406000;fi[32+c]=0;zero[c]=0x404000;zero[32+c]=0;}
    uint32_t rs_off=strtoul(argv[1],0,0),rt_low=strtoul(argv[2],0,0),T=atoi(argv[3]);
    /* column 0 = block 0 (below: block 1), column 1 = block 2 (below: block 3) */
    uint8_t *A0=blk,*B0=blk+0x4000,*A1=blk+0x8000,*B1=blk+0xc000;
    uint32_t wr=T*1024-1;
    memset(out,0xee,0x4000);
    __asm__ volatile(
      "mxclracc\n bias=mxmem2(%[fi])\n"
      "{ activation.ub=mxmem(%[a0],%[t0]):cm\n weight.b=mxmem(%[w],%[wr]) }\n"
      "cvt.ub=acc(%[z])\n mxmem(%[o0],%[z]):cm=cvt\n"
      "{ activation.ub=mxmem(%[a1],%[t1]):cm\n weight.b=mxmem(%[w],%[wr]) }\n"
      "cvt.ub=acc(%[z])\n mxmem(%[o1],%[z]):cm=cvt\n"
      "cvt.ub=acc(%[z])\n mxmem(%[o2],%[z]):cm=cvt\n"
      ::[fi]"r"(fi),[a0]"r"((uint32_t)A0|rs_off),[t0]"r"((uint32_t)(B0-A0)|rt_low),[a1]"r"((uint32_t)A1|rs_off),[t1]"r"((uint32_t)(B1-A1)|rt_low),
        [w]"r"(w),[wr]"r"(wr),[z]"r"(0),[o0]"r"(out),[o1]"r"(out+0x1000),[o2]"r"(out+0x2000):"memory");
    fin();
    printf("rs_off=0x%lx rt_low=0x%lx T=%lu (rows 0,1 shown)\n",(unsigned long)rs_off,(unsigned long)rt_low,(unsigned long)T);
    show("out0",out,T);show("out1",out+0x1000,T);show("out2 (cvt w/o load)",out+0x2000,T);
    return 0;
}
