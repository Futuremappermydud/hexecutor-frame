#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <hexagon_standalone.h>
extern void qd_hmx_finish(void *,void *);
static uint8_t *done;static int32_t *zero;
static void fin(void){*(volatile uint32_t *)done=0xffffffff;qd_hmx_finish(zero,done);uint32_t t=0;while(*(volatile uint32_t *)done&&t++<100000)__asm__ volatile("nop");if(*(volatile uint32_t*)done)puts("NO COMPLETION");}
static void show(const char *n,uint8_t *o){printf("%s\n",n);for(int y=0;y<3;y++){printf("  ");for(int x=0;x<8;x++){uint8_t v=o[(y*8+x)*32];if(v==0xee)printf("  ..");else printf(" %d%d%d",v/64,(v/8)%8,v%8);}printf("\n");}}
int main(int argc,char **argv) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x100000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint8_t *blk=(uint8_t *)base,*w=(uint8_t *)base+0x20000,*out=(uint8_t *)base+0x30000;
    int32_t *fi=(int32_t *)(base+0x38000);zero=(int32_t *)(base+0x38800);done=(uint8_t *)base+0x39000;
    for(int b=0;b<4;b++){uint8_t *p=blk+b*0x4000;memset(p,0,2048);for(int y=0;y<8;y++)for(int x=0;x<8;x++){p[(y*8+x)*32+0]=(uint8_t)(b*64+y*8+x);}}
    memset(w,0,4096);for(int k=0;k<32;k++)w[((k/4)*32+k)*4+k%4]=1; /* identity */
    for(int c=0;c<32;c++){fi[c]=0x406000;fi[32+c]=0;zero[c]=0x404000;zero[32+c]=0;}
    uint32_t so=strtoul(argv[1],0,0),st=strtoul(argv[2],0,0);
    uint8_t *O0=out,*O1=out+0x1000,*O2=out+0x2000;
    memset(out,0xee,0x3000);
    /* two tiles: block0 then block2 through the sliding accumulator (no shift taps) */
    __asm__ volatile("mxclracc\n bias=mxmem2(%0)\n { activation.ub=mxmem(%1,%2):cm\n weight.b=mxmem(%3,%4) }\n cvt.ub=acc(%5)\n mxmem(%6,%7):cm=cvt\n"
      "{ activation.ub=mxmem(%8,%2):cm\n weight.b=mxmem(%3,%4) }\n cvt.ub=acc(%5)\n mxmem(%9,%7):cm=cvt\n"
      ::"r"(fi),"r"(blk),"r"(0x71f),"r"(w),"r"(1023),"r"(0),"r"((uint32_t)O0|so),"r"((uint32_t)(O1-O0)|st),"r"(blk+0x8000),"r"((uint32_t)O1|so):"memory");
    fin();
    printf("store Rs|0x%lx Rt=(delta 0x1000)|0x%lx\n",(unsigned long)so,(unsigned long)st);
    show("O0",O0);show("O1",O1);show("O2",O2);
    return 0;
}
