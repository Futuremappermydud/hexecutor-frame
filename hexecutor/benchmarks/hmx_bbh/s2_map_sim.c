/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Which Rs/Rt low bits give stride-2 tap walks on :cm loads? Prints tap maps by
 * pixel value (b y x). Tile slide: output of the first cvt only (pixels x1.. of block). */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <hexagon_standalone.h>
extern void qd_hmx_finish(void *,void *);
static uint8_t *done;static int32_t *zero;
static void fin(void){*(volatile uint32_t *)done=0xffffffff;qd_hmx_finish(zero,done);uint32_t t=0;while(*(volatile uint32_t *)done&&t++<100000)__asm__ volatile("nop");if(*(volatile uint32_t*)done)puts("NO COMPLETION");}
int main(void) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x100000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint8_t *blk=(uint8_t *)base;uint8_t *w=(uint8_t *)base+0x20000,*out=(uint8_t *)base+0x30000;
    int32_t *fi=(int32_t *)(base+0x38000);zero=(int32_t *)(base+0x38800);done=(uint8_t *)base+0x39000;
    for(int b=0;b<4;b++){uint8_t *p=blk+b*0x4000;memset(p,0,2048);for(int y=0;y<8;y++)for(int x=0;x<8;x++){p[(y*8+x)*32+0]=(uint8_t)(b*64+y*8+x);p[(y*8+x)*32+1]=1;}}
    memset(w,0,16*1024);
    for(int t=0;t<16;t++){w[t*1024+((0/4)*32+t)*4+0]=1;w[t*1024+((1/4)*32+16+t)*4+1]=1;}
    for(int c=0;c<32;c++){fi[c]=0x406000;fi[32+c]=0;zero[c]=0x404000;zero[32+c]=0;}
    uint8_t *A0=blk,*B0=blk+0x4000,*A1=blk+0x8000,*B1=blk+0xc000;
    static const uint32_t rts[]={0x71f,0x73f,0x75f,0x77f,0x79f,0x7bf,0x7df,0x7ff};
    static const uint32_t rss[]={0x000,0x020,0x040,0x100,0x120,0x140};
    const int T=4;const uint32_t wr=T*1024-1;
    for(unsigned ri=0;ri<8;ri++)for(unsigned si=0;si<6;si++) {
        uint32_t a0=(uint32_t)A0|rss[si],t0=(uint32_t)(B0-A0)|rts[ri],a1=(uint32_t)A1|rss[si],t1=(uint32_t)(B1-A1)|rts[ri];
        memset(out,0xee,0x2000);
        __asm__ volatile(
         "mxclracc\n bias=mxmem2(%[fi])\n"
         "{ activation.ub=mxmem(%[a0],%[t0]):cm\n weight.b=mxmem(%[w],%[wr]) }\n"
         "cvt.ub=acc(%[z])\n mxmem(%[o0],%[z]):cm=cvt\n"
         "{ activation.ub=mxmem(%[a1],%[t1]):cm\n weight.b=mxmem(%[w],%[wr]) }\n"
         "cvt.ub=acc(%[z])\n mxmem(%[o1],%[z]):cm=cvt\n"
         ::[fi]"r"(fi),[a0]"r"(a0),[t0]"r"(t0),[a1]"r"(a1),[t1]"r"(t1),[w]"r"(w),[wr]"r"(wr),[z]"r"(0),[o0]"r"(out),[o1]"r"(out+0x1000):"memory");
        fin();
        printf("rt=%03lx rs=%03lx:",(unsigned long)rts[ri],(unsigned long)rss[si]);
        for(int t=0;t<T;t++){
            printf(" T%d[",t);
            for(int k=0;k<2;k++){uint8_t *o=out+k*0x1000;
              /* tile k: pixels (0,0) (0,6) (0,7) (7,7) of channel t */
              static const int px[4][2]={{0,0},{0,6},{0,7},{7,7}};
              for(int q=0;q<4;q++){int y=px[q][0],x=px[q][1];uint8_t v=o[(y*8+x)*32+t],m=o[(y*8+x)*32+16+t];
                if(!m)printf("--- ");else printf("%d%d%d ",v/64,(v/8)%8,v%8);}
              if(!k)printf("/ ");
            }
            printf("]");
        }
        printf("\n");
    }
    return 0;
}
