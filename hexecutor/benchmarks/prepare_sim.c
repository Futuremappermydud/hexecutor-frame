/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* ISA-simulator check: scatter-free hmx_prepare_fast must produce byte-identical
 * activation tiles to the original scatter hmx_prepare, for several geometries. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "../native/qd_conv.h"
#include "../native/qd_hmx_layout.h"
static uint64_t cycles(void){uint64_t value;__asm__ volatile("%0=c15:14":"=r"(value));return value;}
static void barrier(void){__asm__ volatile("barrier":::"memory");}
#include "../native/qd_hmx_kernel.h"
static uint32_t table[QDC_PREPARE_TABLE];
static uint8_t reference[1u<<20];
int main(void) {
    uint32_t base=__rdcfg(__vtcm_base)<<16,pn=base>>12,index=120;
    uint64_t entry=((uint64_t)(0xc0000000u|pn)<<32)|(0xf6000000u|(pn<<1)|0x40u);
    __asm__ volatile("tlbw(%0,%1)\n isync"::"r"(entry),"r"(index):"memory");
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint8_t *vtcm=(uint8_t *)base;
    /* kh, kw, g, oh, owp */
    static const uint32_t cases[][5]={{3,3,8,8,128},{3,3,3,8,256},{1,1,3,1,2048},{1,1,16,1,512},{2,2,32,8,128},
                                      {3,3,16,8,64},{3,3,1,8,256},{3,3,32,8,32},{1,1,64,1,256},{2,2,8,8,64}};
    uint32_t seed=12345,failures=0;
    for(uint32_t t=0;t<sizeof cases/sizeof cases[0];t++) {
        QdConv c;memset(&c,0,sizeof c);
        c.kh=cases[t][0];c.kw=cases[t][1];c.g=cases[t][2];c.oh=cases[t][3];c.owp=cases[t][4];c.n=32;
        c.h=c.oh+c.kh-1;c.wp=c.owp+c.kw-1;
        uint32_t xb=c.h*c.g*c.wp*4;
        QdHmxLayout l=hmx_layout(&c,xb);
        for(uint32_t i=0;i<xb;i++){seed=seed*1103515245u+12345u;vtcm[i]=(uint8_t)(seed>>16);}
        memset(vtcm+xb,0,128); /* host packets zero this over-read pad */
        uint32_t bytes=l.mb*l.kb*2048;
        if(bytes>sizeof reference){printf("case %u too large\n",(unsigned)t);return 2;}
        for(uint32_t tid=0;tid<3;tid++)hmx_zero(vtcm+l.a,bytes,tid);
        barrier();
        uint64_t t0=cycles();
        for(uint32_t tid=0;tid<3;tid++)hmx_prepare_scatter(&c,&l,vtcm,tid);
        uint64_t t1=cycles();
        memcpy(reference,vtcm+l.a,bytes);
        memset(vtcm+l.a,0xa5,bytes);
        uint64_t t2=cycles();
        for(uint32_t tid=0;tid<3;tid++)if(!hmx_prepare_fast(&c,&l,vtcm,table,tid))return 3;
        uint64_t t3=cycles();
        uint32_t bad=0;
        for(uint32_t i=0;i<bytes;i++)if(vtcm[l.a+i]!=reference[i])bad++;
        uint32_t bad32=0,tested32=0;
        if(!(c.g&7)) {
            /* Repack the same activations as [h][g/8][wp][32] and rebuild the tiles. */
            static uint8_t original[1u<<20];
            memcpy(original,vtcm,xb);
            for(uint32_t h=0;h<c.h;h++)for(uint32_t g=0;g<c.g;g++)for(uint32_t w=0;w<c.wp;w++)
                memcpy(vtcm+((h*(c.g/8)+g/8)*c.wp+w)*32+(g%8)*4,original+((h*c.g+g)*c.wp+w)*4,4);
            memset(vtcm+l.a,0x5a,bytes);
            for(uint32_t tid=0;tid<3;tid++)if(!hmx_prepare_x32(&c,&l,vtcm,table,tid))return 4;
            for(uint32_t i=0;i<bytes;i++)if(vtcm[l.a+i]!=reference[i])bad32++;
            tested32=1;
            memcpy(vtcm,original,xb);
        }
        if(tested32)printf("  x32 layout: %s (%u bad bytes)\n",bad32?"FAIL":"PASS",(unsigned)bad32);
        failures+=bad32!=0;
        printf("case %u kh=%u kw=%u g=%u oh=%u owp=%u tiles=%u: %s (%u bad bytes) scatter=%u fast=%u cycles\n",
               (unsigned)t,(unsigned)c.kh,(unsigned)c.kw,(unsigned)c.g,(unsigned)c.oh,(unsigned)c.owp,(unsigned)(l.mb*l.kb),
               bad?"FAIL":"PASS",(unsigned)bad,(unsigned)(t1-t0),(unsigned)(t3-t2));
        failures+=bad!=0;
    }
    SIM_RELEASE_HVX;
    printf(failures?"FAILURES\n":"ALL PASS\n");
    return failures?1:0;
}
