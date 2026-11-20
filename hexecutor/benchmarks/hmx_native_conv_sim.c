/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Independent convolution/layout gate for native 64-pixel integer HMX. */
#define QDC_NATIVE_BUILD 1
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "../native/qd_conv.h"
#include "../native/qd_hmx_layout.h"
static uint64_t cycles(void){uint64_t v;__asm__ volatile("%0=upcycle":"=r"(v));return v;}
static void barrier(void){__asm__ volatile("barrier":::"memory");}
#include "../native/qd_hmx_kernel.h"
#include "../native/qd_hmx_native_kernel.h"
static uint8_t input[0x2d0000] __attribute__((aligned(128))),output[1<<20] __attribute__((aligned(128))),blocked[1<<20] __attribute__((aligned(128)));
static uint32_t table[3][4096] __attribute__((aligned(128)));
static int check_blocks(QdConv c,const QdHmxLayout *l,uint8_t *vtcm,uint32_t bytes) {
    uint32_t height=c.oh==1?8:c.oh,widths[]={8,16,24,c.owp};
    if(height&7)return 0;
    for(uint32_t test=0;test<4;test++) {
        uint32_t width=widths[test],n=height*width*c.n;
        if(n>sizeof blocked || (c.oh==1?height*width>c.owp:width>c.owp))continue;
        memset(blocked,0xa5,n+128);c.reserved[0]=(c.reserved[0]&QDC_X32)|QDC_CROUTON8|(height<<8)|(width<<16);
        for(uint32_t tid=0;tid<3;tid++)hmx_native_collect(&c,l,vtcm,blocked,(tid+2)%3);
        for(uint32_t y=0;y<height;y++)for(uint32_t x=0;x<width;x++)for(uint32_t d=0;d<c.n;d++) {
            uint32_t p=y*(c.oh==1?width:c.owp)+x;
            uint32_t src=((p/32*l->nb+d/32)*32+p%32)*32+d%32;
            uint32_t dst=(((y/8)*(width/8)+x/8)*l->nb+d/32)*2048+(y%8)*256+(x%8)*32+d%32;
            if(src>=bytes || blocked[dst]!=output[src]){printf("FAIL blocked width%u y%u x%u d%u\n",(unsigned)width,(unsigned)y,(unsigned)x,(unsigned)d);fflush(stdout);return 1;}
        }
        for(uint32_t i=n;i<n+128;i++)if(blocked[i]!=0xa5){puts("FAIL blocked guard");fflush(stdout);return 2;}
    }
    return 0;
}
int main(void) {
    FILE *in=fopen("/src/build/hmx-conv-input.bin","rb"),*out=fopen(
#ifdef TEST_NATIVE_CM
    "/src/build/hmx-native-cm-conv-output.bin",
#else
    "/src/build/hmx-native-conv-output.bin",
#endif
    "wb");
    if(!in || !out)return 1;
    uint32_t base=__rdcfg(__vtcm_base)<<16,pn=base>>12,index=120;
    uint64_t entry=((uint64_t)(0xc0000000u|pn)<<32)|(0xf6000000u|(pn<<1)|0x40u);
    __asm__ volatile("tlbw(%0,%1)\n isync"::"r"(entry),"r"(index):"memory");
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint32_t count;if(fread(&count,4,1,in)!=1)return 2;uint8_t *vtcm=(uint8_t *)base;
    for(uint32_t test=0;test<count;test++) {
        QdConv c;if(fread(&c,sizeof c,1,in)!=1){puts("FAIL header read");return 3;}c.version=QDC_NATIVE_VERSION;
#ifdef TEST_NATIVE_CM
        c.version=QDC_NATIVE_CM_VERSION;
#endif
        QdHmxLayout l=hmx_layout(&c,c.h*c.g*c.wp*4);QdQuant *q=(QdQuant *)(vtcm+l.bytes);
        if(fread(input,1,l.a,in)!=l.a || fread(vtcm+l.b,1024,l.nb*l.kb,in)!=l.nb*l.kb || fread(q,sizeof *q,c.n,in)!=c.n)return 4;
        for(uint32_t x32=0;x32<2;x32++) {
            printf("start conv%u x32%u\n",(unsigned)test,(unsigned)x32);fflush(stdout);
            if(x32 && (c.g&7))continue;
            c.reserved[0]=x32?QDC_X32:0;
            if(!x32)memcpy(vtcm,input,l.a);
            else {
                memset(vtcm,0,l.a);
                for(uint32_t y=0;y<c.h;y++)for(uint32_t cb=0;cb<c.g/8;cb++)for(uint32_t x=0;x<c.wp;x++)for(uint32_t ch=0;ch<32;ch++)
                    vtcm[((y*(c.g/8)+cb)*c.wp+x)*32+ch]=input[((y*c.g+cb*8+ch/4)*c.wp+x)*4+ch%4];
            }
            for(uint32_t tid=0;tid<3;tid++)if(!hmx_native_prepare(&c,&l,vtcm,table[tid],tid))return 5;
            if(hmx_native_calculate(&c,&l,vtcm,q)){puts("FAIL HMX completion");fflush(stdout);return 6;}
            uint32_t bytes=c.oh*(c.owp/32)*l.nb*1024;memset(output,0xa5,bytes+128);
            for(uint32_t tid=0;tid<3;tid++)hmx_native_collect(&c,&l,vtcm,output,tid);
            for(uint32_t i=bytes;i<bytes+128;i++)if(output[i]!=0xa5){puts("FAIL output guard");fflush(stdout);return 7;}
            if(check_blocks(c,&l,vtcm,bytes))return 8;
            uint32_t header[]={test,x32,bytes};fwrite(header,4,3,out);fwrite(output,1,bytes,out);
            printf("native conv %u x32=%u %u bytes PASS layouts/guards\n",(unsigned)test,(unsigned)x32,(unsigned)bytes);
        }
    }
    fclose(in);fclose(out);SIM_RELEASE_HVX;puts("ALL LAYOUTS PASS");return 0;
}
