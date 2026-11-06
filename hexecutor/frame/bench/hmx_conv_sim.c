/* Offline ISA simulation of the actual HVX im2col + integer HMX convolution. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "../qd_conv.h"
#include "../qd_hmx_layout.h"
static uint64_t cycles(void){uint64_t value;__asm__ volatile("%0=upcycle":"=r"(value));return value;}
static void barrier(void){__asm__ volatile("barrier":::"memory");}
#include "../qd_hmx_kernel.h"
static uint8_t blocked_output[1u<<20] __attribute__((aligned(128)));
/* Compare every written byte with the independently checked legacy tile
 * layout. The output lives in DDR, just like the real arena. Padding outside
 * the logical croutons must retain its canary. Width 24 exercises tile/row
 * boundaries that do not coincide with a 32-pixel HMX tile. */
static int check_blocked(QdConv c,const QdHmxLayout *l,uint8_t *vtcm,const QdQuant *q) {
    const uint8_t *raw=vtcm+l->q;
    uint32_t widths[4]={8,16,24,c.owp};
    uint32_t h=c.oh==1?8:c.oh;
    if(h&7)return 0;
    for(uint32_t test=0;test<4;test++) {
        uint32_t w=widths[test];
        if((c.oh==1 && h*w>c.owp) || (c.oh!=1 && w>c.owp))continue;
        uint32_t bytes=h*w*c.n;
        if(bytes>sizeof blocked_output)return 1;
        memset(blocked_output,0xa5,sizeof blocked_output);
        c.reserved[0]=QDC_CROUTON8|(h<<8)|(w<<16);
        for(uint32_t t=0;t<3;t++)hmx_quantize_to(&c,l,vtcm,q,blocked_output,(t+2)%3);
        for(uint32_t y=0;y<h;y++)for(uint32_t x=0;x<w;x++)for(uint32_t n=0;n<c.n;n++) {
            uint32_t p=y*(c.oh==1?w:c.owp)+x;
            uint32_t src=(((p/32)*l->nb+n/32)*32+p%32)*32+n%32;
            uint32_t dst=(((y/8)*(w/8)+x/8)*l->nb+n/32)*2048+(y%8)*256+(x%8)*32+n%32;
            if(blocked_output[dst]!=raw[src]) {
                printf("blocked mismatch h=%u w=%u y=%u x=%u n=%u\n",(unsigned)h,(unsigned)w,(unsigned)y,(unsigned)x,(unsigned)n);
                return 1;
            }
        }
        for(uint32_t i=bytes;i<sizeof blocked_output;i++)if(blocked_output[i]!=0xa5)return 1;
        printf("  crouton output %ux%ux%u PASS\n",(unsigned)h,(unsigned)w,(unsigned)c.n);
    }
    return 0;
}
int main(void) {
    FILE *in=fopen("/src/build/hmx-conv-input.bin","rb"),*out=fopen("/src/build/hmx-conv-output.bin","wb");
    FILE *ao=fopen("/src/build/hmx-conv-activations.bin","wb");
    if(!in || !out || !ao)return 1;
    uint32_t base=__rdcfg(__vtcm_base)<<16,pn=base>>12,index=120;
    uint64_t entry=((uint64_t)(0xc0000000u|pn)<<32)|(0xf6000000u|(pn<<1)|0x40u);
    __asm__ volatile("tlbw(%0,%1)\n isync"::"r"(entry),"r"(index):"memory");
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint32_t count;if(fread(&count,4,1,in)!=1)return 2;
    uint8_t *vtcm=(uint8_t *)base;
    for(uint32_t test=0;test<count;test++) {
        QdConv c;if(fread(&c,sizeof(c),1,in)!=1)return 3;
        QdHmxLayout l=hmx_layout(&c,c.h*c.g*c.wp*4);
        QdQuant *q=(QdQuant *)(vtcm+l.bytes);
        if(fread(vtcm,1,l.a,in)!=l.a || fread(vtcm+l.b,1024,l.nb*l.kb,in)!=l.nb*l.kb || fread(q,sizeof(QdQuant),c.n,in)!=c.n)return 4;
        for(uint32_t tid=0;tid<3;tid++)hmx_zero(vtcm+l.a,l.mb*l.kb*2048,tid);
        barrier();
        for(uint32_t tid=0;tid<3;tid++)hmx_prepare_scatter(&c,&l,vtcm,tid);
        fwrite(vtcm+l.a,2048,l.mb*l.kb,ao);
        if(hmx_calculate(&c,&l,vtcm,q))return 5;
        for(uint32_t tid=0;tid<3;tid++)hmx_quantize_to(&c,&l,vtcm,q,vtcm+l.q,tid);
        if(check_blocked(c,&l,vtcm,q))return 7;
        uint32_t bytes=l.mb*l.nb*1024;
        if(fwrite(vtcm+l.q,1,bytes,out)!=bytes)return 6;
        printf("HMX conv %u: %u bytes\n",(unsigned)test,(unsigned)bytes);
    }
    fclose(in);fclose(out);fclose(ao);SIM_RELEASE_HVX;return 0;
}
