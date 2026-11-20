/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <stdio.h>
#include <hexagon_standalone.h>
#include "../native/qd_pack_hvx.h"
static uint8_t expected[1<<20] __attribute__((aligned(128)));
int main(void) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x2d0000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint16_t *src=(uint16_t *)base;uint8_t *got=(uint8_t *)(base+0x100000);
    HVX_Vector *tmp=(HVX_Vector *)(base+0x2cff00);
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    const uint32_t widths[]={1,3,7,32,33,35},channels[]={4,20,32,96,128},heights[]={1,3};uint32_t cases=0;
    for(uint32_t j=0;j<128*1024;j++)src[j]=j*733+(j>>7)*541;
    for(uint32_t hi=0;hi<2;hi++)for(uint32_t wi=0;wi<6;wi++)for(uint32_t ci=0;ci<5;ci++)for(uint32_t kw=1;kw<=3;kw++) {
        if(widths[wi]<kw)continue;
        uint32_t h=heights[hi],w=widths[wi],c=channels[ci],wp=((w-kw+32)&~31u)+kw-1,plane=h*wp*c;
        memset(got,0xa5,plane*2+128);memset(expected,0,plane*2);
        for(uint32_t y=0;y<h;y++)for(uint32_t g=0;g<c/4;g++)for(uint32_t x=0;x<w;x++)for(uint32_t q=0;q<4;q++) {
            uint32_t at=((y*(c/4)+g)*wp+x)*4+q;uint16_t v=src[(y*w+x)*c+g*4+q];
            expected[at]=(uint8_t)v;expected[plane+at]=v>>8;
        }
        dsp_pack_u16_gather(got,plane,src,h,w,c,wp,tmp);
        if(memcmp(got,expected,plane*2)) {
            for(uint32_t j=0;j<plane*2;j++)if(got[j]!=expected[j]){
                printf("FAIL h%u w%u c%u wp%u byte%u got%u want%u\n",(unsigned)h,(unsigned)w,(unsigned)c,(unsigned)wp,(unsigned)j,(unsigned)got[j],(unsigned)expected[j]);break;}
            return 1;
        }
        for(uint32_t j=plane*2;j<plane*2+128;j++)if(got[j]!=0xa5){puts("FAIL guard");return 2;}
        cases++;
    }
    printf("PASS %u u16 gather packet shapes, byte planes, padding and guards\n",(unsigned)cases);
    SIM_RELEASE_HVX;return 0;
}
