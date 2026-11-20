/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "../native/qd_glue_hvx.h"
static uint8_t src[128*1024] __attribute__((aligned(128)));
static uint8_t got[256*1024] __attribute__((aligned(128)));
static uint8_t ref[256*1024] __attribute__((aligned(128)));
int main(void) {
    const uint32_t widths[]={1,3,7,8,17,32,35},channels[]={32,64,96,128,160,256},heights[]={1,3,8};
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    uint32_t cases=0;
    for(uint32_t i=0;i<sizeof src;i++)src[i]=i*73+(i>>8)*41;
    for(uint32_t hi=0;hi<3;hi++)for(uint32_t wi=0;wi<7;wi++)for(uint32_t ci=0;ci<6;ci++)for(uint32_t kw=1;kw<=3;kw++) {
        if(widths[wi]<kw)continue;
        uint32_t h=heights[hi],w=widths[wi],c=channels[ci],wp=((w-kw+32)&~31u)+kw-1,n=h*wp*c;
        memset(got,0xa5,n+128);memset(ref,0,n);
        for(uint32_t y=0;y<h;y++)for(uint32_t cb=0;cb<c/32;cb++)for(uint32_t x=0;x<w;x++)
            memcpy(ref+((y*(c/32)+cb)*wp+x)*32,src+(y*w+x)*c+cb*32,32);
        dsp_pack_x32(got,src,h,w,c,wp);
        if(memcmp(got,ref,n)){printf("FAIL h%u w%u c%u\n",(unsigned)h,(unsigned)w,(unsigned)c);return 1;}
        for(uint32_t i=n;i<n+128;i++)if(got[i]!=0xa5){puts("FAIL guard");return 2;}
        cases++;
    }
    for(uint32_t i=0;i<sizeof src;i++)if(src[i]!=(uint8_t)(i*73+(i>>8)*41)){puts("FAIL source");return 3;}
    printf("PASS %u packet shapes, every byte, padding and guards\n",(unsigned)cases);
    SIM_RELEASE_HVX;return 0;
}
