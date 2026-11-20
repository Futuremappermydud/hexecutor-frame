/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <stdio.h>
#include <hexagon_standalone.h>
#include "../native/qd_glue_hvx.h"
static uint8_t a[0x100000] __attribute__((aligned(128))),b[0x100000] __attribute__((aligned(128)));
static uint8_t ar[0x100000],br[0x100000],dense[65536] __attribute__((aligned(128))),ref[65536];
static uint32_t blocks[256];
int main(void) {
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    int32_t m[4]={2,24,32,160},origin[4]={0,3,2,32};
    const int32_t shapes[][4]={{1,1,1,32},{1,9,7,96},{2,5,9,64},{1,16,16,128}};
    uint8_t *pools[7]={0,a,0,b,0,0,0};
    uint32_t seed=1234;
    for(uint32_t s=0;s<4;s++)for(int mode=0;mode<3;mode++) {
        const int32_t *n=shapes[s];size_t total=(size_t)n[0]*n[1]*n[2]*n[3];
        for(uint32_t i=0;i<256;i++)blocks[i]=((255-i)*2048)|(i&1?1:3);
        if(mode==1)for(uint32_t i=0;i<256;i+=11)blocks[i]=0;
        if(mode==2)for(uint32_t i=4;i<256;i+=5)blocks[i]=blocks[i-3];
        for(uint32_t i=0;i<sizeof a;i++){seed=seed*1103515245u+12345u;a[i]=seed>>16;b[i]=seed>>20;}
        memcpy(ar,a,sizeof a);memcpy(br,b,sizeof b);memset(dense,0xa5,sizeof dense);
        size_t pos=0;
        for(int y=0;y<n[0];y++)for(int h=0;h<n[1];h++)for(int w=0;w<n[2];w++)for(int d=0;d<n[3];d++,pos++) {
            int B=y+origin[0],H=h+origin[1],W=w+origin[2],D=d+origin[3];
            uint32_t e=blocks[(((B*3+H/8)*4+W/8)*5)+3+D/32];
            uint32_t off=(e&~127u)+(H%8)*256+(W%8)*32+D%32;
            ref[pos]=(e&127)==1?a[off]:(e&127)==3?b[off]:0;
        }
        dsp_block_u8(pools,blocks,256,3,m,n,origin,dense+1,0);
        if(memcmp(dense+1,ref,total) || dense[0]!=0xa5 || dense[total+1]!=0xa5){for(size_t j=0;j<total;j++)if(dense[j+1]!=ref[j]){printf("READ FAIL %u %d at %u got %u want %u\n",(unsigned)s,mode,(unsigned)j,dense[j+1],ref[j]);break;}return 1;}
        for(uint32_t i=0;i<total;i++)dense[1+i]=i*73+i/41;
        pos=0;
        for(int y=0;y<n[0];y++)for(int h=0;h<n[1];h++)for(int w=0;w<n[2];w++)for(int d=0;d<n[3];d++,pos++) {
            int B=y+origin[0],H=h+origin[1],W=w+origin[2],D=d+origin[3];
            uint32_t e=blocks[(((B*3+H/8)*4+W/8)*5)+3+D/32];
            uint32_t off=(e&~127u)+(H%8)*256+(W%8)*32+D%32;
            if((e&127)==1)ar[off]=dense[1+pos];if((e&127)==3)br[off]=dense[1+pos];
        }
        dsp_block_u8(pools,blocks,256,3,m,n,origin,dense+1,1);
        if(memcmp(a,ar,sizeof a)||memcmp(b,br,sizeof b)){printf("WRITE FAIL %u %d\n",(unsigned)s,mode);return 2;}
    }
    puts("PASS: 12 blocked u8 read/write cases, holes, aliases, tails and padding");
    SIM_RELEASE_HVX;return 0;
}
