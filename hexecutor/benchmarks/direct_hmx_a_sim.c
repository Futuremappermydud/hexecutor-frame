#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "../native/qd_direct_hmx.h"
static uint8_t pool1[512*1024] __attribute__((aligned(128))),pool3[512*1024] __attribute__((aligned(128)));
static uint8_t got[2*1024*1024] __attribute__((aligned(128))),want[2*1024*1024] __attribute__((aligned(128)));
static uint32_t blocks[256];
int main(void) {
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    uint8_t *pools[7]={0,pool1,0,pool3};uint32_t cases=0;
    for(uint32_t i=0;i<sizeof pool1;i++){pool1[i]=i*73+(i>>8)*23;pool3[i]=i*41+(i>>9)*7;}
    const uint32_t widths[]={8,24,32},channels[]={32,128},origins[]={0,1,7,8};
    for(uint32_t wi=0;wi<3;wi++)for(uint32_t ci=0;ci<2;ci++)for(uint32_t kernel=1;kernel<=3;kernel++)for(uint32_t oi=0;oi<4;oi++)for(int flatten=0;flatten<=(kernel==1);flatten++) {
        uint32_t h=8,w=widths[wi],c=channels[ci],kh=kernel,kw=kernel,owp=(w+31)&~31u;
        int32_t m[4]={1,(int32_t)(h+kh-1+16),(int32_t)(w+kw-1+16),(int32_t)c+32},origin[4]={0,origins[oi],origins[oi],0};
        uint32_t nbh=(m[1]+7)/8,nbw=(m[2]+7)/8,nbd=(m[3]+31)/32,count=nbh*nbw*nbd,slice=2,cbs=c/32,kb=kh*kw*cbs;
        uint32_t mbs=flatten?h*w/32:h*owp/32,bytes=mbs*kb*2048;
        if(count+slice>256 || bytes+128>sizeof got)return 3;
        for(uint32_t i=0;i<count+slice;i++)blocks[i]=i%13?((count+slice-1-i)*2048)|(i%2?1:3):0;
        for(uint32_t i=7;i<count;i+=11)blocks[i]=blocks[i-3];
        memset(got,0xa5,bytes+128);memset(want,0,bytes);
        for(uint32_t mb=0;mb<mbs;mb++)for(uint32_t ky=0;ky<kh;ky++)for(uint32_t kx=0;kx<kw;kx++)for(uint32_t cb=0;cb<cbs;cb++)for(uint32_t i=0;i<32;i++)for(uint32_t ch=0;ch<32;ch++) {
            uint32_t pixel=mb*32+i,y=flatten?pixel/w:mb/(owp/32),x=flatten?pixel%w:(mb%(owp/32))*32+i;
            uint8_t v=0;
            if(x+kx<w+kw-1) {
                uint32_t H=y+ky+origin[1],W=x+kx+origin[2],index=((H/8)*nbw+W/8)*nbd+slice+cb,e=blocks[index],p=e&127;
                if(p)v=pools[p][(e&~127u)+(H%8)*256+(W%8)*32+ch];
            }
            uint32_t k=(ky*kw+kx)*cbs+cb;
            want[(mb*kb+k)*2048+(i/2)*128+ch*4+(i%2)*2+1]=v;
        }
        dsp_direct_hmx_a(got,pools,blocks,count+slice,slice,m,origin,h,w,c,kh,kw,owp,flatten,0);
        if(memcmp(got,want,bytes)){printf("FAIL w%u c%u kernel%u origin%u flatten%u\n",(unsigned)w,(unsigned)c,(unsigned)kernel,(unsigned)origins[oi],(unsigned)flatten);return 1;}
        for(uint32_t i=bytes;i<bytes+128;i++)if(got[i]!=0xa5){puts("FAIL guard");return 2;}
        cases++;
    }
    printf("PASS %u fused graph-block to HMX-A shapes, taps, unaligned origins, holes, aliases, padding and guards\n",(unsigned)cases);
    SIM_RELEASE_HVX;return 0;
}
