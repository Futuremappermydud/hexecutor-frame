#include "sign_blocked.h"
#include <hexagon_types.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint8_t inmem[131200] __attribute__((aligned(128)));
static uint8_t outmem[131200] __attribute__((aligned(128)));
static uint8_t crop_inmem[40960] __attribute__((aligned(128)));
static uint8_t crop_outmem[40960] __attribute__((aligned(128)));
static uint8_t crop_written[40960];
static uint8_t *inpools[128],*outpools[128];
static uint32_t inblocks[64],outblocks[64];
static uint8_t *crop_inpools[128],*crop_outpools[128];
static uint32_t crop_inblocks[4],crop_outblocks[18];
static QdBlockedU16 desc(uint8_t **p,uint32_t *bl) {
    QdBlockedU16 t={.pools=p,.blocks=bl,.block_count=64,.slice=0,
        .meta={1,8,4,2048},.dims={1,8,4,2048},.origin={0,0,0,0}};
    return t;
}
int main(void) {
    inpools[1]=inmem;outpools[1]=outmem;
    for(unsigned i=0;i<64;i++)inblocks[i]=outblocks[i]=1u+(i*2048u);
    for(unsigned i=0;i<65536;i++)((uint16_t*)inmem)[i]=(uint16_t)i;
    memset(outmem,0xa5,sizeof outmem);
    QdBlockedU16 in=desc(inpools,inblocks),out=desc(outpools,outblocks);
    QdBlockedSignU16 prep;QdSignLevels lev={1,32768,65535};int zp=10701;
    if(!qd_blocked_sign_u16_prepare(&in,&out,&prep)||!qd_blocked_sign_u16_apply(&prep,zp,&lev))return 1;
    for(unsigned i=0;i<65536;i++) {
        uint16_t x=((uint16_t*)inmem)[i],want=x<zp?1:x==zp?32768:65535;
        if(((uint16_t*)outmem)[i]!=want){printf("FAIL lane %u got %u want %u\n",i,((uint16_t*)outmem)[i],want);return 2;}
    }
    for(unsigned i=131072;i<sizeof outmem;i++)if(outmem[i]!=0xa5)return 3;
    /* Touched output alias must be rejected. */
    uint32_t dup[64];for(unsigned i=0;i<64;i++)dup[i]=1;QdBlockedU16 alias=desc(inpools,dup);
    if(qd_blocked_sign_u16_prepare(&in,&alias,&prep))return 4;
    /* Exact same descriptor is safe in-place because every physical vector is
     * consumed before its predicated store. */
    if(!qd_blocked_sign_u16_prepare(&in,&in,&prep)||!qd_blocked_sign_u16_apply(&prep,zp,&lev))return 5;
    for(unsigned i=0;i<65536;i++) {
        uint16_t x=(uint16_t)i,want=x<zp?1:x==zp?32768:65535;
        if(((uint16_t*)inmem)[i]!=want)return 6;
    }
    /* Empty region is a successful no-op, even with absent table. */
    QdBlockedU16 empty=out;empty.dims[0]=0;
    if(!qd_blocked_sign_u16_prepare(&in,&empty,&prep)||!qd_blocked_sign_u16_apply(&prep,zp,&lev))return 7;

    /* Independent metadata/slices and shifted origins: W moves by two,
     * depth by 32, and both crops cross W/depth crouton boundaries. The
     * second input depth block is absent (numerical zero). */
    crop_inpools[1]=crop_inmem;crop_outpools[1]=crop_outmem;
    QdBlockedU16 ci={.pools=crop_inpools,.blocks=crop_inblocks,.block_count=4,.slice=0,
        .meta={1,8,8,64},.dims={1,3,3,33},.origin={0,3,1,17}};
    QdBlockedU16 co={.pools=crop_outpools,.blocks=crop_outblocks,.block_count=18,.slice=0,
        .meta={1,16,12,96},.dims={1,3,3,33},.origin={0,9,3,49}};
    for(unsigned j=0;j<4;j++)crop_inblocks[j]=0;
    /* Input table: H-block0, W-block0, D-block0 has a real source. */
    crop_inblocks[0]=1;
    for(unsigned j=0;j<18;j++)crop_outblocks[j]=1u+(j*2048u);
    crop_outblocks[14]=0; /* one touched output block is intentionally absent */
    /* Clear to a sentinel, then initialize the sole present input crouton. */
    memset(crop_inmem,0,sizeof crop_inmem);memset(crop_outmem,0xa5,sizeof crop_outmem);
    for(unsigned j=0;j<1024;j++)((uint16_t*)crop_inmem)[j]=(uint16_t)(j*53u+19u);
    QdBlockedSignU16 cp;
    if(!qd_blocked_sign_u16_prepare(&ci,&co,&cp)||!qd_blocked_sign_u16_apply(&cp,zp,&lev))return 8;
    memset(crop_written,0,sizeof crop_written);
    for(int h=0;h<3;h++)for(int w=0;w<3;w++)for(int d=0;d<33;d++) {
        unsigned src=((unsigned)(3+h)%8)*128u+((unsigned)(1+w)%4)*32u+(unsigned)(17+d)%32u;
        unsigned dst=((unsigned)(9+h)%8)*128u+((unsigned)(3+w)%4)*32u+(unsigned)(49+d)%32u;
        unsigned H=9u+(unsigned)h,W=3u+(unsigned)w,D=49u+(unsigned)d;
        unsigned ix=((H/8)*3u+W/4u)*3u+D/32u;
        unsigned off=ix*2048u+(((H%8)*128u+((W%4)/2u)*64u+(W%2u)*32u+D%32u)*2u);
        if(ix!=14)crop_written[off]=crop_written[off+1]=1;
        uint16_t x=d<15?((uint16_t*)crop_inmem)[src]:0;
        uint16_t want=x<zp?1:x==zp?32768:65535;
        if(ix!=14&&((uint16_t*)crop_outmem)[dst]!=want)return 9;
    }
    for(unsigned j=0;j<sizeof crop_outmem;j++)if(!crop_written[j]&&crop_outmem[j]!=0xa5)return 10;
    puts("PASS blocked HVX sign selector: values, alias rejection, empty crop, guards");
    return 0;
}
