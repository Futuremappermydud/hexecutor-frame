/* One-hot center tap probes for the aligned generic spatial leaf. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>

extern void qd_qnn_hmx_3x3(uint32_t *,uint32_t *,void *,void *,uint32_t *,uint32_t *);
extern void qd_hmx_finish(void *,void *);
extern void qd_set_hmx_params_stride1(uint32_t *,uint32_t,uint32_t,uint32_t,
                                     uint32_t,uint32_t,uint32_t,uint32_t);
#ifndef KH
#define KH 1
#endif
#ifndef KW
#define KW 3
#endif
#ifndef TAPY
#define TAPY (KH/2)
#endif
#ifndef TAPX
#define TAPX (KW/2)
#endif
#ifndef SWAP_HW
#define SWAP_HW 0
#endif
#ifndef ROWSTRIDE_OVERRIDE
#define ROWSTRIDE_OVERRIDE 0
#endif
#ifndef TABLE_COLUMN_MAJOR
#define TABLE_COLUMN_MAJOR 0
#endif
#define OH 8u
#define OW 8u
#define CI 32u
#define CO 32u

int main(void) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x80000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    memset((void *)base,0,0x80000);
    uint8_t *output=(uint8_t *)base,*input=output+0x10000,*weights=input+0x10000;
    int32_t *fi=(int32_t *)(input+0x20000),*zero=(int32_t *)(input+0x21000);
    uint32_t *inblocks=(uint32_t *)(input+0x22000),*outblocks=(uint32_t *)(input+0x22400);
    uint8_t *guard=output+0x4000;uint32_t *done=(uint32_t *)(input+0x23000);
    const uint32_t IH=OH+KH-1,IW=OW+KW-1,NBW=(IW+7)/8,NBH=(IH+7)/8,IB=NBW*NBH;
    const uint32_t O_PHYS=IB;
    memset(output,0xa5,O_PHYS*2048);memset(guard,0xa5,2048);
    for(uint32_t i=0;i<256;i++){inblocks[i]=(uint32_t)(input+0x24000);outblocks[i]=(uint32_t)guard;}
    for(uint32_t b=0;b<IB;b++)inblocks[b]=(uint32_t)(input+b*2048);
    for(uint32_t b=0;b<O_PHYS;b++)outblocks[b]=(uint32_t)(output+b*2048);
    memset(weights,0,9*1024);memset(fi,0,64*sizeof(*fi));
    for(uint32_t y=0;y<IH;y++)for(uint32_t x=0;x<IW;x++)for(uint32_t c=0;c<CI;c++) {
        uint32_t b=TABLE_COLUMN_MAJOR?(x/8)*NBH+y/8:(y/8)*NBW+x/8;
        uint32_t off=(y%8)*256+(x%8)*32+c;
        ((uint8_t *)inblocks[b])[off]=(uint8_t)(c==0?1+y*IW+x:0);
    }
    uint32_t ky=TAPY,kx=TAPX;
    uint32_t k=(ky*KW+kx)*CI;
    weights[(((k/32)*8+(k%32)/4)*32)*4+k%4]=1;
    fi[0]=0x406000;zero[0]=0x404000;
    /* +8 is row stride. Independently test the alternate H/W field order. */
    const uint32_t DH=SWAP_HW?IW:IH,DW=SWAP_HW?IH:IW;
    const uint32_t rowstride=ROWSTRIDE_OVERRIDE?ROWSTRIDE_OVERRIDE:(TABLE_COLUMN_MAJOR?NBH:(DW+7)/8);
    uint32_t in[]={ (uint32_t)inblocks,1,rowstride,DH,DW,CI };
    uint32_t out[]={ (uint32_t)outblocks,1,1,OH,OW,CO };
    uint32_t cfg[16]={0},mode[]={1,0};
    qd_set_hmx_params_stride1(cfg,0x7e0,KH,KW,CI,128,0,0);
    printf("tap kh=%u kw=%u cfg=",KH,KW);for(uint32_t i=0;i<13;i++)printf("%s%x",i?",":"",cfg[i]);puts("");
    __asm__ volatile("mxclracc":::"memory");qd_qnn_hmx_3x3(out,in,weights,fi,cfg,mode);
    *done=0xffffffff;qd_hmx_finish(zero,done);uint32_t tries=0;while(*done&&tries++<100001)__asm__ volatile("nop");
    if(*done){printf("FAIL completion=%x tries=%u output=%u\n",*done,(unsigned)tries,output[0]);return 1;}
    puts("output c0:");for(uint32_t y=0;y<OH;y++){for(uint32_t x=0;x<OW;x++)printf("%s%u",x?" ":"",output[y*256+x*32]);puts("");}
    uint32_t written_extra=0;for(uint32_t b=1;b<O_PHYS;b++)for(uint32_t i=0;i<2048;i++)if(output[b*2048+i]!=0xa5)written_extra++;
    printf("extra physical crouton bytes written=%u\n",(unsigned)written_extra);
    for(uint32_t i=0;i<2048;i++)if(guard[i]!=0xa5){printf("FAIL guard=%u\n",(unsigned)i);return 3;}
    for(uint32_t i=IB;i<256;i++)if(inblocks[i]!=(uint32_t)(input+0x24000)){printf("FAIL input table tail=%u\n",(unsigned)i);return 4;}
    for(uint32_t i=O_PHYS;i<256;i++)if(outblocks[i]!=(uint32_t)guard){printf("FAIL output table tail=%u\n",(unsigned)i);return 5;}
    for(uint32_t y=0;y<OH;y++)for(uint32_t x=0;x<OW;x++) {
        uint32_t got=output[y*256+x*32],want=(uint8_t)(1+(y+ky)*IW+(x+kx));
        if(got!=want){printf("FAIL output y=%u x=%u c=0 got=%u want=%u\n",(unsigned)y,(unsigned)x,(unsigned)got,(unsigned)want);return 2;}
    }
    puts("PASS one-hot center tap outputs and guards");return 0;
}
