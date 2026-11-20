/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Validate the original aligned QNN leaf's descriptor ABI without its runtime. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
extern void qd_qnn_hmx_1x1(uint32_t *,uint32_t *,void *,void *,uint32_t *,uint32_t *);
extern void qd_hmx_finish(void *,void *);
static uint32_t inblocks[256],outblocks[256];
static uint32_t seed=0x1549763u;
static uint32_t random32(void){seed=seed*1664525u+1013904223u;return seed;}
int main(void) {
    FILE *results=fopen("/src/native_kernels/qnn-hmx-1x1-graph-output.bin","wb");
    if(!results){puts("FAIL output file");return 4;}
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x80000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint8_t *input=(uint8_t *)base,*output=input+0x20000,*weights=input+0x40000,*bias=input+0x50000,*done=input+0x52000,*guard=input+0x54000;
    int32_t *zero=(int32_t *)(input+0x51000);uint32_t cases=0;
    const uint32_t shapes[][4]={{8,8,128,64},{8,8,128,128},{8,16,128,128},{8,32,128,128},{8,64,128,128},{4,16,128,128},{2,8,128,128}};
    for(uint32_t si=0;si<sizeof(shapes)/sizeof(shapes[0]);si++)for(uint32_t random=0;random<2;random++) {
        uint32_t h=shapes[si][0],w=shapes[si][1],ci=shapes[si][2],co=shapes[si][3];
        uint32_t hp=(h+7)&~7u,wp=(w+7)&~7u;
        uint32_t icb=ci/32,ocb=co/32,nbw=wp/8,nbh=hp/8,ib=nbh*nbw*icb,ob=nbh*nbw*ocb;
        memset(input,0,0x80000);memset(output,0xa5,0x20000);memset(guard,0xa5,2048);memset(input+0x56000,0xa5,2048);
        for(uint32_t i=0;i<256;i++){inblocks[i]=(uint32_t)(input+0x56000);outblocks[i]=(uint32_t)guard;}
        for(uint32_t i=0;i<ib;i++)inblocks[i]=(uint32_t)(input+(ib-1-i)*2048);
        for(uint32_t i=0;i<ob;i++)outblocks[i]=(uint32_t)(output+(ob-1-i)*2048);
        for(uint32_t y=0;y<h;y++)for(uint32_t x=0;x<w;x++)for(uint32_t d=0;d<ci;d++)
            ((uint8_t *)inblocks[((y/8)*nbw+x/8)*icb+d/32])[(y%8)*256+(x%8)*32+d%32]=random?random32()>>24:(y*37+x*19+d*11)&255;
        for(uint32_t n=0;n<co;n++) {
            for(uint32_t k=0;k<ci;k++)
                weights[(((n/32*icb+k/32)*8+(k%32)/4)*32+n%32)*4+k%4]=random?(int32_t)(random32()>>24)%17-8:k==n%ci;
            /* Packet operands read the old r3 while the same packet advances
             * it: the first bias row is at bias, not bias+256. */
            ((int32_t *)bias)[n/32*64+n%32]=random?0x404200:0x406000;
            ((int32_t *)bias)[n/32*64+32+n%32]=random?20000+(int32_t)(random32()%2001)-1000:0;
        }
        for(uint32_t n=0;n<32;n++)zero[n]=0x404000;
        /* A single 64-pixel column with all spatial croutons flattened into
         * the leaf's block loop. Logical image coordinates live in the tables. */
        uint32_t in[]={ (uint32_t)inblocks,icb,ib,nbh*nbw,64,ci };
        uint32_t out[]={ (uint32_t)outblocks,ocb,ob,nbh*nbw,64,co };
        uint32_t cfg[16]={0};cfg[3]=2047;cfg[6]=1023;uint32_t mode[]={1,0};
        uint32_t header[]={h,w,ci,co,random};
        fwrite(header,sizeof header,1,results);
        for(uint32_t i=0;i<ib;i++)fwrite((void *)inblocks[i],2048,1,results);
        fwrite(weights,ci*co,1,results);fwrite(bias,ocb*256,1,results);
        printf("start %ux%u ci%u co%u random%u\n",(unsigned)h,(unsigned)w,(unsigned)ci,(unsigned)co,(unsigned)random);fflush(stdout);
        __asm__ volatile("mxclracc":::"memory");
        qd_qnn_hmx_1x1(out,in,weights,bias,cfg,mode);
        *(volatile uint32_t *)done=0xffffffff;qd_hmx_finish(zero,done);
        uint32_t tries=0;while(*(volatile uint32_t *)done && tries++<10000)__asm__ volatile("nop");
        if(*(volatile uint32_t *)done){puts("FAIL completion");return 1;}
        for(uint32_t y=0;y<nbh*8;y++)for(uint32_t x=0;x<nbw*8;x++)for(uint32_t d=0;d<co;d++) {
            uint32_t got=((uint8_t *)outblocks[((y/8)*nbw+x/8)*ocb+d/32])[(y%8)*256+(x%8)*32+d%32];
            int64_t want=(y<h && x<w)?((y*37+x*19+(d%ci)*11)&255):0;
            /* The independent Python checker validates every random output
             * against int64 matrix arithmetic, outside the ISA simulator. */
            if(!random && got!=want) {printf("FAIL y%u x%u d%u got%u want%u\n",(unsigned)y,(unsigned)x,(unsigned)d,(unsigned)got,(unsigned)want);return 2;}
        }
        for(uint32_t i=0;i<ob;i++)fwrite((void *)outblocks[i],2048,1,results);
        for(uint32_t i=0;i<2048;i++)if(guard[i]!=0xa5){puts("FAIL block-table tail guard");return 3;}
        for(uint32_t i=ob*2048;i<0x20000;i++)if(output[i]!=0xa5){puts("FAIL output-allocation tail");return 4;}
        for(uint32_t i=ib;i<256;i++)if(inblocks[i]!=(uint32_t)(input+0x56000)){puts("FAIL input pointer table tail");return 5;}
        for(uint32_t i=ob;i<256;i++)if(outblocks[i]!=(uint32_t)guard){puts("FAIL output pointer table tail");return 6;}
        cases++;
    }
    fclose(results);printf("PASS %u graph geometry native 1x1 fixtures, identity layouts and output guards; random arithmetic requires Python checker\n",(unsigned)cases);return 0;
}
