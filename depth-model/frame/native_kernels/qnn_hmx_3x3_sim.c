/* Probe one padded input tile against the local aligned QNN 3x3 leaf.
 * This is intentionally an ABI experiment, not a claimed supported fixture. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
extern void qd_qnn_hmx_3x3(uint32_t *,uint32_t *,void *,void *,uint32_t *,uint32_t *);
extern void qd_hmx_finish(void *,void *);
extern void qd_set_hmx_params_stride1(uint32_t *,uint32_t,uint32_t,uint32_t,
                                      uint32_t,uint32_t,uint32_t,uint32_t);
#ifndef TEST_FLAGS
#define TEST_FLAGS 0x7e0u
#endif
static uint32_t *inblocks,*outblocks;
static uint8_t *input,*output,*weights;
static int32_t *bias,*zero;
static uint32_t cfg[16],mode[2]={1,0};
static uint8_t *done,*guard;
int main(void) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    output=(uint8_t *)base;input=output+0x10000;weights=input+0x10000;bias=(int32_t *)(input+0x20000);zero=(int32_t *)(input+0x21000);done=input+0x22000;guard=input+0x23000;inblocks=(uint32_t *)(input+0x24000);outblocks=(uint32_t *)(input+0x24400);
    for(uint32_t i=0;i<0x80000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    for(uint32_t b=0;b<256;b++)inblocks[b]=(uint32_t)input;
    for(uint32_t b=0;b<4;b++)inblocks[b]=(uint32_t)(input+b*2048);
    outblocks[0]=(uint32_t)output;outblocks[1]=(uint32_t)(output+2048);
    memset(input,0,4*2048);memset(output,0xa5,2048);memset(guard,0xa5,2048);
    for(uint32_t y=0;y<10;y++)for(uint32_t x=0;x<10;x++)for(uint32_t c=0;c<32;c++) {
        uint32_t b=(y/8)*2+x/8,at=(y%8)*256+(x%8)*32+c;
        input[b*2048+at]=(uint8_t)(1+y*11+x*7+c%5);
    }
    /* Generalized 1x1 packing hypothesis: K is flattened across ky,kx,ci. */
    for(uint32_t n=0;n<32;n++)for(uint32_t c=0;c<32;c++) {
        uint32_t k=4*32+c;
        weights[(((k/32)*8+(k%32)/4)*32+n)*4+k%4]=(n==c)?1:0;
    }
    for(uint32_t n=0;n<32;n++)bias[n]=0x406000;
    for(uint32_t n=0;n<32;n++)zero[n]=0x404000;
    uint32_t in[]={ (uint32_t)inblocks,1,2,10,10,32 };
    uint32_t out[]={ (uint32_t)outblocks,1,1,8,8,32 };
    qd_set_hmx_params_stride1(cfg,TEST_FLAGS,3,3,32,128,0,0);
    printf("probe flags=%x cfg0..12=",TEST_FLAGS);for(uint32_t i=0;i<13;i++)printf("%s%x",i?",":"",cfg[i]);
    printf(" in[H,W]=10,10 out[H,W]=8,8 input=%x output=%x table=%x entry=%x\n",(unsigned)input,(unsigned)output,(unsigned)outblocks,(unsigned)outblocks[0]);fflush(stdout);
    __asm__ volatile("mxclracc":: :"memory");
    qd_qnn_hmx_3x3(out,in,weights,bias,cfg,mode);
    *(volatile uint32_t *)done=0xffffffff;qd_hmx_finish(zero,done);
    uint32_t tries=0;while(*(volatile uint32_t *)done && tries++<100000)__asm__ volatile("nop");
    if(*(volatile uint32_t *)done){printf("FAIL completion done=%x tries=%u output=%u\n",*(volatile uint32_t *)done,(unsigned)tries,(unsigned)output[0]);return 1;}
    for(uint32_t y=0;y<8;y++)for(uint32_t x=0;x<8;x++)for(uint32_t c=0;c<32;c++) {
        uint32_t got=output[y*256+x*32+c],want=(uint8_t)(1+(y+1)*11+(x+1)*7+c%5);
        if(got!=want){printf("MISMATCH y%u x%u c%u got%u want%u\n",(unsigned)y,(unsigned)x,(unsigned)c,(unsigned)got,(unsigned)want);return 2;}
    }
    for(uint32_t i=0;i<2048;i++)if(guard[i]!=0xa5){printf("FAIL guard %u\n",(unsigned)i);return 3;}
    puts("PASS padded 3x3 center identity");return 0;
}
