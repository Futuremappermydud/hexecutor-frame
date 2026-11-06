/* Calibrate HMX's native crouton load/store, with identity weights. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
extern void qd_hmx_finish(void *,void *);
static void dot(void *a,void *b,void *fi,void *out,uint32_t input,uint32_t output) {
    __asm__ volatile("mxclracc\n bias=mxmem2(%2)\n { activation.ub=mxmem(%0,%4):deep:cm\n weight.b=mxmem(%1,%6) }\n cvt.ub=acc(%7)\n mxmem(%3,%5):cm=cvt\n"::"r"(a),"r"(b),"r"(fi),"r"(out),"r"(input),"r"(output),"r"(1023),"r"(0):"memory");
}
int main(void) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x80000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint8_t *a=(uint8_t *)base,*b=a+2048,*out=a+0x4000,*done=a+0x50000;
    int32_t *fi=(int32_t *)(a+3072),*zero=(int32_t *)(a+3328);
    memset(a,0,0x80000);
    for(uint32_t p=0;p<64;p++)for(uint32_t c=0;c<32;c++)a[p*32+c]=(p*37+c*11)&255;
    for(uint32_t k=0;k<32;k++)b[((k/4)*32+k)*4+k%4]=1;
    for(uint32_t c=0;c<32;c++){fi[c]=0x406000;zero[c]=0x404000;}
    const uint32_t ctl[]={31,63,127,255,511,1023,2047,0};
    uint32_t winners=0;
    for(uint32_t i=0;i<8;i++)for(uint32_t j=0;j<8;j++) {
        memset(out,0xa5,0x40000);*(volatile uint32_t *)done=0xffffffff;
        dot(a,b,fi,out,ctl[i],ctl[j]);qd_hmx_finish(zero,done);
        uint32_t tries=0;while(*(volatile uint32_t *)done && tries++<10000)__asm__ volatile("nop");
        if(*(volatile uint32_t *)done){puts("FAIL completion");return 1;}
        uint32_t diff=0,spill=0;
        for(uint32_t p=0;p<2048;p++)if(out[p]!=a[p])diff++;
        for(uint32_t p=2048;p<0x40000;p++)if(out[p]!=0xa5)spill++;
        printf("input=%x output=%x different=%u spill=%u first=%u,%u,%u,%u\n",(unsigned)ctl[i],(unsigned)ctl[j],(unsigned)diff,(unsigned)spill,(unsigned)out[0],(unsigned)out[1],(unsigned)out[2],(unsigned)out[3]);
        if(!diff && !spill)winners++;
    }
    printf("DONE %u exact crouton configurations\n",(unsigned)winners);return 0;
}
