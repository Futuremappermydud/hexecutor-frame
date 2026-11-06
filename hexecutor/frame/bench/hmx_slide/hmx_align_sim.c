/* Does HMX activation load (:deep:cm and :deep) honour an unaligned start?
 * Identity weights; compare load(a+off) vs load(aligned copy of a[off..]). */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
extern void qd_hmx_finish(void *,void *);
static void dot_cm(void *a,void *b,void *fi,void *out,uint32_t n) {
    __asm__ volatile("mxclracc\n bias=mxmem2(%2)\n { activation.ub=mxmem(%0,%4):deep:cm\n weight.b=mxmem(%1,%5) }\n cvt.ub=acc(%6)\n mxmem(%3,%6):cm=cvt\n"::"r"(a),"r"(b),"r"(fi),"r"(out),"r"(n),"r"(1023),"r"(0):"memory");
}
static void dot_pl(void *a,void *b,void *fi,void *out,uint32_t n) {
    __asm__ volatile("mxclracc\n bias=mxmem2(%2)\n { activation.ub=mxmem(%0,%4):deep\n weight.b=mxmem(%1,%5) }\n cvt.ub=acc(%6)\n mxmem(%3,%6)=cvt\n"::"r"(a),"r"(b),"r"(fi),"r"(out),"r"(n),"r"(1023),"r"(0):"memory");
}
static uint8_t *done;static int32_t *zero;
static void fin(void){*(volatile uint32_t *)done=0xffffffff;qd_hmx_finish(zero,done);uint32_t t=0;while(*(volatile uint32_t *)done&&t++<100000)__asm__ volatile("nop");}
int main(void) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x100000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint8_t *src=(uint8_t *)base,*b=src+0x8000,*cp=src+0x10000,*o1=src+0x20000,*o2=src+0x28000;
    int32_t *fi=(int32_t *)(src+0x9000);zero=(int32_t *)(src+0x9800);done=src+0x30000;
    for(uint32_t i=0;i<0x8000;i++)src[i]=(uint8_t)((i*2654435761u)>>13);
    memset(b,0,1024);for(uint32_t k=0;k<32;k++)b[((k/4)*32+k)*4+k%4]=1;
    for(uint32_t c=0;c<32;c++){fi[c]=0x406000;fi[32+c]=0;zero[c]=0x404000;zero[32+c]=0;}
    const uint32_t offs[]={0,32,64,96,128,256,512,1024,2048+32,2048+256};
    for(int mode=0;mode<2;mode++)for(unsigned i=0;i<sizeof offs/4;i++) {
        uint32_t off=offs[i];
        memcpy(cp,src+off,2048);
        memset(o1,0xa5,4096);memset(o2,0x5a,4096);
        if(mode){dot_cm(src+off,b,fi,o1,2047);fin();dot_cm(cp,b,fi,o2,2047);fin();}
        else {dot_pl(src+off,b,fi,o1,2047);fin();dot_pl(cp,b,fi,o2,2047);fin();}
        uint32_t diff=0,same_aligned=0;for(uint32_t p=0;p<2048;p++)diff+=o1[p]!=o2[p];
        /* what does an aligned-down load give? */
        memcpy(cp,src+(off&~2047u),2048);memset(o2,0x5a,4096);
        if(mode){dot_cm(cp,b,fi,o2,2047);}else dot_pl(cp,b,fi,o2,2047);fin();
        for(uint32_t p=0;p<2048;p++)same_aligned+=o1[p]==o2[p];
        printf("mode=%s off=%u diff_vs_copy=%u same_as_aligned_down=%u o1[0..3]=%u,%u,%u,%u\n",mode?"cm":"plain",off,diff,same_aligned,o1[0],o1[1],o1[2],o1[3]);
    }
    /* 2-tile deep from an offset (K=64): is the second tile at off+2048? */
    return 0;
}
