/* Tap map of QNN's 16-bit slide activation load (hmx_v75_convhbh_stride1_aligned):
 * activation.ub=mxmem(Rs,Rt)[:above] + weight.b=mxmem(w,r):dilate, cvt.uh:2x1.
 * A block: u16 crouton [row y][pair pp][k*4+2j+{0,1}]: value in the high byte.
 * args: rs_off rt_low T mode(0 plain,1 above) dil(0/1) stlow */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <hexagon_standalone.h>
extern void qd_hmx_finish(void *,void *);
static uint8_t *done;static int32_t *zero;
static void fin(void){*(volatile uint32_t *)done=0xffffffff;qd_hmx_finish(zero,done);uint32_t t=0;while(*(volatile uint32_t *)done&&t++<100000)__asm__ volatile("nop");if(*(volatile uint32_t*)done)puts("NO COMPLETION");}
static void show(const char *n,uint8_t *out,int T){
    printf("%s\n",n);
    uint16_t *o=(uint16_t *)out;
    for(int t=0;t<T;t++){printf(" tap %d:",t);
        for(int r=0;r<2;r++){printf(" |");for(int p=0;p<4;p++){int pp=p>>1,j=p&1;
            uint16_t v=o[(r*2+pp)*64+t*2+j],m=o[(r*2+pp)*64+(16+t)*2+j];
            if(v==0xeeee) printf("  ee");else if(!m)printf("  --");else printf(" %d%d%d",v/64,(v/8)%8,v%8);}}
        printf("\n");}
}
int main(int argc,char **argv) {
    uint32_t base=__rdcfg(__vtcm_base)<<16;
    for(uint32_t i=0;i<0x100000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    uint8_t *blk=(uint8_t *)base;
    uint8_t *w=(uint8_t *)base+0x20000,*out=(uint8_t *)base+0x30000;
    int32_t *fi=(int32_t *)(base+0x38000);zero=(int32_t *)(base+0x38800);done=(uint8_t *)base+0x39000;
    for(int b=0;b<4;b++){uint8_t *p=blk+b*0x4000;memset(p,0,2048);
        for(int y=0;y<8;y++)for(int x=0;x<4;x++){int pp=x>>1,j=x&1;uint8_t *q=p+(y*2+pp)*128;
            q[0*4+2*j+1]=(uint8_t)(b*64+y*8+x);q[1*4+2*j+1]=1;}}
    memset(w,0,32*1024);
    for(int t=0;t<16;t++){w[t*1024+((0/4)*32+t)*4+0]=1;w[t*1024+((1/4)*32+16+t)*4+1]=1;}
    for(int c=0;c<32;c++){fi[c]=0x404000;fi[32+c]=0;zero[c]=0x404000;zero[32+c]=0;}
    uint32_t rs_off=strtoul(argv[1],0,0),rt_low=strtoul(argv[2],0,0),T=atoi(argv[3]);int above=atoi(argv[4]),dil=atoi(argv[5]);uint32_t stl=argc>6?strtoul(argv[6],0,0):0;
    uint8_t *A0=blk,*B0=blk+0x4000,*A1=blk+0x8000,*B1=blk+0xc000;
    uint32_t wr=T*1024-1;
    memset(out,0xee,0x4000);
#define LOAD(MOD,WMOD,a,t) __asm__ volatile("{ activation.ub=mxmem(%0,%1)" MOD "\n weight.b=mxmem(%2,%3)" WMOD " }\n"::"r"(a),"r"(t),"r"(w),"r"(wr):"memory")
#define L(a,t) do{ if(above){if(dil)LOAD(":above",":dilate",a,t);else LOAD(":above","",a,t);}else{if(dil)LOAD("",":dilate",a,t);else LOAD("","",a,t);} }while(0)
#define ST(o) __asm__ volatile("cvt.uh=acc(%1):2x1\n mxmem(%0,%2)=cvt\n"::"r"(o),"r"(0),"r"(stl):"memory")
    __asm__ volatile("mxclracc\n bias=mxmem2(%0)\n"::"r"(fi):"memory");
    L((uint32_t)A0|rs_off,(uint32_t)(B0-A0)|rt_low);
    ST(out);
    L((uint32_t)A1|rs_off,(uint32_t)(B1-A1)|rt_low);
    ST(out+0x1000);
    ST(out+0x2000);
    fin();
    printf("rs_off=0x%lx rt_low=0x%lx T=%lu above=%d dil=%d st=%lx\n",(unsigned long)rs_off,(unsigned long)rt_low,(unsigned long)T,above,dil,(unsigned long)stl);
    show("out0",out,T);show("out1",out+0x1000,T);show("out2 (no load)",out+0x2000,T);
    return 0;
}
