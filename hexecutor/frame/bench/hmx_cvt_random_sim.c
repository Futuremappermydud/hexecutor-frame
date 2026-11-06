/* Native integer HMX conversion characterisation (V75 ISS).
 * A: one crouton block (64 px x 32 ch u8, byte p*32+k), :deep:cm load.
 * B: weight.b block, byte ((k/4)*32+n)*4+k%4.  Fi: 32 scale words, 32 bias.
 * C: cvt.ub, :cm store, byte p*32+n.  Record per case: fi[64], A[2048], B[1024], C[2048]. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <hexagon_standalone.h>
static uint32_t seed=0x2545f491u;
static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
static void run(uint8_t *a,uint8_t *b,int32_t *fi,uint8_t *c,int32_t *zero,uint8_t *comp,int mode){
  __asm__ volatile("mxclracc\n bias = mxmem2(%2)\n"
    "{ activation.ub = mxmem(%0,%3):deep:cm\n weight.b = mxmem(%1,%4) }\n"
    ::"r"(a),"r"(b),"r"(fi),"r"(2047),"r"(1023):"memory");
  if(mode==0) __asm__ volatile("cvt.ub=acc(%1)\n mxmem(%0,%1):cm=cvt"::"r"(c),"r"(0):"memory");
  else if(mode==1) __asm__ volatile("cvt.ub=acc(%1):sc0\n mxmem(%0,%1):cm=cvt"::"r"(c),"r"(0):"memory");
  else __asm__ volatile("cvt.ub=acc(%1):sc1\n mxmem(%0,%1):cm=cvt"::"r"(c),"r"(0):"memory");
  for(int i=0;i<2048;i+=4)*(volatile uint32_t*)(comp+i)=0xffffffffu;
  __asm__ volatile("mxclracc\n bias = mxmem2(%0)\n cvt.uh=acc(%2):2x1\n mxmem(%1,%2)=cvt"::"r"(zero),"r"(comp),"r"(0):"memory");
  while(*(volatile uint32_t*)comp);
}
int main(int argc,char **argv){
  int cases=argc>1?atoi(argv[1]):2000, mode=argc>2?atoi(argv[2]):0;
  FILE *o=fopen(argc>3?argv[3]:"/repo/.work/hmxq/out.bin","wb"); if(!o) return 2;
  uint32_t base=__rdcfg(__vtcm_base)<<16;
  for(uint32_t i=0;i<0x20000;i+=4096)add_translation((void*)(base+i),(void*)(base+i),6);
  uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
  uint8_t *a=(uint8_t*)base,*b=a+0x1000,*c=a+0x2000,*comp=a+0x3000;int32_t *fi=(int32_t*)(a+0x4000),*zero=(int32_t*)(a+0x4800);
  for(int k=0;k<64;k++)zero[k]=k<32?0x404000:0;
  for(int t=0;t<cases;t++){
    int kind=t%4; /* 0: sparse small acc, 1: dense random, 2: bias-only sweep, 3: dense + near-zero bias */
    for(int i=0;i<2048;i++)a[i]=kind==2?0:(kind==0?((i&31)<2?rnd():0):rnd());
    for(int i=0;i<1024;i++)b[i]=kind==0?((i&3)==0?rnd():0):rnd();
    for(int n=0;n<32;n++){
      uint32_t pre=rnd()%14, m=rnd()%1024, s=rnd();
      if(n<4) m=n==0?0:(n==1?1023:512);
      fi[n]=(int32_t)(0x400000u|((20u-pre)<<10)|m);
      int32_t bias=(int32_t)(rnd()%(1u<<(pre+10)))-(int32_t)(1u<<(pre+4));
      if(kind==2) bias=(int32_t)((s&0xfffff)) - 0x8000 + n*7;
      if(kind==3) bias=(int32_t)(rnd()%200001)-100000;
      fi[32+n]=bias;
    }
    memset(c,0xee,2048);run(a,b,fi,c,zero,comp,mode);
    fwrite(fi,4,64,o);fwrite(a,1,2048,o);fwrite(b,1,1024,o);fwrite(c,1,2048,o);
  }
  fclose(o);printf("done %d cases mode %d\n",cases,mode);return 0;
}
