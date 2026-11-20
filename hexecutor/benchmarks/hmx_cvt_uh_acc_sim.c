/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* cvt.uh:2x1 with real accumulators. Mode 0: one-hot A byte i (value 1) with
 * B[k][n] = 1 for k==n%32 ... records C for each i. Mode 1: random A, B, bias,
 * scale (e, m). Record: fi[64], A[2048], B[1024], C[2048]. argv: mode cases out */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <hexagon_standalone.h>
static uint32_t seed=0x9e3779b9u;
static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
int main(int argc,char **argv){
  int mode=atoi(argv[1]),cases=atoi(argv[2]);FILE *o=fopen(argv[3],"wb"); if(!o) return 2;
  uint32_t base=__rdcfg(__vtcm_base)<<16;
  for(uint32_t i=0;i<0x20000;i+=4096)add_translation((void*)(base+i),(void*)(base+i),6);
  uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
  uint8_t *a=(uint8_t*)base,*b=a+0x1000,*c=a+0x2000,*comp=a+0x3000;int32_t *fi=(int32_t*)(a+0x4000),*zero=(int32_t*)(a+0x4800);
  for(int k=0;k<64;k++)zero[k]=k<32?0x404000:0;
  for(int t=0;t<cases;t++){
    if(mode==0){ /* one-hot activation byte t, weights: B byte j = (j&127)+1 distinct per (k,n) */
      memset(a,0,2048);a[t]=1;
      for(int j=0;j<1024;j++)b[j]=(uint8_t)((j%127)+1);
      for(int n=0;n<32;n++){fi[n]=0x400000|(26<<10);fi[32+n]=0;} /* e=26: out = x*2^18*1024 >>26 = x*2^2... */
    } else {
      for(int i=0;i<2048;i++)a[i]=rnd();
      for(int i=0;i<1024;i++)b[i]=rnd();
      for(int n=0;n<32;n++){uint32_t e=8+rnd()%12,m=rnd()%1024;fi[n]=(int32_t)(0x400000u|(e<<10)|m);fi[32+n]=(int32_t)(rnd()%400001)-200000;}
    }
    __asm__ volatile("mxclracc\n bias = mxmem2(%2)\n"
      "{ activation.ub = mxmem(%0,%3):deep\n weight.b = mxmem(%1,%4) }\n"
      "cvt.uh=acc(%5):2x1\n mxmem(%6,%5)=cvt\n"
      ::"r"(a),"r"(b),"r"(fi),"r"(2047),"r"(1023),"r"(0),"r"(c):"memory");
    for(int i=0;i<2048;i+=4)*(volatile uint32_t*)(comp+i)=0xffffffffu;
    __asm__ volatile("mxclracc\n bias = mxmem2(%0)\n cvt.uh=acc(%2):2x1\n mxmem(%1,%2)=cvt"::"r"(zero),"r"(comp),"r"(0):"memory");
    while(*(volatile uint32_t*)comp);
    fwrite(fi,4,64,o);fwrite(a,1,2048,o);fwrite(b,1,1024,o);fwrite(c,1,2048,o);
  }
  fclose(o);return 0;
}
