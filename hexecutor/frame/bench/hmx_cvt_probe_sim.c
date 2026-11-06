/* Contiguous acc sweep: acc[p][n] = x0 + 2048*t + 64*n + p, one (pre, m) per run.
 * Output per case: C[2048] only (p*32+n). argv: cases pre m x0 out */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <hexagon_standalone.h>
int main(int argc,char **argv){
  int cases=atoi(argv[1]),pre=atoi(argv[2]),m=atoi(argv[3]);uint32_t extra=(uint32_t)strtoul(argv[6],0,16);int mode=atoi(argv[7]);int32_t x0=atoi(argv[4]);
  FILE *o=fopen(argv[5],"wb"); if(!o) return 2;
  uint32_t base=__rdcfg(__vtcm_base)<<16;
  for(uint32_t i=0;i<0x20000;i+=4096)add_translation((void*)(base+i),(void*)(base+i),6);
  uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
  uint8_t *a=(uint8_t*)base,*b=a+0x1000,*c=a+0x2000,*comp=a+0x3000;int32_t *fi=(int32_t*)(a+0x4000),*zero=(int32_t*)(a+0x4800);
  for(int k=0;k<64;k++)zero[k]=k<32?0x404000:0;
  memset(a,0,2048);memset(b,0,1024);
  for(int p=0;p<64;p++)a[p*32]=(uint8_t)p;
  for(int n=0;n<32;n++)b[n*4]=1;           /* k=0 row: ((0/4)*32+n)*4+0 */
  for(int n=0;n<32;n++)fi[n]=(int32_t)(0x400000u|((uint32_t)(20-pre)<<10)|(uint32_t)m)^extra;
  for(int t=0;t<cases;t++){
    for(int n=0;n<32;n++)fi[32+n]=x0+2048*t+64*n;
    __asm__ volatile("mxclracc\n bias = mxmem2(%2)\n"
      "{ activation.ub = mxmem(%0,%3):deep:cm\n weight.b = mxmem(%1,%4) }\n"
      
      ::"r"(a),"r"(b),"r"(fi),"r"(2047),"r"(1023):"memory");
    if(mode==0)__asm__ volatile("cvt.ub=acc(%1)\n mxmem(%0,%1):cm=cvt"::"r"(c),"r"(0):"memory");
    else if(mode==1)__asm__ volatile("cvt.ub=acc(%1):sc0\n mxmem(%0,%1):cm=cvt"::"r"(c),"r"(0):"memory");
    else __asm__ volatile("cvt.ub=acc(%1):sc1\n mxmem(%0,%1):cm=cvt"::"r"(c),"r"(0):"memory");
    for(int i=0;i<2048;i+=4)*(volatile uint32_t*)(comp+i)=0xffffffffu;
    __asm__ volatile("mxclracc\n bias = mxmem2(%0)\n cvt.uh=acc(%2):2x1\n mxmem(%1,%2)=cvt"::"r"(zero),"r"(comp),"r"(0):"memory");
    while(*(volatile uint32_t*)comp);
    fwrite(c,1,2048,o);
  }
  fclose(o);return 0;
}
