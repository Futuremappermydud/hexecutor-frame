/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Calibrate integer HMX against model arithmetic in Qualcomm's ISA simulator.
 * Input: case count followed by {convert,u8 A[64,32],s8 packed B[32,32],Fi[64]}.
 * Output: one native u8 [64,32] tile per case. No physical device access. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
static void dot(void *a,void *b,void *fi,void *out,uint32_t convert,uint32_t blocks) {
    __asm__ volatile("mxclracc\n bias = mxmem2(%2)\n"
                     "{ activation.ub = mxmem(%0,%3):deep\n weight.b = mxmem(%1,%4) }\n"
                     ::"r"(a),"r"(b),"r"(fi),"r"(blocks*2048-1),"r"(blocks*1024-1):"memory");
    if(convert&256)__asm__ volatile("cvt.uh=acc(%1):2x1\n mxmem(%0,%1)=cvt"::"r"(out),"r"(0):"memory");
    else __asm__ volatile("cvt.ub=acc(%1)\n mxmem(%0,%2)=cvt"::"r"(out),"r"(convert),"r"(0):"memory");
    for(uint32_t i=0;i<128;i++)__asm__ volatile("nop");
}
int main(int argc,char **argv) {
    const char *ip=argc==3?argv[1]:"/src/build/hmx-int-input.bin";
    const char *op=argc==3?argv[2]:"/src/build/hmx-int-output.bin";
    FILE *in=fopen(ip,"rb"),*output=fopen(op,"wb");
    if(!in || !output)return 2;
    uint32_t base=__rdcfg(__vtcm_base)<<16,count;
    for(uint32_t i=0;i<0x50000;i+=4096)add_translation((void *)(base+i),(void *)(base+i),6);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    unsigned char *a=(unsigned char *)base,*b=a+0x20000,*out=a+0x40000;
    int32_t *fi=(int32_t *)(base+0x30000);
    if(fread(&count,4,1,in)!=1)return 3;
    for(uint32_t c=0;c<count;c++) {
        uint32_t convert,blocks;
        if(fread(&convert,4,1,in)!=1 || fread(&blocks,4,1,in)!=1 || blocks>64 || fread(a,2048,blocks,in)!=blocks || fread(b,1024,blocks,in)!=blocks || fread(fi,1,256,in)!=256)return 4;
        memset(out,0xee,2048);dot(a,b,fi,out,convert,blocks);
        if(fwrite(out,1,2048,output)!=2048)return 5;
    }
    fclose(in);fclose(output);printf("Integer HMX: %u cases\n",(unsigned)count);
    return 0;
}
