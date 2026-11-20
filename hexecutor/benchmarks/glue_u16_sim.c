/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "../native/qd_glue_hvx.h"
static uint16_t a[384] __attribute__((aligned(128))),b[384] __attribute__((aligned(128))),out[384] __attribute__((aligned(128)));
static uint8_t out8[384] __attribute__((aligned(128)));
static uint32_t ref(uint16_t x,int32_t *k) {
    int64_t v=((((int64_t)x-k[1])*k[0]+(1ll<<(k[3]-1)))>>k[3])+k[2],hi=k[4]?255:65535;
    return v<0?0:v>hi?hi:v;
}
int main(void) {
    FILE *f=fopen("/src/build/glue-u16-cases.bin","rb");uint32_t count[2],checked=0;
    if(!f || fread(count,4,2,f)!=2)return 1;
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    for(uint32_t t=0;t<count[0];t++) {
        int32_t k[5];if(fread(k,4,5,f)!=5)return 2;
        if(!dsp_affine_fits(k[0],k[1],k[2],k[3])){printf("FAIL bounds case %u\n",(unsigned)t);return 3;}
        for(uint32_t x=0;x<65536;x+=128) {
            for(uint32_t j=0;j<128;j++)a[j]=x+j;
            dsp_affine_u16(k[4]?(void *)out8:(void *)out,a,128,k[0],k[1],k[2],k[3],k[4]);
            for(uint32_t j=0;j<128;j++)if((k[4]?out8[j]:out[j])!=ref(a[j],k)){
                printf("FAIL affine %u x%u got%u want%u\n",(unsigned)t,(unsigned)a[j],(unsigned)(k[4]?out8[j]:out[j]),(unsigned)ref(a[j],k));return 4;
            }
            checked+=128;
        }
        memset(out,0xa5,sizeof out);memset(out8,0xa5,sizeof out8);
        for(uint32_t j=0;j<384;j++)a[j]=j*733;
        dsp_affine_u16(k[4]?(void *)(out8+1):(void *)(out+1),a+1,129,k[0],k[1],k[2],k[3],k[4]);
        if(k[4]?(out8[0]!=0xa5 || out8[130]!=0xa5):(out[0]!=0xa5a5 || out[130]!=0xa5a5))return 5;
        for(uint32_t j=1;j<130;j++)if((k[4]?out8[j]:out[j])!=ref(a[j],k))return 6;
    }
    for(uint32_t t=0;t<count[1];t++) {
        uint32_t k[3];if(fread(k,4,3,f)!=3)return 7;
        for(uint32_t x=0;x<65536;x+=128) {
            for(uint32_t j=0;j<128;j++){a[j]=x+j;b[j]=(x+j)*173+541;}
            dsp_add_u16(out,a,b,128,k[0],k[1],k[2]);
            for(uint32_t j=0;j<128;j++){
                uint64_t v=((uint64_t)a[j]*k[0]+(uint64_t)b[j]*k[1]+(1ull<<(k[2]-1)))>>k[2];v=v>65535?65535:v;
                if(out[j]!=v){printf("FAIL add %u x%u\n",(unsigned)t,(unsigned)(x+j));return 8;}
            }
            checked+=128;
        }
        memset(out,0xa5,sizeof out);dsp_add_u16(out+1,a+1,b+1,129,k[0],k[1],k[2]);
        if(out[0]!=0xa5a5 || out[130]!=0xa5a5)return 9;
        for(uint32_t j=1;j<130;j++){
            uint64_t v=((uint64_t)a[j]*k[0]+(uint64_t)b[j]*k[1]+(1ull<<(k[2]-1)))>>k[2];v=v>65535?65535:v;
            if(out[j]!=v)return 10;
        }
    }
    if(dsp_affine_fits(65535,0,0,1) || dsp_affine_fits(-65535,0,0,1) || dsp_affine_fits(1,0,0,0))return 11;
    printf("PASS %u affine + %u add parameter sets, %u comparisons, unaligned tails/canaries and overflow rejection\n",(unsigned)count[0],(unsigned)count[1],(unsigned)checked);
    SIM_RELEASE_HVX;fclose(f);return 0;
}
