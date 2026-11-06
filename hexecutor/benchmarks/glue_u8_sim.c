#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "../native/qd_glue_hvx.h"
static uint8_t a[384] __attribute__((aligned(128))),b[384] __attribute__((aligned(128))),out[384] __attribute__((aligned(128)));
int main(void) {
    FILE *f=fopen("/src/build/glue-u8-cases.bin","rb");uint32_t cases;
    if(!f || fread(&cases,4,1,f)!=1)return 1;
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    uint32_t checked=0;
    for(uint32_t t=0;t<cases;t++) {
        uint32_t k[3];if(fread(k,4,3,f)!=3)return 2;
        for(uint32_t p=0;p<65536;p+=128) {
            for(uint32_t j=0;j<128;j++){a[j]=(p+j)/256;b[j]=(p+j)%256;}
            dsp_add_u8(out,a,b,128,k[0],k[1],k[2]);
            for(uint32_t j=0;j<128;j++) {
                uint64_t v=((uint64_t)a[j]*k[0]+(uint64_t)b[j]*k[1]+(1ull<<(k[2]-1)))>>k[2];if(v>255)v=255;
                if(out[j]!=v){printf("FAIL case %u pair %u got %u want %u\n",(unsigned)t,(unsigned)(p+j),(unsigned)out[j],(unsigned)v);return 3;}
                checked++;
            }
        }
        for(uint32_t j=0;j<384;j++){a[j]=j*73;b[j]=j*41;out[j]=0xa5;}
        dsp_add_u8(out+1,a+1,b+1,129,k[0],k[1],k[2]);
        if(out[0]!=0xa5 || out[130]!=0xa5)return 4;
        for(uint32_t j=1;j<130;j++) {
            uint64_t v=((uint64_t)a[j]*k[0]+(uint64_t)b[j]*k[1]+(1ull<<(k[2]-1)))>>k[2];if(v>255)v=255;
            if(out[j]!=v)return 5;
        }
    }
    printf("PASS %u cases, %u exhaustive comparisons, plus unaligned/tail canaries\n",(unsigned)cases,(unsigned)checked);
    SIM_RELEASE_HVX;fclose(f);return 0;
}
