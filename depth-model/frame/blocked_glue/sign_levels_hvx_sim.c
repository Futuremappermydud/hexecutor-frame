#include "sign_levels.h"
#include <hexagon_standalone.h>
#include <stdio.h>
#include <string.h>

static uint16_t input[65536+4] __attribute__((aligned(128)));
static uint16_t output[65536+4] __attribute__((aligned(128)));
static uint16_t inplace[65536+4] __attribute__((aligned(128)));
static const int32_t zps[]={65432,9196,10701,61378};
int main(void) {
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    QdSignLevels levels;
    if(!qd_sign_prepare_levels(3.0518509447574615e-05,32768,&levels))return 1;
    for(uint32_t x=0;x<65536;x++)input[x]=(uint16_t)x;
    input[65536]=0;input[65537]=32768;input[65538]=65535;
    for(unsigned z=0;z<sizeof zps/sizeof zps[0];z++) {
        memset(output,0xa5,sizeof output);
        if(!qd_sign_apply_u16_u16(input,output,65539,zps[z],&levels))return 2;
        for(uint32_t x=0;x<65536;x++) {
            uint16_t want=(int64_t)x<zps[z]?1:(int64_t)x>zps[z]?65535:32768;
            if(output[x]!=want){printf("mismatch zp%ld x%lu got%u want%u\n",(long)zps[z],(unsigned long)x,(unsigned)output[x],(unsigned)want);return 3;}
        }
        for(int t=0;t<3;t++) {
            uint16_t x=input[65536+t],want=x<zps[z]?1:x>zps[z]?65535:32768;
            if(output[65536+t]!=want)return 4;
        }
        if(output[65539]!=0xa5a5)return 5;
    }
    memcpy(inplace,input,sizeof input);
    if(!qd_sign_apply_u16_u16(inplace,inplace,65536,zps[1],&levels))return 6;
    for(uint32_t x=0;x<65536;x++) {
        uint16_t want=x<zps[1]?1:x>zps[1]?65535:32768;
        if(inplace[x]!=want)return 7;
    }
    puts("PASS HVX sign selector: all u16 inputs x 4 zps, tail guards, in-place");
    SIM_RELEASE_HVX;return 0;
}
