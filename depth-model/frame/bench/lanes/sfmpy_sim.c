/* ISS: is Q6_Vsf_vmpy_VsfVsf bit-identical to scalar IEEE float multiply for the values combine_scales sees? */
#include <stdint.h>
#include <stdio.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
static uint32_t rs=0x12345;static uint32_t rnd(void){rs^=rs<<13;rs^=rs>>17;rs^=rs<<5;return rs;}
static float A[32] __attribute__((aligned(128))),B[32] __attribute__((aligned(128))),O[32] __attribute__((aligned(128)));
int main(void){
    unsigned long n=0,bad=0,badq=0;
    static float O2[32] __attribute__((aligned(128)));
    for(int it=0;it<200000;it++){
        for(int i=0;i<32;i++){
            uint32_t a,b;
            switch(it&3){
            case 0: a=0x3f800000u+(rnd()&0x7fffffu);b=0x3f800000u+(rnd()&0x7fffffu);break;      /* [1,2) */
            case 1: a=0x37000000u+(rnd()&0x0fffffffu);b=0x3f000000u+(rnd()&0x01ffffffu);break;  /* small * near one */
            case 2: a=0x3f800000u+((rnd()&0x7ffu)<<12);b=0x37800000u+(rnd()&0x7fffffu);break;   /* 12-bit mantissas */
            default: a=0x30000000u+(rnd()&0x1fffffffu);b=0x30000000u+(rnd()&0x1fffffffu);break;
            }
            ((uint32_t*)A)[i]=a;((uint32_t*)B)[i]=b;
        }
        *(HVX_Vector*)O=Q6_Vsf_vmpy_VsfVsf(*(HVX_Vector*)A,*(HVX_Vector*)B);
        *(HVX_Vector*)O2=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector*)A,*(HVX_Vector*)B));
        for(int i=0;i<32;i++){
            volatile float p=A[i]*B[i];
            uint32_t w;float pf=p;__builtin_memcpy(&w,&pf,4);
            n++;bad+=((uint32_t*)O)[i]!=w;badq+=((uint32_t*)O2)[i]!=w;
        }
    }
    printf("samples %lu  vmpy.sf mismatches %lu  qf32 path mismatches %lu\n",n,bad,badq);
    return 0;
}
