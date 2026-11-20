/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QNN16_HALF_H
#define QNN16_HALF_H
#include <stdint.h>
/* Inline IEEE binary16 conversions. The engine builds with -fno-builtin, so
 * _Float16 casts and two-byte memcpy would each be a library call per
 * element. fh rounds to nearest even (F. Giesen, float_to_half_fast3_rtne);
 * both are checked against the compiler's conversions in test_qnn16_hvx.c. */
typedef union { uint32_t u; float f; } Bits32;
static inline __attribute__((unused)) float hf(uint16_t h){
 uint32_t sign=(uint32_t)(h&0x8000u)<<16,e=(h>>10)&0x1fu,m=h&0x3ffu;Bits32 v;
 if(!e){float f=(float)m*5.9604644775390625e-8f;return sign?-f:f;}
 v.u=e==31u?sign|0x7f800000u|(m<<13):sign|((e+112u)<<23)|(m<<13);
 return v.f;
}
static inline __attribute__((unused)) uint16_t fh(float f){
 Bits32 x={.f=f},magic={.u=((127u-15u)+(23u-10u)+1u)<<23};
 uint32_t sign=x.u&0x80000000u;uint16_t o;
 x.u^=sign;
 if(x.u>=(127u+16u)<<23)o=x.u>255u<<23?0x7e00u:0x7c00u;
 else if(x.u<113u<<23){x.f+=magic.f;o=(uint16_t)(x.u-magic.u);}
 else{uint32_t odd=(x.u>>13)&1u;x.u+=((uint32_t)(15-127)<<23)+0xfffu;x.u+=odd;o=(uint16_t)(x.u>>13);}
 return (uint16_t)(o|(sign>>16));
}
#endif
