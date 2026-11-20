/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef NPRG2_PREPARED_VCOPY_H
#define NPRG2_PREPARED_VCOPY_H
/* memmove-compatible byte copy for large DDR->VTCM/L2 transfers.
 *
 * On Hexagon with 128-byte HVX the copy keeps eight vector loads in flight per
 * iteration (a Frame DDR load costs ~190 cycles and nothing caches it), writes
 * only 128-byte aligned destination vectors in the body, and moves the head
 * (up to 127 bytes) and tail (masked store) so that no byte outside
 * [dst,dst+n) is written. A source that is not 128-byte aligned is realigned
 * with valign. The only source bytes read beyond [src,src+n) are inside the
 * aligned 128-byte lines holding its first and last byte, never a further page.
 * Ranges where dst lies inside (src,src+n) keep memmove's backward semantics
 * by calling memmove. Every other build (host) is memmove. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if defined(__HVX__) && __HVX_LENGTH__==128
#include <hexagon_types.h>
#include <hexagon_protos.h>
#endif

static inline void qd_vcopy(void *dstv, const void *srcv, size_t n) {
#if defined(__HVX__) && __HVX_LENGTH__==128 && !defined(QD_NO_VCOPY)
    uint8_t *dst=(uint8_t *)dstv;
    const uint8_t *src=(const uint8_t *)srcv;
    if(n<512 || (dst>src && (size_t)(dst-src)<n)) {memmove(dst,src,n);return;}
    size_t head=(size_t)(-(uintptr_t)dst)&127u;
    if(head){memmove(dst,src,head);dst+=head;src+=head;n-=head;}
    size_t nvec=n>>7,tail=n&127u;
    uint32_t sh=(uint32_t)((uintptr_t)src&127u);
    const HVX_Vector *sp=(const HVX_Vector *)((uintptr_t)src&~(uintptr_t)127);
    HVX_Vector *dp=(HVX_Vector *)dst;
    size_t i=0;
    if(!sh) {
        for(;i+8<=nvec;i+=8) {
            HVX_Vector a0=sp[i],a1=sp[i+1],a2=sp[i+2],a3=sp[i+3],a4=sp[i+4],a5=sp[i+5],a6=sp[i+6],a7=sp[i+7];
            dp[i]=a0;dp[i+1]=a1;dp[i+2]=a2;dp[i+3]=a3;dp[i+4]=a4;dp[i+5]=a5;dp[i+6]=a6;dp[i+7]=a7;
        }
        for(;i<nvec;i++)dp[i]=sp[i];
    } else {
        HVX_Vector prev=sp[0];
        for(;i+8<=nvec;i+=8) {
            HVX_Vector a1=sp[i+1],a2=sp[i+2],a3=sp[i+3],a4=sp[i+4],a5=sp[i+5],a6=sp[i+6],a7=sp[i+7],a8=sp[i+8];
            dp[i]=Q6_V_valign_VVR(a1,prev,(int32_t)sh);dp[i+1]=Q6_V_valign_VVR(a2,a1,(int32_t)sh);
            dp[i+2]=Q6_V_valign_VVR(a3,a2,(int32_t)sh);dp[i+3]=Q6_V_valign_VVR(a4,a3,(int32_t)sh);
            dp[i+4]=Q6_V_valign_VVR(a5,a4,(int32_t)sh);dp[i+5]=Q6_V_valign_VVR(a6,a5,(int32_t)sh);
            dp[i+6]=Q6_V_valign_VVR(a7,a6,(int32_t)sh);dp[i+7]=Q6_V_valign_VVR(a8,a7,(int32_t)sh);
            prev=a8;
        }
        for(;i<nvec;i++) {
            HVX_Vector nx=sp[i+1];
            dp[i]=Q6_V_valign_VVR(nx,prev,(int32_t)sh);prev=nx;
        }
    }
    if(tail) {
        /* The last tail bytes come from at most two aligned source vectors. */
        const uint8_t *ts=src+(nvec<<7);
        uintptr_t a=(uintptr_t)ts;
        const HVX_Vector *lo=(const HVX_Vector *)(a&~(uintptr_t)127),*hi=(const HVX_Vector *)((a+tail-1)&~(uintptr_t)127);
        HVX_Vector v=(a&127u)?Q6_V_valign_VVR(*hi,*lo,(int32_t)(a&127u)):*lo;
        Q6_vmem_QRIV(Q6_Q_vsetq_R((int32_t)tail),dp+nvec,v);
    }
#else
    memmove(dstv,srcv,n);
#endif
}

/* Same contract as qd_vcopy with 16 aligned vector loads in flight (12 when
 * the source needs realigning) for the big DDR -> VTCM weight copies. */
static inline void qd_vcopy_deep(void *dstv, const void *srcv, size_t n) {
#if defined(__HVX__) && __HVX_LENGTH__==128 && !defined(QD_NO_VCOPY)
    uint8_t *dst=(uint8_t *)dstv;
    const uint8_t *src=(const uint8_t *)srcv;
    if(n<2048 || (dst>src && (size_t)(dst-src)<n)) {qd_vcopy(dst,src,n);return;}
    size_t head=(size_t)(-(uintptr_t)dst)&127u;
    if(head){memmove(dst,src,head);dst+=head;src+=head;n-=head;}
    size_t nvec=n>>7,tail=n&127u;
    uint32_t sh=(uint32_t)((uintptr_t)src&127u);
    const HVX_Vector *sp=(const HVX_Vector *)((uintptr_t)src&~(uintptr_t)127);
    HVX_Vector *dp=(HVX_Vector *)dst;
    size_t i=0;
    if(!sh) {
        for(;i+16<=nvec;i+=16) {
            HVX_Vector a0=sp[i],a1=sp[i+1],a2=sp[i+2],a3=sp[i+3],a4=sp[i+4],a5=sp[i+5],a6=sp[i+6],a7=sp[i+7];
            HVX_Vector a8=sp[i+8],a9=sp[i+9],a10=sp[i+10],a11=sp[i+11],a12=sp[i+12],a13=sp[i+13],a14=sp[i+14],a15=sp[i+15];
            dp[i]=a0;dp[i+1]=a1;dp[i+2]=a2;dp[i+3]=a3;dp[i+4]=a4;dp[i+5]=a5;dp[i+6]=a6;dp[i+7]=a7;
            dp[i+8]=a8;dp[i+9]=a9;dp[i+10]=a10;dp[i+11]=a11;dp[i+12]=a12;dp[i+13]=a13;dp[i+14]=a14;dp[i+15]=a15;
        }
        for(;i<nvec;i++)dp[i]=sp[i];
    } else {
        HVX_Vector prev=sp[0];
        for(;i+12<=nvec;i+=12) {
            HVX_Vector a1=sp[i+1],a2=sp[i+2],a3=sp[i+3],a4=sp[i+4],a5=sp[i+5],a6=sp[i+6],a7=sp[i+7],a8=sp[i+8];
            HVX_Vector a9=sp[i+9],a10=sp[i+10],a11=sp[i+11],a12=sp[i+12];
            dp[i]=Q6_V_valign_VVR(a1,prev,(int32_t)sh);dp[i+1]=Q6_V_valign_VVR(a2,a1,(int32_t)sh);
            dp[i+2]=Q6_V_valign_VVR(a3,a2,(int32_t)sh);dp[i+3]=Q6_V_valign_VVR(a4,a3,(int32_t)sh);
            dp[i+4]=Q6_V_valign_VVR(a5,a4,(int32_t)sh);dp[i+5]=Q6_V_valign_VVR(a6,a5,(int32_t)sh);
            dp[i+6]=Q6_V_valign_VVR(a7,a6,(int32_t)sh);dp[i+7]=Q6_V_valign_VVR(a8,a7,(int32_t)sh);
            dp[i+8]=Q6_V_valign_VVR(a9,a8,(int32_t)sh);dp[i+9]=Q6_V_valign_VVR(a10,a9,(int32_t)sh);
            dp[i+10]=Q6_V_valign_VVR(a11,a10,(int32_t)sh);dp[i+11]=Q6_V_valign_VVR(a12,a11,(int32_t)sh);
            prev=a12;
        }
        for(;i<nvec;i++) {
            HVX_Vector nx=sp[i+1];
            dp[i]=Q6_V_valign_VVR(nx,prev,(int32_t)sh);prev=nx;
        }
    }
    if(tail) {
        const uint8_t *ts=src+(nvec<<7);
        uintptr_t a=(uintptr_t)ts;
        const HVX_Vector *lo=(const HVX_Vector *)(a&~(uintptr_t)127),*hi=(const HVX_Vector *)((a+tail-1)&~(uintptr_t)127);
        HVX_Vector v=(a&127u)?Q6_V_valign_VVR(*hi,*lo,(int32_t)(a&127u)):*lo;
        Q6_vmem_QRIV(Q6_Q_vsetq_R((int32_t)tail),dp+nvec,v);
    }
#else
    memmove(dstv,srcv,n);
#endif
}
#endif
