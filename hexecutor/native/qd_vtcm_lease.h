/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_VTCM_LEASE_H
#define QD_VTCM_LEASE_H

/* Exclusive-session lease for the complete 8 MiB SM8650 VTCM window.
 * No libc, allocation, or firmware constants are used. The caller must hold
 * the FrameTransport exclusive callback lease and keep the eye client paused
 * until qdvt_release_verified() succeeds. */
#include <stdint.h>
#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

#define QDVT_BASE          0x31000000u
#define QDVT_BYTES         0x00800000u
#define QDVT_BACKUP_BASE   0xa6000000u
#define QDVT_BACKUP_BYTES  0x00800000u
#define QDVT_TLB8          8u
#define QDVT_TLB9          9u
#define QDVT_TLB8_VALUE    0xc0031000f7062020ull
#define QDVT_TLB9_VALUE    0xc0031400f7062820ull
#define QDVT_MAGIC         0x31545651u

typedef struct {
    volatile uint32_t magic;
    volatile uint32_t phase; /* 0 empty, 1 saved, 2 restore verified */
    volatile uint32_t error;
    volatile uint32_t reserved;
    uint64_t tlb8, tlb9;
    uint32_t base, bytes, backup_base;
    /* XOR plus position-weighted (vector_index+1) 32-bit word sum. The
     * positional term detects backup chunk reordering that plain reductions
     * cannot distinguish. */
    uint32_t checksum_xor, checksum_weighted;
    uint8_t *backup;
} QdVtcmLease;

enum {
    QDVT_OK=0, QDVT_BAD_STATE=1, QDVT_BAD_RANGE=2,
    QDVT_MAPPING_CHANGED=3, QDVT_RESTORE_MISMATCH=4
};

#ifdef __hexagon__
static uint64_t qdvt_read_tlb(uint32_t i) {
    uint64_t e;
    __asm__ volatile("%0=tlbr(%1)" : "=r"(e) : "r"(i) : "memory");
    return e;
}
static void qdvt_sync(void) { __asm__ volatile("syncht\n barrier" ::: "memory"); }
#else
/* Host harness may install a test reader; the target implementation reads
 * the architectural TLB registers directly. */
extern uint64_t qdvt_test_read_tlb(uint32_t);
static uint64_t qdvt_read_tlb(uint32_t i) { return qdvt_test_read_tlb(i); }
static void qdvt_sync(void) { __asm__ volatile("" ::: "memory"); }
#endif

#ifdef __hexagon__
/* Low 32 bits of lane-wise word multiplication by a scalar coefficient. */
static HVX_Vector qdvt_mul_u32(HVX_Vector v,uint32_t m) {
    HVX_Vector lo=Q6_V_vsplat_R(m&65535u), hi=Q6_V_vsplat_R(m>>16);
    return Q6_Vw_vadd_VwVw(Q6_Vw_vmpyie_VwVuh(v,lo),
            Q6_Vw_vasl_VwR(Q6_Vw_vmpyie_VwVuh(v,hi),16));
}
#endif

static int qdvt_range_valid(uint32_t base, uint32_t bytes, uint32_t backup_base,
                            volatile uint8_t *vtcm, uint8_t *backup) {
    if (base != QDVT_BASE || bytes != QDVT_BYTES || backup_base != QDVT_BACKUP_BASE ||
        !vtcm || !backup || ((uintptr_t)backup & 127u) || ((uintptr_t)vtcm & 127u)) return 0;
#ifdef __hexagon__
    /* On target the architectural pointers must be exactly the validated
     * TCM and private cdsp_mem addresses. */
    if ((uintptr_t)vtcm != QDVT_BASE || (uintptr_t)backup != QDVT_BACKUP_BASE) return 0;
#else
    /* Host builds are explicitly a mock backend: pointers may be ordinary
     * aligned allocations, but the requested architectural ranges remain
     * fixed and the mock buffers must not overlap. */
    uintptr_t v=(uintptr_t)vtcm, b=(uintptr_t)backup;
    if (v > UINTPTR_MAX-QDVT_BYTES || b > UINTPTR_MAX-QDVT_BACKUP_BYTES ||
        (v < b+QDVT_BACKUP_BYTES && b < v+QDVT_BYTES)) return 0;
#endif
    /* The backup occupies only the dedicated free 96..104 MiB range in the
     * private 128 MiB cdsp_mem mapping. */
    if (backup_base < 0xa0000000u || bytes > 0xa8000000u - backup_base) return 0;
    return 1;
}
static void qdvt_copy_fingerprint(volatile uint8_t *src, uint8_t *dst,
                                 uint32_t bytes, uint32_t *xor_out, uint32_t *sum_out) {
#ifdef __hexagon__
    volatile HVX_Vector *sv=(volatile HVX_Vector *)src;
    volatile HVX_Vector *dv=(volatile HVX_Vector *)dst;
    HVX_Vector x=Q6_V_vzero(), weighted=Q6_V_vzero();
    for (uint32_t i=0;i<bytes/128u;i+=8u) {
        /* Issue eight independent DDR reads before any store or dependent
         * vector operation so the memory system can overlap read latency. */
        HVX_Vector v0=sv[i+0], v1=sv[i+1], v2=sv[i+2], v3=sv[i+3];
        HVX_Vector v4=sv[i+4], v5=sv[i+5], v6=sv[i+6], v7=sv[i+7];
        dv[i+0]=v0; dv[i+1]=v1; dv[i+2]=v2; dv[i+3]=v3;
        dv[i+4]=v4; dv[i+5]=v5; dv[i+6]=v6; dv[i+7]=v7;
        x=Q6_V_vxor_VV(x,Q6_V_vxor_VV(Q6_V_vxor_VV(v0,v1),Q6_V_vxor_VV(v2,v3)));
        x=Q6_V_vxor_VV(x,Q6_V_vxor_VV(Q6_V_vxor_VV(v4,v5),Q6_V_vxor_VV(v6,v7)));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v0,i+1u));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v1,i+2u));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v2,i+3u));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v3,i+4u));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v4,i+5u));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v5,i+6u));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v6,i+7u));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v7,i+8u));
    }
    uint32_t lanes[32] __attribute__((aligned(128)));
    *(HVX_Vector *)lanes=x; uint32_t fx=0;
    for (uint32_t i=0;i<32;i++) fx^=lanes[i];
    *(HVX_Vector *)lanes=weighted; uint32_t fw=0;
    for (uint32_t i=0;i<32;i++) fw+=lanes[i];
    *xor_out=fx; *sum_out=fw;
#else
    uint32_t x=0, weighted=0;
    for (uint32_t i=0;i<bytes;i+=4) {
        uint32_t v=(uint32_t)src[i] | ((uint32_t)src[i+1]<<8) |
                   ((uint32_t)src[i+2]<<16) | ((uint32_t)src[i+3]<<24);
        dst[i]=src[i]; dst[i+1]=src[i+1]; dst[i+2]=src[i+2]; dst[i+3]=src[i+3];
        x^=v; weighted+=v*(i/128u+1u);
    }
    *xor_out=x; *sum_out=weighted;
#endif
}
static int qdvt_fingerprint_matches(const QdVtcmLease *s) {
#ifdef __hexagon__
    volatile HVX_Vector *p=(volatile HVX_Vector *)s->backup;
    HVX_Vector x=Q6_V_vzero(), weighted=Q6_V_vzero();
    for (uint32_t i=0;i<s->bytes/128u;i+=8u) {
        HVX_Vector v0=p[i+0], v1=p[i+1], v2=p[i+2], v3=p[i+3];
        HVX_Vector v4=p[i+4], v5=p[i+5], v6=p[i+6], v7=p[i+7];
        x=Q6_V_vxor_VV(x,Q6_V_vxor_VV(Q6_V_vxor_VV(v0,v1),Q6_V_vxor_VV(v2,v3)));
        x=Q6_V_vxor_VV(x,Q6_V_vxor_VV(Q6_V_vxor_VV(v4,v5),Q6_V_vxor_VV(v6,v7)));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v0,i+1u));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v1,i+2u));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v2,i+3u));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v3,i+4u));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v4,i+5u));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v5,i+6u));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v6,i+7u));
        weighted=Q6_Vw_vadd_VwVw(weighted,qdvt_mul_u32(v7,i+8u));
    }
    uint32_t lanes[32] __attribute__((aligned(128))); *(HVX_Vector *)lanes=x;
    uint32_t fx=0; for (uint32_t i=0;i<32;i++) fx^=lanes[i];
    *(HVX_Vector *)lanes=weighted; uint32_t fw=0; for (uint32_t i=0;i<32;i++) fw+=lanes[i];
    return fx==s->checksum_xor && fw==s->checksum_weighted;
#else
    uint32_t x=0,weighted=0;
    for (uint32_t i=0;i<s->bytes;i+=4) {
        uint32_t v=(uint32_t)s->backup[i] | ((uint32_t)s->backup[i+1]<<8) |
                   ((uint32_t)s->backup[i+2]<<16) | ((uint32_t)s->backup[i+3]<<24);
        x^=v; weighted+=v*(i/128u+1u);
    }
    return x==s->checksum_xor && weighted==s->checksum_weighted;
#endif
}
static int qdvt_vectors_equal(volatile uint8_t *a, volatile uint8_t *b, uint32_t bytes) {
#ifdef __hexagon__
    volatile HVX_Vector *av=(volatile HVX_Vector *)a, *bv=(volatile HVX_Vector *)b;
    HVX_Vector diff=Q6_V_vzero();
    for (uint32_t i=0;i<bytes/128u;i+=8u) {
        HVX_Vector a0=av[i+0],a1=av[i+1],a2=av[i+2],a3=av[i+3];
        HVX_Vector a4=av[i+4],a5=av[i+5],a6=av[i+6],a7=av[i+7];
        HVX_Vector b0=bv[i+0],b1=bv[i+1],b2=bv[i+2],b3=bv[i+3];
        HVX_Vector b4=bv[i+4],b5=bv[i+5],b6=bv[i+6],b7=bv[i+7];
        diff=Q6_V_vor_VV(diff,Q6_V_vxor_VV(a0,b0));
        diff=Q6_V_vor_VV(diff,Q6_V_vxor_VV(a1,b1));
        diff=Q6_V_vor_VV(diff,Q6_V_vxor_VV(a2,b2));
        diff=Q6_V_vor_VV(diff,Q6_V_vxor_VV(a3,b3));
        diff=Q6_V_vor_VV(diff,Q6_V_vxor_VV(a4,b4));
        diff=Q6_V_vor_VV(diff,Q6_V_vxor_VV(a5,b5));
        diff=Q6_V_vor_VV(diff,Q6_V_vxor_VV(a6,b6));
        diff=Q6_V_vor_VV(diff,Q6_V_vxor_VV(a7,b7));
    }
    uint32_t lanes[32] __attribute__((aligned(128))); *(HVX_Vector *)lanes=diff;
    uint32_t any=0; for (uint32_t i=0;i<32;i++) any|=lanes[i];
    return any==0;
#else
    for (uint32_t i=0;i<bytes;i++) if (a[i]!=b[i]) return 0;
    return 1;
#endif
}
static int qdvt_mapping_valid(const QdVtcmLease *s) {
    return qdvt_read_tlb(QDVT_TLB8) == s->tlb8 &&
           qdvt_read_tlb(QDVT_TLB9) == s->tlb9 &&
           s->tlb8 == QDVT_TLB8_VALUE && s->tlb9 == QDVT_TLB9_VALUE;
}
static uint32_t qdvt_save(QdVtcmLease *s, volatile uint8_t *vtcm,
                          uint32_t base, uint32_t bytes, uint32_t backup_base,
                          uint8_t *backup) {
    if (!s || s->phase || (s->magic && s->magic != QDVT_MAGIC)) return QDVT_BAD_STATE;
    if (!qdvt_range_valid(base, bytes, backup_base, vtcm, backup)) return QDVT_BAD_RANGE;
    s->base=base; s->bytes=bytes; s->backup_base=backup_base; s->backup=backup;
    s->tlb8=qdvt_read_tlb(QDVT_TLB8); s->tlb9=qdvt_read_tlb(QDVT_TLB9);
    if (s->tlb8 != QDVT_TLB8_VALUE || s->tlb9 != QDVT_TLB9_VALUE) {
        s->error=QDVT_MAPPING_CHANGED; return QDVT_MAPPING_CHANGED;
    }
    s->magic=QDVT_MAGIC;
    qdvt_copy_fingerprint(vtcm,backup,bytes,&s->checksum_xor,&s->checksum_weighted);
    qdvt_sync();
    if (!qdvt_mapping_valid(s)) { s->error=QDVT_MAPPING_CHANGED; return QDVT_MAPPING_CHANGED; }
    s->phase=1; s->error=0; qdvt_sync();
    return QDVT_OK;
}
static uint32_t qdvt_restore_verify(QdVtcmLease *s, volatile uint8_t *vtcm) {
    if (!s || s->magic!=QDVT_MAGIC || s->phase!=1 || !vtcm ||
        !qdvt_range_valid(s->base,s->bytes,s->backup_base,vtcm,s->backup)) {
        if (s) s->error=QDVT_BAD_STATE;
        return QDVT_BAD_STATE;
    }
    if (!qdvt_mapping_valid(s)) { s->error=QDVT_MAPPING_CHANGED; return QDVT_MAPPING_CHANGED; }
    if (!qdvt_fingerprint_matches(s)) { s->error=QDVT_RESTORE_MISMATCH; return QDVT_RESTORE_MISMATCH; }
#ifdef __hexagon__
    volatile HVX_Vector *dv=(volatile HVX_Vector *)vtcm;
    volatile HVX_Vector *sv=(volatile HVX_Vector *)s->backup;
    for (uint32_t i=0;i<s->bytes/128u;i+=8u) {
        HVX_Vector v0=sv[i+0],v1=sv[i+1],v2=sv[i+2],v3=sv[i+3];
        HVX_Vector v4=sv[i+4],v5=sv[i+5],v6=sv[i+6],v7=sv[i+7];
        dv[i+0]=v0; dv[i+1]=v1; dv[i+2]=v2; dv[i+3]=v3;
        dv[i+4]=v4; dv[i+5]=v5; dv[i+6]=v6; dv[i+7]=v7;
    }
#else
    for (uint32_t i=0;i<s->bytes;i++) vtcm[i]=s->backup[i];
#endif
    qdvt_sync();
    if (!qdvt_vectors_equal(vtcm,s->backup,s->bytes)) {
        s->error=QDVT_RESTORE_MISMATCH; return QDVT_RESTORE_MISMATCH;
    }
    if (!qdvt_mapping_valid(s)) { s->error=QDVT_MAPPING_CHANGED; return QDVT_MAPPING_CHANGED; }
    s->phase=2; s->error=0; qdvt_sync();
    return QDVT_OK;
}
static uint32_t qdvt_release_verified(QdVtcmLease *s) {
    if (!s || s->magic!=QDVT_MAGIC || s->phase!=2 || s->error) return QDVT_BAD_STATE;
    /* Keep metadata until host confirms session close; the 8 MiB snapshot is
     * never erased here, including failure paths. */
    return QDVT_OK;
}

#endif
