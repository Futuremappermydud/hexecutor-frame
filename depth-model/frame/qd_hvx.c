/* Bare-metal Frame callback. No QNN, QuRT, libc or HMX dependency.
 * QD_HOST_TEST compiles the same dispatcher with a scalar vector substitute
 * for local protocol/arithmetic tests; it does not emulate HVX hardware. */
#include "qd_protocol.h"
#ifndef QD_HOST_TEST
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

static uint32_t in_range(uint32_t off, uint32_t size, uint32_t limit) {
    return off >= QD_DATA_OFFSET && off <= limit && size <= limit-off;
}
static uint32_t overlap(uint32_t a, uint32_t na, uint32_t b, uint32_t nb) {
    return a < b+nb && b < a+na;
}
static void barrier(void) {
#ifndef QD_HOST_TEST
    __asm__ volatile("barrier" ::: "memory");
#endif
}
static void invalidate(const void *p, uint32_t bytes) {
#ifndef QD_HOST_TEST
    uint32_t lo=(uint32_t)p & ~31u, end=(uint32_t)p+bytes;
    for (;lo<end;lo+=32) __asm__ volatile("dcinva(%0)" :: "r"(lo) : "memory");
#else
    (void)p; (void)bytes;
#endif
}
static void clean(const void *p) {
#ifndef QD_HOST_TEST
    for (uint32_t i=0;i<128;i+=32)
        __asm__ volatile("dccleana(%0)" :: "r"((uint32_t)p+i) : "memory");
#else
    (void)p;
#endif
}

static void run(uint8_t *base, volatile QdCommand *cmd, volatile QdStatus *status, uint32_t tid) {
    if(tid>=3) return;
    uint32_t seq=cmd->sequence, m=cmd->m, n=cmd->n, k=cmd->k;
    uint32_t ao=cmd->a_offset, bo=cmd->b_offset, co=cmd->c_offset;
    uint32_t error=QD_OK, rows=0;
    if(cmd->magic!=QD_MAGIC || cmd->version!=QD_VERSION || !seq || cmd->arena_bytes!=QD_ARENA_BYTES)
        error=QD_BAD_HEADER;
    else if(!m || !n || !k || m>QD_MAX_M || n>QD_MAX_N || k>QD_MAX_K || (n&31))
        error=QD_BAD_SHAPE;
    if(!error) {
        uint32_t chunks=(k+255)>>8;
        /* Products are bounded by the shape limits above and fit uint32. */
        uint32_t ab=m*k*2, bb=k*n*4, cb=chunks*m*n*4;
        if((ao&127) || (bo&127) || (co&127) ||
           !in_range(ao,ab,QD_ARENA_BYTES) || !in_range(bo,bb,QD_ARENA_BYTES) ||
           !in_range(co,cb,QD_ARENA_BYTES) || overlap(ao,ab,bo,bb) ||
           overlap(ao,ab,co,cb) || overlap(bo,bb,co,cb)) error=QD_BAD_RANGE;
    }
    if(!error) {
        const int16_t *a=(const int16_t *)(base+ao);
        const int32_t *b=(const int32_t *)(base+bo);
        int32_t *c=(int32_t *)(base+co);
        invalidate(a,m*k*2); invalidate(b,k*n*4);
        barrier();
        for(uint32_t row=tid;row<m;row+=3) {
            for(uint32_t chunk=0;chunk<(k+255)>>8;chunk++) {
                uint32_t end=(chunk+1)*256;
                if(end>k) end=k;
                for(uint32_t col=0;col<n;col+=32) {
#ifdef QD_HOST_TEST
                    int32_t acc[32]={0};
                    for(uint32_t j=chunk*256;j<end;j++)
                        for(uint32_t lane=0;lane<32;lane++)
                            acc[lane]+=(int32_t)a[row*k+j]*b[j*n+col+lane];
                    for(uint32_t lane=0;lane<32;lane++) c[(chunk*m+row)*n+col+lane]=acc[lane];
#else
                    HVX_Vector acc=Q6_V_vzero();
                    for(uint32_t j=chunk*256;j<end;j++) {
                        uint32_t av=(uint32_t)(uint16_t)a[row*k+j]*0x10001u;
                        HVX_Vector weights=*(const HVX_Vector *)(b+j*n+col);
                        acc=Q6_Vw_vmpyiacc_VwVwRh(acc,weights,av);
                    }
                    HVX_Vector *dst=(HVX_Vector *)(c+(chunk*m+row)*n+col);
                    *dst=acc;
#endif
                    clean(c+(chunk*m+row)*n+col);
                }
            }
            rows++;
        }
        barrier();
    }
    status[tid].error=error; status[tid].rows=rows; status[tid].reserved=0;
    barrier(); status[tid].done_sequence=seq; barrier();
}

#ifdef QD_HOST_TEST
void qd_host_run(void *arena, uint32_t tid) {
    run(arena,(volatile QdCommand *)arena,(volatile QdStatus *)((uint8_t *)arena+QD_STATUS_OFFSET),tid);
}
#else
__attribute__((section(".text.entry"))) void callback(void) {
    uint32_t tid;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    run((uint8_t *)(0x87000000u+QD_ARENA_OFFSET),
        (volatile QdCommand *)(0xf0000000u+QD_ARENA_OFFSET),
        (volatile QdStatus *)(0xf0000000u+QD_ARENA_OFFSET+QD_STATUS_OFFSET),tid);
}
#endif
