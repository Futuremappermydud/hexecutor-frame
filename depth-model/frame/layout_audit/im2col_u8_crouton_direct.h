#ifndef QD_LAYOUT_AUDIT_IM2COL_U8_CROUTON_DIRECT_H
#define QD_LAYOUT_AUDIT_IM2COL_U8_CROUTON_DIRECT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef uint8_t *(*qd_layout_block_access_fn)(uint32_t encoded, int store, void *ctx);
#define QD_LAYOUT_POOL_COUNT 7u

/*
 * Fused version of the exact hot QDRX im2col geometry:
 *   input: [1,10,258,1] u8, contiguous NHWC
 *   kernel/stride: 3x3x1 / 1x1
 *   logical output: [1,8,256,9] u8, crouton max [1,8,256,32]
 *
 * The function computes nine im2col bytes directly into each 32-byte u8
 * crouton fragment and zeros bytes 9..31, matching t_write's crouton depth
 * extension. It resolves an output block once per 8x8 tile using the tensor's
 * block-list slice offset. `pools` and `pool_bytes` are arrays of exactly
 * QD_LAYOUT_POOL_COUNT (7) entries; encoded pool IDs must be 1..6. They are
 * used only to preflight physical block bounds and source
 * aliasing; actual writes use `block_access` so QDG_DSP can preserve its block
 * cache semantics. Returns 1 if handled, 0 before any block writes otherwise.
 */
static inline int qd_layout_im2col_u8_crouton_direct(
        const uint8_t *src, size_t src_bytes,
        const int32_t in[4], const int32_t out[4], const int32_t out_max[4],
        const int32_t kernel[4], const int32_t stride[4],
        const uint32_t *blocks, uint32_t nblocks, uint32_t slice,
        uint8_t *const *pools, const size_t *pool_bytes,
        qd_layout_block_access_fn block_access, void *ctx) {
    static const int32_t want_in[4]={1,10,258,1};
    static const int32_t want_out[4]={1,8,256,9};
    static const int32_t want_max[4]={1,8,256,32};
    static const int32_t want_kernel[4]={3,3,1,32};
    static const int32_t want_stride[4]={1,1,1,1};
    const size_t block_bytes=2048;
    uintptr_t sb=(uintptr_t)src;
    uintptr_t physical[32];
    if (!src || !in || !out || !out_max || !kernel || !stride || !blocks
            || !pools || !pool_bytes || !block_access) return 0;
    if (src_bytes!=2580 || slice>nblocks || nblocks-slice<32) return 0;
    for (int i=0;i<4;i++)
        if (in[i]!=want_in[i] || out[i]!=want_out[i]
                || out_max[i]!=want_max[i] || kernel[i]!=want_kernel[i]
                || stride[i]!=want_stride[i]) return 0;
    if (src_bytes>UINTPTR_MAX-sb) return 0;
    for (uint32_t i=0;i<32;i++) {
        uint32_t e=blocks[slice+i], pool=e&127u, offset=e&~127u;
        if (!pool || pool>=QD_LAYOUT_POOL_COUNT || !pools[pool]
                || offset%block_bytes || offset>pool_bytes[pool]
                || block_bytes>pool_bytes[pool]-offset) return 0;
        uintptr_t pb=(uintptr_t)pools[pool];
        if (offset>UINTPTR_MAX-pb) return 0;
        uintptr_t db=pb+offset;
        if (block_bytes>UINTPTR_MAX-db) return 0;
        if (sb<db+block_bytes && db<sb+src_bytes) return 0;
        for (uint32_t j=0;j<i;j++)
            if (db<physical[j]+block_bytes && physical[j]<db+block_bytes) return 0;
        physical[i]=db;
    }
    /* Resolve all entries before writes so callback/pool failures reject the
     * fast path without leaving a partially produced output. */
    for (uint32_t i=0;i<32;i++)
        if (!block_access(blocks[slice+i],0,ctx)) return 0;

    /* h extent is exactly one crouton block, w extent is 32 blocks wide. */
    for (uint32_t wb=0;wb<32;wb++) {
        uint8_t *block=block_access(blocks[slice+wb],1,ctx);
        if (!block) return 0; /* callback contract: successful preflight is stable */
        for (uint32_t h=0;h<8;h++) for (uint32_t x=0;x<8;x++) {
            uint32_t w=wb*8+x;
            uint8_t *d=block+h*256+x*32;
            uint32_t c=0;
            for (uint32_t ky=0;ky<3;ky++)
                for (uint32_t kx=0;kx<3;kx++)
                    d[c++]=src[(h+ky)*258+w+kx];
            memset(d+9,0,23);
        }
    }
    return 1;
}

#endif
