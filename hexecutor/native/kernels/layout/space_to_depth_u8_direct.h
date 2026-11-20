/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_LAYOUT_AUDIT_SPACE_TO_DEPTH_U8_DIRECT_H
#define QD_LAYOUT_AUDIT_SPACE_TO_DEPTH_U8_DIRECT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define QD_LAYOUT_S2D_POOL_COUNT 7u
#define QD_LAYOUT_S2D_MAX_TOUCHED 128u
#define QD_LAYOUT_S2D_MAX_TILES 17u
#define QD_LAYOUT_S2D_MAX_TILE_SOURCES 9u
#define QD_LAYOUT_S2D_TILE_BYTES (16u*16u*32u)
#define QD_LAYOUT_S2D_PLAN_MAGIC 0x53324450u

typedef uint8_t *(*qd_layout_s2d_block_access_fn)(uint32_t encoded, int store, void *ctx);
typedef uint64_t (*qd_layout_s2d_clock_fn)(void *ctx);

typedef struct {
    uint32_t encoded;
    uint8_t tile_y, tile_x, source_y, source_x, rows, columns;
} QdLayoutS2dSourceSpan;

typedef struct {
    uint16_t output_w;
    uint8_t output_h, tile_h, tile_w, source_count;
    QdLayoutS2dSourceSpan source[QD_LAYOUT_S2D_MAX_TILE_SOURCES];
    uint32_t output[4];
} QdLayoutS2dTile;

typedef struct {
    uint32_t magic;
    uint16_t output_h, output_w;
    uint8_t tile_count;
    uintptr_t pool_base[QD_LAYOUT_S2D_POOL_COUNT];
    size_t pool_bytes[QD_LAYOUT_S2D_POOL_COUNT];
    QdLayoutS2dTile tile[QD_LAYOUT_S2D_MAX_TILES];
} QdLayoutS2dPlan;

typedef struct {
    uint64_t proof_ticks, prefetch_ticks, gather_ticks, scatter_ticks;
    uint32_t prefetch_block_calls, source_block_calls, output_block_calls;
    uint64_t source_bytes, output_bytes;
} QdLayoutS2dStats;

static inline int qd_layout_s2d_add_block(uint32_t encoded,
        uint8_t *const *pools, const size_t *pool_bytes,
        uint32_t *ids, uintptr_t *addresses, uint32_t *count,
        int reject_physical_duplicate) {
    const size_t block_bytes=2048;
    uint32_t pool=encoded&127u, offset=encoded&~127u;
    if (!pool || pool>=QD_LAYOUT_S2D_POOL_COUNT || !pools[pool]
            || offset%block_bytes || offset>pool_bytes[pool]
            || block_bytes>pool_bytes[pool]-offset) return 0;
    uintptr_t base=(uintptr_t)pools[pool];
    if (offset>UINTPTR_MAX-base) return 0;
    uintptr_t addr=base+offset;
    if (block_bytes>UINTPTR_MAX-addr) return 0;
    for (uint32_t j=0;j<*count;j++) {
        int overlap=addr<addresses[j]+block_bytes && addresses[j]<addr+block_bytes;
        if (overlap) {
            if (reject_physical_duplicate) return 0;
            if (addr==addresses[j] && encoded==ids[j]) return 2;
            if (addr==addresses[j]) return 0; /* cross-pool aliases need fallback */
            return 0; /* partial overlap is never a valid block map */
        }
    }
    if (*count>=QD_LAYOUT_S2D_MAX_TOUCHED) return 0;
    ids[*count]=encoded;addresses[*count]=addr;(*count)++;
    return 1;
}

/*
 * Prepare once for an immutable command/pool map. This strict u8 family has
 * input channels 32, output channels 128, input padding [0,7,7,0], output
 * padding zero, output height <=8, input max dimensions <=272 and output max
 * width <=136. `ow` deliberately follows the original kernel and can exceed
 * out_d[2] while remaining within out_m[2].
 *
 * The plan contains encoded block IDs and row spans, never block-cache
 * pointers. It proves all input/output block ranges once and rejects output
 * aliases and any input/output overlap. Repeated identical read-only input
 * block IDs are allowed. `pools` and `pool_bytes` each have seven entries.
 */
static inline int qd_layout_s2d_prepare_profiled(
        const int32_t in_m[4], const int32_t in_p[4],
        const int32_t out_d[4], const int32_t out_m[4], const int32_t out_p[4],
        const uint32_t *in_blocks, uint32_t in_nblocks, uint32_t in_slice,
        const uint32_t *out_blocks, uint32_t out_nblocks, uint32_t out_slice,
        uint8_t *const *pools, const size_t *pool_bytes,
        QdLayoutS2dPlan *plan, qd_layout_s2d_clock_fn clock, void *clock_ctx,
        QdLayoutS2dStats *stats) {
    uint32_t in_ids[QD_LAYOUT_S2D_MAX_TOUCHED], out_ids[QD_LAYOUT_S2D_MAX_TOUCHED];
    uintptr_t in_addr[QD_LAYOUT_S2D_MAX_TOUCHED], out_addr[QD_LAYOUT_S2D_MAX_TOUCHED];
    uint32_t in_count=0,out_count=0;
    uint64_t proof_start=clock?clock(clock_ctx):0;
    if (stats) memset(stats,0,sizeof *stats);
    if (!plan) return 0;
    memset(plan,0,sizeof *plan);
    if (!in_m || !in_p || !out_d || !out_m || !out_p || !in_blocks || !out_blocks
            || !pools || !pool_bytes) return 0;
    if (in_m[0]!=1 || in_m[1]<=0 || in_m[1]>272 || in_m[2]<=0 || in_m[2]>272 || in_m[3]!=32
            || in_p[0]!=0 || in_p[1]!=7 || in_p[2]!=7 || in_p[3]!=0
            || out_d[0]!=1 || out_d[1]<=0 || out_d[1]>8 || out_d[3]!=128
            || out_m[0]!=1 || out_m[1]!=8 || out_m[2]<=0 || out_m[2]>136 || out_m[3]!=128
            || out_p[0]!=0 || out_p[1]!=0 || out_p[2]!=0 || out_p[3]!=0) return 0;
    int32_t oh=out_d[1],ph=in_p[1],pw=in_p[2];
    if (out_d[2]<0 || out_d[2]>out_m[2] || out_d[3]>out_m[3]) return 0;
    int64_t ow64=((int64_t)in_m[2]-pw)/2;
    if (ow64>out_m[2]) ow64=out_m[2];
    if (ow64<=0 || ow64>INT32_MAX) return 0;
    int32_t ow=(int32_t)ow64;
    if ((int64_t)ph+2ll*(oh-1)+1>=in_m[1]
            || (int64_t)pw+2ll*(ow-1)>=in_m[2]
            || oh>out_m[1]) return 0;
    if (in_slice>in_nblocks || out_slice>out_nblocks) return 0;

    uint32_t in_nbw=((uint32_t)in_m[2]+7u)/8u;
    uint32_t out_nbw=((uint32_t)out_m[2]+7u)/8u;
    const uint32_t out_nbd=4;
    int32_t in_h0=ph,in_h1=ph+2*oh;
    int32_t in_w0=pw-1,in_w1=pw-1+2*ow;
    uint32_t in_bh0=(uint32_t)in_h0/8,in_bh1=(uint32_t)(in_h1-1)/8;
    uint32_t in_bw0=(uint32_t)in_w0/8,in_bw1=(uint32_t)(in_w1-1)/8;
    uint32_t out_bw1=(uint32_t)(ow-1)/8;
    if (out_bw1+1u>QD_LAYOUT_S2D_MAX_TILES) return 0;

    /* Cache the seven pool identities so apply can validate its proof cheaply. */
    for (uint32_t p=0;p<QD_LAYOUT_S2D_POOL_COUNT;p++) {
        plan->pool_base[p]=(uintptr_t)pools[p];
        plan->pool_bytes[p]=pool_bytes[p];
    }

    /* Validate and record the complete source read-block rectangle. */
    for (uint32_t bh=in_bh0;bh<=in_bh1;bh++)
        for (uint32_t bw=in_bw0;bw<=in_bw1;bw++) {
            uint64_t index=((uint64_t)bh*in_nbw+bw)+in_slice;
            if (index>=in_nblocks) return 0;
            int added=qd_layout_s2d_add_block(in_blocks[index],pools,pool_bytes,
                    in_ids,in_addr,&in_count,0);
            if (!added && added!=2) return 0;
        }

    /* Precompute one tile's source row spans and four encoded output blocks. */
    for (int32_t h0=0;h0<oh;h0+=8) {
        uint32_t th=(uint32_t)(oh-h0<8?oh-h0:8);
        for (int32_t w0=0;w0<ow;w0+=8) {
            if (plan->tile_count>=QD_LAYOUT_S2D_MAX_TILES) return 0;
            QdLayoutS2dTile *t=&plan->tile[plan->tile_count++];
            t->output_h=(uint8_t)h0;t->output_w=(uint16_t)w0;
            t->tile_h=(uint8_t)th;t->tile_w=(uint8_t)(ow-w0<8?ow-w0:8);
            int32_t sy0=ph+2*h0,sx0=pw-1+2*w0;
            int32_t sy1=sy0+(int32_t)(2*th),sx1=sx0+(int32_t)(2*t->tile_w);
            uint32_t sbh0=(uint32_t)sy0/8,sbh1=(uint32_t)(sy1-1)/8;
            uint32_t sbw0=(uint32_t)sx0/8,sbw1=(uint32_t)(sx1-1)/8;
            for (uint32_t bh=sbh0;bh<=sbh1;bh++)
                for (uint32_t bw=sbw0;bw<=sbw1;bw++) {
                    uint64_t index=((uint64_t)bh*in_nbw+bw)+in_slice;
                    if (index>=in_nblocks || t->source_count>=QD_LAYOUT_S2D_MAX_TILE_SOURCES) return 0;
                    uint32_t e=in_blocks[index];
                    int32_t ylo=sy0>(int32_t)(bh*8)?sy0:(int32_t)(bh*8);
                    int32_t yhi=sy1<(int32_t)(bh*8+8)?sy1:(int32_t)(bh*8+8);
                    int32_t xlo=sx0>(int32_t)(bw*8)?sx0:(int32_t)(bw*8);
                    int32_t xhi=sx1<(int32_t)(bw*8+8)?sx1:(int32_t)(bw*8+8);
                    QdLayoutS2dSourceSpan *s=&t->source[t->source_count++];
                    s->encoded=e;s->tile_y=(uint8_t)(ylo-sy0);s->tile_x=(uint8_t)(xlo-sx0);
                    s->source_y=(uint8_t)(ylo%8);s->source_x=(uint8_t)(xlo%8);
                    s->rows=(uint8_t)(yhi-ylo);s->columns=(uint8_t)(xhi-xlo);
                }
            for (uint32_t dy=0;dy<2;dy++) for (uint32_t dx=0;dx<2;dx++) {
                uint32_t j=2*dy+dx;
                uint64_t index=((uint64_t)(h0/8)*out_nbw+(uint32_t)(w0/8))*out_nbd
                              +out_slice+j;
                if (index>=out_nblocks) return 0;
                t->output[j]=out_blocks[index];
                if (!qd_layout_s2d_add_block(t->output[j],pools,pool_bytes,
                        out_ids,out_addr,&out_count,1)) return 0;
            }
        }
    }
    for (uint32_t a=0;a<in_count;a++) for (uint32_t b=0;b<out_count;b++)
        if (in_addr[a]<out_addr[b]+2048 && out_addr[b]<in_addr[a]+2048) return 0;

    plan->output_h=(uint16_t)oh;plan->output_w=(uint16_t)ow;
    plan->magic=QD_LAYOUT_S2D_PLAN_MAGIC;
    if (stats && clock) stats->proof_ticks=clock(clock_ctx)-proof_start;
    return 1;
}

static inline int qd_layout_s2d_prepare(
        const int32_t in_m[4], const int32_t in_p[4],
        const int32_t out_d[4], const int32_t out_m[4], const int32_t out_p[4],
        const uint32_t *in_blocks, uint32_t in_nblocks, uint32_t in_slice,
        const uint32_t *out_blocks, uint32_t out_nblocks, uint32_t out_slice,
        uint8_t *const *pools, const size_t *pool_bytes, QdLayoutS2dPlan *plan) {
    return qd_layout_s2d_prepare_profiled(in_m,in_p,out_d,out_m,out_p,
            in_blocks,in_nblocks,in_slice,out_blocks,out_nblocks,out_slice,
            pools,pool_bytes,plan,NULL,NULL,NULL);
}

/* Apply a prepared immutable plan. The callback is the DSP block-cache
 * accessor: it is known to return a pointer for every prevalidated ID. Never
 * retain any returned pointer across another callback, because block-cache
 * slots are direct-mapped. */
static inline int qd_layout_s2d_apply_profiled(const QdLayoutS2dPlan *plan,
        uint8_t *const *pools, const size_t *pool_bytes,
        uint8_t *tile, size_t tile_bytes,
        qd_layout_s2d_block_access_fn block_access, void *ctx,
        qd_layout_s2d_clock_fn clock, void *clock_ctx, QdLayoutS2dStats *stats) {
    if (stats) memset(stats,0,sizeof *stats);
    if (!plan || plan->magic!=QD_LAYOUT_S2D_PLAN_MAGIC || !pools || !pool_bytes
            || !tile || tile_bytes<QD_LAYOUT_S2D_TILE_BYTES || !block_access) return 0;
    for (uint32_t p=0;p<QD_LAYOUT_S2D_POOL_COUNT;p++)
        if ((uintptr_t)pools[p]!=plan->pool_base[p] || pool_bytes[p]!=plan->pool_bytes[p]) return 0;

    for (uint32_t ti=0;ti<plan->tile_count;ti++) {
        const QdLayoutS2dTile *t=&plan->tile[ti];
        uint64_t start=clock?clock(clock_ctx):0;
        for (uint32_t si=0;si<t->source_count;si++) {
            const QdLayoutS2dSourceSpan *s=&t->source[si];
            uint8_t *src=block_access(s->encoded,0,ctx);
            if (!src) return 0; /* prevalidated DSP accessor contract */
            if (stats) stats->source_block_calls++;
            for (uint32_t row=0;row<s->rows;row++)
                memcpy(tile+((size_t)(s->tile_y+row)*16u+s->tile_x)*32u,
                       src+(size_t)(s->source_y+row)*256u+(size_t)s->source_x*32u,
                       (size_t)s->columns*32u);
            if (stats) stats->source_bytes+=(uint64_t)s->rows*s->columns*32u;
        }
        if (stats && clock) stats->gather_ticks+=clock(clock_ctx)-start;

        start=clock?clock(clock_ctx):0;
        for (uint32_t j=0;j<4;j++) {
            uint8_t *dst=block_access(t->output[j],1,ctx);
            if (!dst) return 0; /* prevalidated DSP accessor contract */
            if (stats) stats->output_block_calls++;
            for (uint32_t h=0;h<t->tile_h;h++) for (uint32_t w=0;w<t->tile_w;w++) {
                uint32_t dy=j/2,dx=j%2;
                const uint8_t *src=tile+((size_t)(2*h+dy)*16u+(2*w+1u-dx))*32u;
                memcpy(dst+(size_t)h*256u+(size_t)w*32u,src,32);
            }
            if (stats) stats->output_bytes+=(uint64_t)t->tile_h*t->tile_w*32u;
        }
        if (stats && clock) stats->scatter_ticks+=clock(clock_ctx)-start;
    }
    return 1;
}

static inline int qd_layout_s2d_apply(const QdLayoutS2dPlan *plan,
        uint8_t *const *pools, const size_t *pool_bytes,
        uint8_t *tile, size_t tile_bytes,
        qd_layout_s2d_block_access_fn block_access, void *ctx) {
    return qd_layout_s2d_apply_profiled(plan,pools,pool_bytes,tile,tile_bytes,
            block_access,ctx,NULL,NULL,NULL);
}

/* Compatibility wrapper. Call prepare once per command in a production path
 * and retain its plan; this one-shot form deliberately recomputes proof. */
static inline int qd_layout_space_to_depth_u8_direct(
        const int32_t in_m[4], const int32_t in_p[4],
        const int32_t out_d[4], const int32_t out_m[4], const int32_t out_p[4],
        const uint32_t *in_blocks, uint32_t in_nblocks, uint32_t in_slice,
        const uint32_t *out_blocks, uint32_t out_nblocks, uint32_t out_slice,
        uint8_t *const *pools, const size_t *pool_bytes,
        uint8_t *tile, size_t tile_bytes,
        qd_layout_s2d_block_access_fn block_access, void *ctx) {
    QdLayoutS2dPlan plan;
    return qd_layout_s2d_prepare(in_m,in_p,out_d,out_m,out_p,
            in_blocks,in_nblocks,in_slice,out_blocks,out_nblocks,out_slice,
            pools,pool_bytes,&plan)
        && qd_layout_s2d_apply(&plan,pools,pool_bytes,tile,tile_bytes,block_access,ctx);
}

#endif
