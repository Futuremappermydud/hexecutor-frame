/* Exact fast paths for the data-movement reference kernels (SpaceToDepth,
 * ResizeNear, Transpose, Slice, SlicePad, Gather). Included by
 * prepared_reference.c after its blocked-layout helpers; not a standalone unit.
 *
 * Each fast path replaces "read the region into a dense int64-agnostic temp,
 * permute, write the region" by direct block-to-block (or row-to-row) moves on
 * the native bytes. The choice depends only on tensor geometry and immutable
 * pool-5 constants, so the dry (validate/measure) and wet passes always agree,
 * and a path that is taken needs no call scratch. It runs the same xfer_check
 * range/pool checks the generic path would, so acceptance is unchanged; any
 * shape or alignment it does not cover, and any source/destination block
 * overlap (the generic path reads everything before it writes), keeps the
 * generic code. Vector data movement uses DV (HVX on Hexagon, a byte array
 * elsewhere, so the host lockstep test runs the identical index logic). */
#ifndef DM_ENABLED
#define DM_ENABLED 1
#endif
#ifdef DM_STATS
#include <stdio.h>
/* Host diagnostics only: output elements that took each fast path. */
static unsigned long long dm_stat[16];
__attribute__((destructor)) static void dm_stat_print(void) {
    for(int i=0;i<16;i++)if(dm_stat[i])printf("DMSTAT %d %llu\n",i,dm_stat[i]);
}
#define DM_COUNT(id,n) (dm_stat[id]+=(unsigned long long)(n))
#else
#define DM_COUNT(id,n) ((void)0)
#endif

#if BLK_HVX
typedef HVX_Vector DV;
static inline DV dv_ld(const uint8_t *p) {return *(const HVX_Vector *)p;}
static inline void dv_st(uint8_t *p, DV v) {*(HVX_Vector *)p=v;}
static inline DV dv_zero(void) {return Q6_V_vzero();}
/* First n (1..127) bytes of the aligned vector at p. */
static inline void dv_stn(uint8_t *p, DV v, uint32_t n) {Q6_vmem_QRIV(Q6_Q_vsetq_R((int32_t)n),(HVX_Vector *)p,v);}
/* Bytes n..n+127 of lo:hi. */
static inline DV dv_align(DV hi, DV lo, uint32_t n) {return Q6_V_valign_VVR(hi,lo,(int32_t)n);}
/* Pixels (32-byte units) of the 8-pixel stream a:b: even ones, odd ones. */
static inline void dv_deal32(DV a, DV b, DV *even, DV *odd) {
    HVX_VectorPair p=Q6_W_vdeal_VVR(b,a,-32);
    *even=Q6_V_lo_W(p);*odd=Q6_V_hi_W(p);
}
/* Each of the 4 pixels twice: [0,0,1,1] then [2,2,3,3]. */
static inline void dv_dup32(DV v, DV *first, DV *second) {
    HVX_VectorPair p=Q6_W_vshuff_VVR(v,v,-32);
    *first=Q6_V_lo_W(p);*second=Q6_V_hi_W(p);
}
/* Word i = (u.half[2i] << 16) | v.half[2i], and the odd halves. */
static inline DV dv_shuffe16(DV u, DV v) {return Q6_Vh_vshuffe_VhVh(u,v);}
static inline DV dv_shuffo16(DV u, DV v) {return Q6_Vh_vshuffo_VhVh(u,v);}
/* Elements of 1<<k bytes: lo:hi interleaves v and u (v0,u0,v1,u1,...). */
static inline void dv_shuffk(DV u, DV v, uint32_t k, DV *lo, DV *hi) {
    HVX_VectorPair p=Q6_W_vshuff_VVR(u,v,-(int32_t)(1u<<k));
    *lo=Q6_V_lo_W(p);*hi=Q6_V_hi_W(p);
}
/* Inverse: lo and hi are the even and odd elements of v:u. */
static inline void dv_dealk(DV u, DV v, uint32_t k, DV *lo, DV *hi) {
    HVX_VectorPair p=Q6_W_vdeal_VVR(u,v,-(int32_t)(1u<<k));
    *lo=Q6_V_lo_W(p);*hi=Q6_V_hi_W(p);
}
/* 128 bytes from / the first n bytes to an arbitrary address. */
static inline DV dv_ldu(const uint8_t *p) {return load_part(p,128);}
static inline void dv_stun(uint8_t *p, DV v, uint32_t n) {store_part(p,v,0,n);}
/* The first n bytes (1..128) from any address; the rest of the vector is unspecified. */
static inline DV dv_ldn(const uint8_t *p, uint32_t n) {return load_part(p,n);}
/* v[src..src+n) to p. */
static inline void dv_stup(uint8_t *p, DV v, uint32_t src, uint32_t n) {store_part(p,v,src,n);}
static inline DV dv_or(DV a, DV b) {return Q6_V_vor_VV(a,b);}
/* Bytes [0,n) of v, the rest zero (0<n<=128). */
static inline DV dv_keep(DV v, uint32_t n) {return Q6_V_vand_QV(Q6_Q_vsetq_R((int32_t)n),v);}
/* v rotated right (towards byte 0) by n bytes. */
static inline DV dv_ror(DV v, uint32_t n) {return Q6_V_vror_VR(v,(int32_t)(n&127u));}
/* Halfword plane split: [a0..a31,b0..b31] of [a0,b0,a1,b1,...] and back. */
static inline DV dv_dealh(DV v) {return Q6_Vh_vdeal_Vh(v);}
static inline DV dv_shuffh(DV v) {return Q6_Vh_vshuff_Vh(v);}
#else
typedef struct { uint8_t b[128]; } DV;
static inline DV dv_ld(const uint8_t *p) {DV v;memcpy(v.b,p,128);return v;}
static inline void dv_st(uint8_t *p, DV v) {memcpy(p,v.b,128);}
static inline DV dv_zero(void) {DV v;memset(v.b,0,128);return v;}
static inline void dv_stn(uint8_t *p, DV v, uint32_t n) {memcpy(p,v.b,n);}
static inline DV dv_align(DV hi, DV lo, uint32_t n) {
    DV r;
    for(uint32_t i=0;i<128;i++)r.b[i]=i+n<128?lo.b[i+n]:hi.b[i+n-128];
    return r;
}
static inline void dv_deal32(DV a, DV b, DV *even, DV *odd) {
    for(uint32_t p=0;p<4;p++) {
        memcpy(even->b+32*p,p<2?a.b+64*p:b.b+64*(p-2),32);
        memcpy(odd->b+32*p,(p<2?a.b+64*p:b.b+64*(p-2))+32,32);
    }
}
static inline void dv_dup32(DV v, DV *first, DV *second) {
    for(uint32_t p=0;p<4;p++) {
        memcpy(first->b+32*p,v.b+32*(p/2),32);
        memcpy(second->b+32*p,v.b+32*(2+p/2),32);
    }
}
static inline DV dv_shuffe16(DV u, DV v) {
    DV r;
    for(uint32_t i=0;i<32;i++){memcpy(r.b+4*i,v.b+4*i,2);memcpy(r.b+4*i+2,u.b+4*i,2);}
    return r;
}
static inline DV dv_shuffo16(DV u, DV v) {
    DV r;
    for(uint32_t i=0;i<32;i++){memcpy(r.b+4*i,v.b+4*i+2,2);memcpy(r.b+4*i+2,u.b+4*i+2,2);}
    return r;
}
static inline void dv_shuffk(DV u, DV v, uint32_t k, DV *lo, DV *hi) {
    const uint32_t e=1u<<k,n=128/e;
    uint8_t t[256];
    for(uint32_t i=0;i<n;i++){memcpy(t+2*i*e,v.b+i*e,e);memcpy(t+(2*i+1)*e,u.b+i*e,e);}
    memcpy(lo->b,t,128);memcpy(hi->b,t+128,128);
}
static inline void dv_dealk(DV u, DV v, uint32_t k, DV *lo, DV *hi) {
    const uint32_t e=1u<<k,n=128/e;
    uint8_t t[256];
    memcpy(t,v.b,128);memcpy(t+128,u.b,128);
    for(uint32_t i=0;i<n;i++){memcpy(lo->b+i*e,t+2*i*e,e);memcpy(hi->b+i*e,t+(2*i+1)*e,e);}
}
static inline DV dv_ldu(const uint8_t *p) {DV v;memcpy(v.b,p,128);return v;}
static inline void dv_stun(uint8_t *p, DV v, uint32_t n) {memcpy(p,v.b,n);}
static inline DV dv_ldn(const uint8_t *p, uint32_t n) {DV v;memset(v.b,0,128);memcpy(v.b,p,n);return v;}
static inline void dv_stup(uint8_t *p, DV v, uint32_t src, uint32_t n) {memcpy(p,v.b+src,n);}
static inline DV dv_or(DV a, DV b) {DV r;for(uint32_t i=0;i<128;i++)r.b[i]=a.b[i]|b.b[i];return r;}
static inline DV dv_keep(DV v, uint32_t n) {DV r;memset(r.b,0,128);memcpy(r.b,v.b,n);return r;}
static inline DV dv_ror(DV v, uint32_t n) {DV r;for(uint32_t i=0;i<128;i++)r.b[i]=v.b[(i+n)&127u];return r;}
static inline DV dv_dealh(DV v) {
    DV r;
    for(uint32_t i=0;i<32;i++){memcpy(r.b+2*i,v.b+4*i,2);memcpy(r.b+64+2*i,v.b+4*i+2,2);}
    return r;
}
static inline DV dv_shuffh(DV v) {
    DV r;
    for(uint32_t i=0;i<32;i++){memcpy(r.b+4*i,v.b+2*i,2);memcpy(r.b+4*i+2,v.b+64+2*i,2);}
    return r;
}
#endif

/* Register-array bit permutations. r[0..n) are vectors; a stage (kind, j, k)
 * pairs r[i], r[i | 1<<j] and replaces them with the interleave (kind 0) or
 * de-interleave (kind 1) of their 1<<k byte elements, which exchanges address
 * bit j of the register index with the top (kind 0) or the k-th (kind 1) bit of
 * the byte offset. The stage lists below were found by a bit-permutation search
 * and are proved by the sim test against the generic kernels. */
static inline __attribute__((always_inline)) void dv_ladder(DV *r, uint32_t n, const uint8_t (*ops)[3], uint32_t nops) {
    _Pragma("clang loop unroll(full)")
    for(uint32_t s=0;s<nops;s++) {
        const uint32_t kind=ops[s][0],bit=1u<<ops[s][1],k=ops[s][2];
        _Pragma("clang loop unroll(full)")
        for(uint32_t i=0;i<n;i++) {
            if(i&bit) continue;
            DV lo,hi;
            if(kind)dv_dealk(r[i|bit],r[i],k,&lo,&hi);else dv_shuffk(r[i|bit],r[i],k,&lo,&hi);
            r[i]=lo;r[i|bit]=hi;
        }
    }
}
/* u16 blocked 32 pixel x 32 channel tile, W<->C (16 pixel-pair vectors in, 16 channel-pair vectors out). */
static const uint8_t dm_lad_wc16[6][3]={{0,0,2},{0,2,3},{0,1,3},{1,0,5},{0,3,5},{0,0,1}};
/* u8 blocked 32 x 32 tile, W<->C (8 four-pixel vectors in and out). */
static const uint8_t dm_lad_wc8[5][3]={{0,0,0},{0,0,0},{0,2,2},{0,1,2},{0,0,0}};
/* 16 x 16 byte transpose: out[m][r+16u] = in[r][8m+u]. */
static const uint8_t dm_lad_t16[4][3]={{0,3,0},{0,2,0},{0,1,0},{0,0,0}};
/* im2col 4x4x2 u16: 4 rows x 8 groups (4 pixels) -> 4 pixel-pair vectors of 4 row segments. */
static const uint8_t dm_lad_i25[4][3]={{0,0,4},{1,0,5},{0,1,5},{0,0,1}};
/* 4 x 4 transpose of 32-byte units. */
static const uint8_t dm_lad_u32[2][3]={{0,1,5},{0,0,5}};
/* ------------------------------------------------------------- threads */
/* The HVX fast paths below split their outermost (flattened) loop over
 * g->call->parallel. Each iteration stores whole, distinct 128-byte vectors of
 * the destination and only reads the source (the blocks are disjoint), so
 * parts are independent. Small jobs stay on the caller. */
#ifndef PAR_DM_MIN_ELEMS
#define PAR_DM_MIN_ELEMS 8192u /* Frame sweep: flat below 8K (t4c) */
#endif
static uint32_t dm_parts(const G *g, uint64_t elems, int64_t items) {
    if(elems<PAR_DM_MIN_ELEMS || items<2) return 1;
    return nprg2_parallel_parts(g->call->parallel,items<8?(uint32_t)items:8u);
}
static void dm_range(int64_t items, uint32_t part, uint32_t parts, int64_t *a, int64_t *b) {
    *a=items*(int64_t)part/(int64_t)parts;*b=items*((int64_t)part+1)/(int64_t)parts;
}

/* ------------------------------------------------------- blocked addressing */
/* Start of row H of block (B, H/8, col, z): the block's 8-row x cw-column x
 * 32-channel tile has 256 bytes per row for every element size. NULL when the
 * block is absent (reads as zero, stores are skipped), as in blocked_impl. */
typedef struct {
    uint8_t *const *pool;
    const uint32_t *blocks;
    int64_t count, nbh, nbw, nbd, slice;
} BK;
static void bk_init(BK *k, const G *g, const T *t) {
    const int64_t cw=t->esize==1?8:t->esize==2?4:2;
    k->pool=g->pool;k->blocks=t->blocks;k->count=t->nblocks;
    k->nbh=((int64_t)t->m[1]+7)/8;k->nbw=((int64_t)t->m[2]+cw-1)/cw;k->nbd=((int64_t)t->m[3]+31)/32;
    k->slice=t->slice;
}
static inline uint8_t *bk_at(const BK *k, int64_t B, int64_t H, int64_t col, int64_t z) {
    int64_t index=((B*k->nbh+H/8)*k->nbw+col)*k->nbd+k->slice+z;
    if(index<0 || index>=k->count) return NULL;
    uint32_t e=k->blocks[index],pool=e&127u;
    return pool?k->pool[pool]+(e&~127u)+(size_t)(H&7)*256:NULL;
}
static inline DV dv_ldz(const uint8_t *p) {return p?dv_ld(p):dv_zero();}
/* Start of block (B, block row Hb, block column col, depth group z); NULL when absent. */
static inline uint8_t *bk_blk(const BK *k, int64_t B, int64_t Hb, int64_t col, int64_t z) {
    int64_t index=((B*k->nbh+Hb)*k->nbw+col)*k->nbd+k->slice+z;
    if(index<0 || index>=k->count) return NULL;
    uint32_t e=k->blocks[index],pool=e&127u;
    return pool?k->pool[pool]+(e&~127u):NULL;
}

static const uint8_t dm_zblk[2048] __attribute__((aligned(128)));
/* Block start by index, absent blocks (NULL) replaced by `alt`. */
static inline uint8_t *bk_blk_or(const BK *k, int64_t index, uint8_t *alt) {
    if(index<0 || index>=k->count) return alt;
    uint32_t e=k->blocks[index],pool=e&127u;
    return pool?k->pool[pool]+(e&~127u):alt;
}
/* Vector accesses assume every pool base is 128-byte aligned (block offsets and
 * the in-block offsets used are). */
static int dm_pools_aligned(const G *g) {
    for(int i=0;i<7;i++)if(g->pool[i] && ((uintptr_t)g->pool[i]&127u)) return 0;
    return 1;
}
/* The blocks a blocked region touches, as raw table entries (present only). */
enum { DM_TOUCH_CAP=384 };
static int dm_touch(const T *t, const int32_t *dims, const int32_t *origin, uint32_t *keys, uint32_t *n) {
    *n=0;
    if(!dims[0] || !dims[1] || !dims[2] || !dims[3]) return 1;
    int64_t cw=t->esize==1?8:t->esize==2?4:2;
    int64_t nbh=((int64_t)t->m[1]+7)/8,nbw=((int64_t)t->m[2]+cw-1)/cw,nbd=((int64_t)t->m[3]+31)/32;
    for(int64_t b=origin[0];b<(int64_t)origin[0]+dims[0];b++)
     for(int64_t h=origin[1]/8;h<=((int64_t)origin[1]+dims[1]-1)/8;h++)
      for(int64_t w=origin[2]/cw;w<=((int64_t)origin[2]+dims[2]-1)/cw;w++)
       for(int64_t z=origin[3]/32;z<=((int64_t)origin[3]+dims[3]-1)/32;z++) {
        int64_t index=((b*nbh+h)*nbw+w)*nbd+t->slice+z;
        if(index<0 || index>=(int64_t)t->nblocks) continue;
        uint32_t e=t->blocks[index];
        if(!(e&127u)) continue;
        if(*n==DM_TOUCH_CAP) return 0;
        keys[(*n)++]=e;
       }
    return 1;
}
/* 1 when no block of a shares bytes with a block of b (blocks are 2048 bytes). */
static int dm_disjoint(const uint32_t *a, uint32_t na, const uint32_t *b, uint32_t nb) {
    uint32_t lo[2]={UINT32_MAX,UINT32_MAX},hi[2]={0,0};
    for(uint32_t i=0;i<na;i++){uint32_t o=a[i]&~127u;if(o<lo[0])lo[0]=o;if(o>hi[0])hi[0]=o;}
    for(uint32_t i=0;i<nb;i++){uint32_t o=b[i]&~127u;if(o<lo[1])lo[1]=o;if(o>hi[1])hi[1]=o;}
    if(!na || !nb || hi[0]+BLOCK_BYTES<=lo[1] || hi[1]+BLOCK_BYTES<=lo[0]) return 1;
    for(uint32_t i=0;i<na;i++)for(uint32_t j=0;j<nb;j++) {
        if((a[i]&127u)!=(b[j]&127u)) continue;
        uint32_t x=a[i]&~127u,y=b[j]&~127u;
        if(x<y+BLOCK_BYTES && y<x+BLOCK_BYTES) return 0;
    }
    return 1;
}
/* Present blocks of a blocked region, as raw table entries. */
typedef void (*DMVisit)(void *ctx, uint32_t entry);
static inline __attribute__((always_inline)) void dm_visit(const T *t, const int32_t *dims, const int32_t *origin, DMVisit f, void *ctx) {
    if(!dims[0] || !dims[1] || !dims[2] || !dims[3]) return;
    const int cws=t->esize==1?3:t->esize==2?2:1;
    const int64_t cw=(int64_t)1<<cws;
    const int64_t nbh=((int64_t)t->m[1]+7)>>3,nbw=((int64_t)t->m[2]+cw-1)>>cws,nbd=((int64_t)t->m[3]+31)>>5;
    if((origin[0]|origin[1]|origin[2]|origin[3])<0 || dims[0]<0 || dims[1]<0 || dims[2]<0 || dims[3]<0) {
        /* sign-sensitive division: general loop */
        for(int64_t b=origin[0];b<(int64_t)origin[0]+dims[0];b++)
         for(int64_t h=origin[1]/8;h<=((int64_t)origin[1]+dims[1]-1)/8;h++)
          for(int64_t w=origin[2]/cw;w<=((int64_t)origin[2]+dims[2]-1)/cw;w++)
           for(int64_t z=origin[3]/32;z<=((int64_t)origin[3]+dims[3]-1)/32;z++) {
            int64_t index=((b*nbh+h)*nbw+w)*nbd+t->slice+z;
            if(index<0 || index>=(int64_t)t->nblocks) continue;
            uint32_t e=t->blocks[index];
            if(e&127u) f(ctx,e);
           }
        return;
    }
    const int64_t b1=(int64_t)origin[0]+dims[0],h0=origin[1]>>3,h1=((int64_t)origin[1]+dims[1]-1)>>3;
    const int64_t w0=origin[2]>>cws,w1=((int64_t)origin[2]+dims[2]-1)>>cws,z0=origin[3]>>5,z1=((int64_t)origin[3]+dims[3]-1)>>5;
    const int64_t cnt=(int64_t)t->nblocks;
    for(int64_t b=origin[0];b<b1;b++)
     for(int64_t h=h0;h<=h1;h++)
      for(int64_t w=w0;w<=w1;w++) {
       const int64_t base=((b*nbh+h)*nbw+w)*nbd+t->slice;
       for(int64_t z=z0;z<=z1;z++) {
        const int64_t index=base+z;
        if((uint64_t)index>=(uint64_t)cnt) continue;
        uint32_t e=t->blocks[index];
        if(e&127u) f(ctx,e);
       }
      }
}
/* Per-pool byte extent [lo, hi] (block start offsets). */
typedef struct { uint32_t lo[8], hi[8]; } DMExt;
static void dm_v_ext(void *c, uint32_t e) {
    DMExt *x=c;
    uint32_t pool=e&127u,o=e&~127u;
    if(pool>=8) return;
    if(o<x->lo[pool])x->lo[pool]=o;
    if(o>x->hi[pool])x->hi[pool]=o;
}
static void dm_extent(const T *t, const int32_t *dims, const int32_t *origin, DMExt *x) {
    for(int i=0;i<8;i++){x->lo[i]=UINT32_MAX;x->hi[i]=0;}
    dm_visit(t,dims,origin,dm_v_ext,x);
}
/* Exact test for 2048-aligned blocks of one pool inside [base, base + slots * 2048): one bit per slot. */
enum { DM_SLOTS=8192 };
typedef struct { uint32_t pool, base, mark, hit, odd; uint32_t bits[DM_SLOTS/32]; } DMBits;
static void dm_v_bits(void *c, uint32_t e) {
    DMBits *k=c;
    if((e&127u)!=k->pool) return;
    uint32_t o=e&~127u;
    if(o&(BLOCK_BYTES-1)){k->odd=1;return;}
    if(o<k->base || (o-k->base)/BLOCK_BYTES>=DM_SLOTS) return;
    uint32_t i=(o-k->base)/BLOCK_BYTES;
    if(k->mark)k->bits[i>>5]|=1u<<(i&31);
    else if(k->bits[i>>5]&(1u<<(i&31)))k->hit=1;
}
/* 1 when no block of region a shares a byte with a block of region b: disjoint
 * per-pool extents, else an exact slot-bitmap test of the overlapping pools
 * (blocks on a 2048 grid), else the pairwise test when both fit dm_touch. */
static int dm_clear_raw(const T *a, const int32_t *da, const int32_t *oa, const T *b, const int32_t *db, const int32_t *ob) {
    DMExt x,y;
    dm_extent(a,da,oa,&x);dm_extent(b,db,ob,&y);
    int exact=0;
    for(uint32_t p=1;p<8;p++) {
        if(!(x.lo[p]<=x.hi[p] && y.lo[p]<=y.hi[p] && x.hi[p]+BLOCK_BYTES>y.lo[p] && y.hi[p]+BLOCK_BYTES>x.lo[p])) continue;
        uint32_t lo=x.lo[p]>y.lo[p]?x.lo[p]:y.lo[p],hi=x.hi[p]<y.hi[p]?x.hi[p]:y.hi[p];
        if(lo>hi) continue;   /* extents touch without a shared start: aligned blocks differ */
        DMBits k;
        memset(&k,0,sizeof k);
        k.pool=p;k.base=lo;k.mark=1;
        dm_visit(a,da,oa,dm_v_bits,&k);
        k.mark=0;
        dm_visit(b,db,ob,dm_v_bits,&k);
        if(k.hit && !k.odd) return 0;
        if(k.odd || (hi-lo)/BLOCK_BYTES>=DM_SLOTS) {exact=1;break;}
    }
    if(!exact) return 1;
    uint32_t ka[DM_TOUCH_CAP],kb[DM_TOUCH_CAP],na,nb;
    return dm_touch(a,da,oa,ka,&na) && dm_touch(b,db,ob,kb,&nb) && dm_disjoint(ka,na,kb,nb);
}
/* dm_clear for the kernel's G: the passing verdict is memoised per command
 * (see chk_hit). Callers have `g` in scope; this keeps their call sites as is. */
static int dm_clear_m(G *g, const T *a, const int32_t *da, const int32_t *oa, const T *b, const int32_t *db, const int32_t *ob) {
    uint32_t *s=chk_slots(g);
    uint32_t tag=0;
    if(s) {
        tag=chk_r(chk_r(chk_t(0x2545f491u,a),da),oa);
        tag=chk_r(chk_r(chk_t(tag,b),db),ob)|1u;
        if(chk_hit(s,tag)) return 1;
    }
    int ok=dm_clear_raw(a,da,oa,b,db,ob);
    if(ok) chk_put(s,tag);
    return ok;
}
#define dm_clear(a,da,oa,b,db,ob) dm_clear_m(g,(a),(da),(oa),(b),(db),(ob))
/* dims the generic t_write uses for a same-shape value of d, or 0 when the
 * value would be padded or broadcast (n != dims). */
static int dm_write_dims_match(const T *t, const int32_t *n) {
    int32_t dims[4];
    if(!(t->flags&F_SHAPE)) return 0;
    memcpy(dims,t->d,sizeof dims);
    if(t->crouton) {
        int32_t e=(t->d[3]+t->p[3]+31)/32*32;
        if(e>t->m[3])e=t->m[3];
        dims[3]=e-t->p[3];
    }
    return !memcmp(dims,n,sizeof dims);
}
static int dm_blocked_u8(const T *t) {
    return (t->flags&F_BLOCKED) && (t->flags&F_SHAPE) && t->esize==1 && t->dtype!=DT_F32;
}

/* ------------------------------------------------------------ SpaceToDepth */
/* u8 blocked in and out, ci a multiple of 32, oc = 4*ci (every output group is
 * written), pw = 7 mod 8, ow a multiple of 4. Output pixel (h,w), group
 * j = cb*4 + 2*dy + dx is input pixel (ph + 2h + dy, pw + 2w - dx), 32 bytes.
 * Seen from one input row, the stream of pixels pw-1.. is t = W - (pw-1);
 * odd t feed dx = 0 and even t feed dx = 1, so each 4 output pixels (128
 * bytes) per group are the even or odd pixels of an 8-pixel window that spans
 * the 3 input vectors (k,1), (k+1,0), (k+1,1) of a row. A rolling window loads
 * each input vector once and stores whole output vectors. */
static int s2d_fast_ok(const G *g, const T *i, const T *o, int32_t ph, int32_t pw, int32_t oh, int32_t ow, int32_t ci, int32_t oc) {
    return DM_ENABLED && dm_blocked_u8(i) && dm_blocked_u8(o) && ci>0 && !(ci&31) && oc==4*ci && ow>0 && !(ow&3) &&
           oh>0 && ph>=0 && pw>=7 && (pw&7)==7 && dm_pools_aligned(g);
}
typedef struct { const G *g; const T *i, *o; int32_t ph, pw, oh, ow, ci; int64_t items; } S2dJob;
static void s2d_fast_part(void *ctx, uint32_t part, uint32_t parts) {
    const S2dJob *J=ctx;
    const G *g=J->g;const int32_t ph=J->ph,pw=J->pw,oh=J->oh,ow=J->ow,ci=J->ci;
    BK bi,bo;
    bk_init(&bi,g,J->i);bk_init(&bo,g,J->o);
    const int64_t s=(pw-7)/8,nk=ow/4,nq=(nk+1)/2;
    int64_t i0,i1;
    dm_range(J->items,part,parts,&i0,&i1);
    for(int64_t it=i0;it<i1;it++) {
     const int64_t cb=it/oh;const int32_t h=(int32_t)(it%oh);
     const int64_t obase=(int64_t)(h>>3)*bo.nbw*bo.nbd+bo.slice;
     const uint32_t roff=256u*(uint32_t)(h&7);
     for(int dy=0;dy<2;dy++) {
        const int64_t H=(int64_t)ph+2*h+dy,j0=cb*4+2*dy;
        const int64_t ibase=(H>>3)*bi.nbw*bi.nbd+bi.slice+cb;
        const uint32_t ro=256u*(uint32_t)(H&7);
        DV A=dv_ld(bk_blk_or(&bi,ibase+s*bi.nbd,(uint8_t *)dm_zblk)+ro+128);
        const uint8_t *ib=bk_blk_or(&bi,ibase+(s+1)*bi.nbd,(uint8_t *)dm_zblk)+ro;
        for(int64_t q=0;q<nq;q++) {
            uint8_t *d1=bk_blk_or(&bo,obase+q*bo.nbd+j0+1,NULL),*d0=bk_blk_or(&bo,obase+q*bo.nbd+j0,NULL);
            if(d1)d1+=roff;
            if(d0)d0+=roff;
            #pragma clang loop unroll(full)
            for(int kk=0;kk<2;kk++) {
                if(kk && 2*q+1>=nk) break;
                const uint8_t *cur=ib;
                DV B=dv_ld(cur),C=dv_ld(cur+128);
                ib=bk_blk_or(&bi,ibase+(s+2+2*q+kk)*bi.nbd,(uint8_t *)dm_zblk)+ro;
                DV E,O;
                dv_deal32(dv_align(B,A,64),dv_align(C,B,64),&E,&O);
                if(d1)dv_st(d1+128*kk,E);
                if(d0)dv_st(d0+128*kk,O);
                A=C;
            }
        }
     }
    }
    (void)ci;
}
static void s2d_fast_run(const G *g, const T *i, const T *o, int32_t ph, int32_t pw, int32_t oh, int32_t ow, int32_t ci) {
    S2dJob J={g,i,o,ph,pw,oh,ow,ci,(int64_t)(ci/32)*oh};
    uint32_t parts=dm_parts(g,(uint64_t)oh*ow*4u*(uint64_t)ci,J.items);
    if(parts>1) nprg2_parallel_for(g->call->parallel,NPRG2_PAR_DM,parts,s2d_fast_part,&J);
    else s2d_fast_part(&J,0,1);
}
/* Whole space-to-depth: -1 when not taken, else 0 or the error code the
 * generic path's checks would give. */
static int s2d_fast(G *g, const T *i, const T *o, int32_t ph, int32_t pw, int32_t oh, int32_t ow, int32_t ci, int32_t oc) {
    if(!s2d_fast_ok(g,i,o,ph,pw,oh,ow,ci,oc)) return -1;
    int32_t rd[4]={1,2*oh,2*ow,ci},ro[4]={0,ph,pw-1,0},wd[4]={1,oh,ow,oc},zero[4]={0,0,0,0};
    int e=xfer_check(g,i,rd,ro,0);
    if(!e)e=xfer_check(g,o,wd,zero,1);
    if(e) return e;
    if(!dm_clear(i,rd,ro,o,wd,zero)) return -1;
    if(!g->dry){s2d_fast_run(g,i,o,ph,pw,oh,ow,ci);DM_COUNT(0,(uint64_t)oh*ow*oc);}
    return 0;
}

/* -------------------------------------------------------------- ResizeNear */
/* u8 blocked in and out. Handles the 2x nearest-neighbour column map
 * (wi[w] = w >> 1: parameters {.., 0x80000000, .., v4 < 0x80000000, 0}) with any
 * row map, and 8-column output chunks: 4 input pixels (one half-block vector)
 * become 8 output pixels (two vectors) with each pixel doubled. The column map
 * is read from the immutable constant tensor, so dry and wet agree. */
enum { RS_HMAX=512 };
static int resize_fast_ok(const G *g, const T *xt, const T *wpt, const T *o) {
    if(!DM_ENABLED || !dm_blocked_u8(xt) || !dm_blocked_u8(o) || !dm_pools_aligned(g)) return 0;
    int64_t v2,v4,v5;
    if(!const_value(g,wpt,2,&v2) || !const_value(g,wpt,4,&v4) || !const_value(g,wpt,5,&v5)) return 0;
    if(v2!=-2147483648LL || (uint32_t)v4>=0x80000000u || v5!=0) return 0; /* (w*2^31 + v4) >> 32 == w >> 1 */
    int32_t c=xt->d[3],n[4]={xt->d[0],o->d[1],o->d[2],c};
    if(c<=0 || (c&31) || xt->p[3] || o->p[3] || n[0]<=0 || n[1]<=0 || n[1]>RS_HMAX || n[2]<=0 || (n[2]&7)) return 0;
    if(xt->d[2]<n[2]/2 || (xt->p[2]&3) || (o->p[2]&7) || xt->d[1]<=0) return 0;
    return dm_write_dims_match(o,n);
}
typedef struct { const G *g; const T *xt, *o; const int32_t *hi; int32_t n[4]; } RsJob;
static void resize_part(void *ctx, uint32_t part, uint32_t parts) {
    const RsJob *J=ctx;const G *g=J->g;const T *xt=J->xt,*o=J->o;const int32_t *hi=J->hi,*n=J->n;
    BK bx,bo;
    bk_init(&bx,g,xt);bk_init(&bo,g,o);
    const int64_t ng=n[3]/32,nq=n[2]/8,xw=xt->p[2]/4,ow=o->p[2]/8;
    int64_t i0,i1;
    dm_range((int64_t)n[0]*n[1]*ng,part,parts,&i0,&i1);
    for(int64_t it=i0;it<i1;it++) {
        const int64_t z=it%ng;const int32_t h=(int32_t)((it/ng)%n[1]),b=(int32_t)(it/ng/n[1]);
        const int64_t H=(int64_t)hi[h]+xt->p[1],B=b+(int64_t)xt->p[0],OB=b+(int64_t)o->p[0],OH=h+(int64_t)o->p[1];
        const int64_t ibase=(B*bx.nbh+(H>>3))*bx.nbw*bx.nbd+bx.slice+z+xt->p[3]/32;
        const uint32_t ro=256u*(uint32_t)(H&7);
        const int64_t obase=(OB*bo.nbh+(OH>>3))*bo.nbw*bo.nbd+bo.slice+z+o->p[3]/32;
        const uint32_t oro=256u*(uint32_t)(OH&7);
        for(int64_t q=0;q<nq;q++) {
            const uint8_t *s=bk_blk_or(&bx,ibase+((xw+q)>>1)*bx.nbd,(uint8_t *)dm_zblk);
            DV lo,hi2;
            dv_dup32(dv_ld(s+ro+128*((xw+q)&1)),&lo,&hi2);
            uint8_t *d=bk_blk_or(&bo,obase+(ow+q)*bo.nbd,NULL);
            if(d){dv_st(d+oro,lo);dv_st(d+oro+128,hi2);}
        }
    }
}
static int resize_fast(G *g, const T *xt, const T *wpt, const T *o, const int32_t *hp) {
    if(!resize_fast_ok(g,xt,wpt,o)) return -1;
    int32_t n[4]={xt->d[0],o->d[1],o->d[2],xt->d[3]};
    int e=xfer_check(g,xt,xt->d,xt->p,0);
    if(!e)e=xfer_check(g,o,n,o->p,1);
    if(e) return e;
    if(!dm_clear(xt,xt->d,xt->p,o,n,o->p)) return -1;
    if(g->dry) return 0;
    int32_t hi[RS_HMAX];
    for(int32_t i=0;i<n[1];i++) {
        int64_t v=(int64_t)((((uint64_t)i*(uint32_t)hp[2])+(uint32_t)hp[4])>>32)+hp[5];
        if(v<0)v+=xt->d[1];
        CHECK(v>=0 && v<xt->d[1],E_RANGE);hi[i]=(int32_t)v;
    }
    DM_COUNT(1,(uint64_t)n[0]*n[1]*n[2]*n[3]);
    RsJob J={g,xt,o,hi,{n[0],n[1],n[2],n[3]}};
    uint32_t parts=dm_parts(g,(uint64_t)n[0]*n[1]*n[2]*n[3],(int64_t)n[0]*n[1]*(n[3]/32));
    if(parts>1) nprg2_parallel_for(g->call->parallel,NPRG2_PAR_DM,parts,resize_part,&J);
    else resize_part(&J,0,1);
    return 0;
}

/* --------------------------------------------------------------- Transpose */
/* u16 blocked, permutation {0,2,1,3} (swap H and W): a 2x2 pixel tile of one
 * 32-channel group is two input vectors (rows h, h+1; pixel pair w, w+1 with
 * channels interleaved as [w c0, w+1 c0, ...]) and two output vectors (rows w,
 * w+1; pixel pair h, h+1), the even and odd halfwords of the two inputs. */
static int tr_hw_fast_ok(const G *g, const T *xt, const T *o) {
    if(!DM_ENABLED || !(xt->flags&F_BLOCKED) || !(xt->flags&F_SHAPE) || !(o->flags&F_BLOCKED) || !(o->flags&F_SHAPE) ||
       xt->esize!=2 || o->esize!=2 || xt->dtype==DT_F32 || !dm_pools_aligned(g)) return 0;
    int32_t c=xt->d[3],n[4]={xt->d[0],xt->d[2],xt->d[1],c};
    if(c<=0 || (c&31) || xt->p[3] || o->p[3] || n[0]<=0 || n[1]<=0 || n[2]<=0 || (n[1]&1) || (n[2]&1)) return 0;
    if((xt->p[1]&1) || (xt->p[2]&1) || (o->p[2]&1)) return 0;
    return dm_write_dims_match(o,n);
}
typedef struct { const G *g; const T *xt, *o; int32_t n[4]; } TrJob;
/* Block-aligned geometry: every 8-row x 4-column input block maps onto half of two output blocks. */
static int tr_hw_al(const T *xt, const T *o) {
    return !(xt->d[1]&7) && !(xt->d[2]&3) && !(xt->p[1]&7) && !(xt->p[2]&3) && !(o->p[1]&3) && !(o->p[2]&7);
}
static void tr_hw_part_al(const TrJob *J, uint32_t part, uint32_t parts) {
    const G *g=J->g;const T *xt=J->xt,*o=J->o;const int32_t *n=J->n;
    BK bx,bo;
    bk_init(&bx,g,xt);bk_init(&bo,g,o);
    const int32_t H=xt->d[1],W=xt->d[2];
    const int64_t nz=n[3]/32,nhb=H/8,nwb=W/4;
    int64_t i0,i1;
    dm_range((int64_t)n[0]*nz*nhb,part,parts,&i0,&i1);
    const int64_t zi=bx.slice+xt->p[3]/32,zo=bo.slice+o->p[3]/32;
    for(int64_t it=i0;it<i1;it++) {
        const int64_t hb=it%nhb,z=(it/nhb)%nz,b=it/nhb/nz;
        const int64_t ib=((b+(int64_t)xt->p[0])*bx.nbh+(hb+(xt->p[1]>>3)))*bx.nbw+(xt->p[2]>>2);
        const int64_t obase=(b+(int64_t)o->p[0])*bo.nbh;
        const int64_t ocol=((int64_t)8*hb+o->p[2])>>2;
        for(int64_t wb=0;wb<nwb;wb++) {
            const uint8_t *ip=bk_blk_or(&bx,(ib+wb)*bx.nbd+z+zi,(uint8_t *)dm_zblk);
            const int64_t r0=4*wb+o->p[1];
            uint8_t *o0=bk_blk_or(&bo,((obase+(r0>>3))*bo.nbw+ocol)*bo.nbd+z+zo,NULL);
            uint8_t *o1=bk_blk_or(&bo,((obase+(r0>>3))*bo.nbw+ocol+1)*bo.nbd+z+zo,NULL);
            const uint32_t rb=256u*(uint32_t)(r0&7);
            #pragma clang loop unroll(full)
            for(int hq=0;hq<4;hq++) {
                uint8_t *ob=hq<2?o0:o1;
                #pragma clang loop unroll(full)
                for(int wp=0;wp<2;wp++) {
                    DV A=dv_ld(ip+256*(2*hq)+128*wp),B=dv_ld(ip+256*(2*hq+1)+128*wp);
                    if(ob) {
                        uint8_t *d=ob+rb+512*wp+128*(hq&1);
                        dv_st(d,dv_shuffe16(B,A));dv_st(d+256,dv_shuffo16(B,A));
                    }
                }
            }
        }
    }
}
static void tr_hw_part(void *ctx, uint32_t part, uint32_t parts) {
    const TrJob *J=ctx;const G *g=J->g;const T *xt=J->xt,*o=J->o;const int32_t *n=J->n;
    if(tr_hw_al(xt,o)){tr_hw_part_al(J,part,parts);return;}
    BK bx,bo;
    bk_init(&bx,g,xt);bk_init(&bo,g,o);
    const int32_t H=xt->d[1],W=xt->d[2];
    const int64_t nz=n[3]/32,hp=H/2;
    int64_t i0,i1;
    dm_range((int64_t)n[0]*nz*hp,part,parts,&i0,&i1);
    const int64_t zi=bx.slice+xt->p[3]/32,zo=bo.slice+o->p[3]/32;
    for(int64_t it=i0;it<i1;it++) {
        const int32_t h=(int32_t)(it%hp)*2,b=(int32_t)(it/hp/nz);const int64_t z=(it/hp)%nz;
        const int64_t AH=(int64_t)h+xt->p[1];
        const int64_t ib=((b+(int64_t)xt->p[0])*bx.nbh+(AH>>3))*bx.nbw;
        const uint32_t iro=256u*(uint32_t)(AH&7);
        const int64_t OW=(int64_t)h+o->p[2],OB=b+(int64_t)o->p[0];
        const int64_t ocol=OW>>2;
        const uint32_t ooff=128u*(uint32_t)((OW>>1)&1);
        const int64_t obase=OB*bo.nbh;
        int64_t icol=-1,crb=-1;
        const uint8_t *iblk=NULL;uint8_t *ocur=NULL;
        if(!(o->p[1]&1)) {
            /* output row pairs never straddle an 8-row block: run in segments that share
             * one input block column and one output block row */
            for(int32_t w=0;w<W;) {
                const int64_t AW=(int64_t)w+xt->p[2],r0=(int64_t)w+o->p[1];
                int32_t np=(int32_t)(4-(AW&3))>>1,npo=(int32_t)(8-(r0&7))>>1;
                if(npo<np)np=npo;
                if((W-w)/2<np)np=(W-w)/2;
                iblk=bk_blk_or(&bx,(ib+(AW>>2))*bx.nbd+z+zi,(uint8_t *)dm_zblk);
                ocur=bk_blk_or(&bo,((obase+(r0>>3))*bo.nbw+ocol)*bo.nbd+z+zo,NULL);
                const uint8_t *ip=iblk+128u*(uint32_t)((AW>>1)&1)+iro;
                uint8_t *dp=ocur?ocur+256u*(uint32_t)(r0&7)+ooff:NULL;
                for(int32_t k=0;k<np;k++) {
                    /* input pixel pairs of one block row alternate halves; output rows advance by two */
                    DV A=dv_ld(ip+128*k),B=dv_ld(ip+128*k+256);
                    if(dp){dv_st(dp+512*k,dv_shuffe16(B,A));dv_st(dp+512*k+256,dv_shuffo16(B,A));}
                }
                w+=2*np;
            }
            continue;
        }
        for(int32_t w=0;w<W;w+=2) {
            const int64_t AW=(int64_t)w+xt->p[2];
            if((AW>>2)!=icol){icol=AW>>2;iblk=bk_blk_or(&bx,(ib+icol)*bx.nbd+z+zi,(uint8_t *)dm_zblk);}
            const uint32_t ioff=128u*(uint32_t)((AW>>1)&1)+iro;
            DV A=dv_ld(iblk+ioff),B=dv_ld(iblk+ioff+256);
            const int64_t r0=(int64_t)w+o->p[1],r1=r0+1;
            if((r0>>3)!=crb){crb=r0>>3;ocur=bk_blk_or(&bo,((obase+crb)*bo.nbw+ocol)*bo.nbd+z+zo,NULL);}
            if(ocur)dv_st(ocur+256u*(uint32_t)(r0&7)+ooff,dv_shuffe16(B,A));
            if((r1>>3)!=crb){crb=r1>>3;ocur=bk_blk_or(&bo,((obase+crb)*bo.nbw+ocol)*bo.nbd+z+zo,NULL);}
            if(ocur)dv_st(ocur+256u*(uint32_t)(r1&7)+ooff,dv_shuffo16(B,A));
        }
    }
}
static int tr_hw_fast(G *g, const T *xt, const T *o) {
    if(!tr_hw_fast_ok(g,xt,o)) return -1;
    int32_t n[4]={xt->d[0],xt->d[2],xt->d[1],xt->d[3]};
    int e=xfer_check(g,xt,xt->d,xt->p,0);
    if(!e)e=xfer_check(g,o,n,o->p,1);
    if(e) return e;
    if(!dm_clear(xt,xt->d,xt->p,o,n,o->p)) return -1;
    if(g->dry) return 0;
    DM_COUNT(2,(uint64_t)n[0]*n[1]*n[2]*n[3]);
    TrJob J={g,xt,o,{n[0],n[1],n[2],n[3]}};
    const int32_t H=xt->d[1];
    uint32_t parts=dm_parts(g,(uint64_t)n[0]*n[1]*n[2]*n[3],(int64_t)n[0]*(n[3]/32)*(tr_hw_al(xt,o)?H/8:H/2));
    if(parts>1) nprg2_parallel_for(g->call->parallel,NPRG2_PAR_DM,parts,tr_hw_part,&J);
    else tr_hw_part(&J,0,1);
    return 0;
}

/* -------------------------------------------------------- flat (direct) data */
/* Direct tensors are dense row-major regions of their pool: m[] strides, d[] at
 * origin p. These helpers move regions between two of them without the dense
 * temporaries of the generic path. */
static int dm_flat(const T *t) {
    return (t->flags&F_DIRECT) && (t->flags&F_SHAPE) && t->dtype!=DT_F32 && (t->esize==1 || t->esize==2 || t->esize==4);
}
static size_t fl_lin(const T *t, const int32_t *o) {
    return (((size_t)o[0]*t->m[1]+(size_t)o[1])*t->m[2]+(size_t)o[2])*t->m[3]+(size_t)o[3];
}
/* Half-open byte range of a nonempty region, relative to its pool. */
static void fl_span(const T *t, const int32_t *dims, const int32_t *org, uint64_t *lo, uint64_t *hi) {
    int32_t last[4];
    for(int i=0;i<4;i++)last[i]=org[i]+dims[i]-1;
    *lo=(uint64_t)t->offset+(uint64_t)fl_lin(t,org)*t->esize;
    *hi=(uint64_t)t->offset+((uint64_t)fl_lin(t,last)+1)*t->esize;
}
/* 1 when two nonempty regions cannot share a byte. */
static int fl_apart(const T *a, const int32_t *da, const int32_t *oa, const T *b, const int32_t *db, const int32_t *ob) {
    if(a->pool!=b->pool) return 1;
    uint64_t al,ah,bl,bh;
    fl_span(a,da,oa,&al,&ah);fl_span(b,db,ob,&bl,&bh);
    return ah<=bl || bh<=al;
}
static int fl_empty(const int32_t *d) {return !d[0] || !d[1] || !d[2] || !d[3];}
static void fl_move(uint8_t *dst, const uint8_t *src, size_t n) {
    if(n>=512)qd_vcopy(dst,src,n);else memcpy(dst,src,n);
}
/* Copies dims of s at so to d at do_; the regions are disjoint. Contiguous
 * trailing runs (axes where both tensors are full width) are merged. */
static void fl_copy(const G *g, const T *d, const int32_t *dorg, const T *s, const int32_t *sorg, const int32_t *dims) {
    uint8_t *db=g->pool[d->pool]+d->offset;
    const uint8_t *sb=g->pool[s->pool]+s->offset;
    const size_t es=d->esize;
    size_t run=(size_t)dims[3];
    int k=3; /* axes k.. are inside the run */
    while(k>0 && dims[k]==s->m[k] && dims[k]==d->m[k]){k--;run*=(size_t)dims[k];}
    for(int32_t i0=0;i0<(k>0?dims[0]:1);i0++)for(int32_t i1=0;i1<(k>1?dims[1]:1);i1++)for(int32_t i2=0;i2<(k>2?dims[2]:1);i2++) {
        int32_t so[4]={sorg[0]+(k>0?i0:0),sorg[1]+(k>1?i1:0),sorg[2]+(k>2?i2:0),sorg[3]};
        int32_t dd[4]={dorg[0]+(k>0?i0:0),dorg[1]+(k>1?i1:0),dorg[2]+(k>2?i2:0),dorg[3]};
        fl_move(db+fl_lin(d,dd)*es,sb+fl_lin(s,so)*es,run*es);
    }
}
/* Fills dims of d at dorg with the element value fill (low esize bytes). */
static void fl_fill(const G *g, const T *d, const int32_t *dorg, const int32_t *dims, int64_t fill) {
    uint8_t *db=g->pool[d->pool]+d->offset;
    const size_t es=d->esize;
    for(int32_t b=0;b<dims[0];b++)for(int32_t h=0;h<dims[1];h++)for(int32_t w=0;w<dims[2];w++) {
        int32_t o[4]={dorg[0]+b,dorg[1]+h,dorg[2]+w,dorg[3]};
        uint8_t *p=db+fl_lin(d,o)*es;
        if(es==1)memset(p,(uint8_t)fill,(size_t)dims[3]);
        else if(es==2){uint16_t v=(uint16_t)fill;for(int32_t k=0;k<dims[3];k++)memcpy(p+2*(size_t)k,&v,2);}
        else {uint32_t v=(uint32_t)fill;for(int32_t k=0;k<dims[3];k++)memcpy(p+4*(size_t)k,&v,4);}
    }
}

/* --------------------------------------------------------------- Slice (flat) */
static int slice_fast(G *g, const T *i, const T *o, const int32_t *size, const int32_t *origin) {
    if(!DM_ENABLED || !dm_flat(i) || !dm_flat(o) || i->esize!=o->esize || !dm_write_dims_match(o,size)) return -1;
    int e=xfer_check(g,i,size,origin,0);
    if(!e)e=xfer_check(g,o,size,o->p,1);
    if(e) return e;
    if(fl_empty(size)) return 0;
    if(!fl_apart(i,size,origin,o,size,o->p)) return -1;
    if(!g->dry){fl_copy(g,o,o->p,i,origin,size);DM_COUNT(3,count4(size));}
    return 0;
}

/* ------------------------------------------------------------ SlicePad (flat) */
/* pad/start/size are shape values. The generic path fills a dense temp with
 * the fill value, copies the overlap of the padded window with the source and
 * writes it all; here the destination region gets the fill (unless the copy
 * covers it) and the overlap is copied straight in. */
static int slice_pad_fast(G *g, const T *xt, const T *o, const T *pad, const T *st, const T *sz, int64_t fill) {
    if(!DM_ENABLED || !dm_flat(xt) || !dm_flat(o) || xt->esize!=o->esize || !dm_write_dims_match(o,sz->sv)) return -1;
    for(int k=0;k<4;k++)if(sz->sv[k]<0) return -1;
    int e=xfer_check(g,xt,xt->d,xt->p,0);
    if(!e)e=xfer_check(g,o,sz->sv,o->p,1);
    if(e) return e;
    if(fl_empty(sz->sv)) return 0;
    if(!fl_empty(xt->d) && !fl_apart(xt,xt->d,xt->p,o,sz->sv,o->p)) return -1;
    if(g->dry) return 0;
    CHECK(fill>=0 && fill<=dtmax(xt->dtype),E_ARGS);
    int32_t lo[4],s0[4],s1[4],cd[4],so[4],dorg[4];
    int covered=1;
    for(int k=0;k<4;k++) {
        lo[k]=st->sv[k]-pad->sv[k];
        s0[k]=lo[k]>0?lo[k]:0;
        s1[k]=lo[k]+sz->sv[k]<xt->d[k]?lo[k]+sz->sv[k]:xt->d[k];
        if(s1[k]<s0[k])s1[k]=s0[k];
        cd[k]=s1[k]-s0[k];
        so[k]=xt->p[k]+s0[k];dorg[k]=o->p[k]+(s0[k]-lo[k]);
        if(cd[k]!=sz->sv[k])covered=0;
    }
    if(!covered)fl_fill(g,o,o->p,sz->sv,fill);
    if(!fl_empty(cd))fl_copy(g,o,dorg,xt,so,cd);
    DM_COUNT(4,count4(sz->sv));
    return 0;
}

/* -------------------------------------------------------------- Gather (flat) */
static int gather_fast(G *g, const T *xt, const T *axt, const T *o, const int32_t *idx, size_t ni) {
    int64_t axis;
    if(!DM_ENABLED || !dm_flat(xt) || !dm_flat(o) || xt->esize!=o->esize || !const_value(g,axt,0,&axis)) return -1;
    if(axis<0)axis+=4;
    if(axis<0 || axis>=4 || ni>(size_t)INT32_MAX) return -1;
    int32_t n[4];
    memcpy(n,xt->d,sizeof n);n[axis]=(int32_t)ni;
    if(!dm_write_dims_match(o,n)) return -1;
    for(int a=(int)axis+1;a<4;a++)if(xt->d[a]!=xt->m[a] || o->d[a]!=o->m[a]) return -1;
    int e=xfer_check(g,xt,xt->d,xt->p,0);
    if(!e)e=xfer_check(g,o,n,o->p,1);
    if(e) return e;
    if(fl_empty(n) || fl_empty(xt->d)) return fl_empty(n)?0:-1;
    if(!fl_apart(xt,xt->d,xt->p,o,n,o->p)) return -1;
    if(g->dry) return 0;
    for(size_t j=0;j<ni;j++) {
        int64_t k=idx[j];
        if(k<0)k+=xt->d[axis];
        CHECK(k>=0 && k<xt->d[axis],E_RANGE);
    }
    size_t inner=1;
    for(int a=(int)axis+1;a<4;a++)inner*=(size_t)xt->d[a];
    inner*=xt->esize;
    uint8_t *db=g->pool[o->pool]+o->offset;
    const uint8_t *sb=g->pool[xt->pool]+xt->offset;
    for(int32_t i0=0;i0<(axis>0?xt->d[0]:1);i0++)for(int32_t i1=0;i1<(axis>1?xt->d[1]:1);i1++)for(int32_t i2=0;i2<(axis>2?xt->d[2]:1);i2++)
     for(size_t j=0;j<ni;j++) {
        int64_t k=idx[j];
        if(k<0)k+=xt->d[axis];
        int32_t so[4]={xt->p[0]+(axis>0?i0:0),xt->p[1]+(axis>1?i1:0),xt->p[2]+(axis>2?i2:0),xt->p[3]},dorg[4]={o->p[0]+(axis>0?i0:0),o->p[1]+(axis>1?i1:0),o->p[2]+(axis>2?i2:0),o->p[3]};
        so[axis]=xt->p[axis]+(int32_t)k;dorg[axis]=o->p[axis]+(int32_t)j;
        fl_move(db+fl_lin(o,dorg)*o->esize,sb+fl_lin(xt,so)*xt->esize,inner);
    }
    DM_COUNT(5,count4(n));
    return 0;
}

/* ------------------------------------------------------------ Transpose (flat) */
/* Any permutation of u8/u16/i32 flat tensors, straight between the pools. The
 * last two output axes are walked in 16x16 tiles so strided reads and writes
 * both stay within a few cache lines. */
#define DM_TILE 16
static void tr_copy_elems(uint8_t *d, size_t ds, const uint8_t *s, size_t ss, int32_t n, size_t es) {
    if(es==1)for(int32_t k=0;k<n;k++)d[(size_t)k*ds]=s[(size_t)k*ss];
    else if(es==2)for(int32_t k=0;k<n;k++)memcpy(d+2*(size_t)k*ds,s+2*(size_t)k*ss,2);
    else for(int32_t k=0;k<n;k++)memcpy(d+4*(size_t)k*ds,s+4*(size_t)k*ss,4);
}
static int tr_flat_fast(G *g, const T *xt, const T *o, const int32_t *perm) {
    if(!DM_ENABLED || !dm_flat(xt) || !dm_flat(o) || xt->esize!=o->esize) return -1;
    int used=0;
    for(int k=0;k<4;k++){if(perm[k]<0 || perm[k]>3 || (used&(1<<perm[k]))) return -1;used|=1<<perm[k];}
    int32_t n[4];
    for(int k=0;k<4;k++)n[k]=xt->d[perm[k]];
    if(!dm_write_dims_match(o,n)) return -1;
    int e=xfer_check(g,xt,xt->d,xt->p,0);
    if(!e)e=xfer_check(g,o,n,o->p,1);
    if(e) return e;
    if(fl_empty(n)) return 0;
    if(!fl_apart(xt,xt->d,xt->p,o,n,o->p)) return -1;
    if(g->dry) return 0;
    const size_t es=xt->esize;
    size_t st[4]={(size_t)xt->m[1]*xt->m[2]*xt->m[3],(size_t)xt->m[2]*xt->m[3],(size_t)xt->m[3],1};
    size_t ds[4]={(size_t)o->m[1]*o->m[2]*o->m[3],(size_t)o->m[2]*o->m[3],(size_t)o->m[3],1};
    size_t ps[4]={st[perm[0]],st[perm[1]],st[perm[2]],st[perm[3]]};
    const uint8_t *sb=g->pool[xt->pool]+xt->offset+fl_lin(xt,xt->p)*es;
    uint8_t *db=g->pool[o->pool]+o->offset+fl_lin(o,o->p)*es;
    for(int32_t a=0;a<n[0];a++)for(int32_t b=0;b<n[1];b++) {
        const uint8_t *s0=sb+((size_t)a*ps[0]+(size_t)b*ps[1])*es;
        uint8_t *d0=db+((size_t)a*ds[0]+(size_t)b*ds[1])*es;
        if(ps[3]==1) {
            for(int32_t c=0;c<n[2];c++)memcpy(d0+(size_t)c*ds[2]*es,s0+(size_t)c*ps[2]*es,(size_t)n[3]*es);
            continue;
        }
        for(int32_t c0=0;c0<n[2];c0+=DM_TILE)for(int32_t d0i=0;d0i<n[3];d0i+=DM_TILE) {
            int32_t c1=c0+DM_TILE<n[2]?c0+DM_TILE:n[2],d1=d0i+DM_TILE<n[3]?d0i+DM_TILE:n[3];
            for(int32_t c=c0;c<c1;c++)
                tr_copy_elems(d0+((size_t)c*ds[2]+(size_t)d0i)*es,1,s0+((size_t)c*ps[2]+(size_t)d0i*ps[3])*es,ps[3],d1-d0i,es);
        }
    }
    DM_COUNT(6,count4(n));
    return 0;
}

/* ------------------------------------------------- blocked W<->C transposes */
/* Permutation {0,2,1,3} of a blocked tensor swaps its width and depth axes.
 * The unit is a tile of 32 pixels x 32 channels: input pixels are packed 2 (u16)
 * or 4 (u8) to a vector with all 32 channels of the group, output pixels (the
 * channels) are packed the same way with the 32 input pixels as lanes, so the
 * tile is a register-array transpose (dv_ladder). Block pointers are resolved
 * once per 8-row block row and tile, then reused for each of its rows. A group
 * of fewer than 32 channels yields only the output pixels it has (a masked
 * store when a vector is partial); lanes the tile does not own are not touched. */
static int tr_wc_common_ok(const G *g, const T *xt, const T *o, uint32_t es, int32_t pixw, int32_t *n) {
    if(!DM_ENABLED || !(xt->flags&F_BLOCKED) || !(xt->flags&F_SHAPE) || !(o->flags&F_BLOCKED) || !(o->flags&F_SHAPE) ||
       xt->esize!=es || o->esize!=es || xt->dtype==DT_F32 || o->dtype==DT_F32 || !dm_pools_aligned(g)) return 0;
    const int32_t W=xt->d[2],C=xt->d[3];
    n[0]=xt->d[0];n[1]=xt->d[1];n[2]=C;n[3]=W;
    if(n[0]<=0 || n[1]<=0 || W<=0 || C<=0 || (W&31)) return 0;
    if((xt->p[2]%pixw) || (xt->p[3]&31) || (o->p[2]%pixw) || (o->p[3]&31)) return 0;
    return dm_write_dims_match(o,n);
}
typedef struct { const G *g; const T *xt, *o; const int32_t *n; int es; } TrWcJob;
/* Row runs (b, h..h+run) are independent: part p takes every parts-th run.
 * ES is a compile-time constant (1 or 2); pixels per vector pb=4/ES... (u16: 2, u8: 4). */
static inline __attribute__((always_inline)) void tr_wc_part_es(const TrWcJob *J, uint32_t part, uint32_t parts, const int ES) {
    const G *g=J->g;const T *xt=J->xt,*o=J->o;const int32_t *n=J->n;
    BK bx,bo;
    bk_init(&bx,g,xt);bk_init(&bo,g,o);
    const int32_t H=xt->d[1],W=xt->d[2],C=xt->d[3];
    const int pb=ES==2?2:4;       /* pixels per vector */
    const int nin=ES==2?16:8;     /* input vectors per tile: 32 pixels / pixels per vector */
    const int cws=ES==2?2:3;      /* log2 pixels per block column */
    const int pbs=ES==2?1:2;      /* log2 pixels per vector */
    const int64_t zi0=xt->p[3]/32;
    const int calign=!(((uint32_t)xt->p[2]|(uint32_t)o->p[2])&((1u<<cws)-1u));
    uint32_t runix=0;
    for(int32_t b=0;b<n[0];b++)for(int32_t h=0;h<H;runix++) {
        const int64_t Hi=(int64_t)h+xt->p[1],Ho=(int64_t)h+o->p[1],B=b+(int64_t)xt->p[0],OB=b+(int64_t)o->p[0];
        int32_t run=8-(int32_t)(Hi&7);
        if(8-(int32_t)(Ho&7)<run)run=8-(int32_t)(Ho&7);
        if(H-h<run)run=H-h;
        if(runix%parts!=part){h+=run;continue;}
        const int64_t ibase=(B*bx.nbh+(Hi>>3))*bx.nbw,obase=(OB*bo.nbh+(Ho>>3))*bo.nbw;
        const uint32_t iro=256u*(uint32_t)(Hi&7),oro=256u*(uint32_t)(Ho&7);
        for(int32_t zc=0;zc*32<C;zc++) {
            const int32_t cn=C-32*zc<32?C-32*zc:32,nout=(cn+pb-1)/pb;
            for(int32_t w0=0;w0<W;w0+=32) {
                const uint8_t *ip[16];
                uint8_t *op[16];
                uint32_t allout=1;
                /* p[2] is a multiple of pb, a vector's pixels share a block, and a block holds
                 * two vectors (halves of 128 bytes): 32 pixels span nin/2 blocks. */
                if(!calign) {
                    #pragma clang loop unroll(full)
                    for(int i=0;i<nin;i++) {
                        const int64_t aw=(int64_t)xt->p[2]+w0+(int64_t)pb*i;
                        const int64_t idx=(ibase+(aw>>cws))*bx.nbd+bx.slice+zc+zi0;
                        ip[i]=bk_blk_or(&bx,idx,(uint8_t *)dm_zblk)+128*((aw>>pbs)&1)+iro;
                    }
                } else {
                    const int64_t awb=((int64_t)xt->p[2]+w0)>>cws;
                    #pragma clang loop unroll(full)
                    for(int k=0;k<nin/2;k++) {
                        uint8_t *blk=bk_blk_or(&bx,(ibase+awb+k)*bx.nbd+bx.slice+zc+zi0,(uint8_t *)dm_zblk)+iro;
                        ip[2*k]=blk;ip[2*k+1]=blk+128;
                    }
                }
                if(cn==32 && calign) {
                    const int64_t ocb=((int64_t)o->p[2]+32*zc)>>cws;
                    const int64_t zo=bo.slice+(o->p[3]+w0)/32;
                    #pragma clang loop unroll(full)
                    for(int k=0;k<nin/2;k++) {
                        uint8_t *blk=bk_blk_or(&bo,(obase+ocb+k)*bo.nbd+zo,NULL);
                        op[2*k]=blk?blk+oro:NULL;op[2*k+1]=blk?blk+oro+128:NULL;
                        allout&=blk!=NULL;
                    }
                } else {
                    for(int i=0;i<nout;i++) {
                        const int64_t oc=(int64_t)o->p[2]+32*zc+(int64_t)pb*i;
                        const int64_t idx=(obase+(oc>>cws))*bo.nbd+bo.slice+(o->p[3]+w0)/32;
                        uint8_t *blk=bk_blk_or(&bo,idx,NULL);
                        op[i]=blk?blk+128*((oc>>pbs)&1)+oro:NULL;
                    }
                    allout=0;
                }
                                for(int32_t r=0;r<run;r++) {
                    DV v[16];
                    #pragma clang loop unroll(full)
                    for(int i=0;i<nin;i++)v[i]=dv_ld(ip[i]+256*r);
                    if(ES==2)dv_ladder(v,16,dm_lad_wc16,6);else dv_ladder(v,8,dm_lad_wc8,5);
                    if(allout) {
                        #pragma clang loop unroll(full)
                        for(int i=0;i<nin;i++)dv_st(op[i]+256*r,v[i]);
                    } else {
                        for(int i=0;i<nout;i++) {
                            if(!op[i]) continue;
                            uint8_t *d=op[i]+256*r;
                            const int32_t m=cn-pb*i<pb?cn-pb*i:pb;
                            if(m==pb)dv_st(d,v[i]);else dv_stun(d,v[i],(uint32_t)m*(ES==2?64u:32u));
                        }
                    }
                }
            }
        }
        h+=run;
    }
}
static void tr_wc_part(void *ctx, uint32_t part, uint32_t parts) {
    const TrWcJob *J=ctx;
    if(J->es==2)tr_wc_part_es(J,part,parts,2);else tr_wc_part_es(J,part,parts,1);
}
static int tr_wc_hook(G *g, const T *xt, const T *o, const int32_t *n, int id, int es) {
    int e=xfer_check(g,xt,xt->d,xt->p,0);
    if(!e)e=xfer_check(g,o,n,o->p,1);
    if(e) return e;
    if(!dm_clear(xt,xt->d,xt->p,o,n,o->p)) return -1;
    if(g->dry) return 0;
    (void)id;DM_COUNT(id,(uint64_t)n[0]*n[1]*n[2]*n[3]);
    TrWcJob J={g,xt,o,n,es};
    int64_t runs=(int64_t)n[0]*((xt->d[1]+7)/8);
    uint32_t parts=dm_parts(g,(uint64_t)n[0]*n[1]*n[2]*n[3],runs);
    if(parts>1) nprg2_parallel_for(g->call->parallel,NPRG2_PAR_DM,parts,tr_wc_part,&J);
    else tr_wc_part(&J,0,1);
    return 0;
}
static int tr_wc16_fast(G *g, const T *xt, const T *o) {
    int32_t n[4];
    /* every channel pair of the group must be whole */
    if(!tr_wc_common_ok(g,xt,o,2,2,n) || (xt->d[3]&1)) return -1;
    return tr_wc_hook(g,xt,o,n,7,2);
}
static int tr_wc8_fast(G *g, const T *xt, const T *o) {
    int32_t n[4];
    if(!tr_wc_common_ok(g,xt,o,1,4,n)) return -1;
    return tr_wc_hook(g,xt,o,n,8,1);
}

/* u8 blocked {0,2,1,3}: pixel (h,w) of a depth group is 32 bytes, so the swap
 * moves 32-byte units; 4 rows x 4 pixels of one group are four vectors whose
 * units transpose. Pixels of the input quad beyond the width are loaded but
 * their output rows are not stored. */
static int tr_hw8_fast(G *g, const T *xt, const T *o) {
    if(!DM_ENABLED || !(xt->flags&F_BLOCKED) || !(xt->flags&F_SHAPE) || !(o->flags&F_BLOCKED) || !(o->flags&F_SHAPE) ||
       xt->esize!=1 || o->esize!=1 || xt->dtype==DT_F32 || o->dtype==DT_F32 || !dm_pools_aligned(g)) return -1;
    const int32_t H=xt->d[1],W=xt->d[2],C=xt->d[3],n[4]={xt->d[0],W,H,C};
    if(n[0]<=0 || H<=0 || W<=0 || C<=0 || (C&31) || (H&3) || (xt->p[1]&3) || (xt->p[2]&3) || (xt->p[3]&31) || (o->p[2]&3) || (o->p[3]&31)) return -1;
    if(!dm_write_dims_match(o,n)) return -1;
    int e=xfer_check(g,xt,xt->d,xt->p,0);
    if(!e)e=xfer_check(g,o,n,o->p,1);
    if(e) return e;
    if(!dm_clear(xt,xt->d,xt->p,o,n,o->p)) return -1;
    if(g->dry) return 0;
    DM_COUNT(9,(uint64_t)n[0]*n[1]*n[2]*n[3]);
    BK bx,bo;
    bk_init(&bx,g,xt);bk_init(&bo,g,o);
    for(int32_t b=0;b<n[0];b++)for(int64_t z=0;z<C/32;z++)for(int32_t h0=0;h0<H;h0+=4)for(int32_t w0=0;w0<W;w0+=4) {
        const int64_t Hi=(int64_t)h0+xt->p[1],Wi=(int64_t)w0+xt->p[2],B=b+(int64_t)xt->p[0],OB=b+(int64_t)o->p[0];
        uint8_t *blk=bk_blk(&bx,B,Hi>>3,Wi>>3,z+xt->p[3]/32);
        DV v[4];
        for(int i=0;i<4;i++)v[i]=blk?dv_ld(blk+256*((Hi&7)+i)+128*((Wi>>2)&1)):dv_zero();
        dv_ladder(v,4,dm_lad_u32,2);
        const int64_t Oc=(int64_t)h0+o->p[2];
        for(int i=0;i<4 && w0+i<W;i++) {
            const int64_t Orow=(int64_t)w0+i+o->p[1];
            uint8_t *ob=bk_blk(&bo,OB,Orow>>3,Oc>>3,z+o->p[3]/32);
            if(ob)dv_st(ob+256*(Orow&7)+128*((Oc>>2)&1),v[i]);
        }
    }
    return 0;
}

/* ----------------------------------------------- flat <-> blocked data moves */
typedef struct { uint32_t pool; uint64_t lo, hi; int hit; } DMFlat;
static void dm_v_flat(void *c, uint32_t e) {
    DMFlat *k=c;
    if((e&127u)!=k->pool) return;
    uint64_t o=e&~127u;
    if(o<k->hi && o+BLOCK_BYTES>k->lo) k->hit=1;
}
/* 1 when the flat region f shares no byte with a block of the blocked region b. */
static int dm_flat_apart_blocked_raw(const T *f, const int32_t *fd, const int32_t *fo, const T *b, const int32_t *bd, const int32_t *bo) {
    if(fl_empty(fd)) return 1;
    DMFlat k;
    k.pool=f->pool;k.hit=0;
    fl_span(f,fd,fo,&k.lo,&k.hi);
    dm_visit(b,bd,bo,dm_v_flat,&k);
    return !k.hit;
}
/* Memoised per command like dm_clear (callers have `g` in scope). */
static int dm_flat_apart_blocked_m(G *g, const T *f, const int32_t *fd, const int32_t *fo, const T *b, const int32_t *bd, const int32_t *bo) {
    uint32_t *s=chk_slots(g);
    uint32_t tag=0;
    if(s) {
        tag=chk_r(chk_r(chk_t(0x7f4a7c15u,f),fd),fo);
        tag=chk_r(chk_r(chk_t(tag,b),bd),bo)|1u;
        if(chk_hit(s,tag)) return 1;
    }
    int ok=dm_flat_apart_blocked_raw(f,fd,fo,b,bd,bo);
    if(ok) chk_put(s,tag);
    return ok;
}
#define dm_flat_apart_blocked(f,fd,fo,b,bd,bo) dm_flat_apart_blocked_m(g,(f),(fd),(fo),(b),(bd),(bo))
/* A blocked u16 tensor written (or read) as pixels with 2 channels in lanes 0..1:
 * the geometry the flow tensors use. */
static int dm_blocked_c2(const T *t, int crouton) {
    return (t->flags&F_BLOCKED) && (t->flags&F_SHAPE) && t->esize==2 && t->dtype!=DT_F32 && t->d[0]==1 && t->d[3]==2 && t->p[3]==0 &&
           (!crouton || (t->crouton && t->m[3]>=32)) && !(t->p[2]&1) && t->d[1]>0 && t->d[2]>0 && !(t->d[2]&31);
}

/* ------------------------------------------------------------ COPY_VALUES */
/* Flat flow (1,H,W,2) u16 -> padded blocked (1,H,W,2 of 32 lanes), and back.
 * 32 pixels are one vector of flat words (c0, c1). dealh splits it into the c0
 * and c1 planes, whose words (c0 of pixel 2p, 2p+1) are exactly the first word
 * of the blocked pixel-pair vector p and (c1 ...) the second: Z = word
 * interleave of the planes holds pair p's two words at bytes 8p. The crouton
 * write zero fills the other 30 lanes, as the generic padded t_write does. */
static int copy_c2_ok(const G *g, const T *f, const T *b, int to_blocked) {
    if(!DM_ENABLED || !(f->flags&F_DIRECT) || !(f->flags&F_SHAPE) || f->esize!=2 || f->dtype==DT_F32 || !dm_pools_aligned(g)) return 0;
    if(!dm_blocked_c2(b,to_blocked)) return 0;
    if(memcmp(f->d,b->d,sizeof f->d) || f->p[3]!=0 || f->m[3]!=2 || f->d[0]!=1) return 0;
    /* the flat value reshapes to the blocked tensor's d (equal here) and the dtypes must match as in k_copy_values */
    return (to_blocked || !f->crouton) && b->dtype==f->dtype;
}
static int copy_c2_fast(G *g, const T *f, const T *b, int to_blocked) {
    if(!copy_c2_ok(g,f,b,to_blocked)) return -1;
    int32_t bd[4]={1,b->d[1],b->d[2],to_blocked?32:2};   /* region the generic path moves in the blocked tensor */
    if(to_blocked && b->m[3]<32) return -1;
    int e=to_blocked?xfer_check(g,f,f->d,f->p,0):xfer_check(g,b,b->d,b->p,0);
    if(!e)e=to_blocked?xfer_check(g,b,bd,b->p,1):xfer_check(g,f,f->d,f->p,1);
    if(e) return e;
    if(!dm_flat_apart_blocked(f,f->d,f->p,b,to_blocked?bd:b->d,b->p)) return -1;
    if(g->dry) return 0;
    DM_COUNT(to_blocked?10:11,(uint64_t)b->d[1]*b->d[2]*2);
    BK bk;
    bk_init(&bk,g,b);
    uint8_t *fb=g->pool[f->pool]+f->offset;
    const int32_t H=b->d[1],W=b->d[2];
    for(int32_t h=0;h<H;) {
        const int64_t Hb=(int64_t)h+b->p[1];
        int32_t run=8-(int32_t)(Hb&7);
        if(H-h<run)run=H-h;
        for(int32_t w0=0;w0<W;w0+=32) {
            uint8_t *bp[16];
            for(int p=0;p<16;p++) {
                const int64_t aw=(int64_t)b->p[2]+w0+2*p;
                uint8_t *blk=bk_blk(&bk,b->p[0],Hb>>3,aw>>2,b->p[3]/32);
                bp[p]=blk?blk+128*((aw>>1)&1):NULL;
            }
            for(int32_t r=0;r<run;r++) {
                uint8_t *fp=fb+fl_lin(f,(int32_t[4]){f->p[0],f->p[1]+h+r,f->p[2]+w0,f->p[3]})*2;
                const uint32_t ro=256u*(uint32_t)((Hb&7)+r);
                if(to_blocked) {
                    DV m=dv_dealh(dv_ldu(fp)),z,zh;
                    dv_shuffk(dv_ror(m,64),m,2,&z,&zh);
                    for(int p=0;p<16;p++)
                        if(bp[p])dv_st(bp[p]+ro,dv_keep(dv_ror(z,8u*(uint32_t)p),8));
                } else {
                    DV z=dv_zero();
                    for(int p=0;p<16;p++)
                        if(bp[p])z=dv_or(z,dv_ror(dv_keep(dv_ld(bp[p]+ro),8),(128u-8u*(uint32_t)p)&127u));
                    DV lo,hi;
                    dv_dealk(z,z,2,&lo,&hi);
                    dv_stun(fp,dv_shuffh(dv_or(dv_keep(lo,64),dv_ror(dv_keep(hi,64),64))),128);
                }
            }
        }
        h+=run;
    }
    return 0;
}

/* COPY_VALUES (a reshape / format change that keeps the dense pixel order and
 * the channel count D) between blocked and blocked, blocked and flat, and flat
 * and blocked, without the strip buffer. Pixels move in runs inside one row of
 * both sides. Same bytes as copy_runs / t_read + t_write; -1 when not covered.
 *   blocked -> blocked: D a multiple of 32 and both sides chunk-aligned in width
 *     (8 u8 / 4 u16 / 2 u32 pixels): a chunk of one depth group is 256 bytes in
 *     the same layout on both sides, so it is two vector copies. An absent source
 *     block reads as zero; an absent destination block is skipped.
 *   blocked <-> flat (u8): a pixel's 32 channels of a depth group are a 32-byte
 *     unit of the block row; units move through partial vector stores. A crouton
 *     destination whose depth widens to a multiple of 32 zero fills the unit's
 *     tail, as the padded t_write does. */
typedef struct { int32_t b, h, w; } CvCur;
static inline void cv_adv(CvCur *c, int32_t n, int32_t d1, int32_t d2) {
    if((c->w+=n)==d2){c->w=0;if(++c->h==d1){c->h=0;c->b++;}}
}
static int copy_vals_fast(G *g, const T *it, const T *ot) {
    if(!DM_ENABLED || !(it->flags&F_SHAPE) || !(ot->flags&F_SHAPE) || it->esize!=ot->esize || it->dtype!=ot->dtype || it->dtype==DT_F32 ||
       !dm_pools_aligned(g)) return -1;
    if(count4(it->d)!=count4(ot->d) || it->d[3]!=ot->d[3] || it->d[3]<=0) return -1;
    for(int i=0;i<4;i++) if(it->d[i]<=0 || ot->d[i]<=0) return -1;
    const int ibk=!!(it->flags&F_BLOCKED),obk=!!(ot->flags&F_BLOCKED);
    if(!ibk && !obk) return -1;
    if((!ibk && !(it->flags&F_DIRECT)) || (!obk && !(ot->flags&F_DIRECT))) return -1;
    const uint32_t es=it->esize;
    if(es!=1 && es!=2 && es!=4) return -1;
    const int32_t D=it->d[3];
    int32_t Dw=D;
    if(ot->crouton) {
        int32_t e=(ot->d[3]+ot->p[3]+31)/32*32;
        if(e>ot->m[3])e=ot->m[3];
        Dw=e-ot->p[3];
        if(Dw<D) return -1;
    }
    const int32_t cw=es==1?8:es==2?4:2,cws=es==1?3:es==2?2:1;
    if((ibk && (it->p[3]&31)) || (obk && (ot->p[3]&31))) return -1;
    if(ibk && obk) {
        if((D&31) || Dw!=D || (it->d[2]%cw) || (ot->d[2]%cw) || (it->p[2]%cw) || (ot->p[2]%cw)) return -1;
    } else {
        if(es!=1 || ibk) return -1;   /* blocked -> flat measured slower than copy_runs */
        if(obk && ((Dw&31) || (!ot->crouton && (D&31)))) return -1;
    }
    const int32_t ow[4]={ot->d[0],ot->d[1],ot->d[2],Dw};
    {
        int e=xfer_check(g,it,it->d,it->p,0);
        if(!e)e=xfer_check(g,ot,ow,ot->p,1);
        if(e) return e;
    }
    if(ibk && obk) {if(!dm_clear(it,it->d,it->p,ot,ow,ot->p)) return -1;}
    else if(ibk) {if(!dm_flat_apart_blocked(ot,ow,ot->p,it,it->d,it->p)) return -1;}
    else {if(!dm_flat_apart_blocked(it,it->d,it->p,ot,ow,ot->p)) return -1;}
    if(g->dry) return 0;
    DM_COUNT(5,count4(it->d));
    BK bi,bo;
    if(ibk)bk_init(&bi,g,it);
    if(obk)bk_init(&bo,g,ot);
    const int32_t nz=(Dw+31)/32;
    const int64_t total=(int64_t)it->d[0]*it->d[1]*it->d[2];
    const uint8_t *fsrc=ibk?NULL:g->pool[it->pool]+it->offset;
    uint8_t *fdst=obk?NULL:g->pool[ot->pool]+ot->offset;
    const size_t fsi=ibk?0:(size_t)it->m[3],fso=obk?0:(size_t)ot->m[3];
    CvCur ci={0,0,0},co={0,0,0};
    for(int64_t P=0;P<total;) {
        int32_t n=cw;
        if(!(ibk && obk)) {
            const int32_t Wb=ibk?ci.w+it->p[2]:co.w+ot->p[2];
            n=cw-(Wb&(cw-1));
            if(it->d[2]-ci.w<n)n=it->d[2]-ci.w;
            if(ot->d[2]-co.w<n)n=ot->d[2]-co.w;
        }
        /* index of depth group 0 and byte offset of the first pixel in the block row */
        int64_t ixi=0,ixo=0;uint32_t rowi=0,rowo=0;int32_t pxi=0,pxo=0;
        if(ibk) {
            const int64_t Hi=(int64_t)ci.h+it->p[1],Wi=(int64_t)ci.w+it->p[2];
            ixi=((((int64_t)ci.b+it->p[0])*bi.nbh+(Hi>>3))*bi.nbw+(Wi>>cws))*bi.nbd+bi.slice+it->p[3]/32;
            rowi=256u*(uint32_t)(Hi&7);pxi=(int32_t)(Wi&(cw-1));
        }
        if(obk) {
            const int64_t Ho=(int64_t)co.h+ot->p[1],Wo=(int64_t)co.w+ot->p[2];
            ixo=((((int64_t)co.b+ot->p[0])*bo.nbh+(Ho>>3))*bo.nbw+(Wo>>cws))*bo.nbd+bo.slice+ot->p[3]/32;
            rowo=256u*(uint32_t)(Ho&7);pxo=(int32_t)(Wo&(cw-1));
        }
        if(ibk && obk) {
            for(int32_t z=0;z<nz;z++) {
                uint8_t *d=bk_blk_or(&bo,ixo+z,NULL);
                if(!d) continue;
                d+=rowo;
                const uint8_t *sb=bk_blk_or(&bi,ixi+z,NULL);
                if(sb){sb+=rowi;DV a=dv_ld(sb),b=dv_ld(sb+128);dv_st(d,a);dv_st(d+128,b);}
                else {dv_st(d,dv_zero());dv_st(d+128,dv_zero());}
            }
        } else if(ibk) {
            uint8_t *fd=fdst+fl_lin(ot,(int32_t[4]){ot->p[0]+co.b,ot->p[1]+co.h,ot->p[2]+co.w,ot->p[3]});
            for(int32_t z=0;z<nz;z++) {
                const uint32_t nb=(D-32*z)<32?(uint32_t)(D-32*z):32u;
                const uint8_t *sb=bk_blk_or(&bi,ixi+z,NULL);
                if(sb) {
                    sb+=rowi;
                    const DV va=dv_ld(sb),vb=dv_ld(sb+128);
                    for(int32_t k=0;k<n;k++) {
                        const int32_t px=pxi+k;
                        dv_stup(fd+(size_t)k*fso+32*(size_t)z,px<4?va:vb,32u*(uint32_t)(px&3),nb);
                    }
                } else {
                    for(int32_t k=0;k<n;k++)dv_stup(fd+(size_t)k*fso+32*(size_t)z,dv_zero(),0,nb);
                }
            }
        } else {
            const uint8_t *fs=fsrc+fl_lin(it,(int32_t[4]){it->p[0]+ci.b,it->p[1]+ci.h,it->p[2]+ci.w,it->p[3]});
            for(int32_t z=0;z<nz;z++) {
                uint8_t *d=bk_blk_or(&bo,ixo+z,NULL);
                if(!d) continue;
                d+=rowo;
                const uint32_t nb=(D-32*z)<32?(uint32_t)(D-32*z):32u;
                for(int32_t k=0;k<n;k++) {
                    DV v=dv_ldn(fs+(size_t)k*fsi+32*(size_t)z,nb);
                    if(nb<32)v=dv_keep(v,nb);
                    dv_stun(d+32*(size_t)(pxo+k),v,32);
                }
            }
        }
        P+=n;cv_adv(&ci,n,it->d[1],it->d[2]);cv_adv(&co,n,ot->d[1],ot->d[2]);
    }
    return 0;
}

/* ------------------------------------------------------------------ IM2COL */
/* 3x3 / stride 1 / depth 1 u8: output lane 3y+x of pixel w is input (h+y, w+x).
 * For 128 pixels the 9 shifted row windows (and 7 zero vectors) are 16 vectors;
 * a 16x16 byte transpose gives vectors of 8 pixels x 16 lanes, and interleaving
 * each with zero 16-byte units yields the two 4-pixel x 32-lane vectors (lanes
 * 9..31 zero, as the padded t_write leaves them) of one block row. */
static int im2col_u8_fast(G *g, const T *i, const T *o, int32_t kh, int32_t kw, int32_t cin, int32_t sh, int32_t sw) {
    if(!DM_ENABLED || kh!=3 || kw!=3 || cin!=1 || sh!=1 || sw!=1 || !dm_pools_aligned(g)) return -1;
    if(!(i->flags&F_DIRECT) || !(i->flags&F_SHAPE) || i->esize!=1 || i->dtype==DT_F32 || i->d[0]!=1 || i->d[3]!=1 || i->m[3]!=1 || i->p[3]!=0) return -1;
    if(!(o->flags&F_BLOCKED) || !(o->flags&F_SHAPE) || o->esize!=1 || o->dtype==DT_F32 || !o->crouton || o->d[0]!=1 || o->d[3]!=9 ||
       o->p[3]!=0 || o->m[3]<32 || (o->p[2]&7)) return -1;
    const int32_t oh=o->d[1],ow=o->d[2];
    if(oh<=0 || ow<=0 || (ow&127) || 2+oh-1>=i->d[1] || 2+ow-1>=i->d[2]) return -1;
    int32_t wd[4]={1,oh,ow,32};
    int e=xfer_check(g,i,i->d,i->p,0);
    if(!e)e=xfer_check(g,o,wd,o->p,1);
    if(e) return e;
    if(!dm_flat_apart_blocked(i,i->d,i->p,o,wd,o->p)) return -1;
    if(g->dry) return 0;
    DM_COUNT(12,(uint64_t)oh*ow*9);
    BK bo;
    bk_init(&bo,g,o);
    const uint8_t *xb=g->pool[i->pool]+i->offset;
    for(int32_t h=0;h<oh;) {
        const int64_t Ho=(int64_t)h+o->p[1];
        int32_t run=8-(int32_t)(Ho&7);
        if(oh-h<run)run=oh-h;
        for(int32_t w0=0;w0<ow;w0+=128) {
            uint8_t *bp[16];
            for(int m=0;m<16;m++)bp[m]=bk_blk(&bo,o->p[0],Ho>>3,((int64_t)o->p[2]+w0)/8+m,0);
            for(int32_t r=0;r<run;r++) {
                DV s[16];
                for(int y=0;y<3;y++)for(int x=0;x<3;x++)
                    s[3*y+x]=dv_ldu(xb+fl_lin(i,(int32_t[4]){i->p[0],i->p[1]+h+r+y,i->p[2]+w0+x,0}));
                for(int l=9;l<16;l++)s[l]=dv_zero();
                dv_ladder(s,16,dm_lad_t16,4);
                const uint32_t ro=256u*(uint32_t)((Ho&7)+r);
                for(int m=0;m<16;m++) {
                    if(!bp[m]) continue;
                    DV lo,hi;
                    dv_shuffk(dv_zero(),s[m],4,&lo,&hi);
                    dv_st(bp[m]+ro,lo);dv_st(bp[m]+ro+128,hi);
                }
            }
        }
        h+=run;
    }
    return 0;
}
/* 4x4 / stride 4 / depth 2 u16 (a 16-pixel patch is 32 lanes): pixel w's lane
 * 8y+2x+c is input (4h+y, 4w+x) channel c. A row vector of 32 input pixels is 8
 * patch rows (4 pixels = 8 halfwords each); four rows y become 4 output
 * pixel-pair vectors whose 32-byte row segments interleave the halfwords of
 * their two patch rows. */
static int im2col_u16_fast(G *g, const T *i, const T *o, int32_t kh, int32_t kw, int32_t cin, int32_t sh, int32_t sw) {
    if(!DM_ENABLED || kh!=4 || kw!=4 || cin!=2 || sh!=4 || sw!=4 || !dm_pools_aligned(g)) return -1;
    if(!(i->flags&F_DIRECT) || !(i->flags&F_SHAPE) || i->esize!=2 || i->dtype==DT_F32 || i->d[0]!=1 || i->d[3]!=2 || i->m[3]!=2 || i->p[3]!=0) return -1;
    if(!(o->flags&F_BLOCKED) || !(o->flags&F_SHAPE) || o->esize!=2 || o->dtype==DT_F32 || o->d[0]!=1 || o->d[3]!=32 ||
       o->p[3]!=0 || o->m[3]<32 || (o->p[2]&1)) return -1;
    const int32_t oh=o->d[1],ow=o->d[2];
    if(oh<=0 || ow<=0 || (ow&7) || 3+4*(oh-1)>=i->d[1] || 3+4*(ow-1)>=i->d[2]) return -1;
    int e=xfer_check(g,i,i->d,i->p,0);
    if(!e)e=xfer_check(g,o,o->d,o->p,1);
    if(e) return e;
    if(!dm_flat_apart_blocked(i,i->d,i->p,o,o->d,o->p)) return -1;
    if(g->dry) return 0;
    DM_COUNT(13,(uint64_t)oh*ow*32);
    BK bo;
    bk_init(&bo,g,o);
    const uint8_t *xb=g->pool[i->pool]+i->offset;
    for(int32_t h=0;h<oh;) {
        const int64_t Ho=(int64_t)h+o->p[1];
        int32_t run=8-(int32_t)(Ho&7);
        if(oh-h<run)run=oh-h;
        for(int32_t w0=0;w0<ow;w0+=8) {
            uint8_t *bp[4];
            for(int q=0;q<4;q++) {
                const int64_t W=(int64_t)o->p[2]+w0+2*q;
                uint8_t *blk=bk_blk(&bo,o->p[0],Ho>>3,W>>2,0);
                bp[q]=blk?blk+128*((W>>1)&1):NULL;
            }
            for(int32_t r=0;r<run;r++) {
                DV v[4];
                for(int y=0;y<4;y++)v[y]=dv_ldu(xb+fl_lin(i,(int32_t[4]){i->p[0],i->p[1]+4*(h+r)+y,i->p[2]+4*w0,0})*2);
                dv_ladder(v,4,dm_lad_i25,4);
                const uint32_t ro=256u*(uint32_t)((Ho&7)+r);
                for(int q=0;q<4;q++)if(bp[q])dv_st(bp[q]+ro,v[q]);
            }
        }
        h+=run;
    }
    return 0;
}

/* ------------------------------------------------- flat 2-plane interleave */
/* {0,2,3,1} of a flat (b,2,H,W) tensor to (b,H,W,2): the output stream is the
 * element-wise interleave of the two planes, one vector shuffle per 128 bytes of
 * each plane. Needs the planes and the output rows contiguous. */
static int tr_flat_il(G *g, const T *xt, const T *o, const int32_t *perm) {
    if(!DM_ENABLED || perm[0]!=0 || perm[1]!=2 || perm[2]!=3 || perm[3]!=1 || !dm_flat(xt) || !dm_flat(o) || xt->esize!=o->esize || xt->d[1]!=2) return -1;
    const int32_t n[4]={xt->d[0],xt->d[2],xt->d[3],2};
    if(!dm_write_dims_match(o,n) || fl_empty(n)) return -1;
    const uint32_t es=xt->esize,k=es==1?0:es==2?1:2;
    if(xt->p[2] || xt->p[3] || xt->m[2]!=xt->d[2] || xt->m[3]!=xt->d[3] || o->p[2] || o->p[3] || o->m[3]!=2 || o->m[2]!=n[2]) return -1;
    int e=xfer_check(g,xt,xt->d,xt->p,0);
    if(!e)e=xfer_check(g,o,n,o->p,1);
    if(e) return e;
    if(!fl_apart(xt,xt->d,xt->p,o,n,o->p)) return -1;
    if(g->dry) return 0;
    DM_COUNT(15,count4(n));
    const size_t plane=(size_t)xt->d[2]*xt->d[3]*es;
    for(int32_t b=0;b<n[0];b++) {
        const uint8_t *s0=g->pool[xt->pool]+xt->offset+fl_lin(xt,(int32_t[4]){xt->p[0]+b,xt->p[1],0,0})*es;
        const uint8_t *s1=g->pool[xt->pool]+xt->offset+fl_lin(xt,(int32_t[4]){xt->p[0]+b,xt->p[1]+1,0,0})*es;
        uint8_t *d=g->pool[o->pool]+o->offset+fl_lin(o,(int32_t[4]){o->p[0]+b,o->p[1],0,0})*es;
        for(size_t at=0;at<plane;at+=128) {
            const uint32_t nb=plane-at<128?(uint32_t)(plane-at):128u;
            DV lo,hi;
            dv_shuffk(dv_ldn(s1+at,nb),dv_ldn(s0+at,nb),k,&lo,&hi);
            dv_stun(d+2*at,lo,2*nb<128?2*nb:128u);
            if(2*nb>128)dv_stun(d+2*at+128,hi,2*nb-128);
        }
    }
    return 0;
}

/* ------------------------------------------------------ flat u8 matrix transpose */
/* {0,1,3,2} of a flat u8 tensor: per (a,b) a R x C byte matrix with R a multiple
 * of 16. A tile is 16 rows x up to 128 columns: 16 row vectors, a 16x16 byte
 * transpose (dm_lad_t16), and each output vector then holds 8 output rows of
 * 16 bytes, stored as masked pieces. */
static int tr_flat_t8(G *g, const T *xt, const T *o, const int32_t *perm) {
    if(!DM_ENABLED || perm[0]!=0 || perm[1]!=1 || perm[2]!=3 || perm[3]!=2 || !dm_flat(xt) || !dm_flat(o) || xt->esize!=1 || o->esize!=1) return -1;
    const int32_t n[4]={xt->d[0],xt->d[1],xt->d[3],xt->d[2]};
    if(!dm_write_dims_match(o,n) || fl_empty(n) || (xt->d[2]&15)) return -1;
    int e=xfer_check(g,xt,xt->d,xt->p,0);
    if(!e)e=xfer_check(g,o,n,o->p,1);
    if(e) return e;
    if(!fl_apart(xt,xt->d,xt->p,o,n,o->p)) return -1;
    if(g->dry) return 0;
    DM_COUNT(14,count4(n));
    const int32_t R=xt->d[2],Cn=xt->d[3];
    for(int32_t a=0;a<n[0];a++)for(int32_t b=0;b<n[1];b++)for(int32_t r0=0;r0<R;r0+=16)for(int32_t c0=0;c0<Cn;c0+=128) {
        const uint32_t nb=Cn-c0<128?(uint32_t)(Cn-c0):128u;
        DV v[16];
        for(int i=0;i<16;i++)
            v[i]=dv_ldn(g->pool[xt->pool]+xt->offset+fl_lin(xt,(int32_t[4]){xt->p[0]+a,xt->p[1]+b,xt->p[2]+r0+i,xt->p[3]+c0}),nb);
        dv_ladder(v,16,dm_lad_t16,4);
        for(int32_t c=0;c<(int32_t)nb;c++)
            dv_stup(g->pool[o->pool]+o->offset+fl_lin(o,(int32_t[4]){o->p[0]+a,o->p[1]+b,o->p[2]+c0+c,o->p[3]+r0}),v[c>>3],16u*(uint32_t)(c&7),16);
    }
    return 0;
}

/* The fast transposes in the order the kernel tries them; -1 when none applies. */
static int tr_fast_dispatch(G *g, const T *xt, const T *o, const int32_t *pm) {
    int fr=-1;
    if(pm[0]==0 && pm[1]==2 && pm[2]==1 && pm[3]==3) {
        fr=tr_hw_fast(g,xt,o);
        if(fr<0)fr=tr_hw8_fast(g,xt,o);
    } else if(pm[0]==0 && pm[1]==1 && pm[2]==3 && pm[3]==2) {
        fr=tr_wc16_fast(g,xt,o);
        if(fr<0)fr=tr_wc8_fast(g,xt,o);
        if(fr<0)fr=tr_flat_t8(g,xt,o,pm);
    } else if(pm[0]==0 && pm[1]==2 && pm[2]==3 && pm[3]==1) fr=tr_flat_il(g,xt,o,pm);
    if(fr<0)fr=tr_flat_fast(g,xt,o,pm);
    return fr;
}
