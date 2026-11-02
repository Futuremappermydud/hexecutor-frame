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
#endif

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
static void s2d_fast_run(const G *g, const T *i, const T *o, int32_t ph, int32_t pw, int32_t oh, int32_t ow, int32_t ci) {
    BK bi,bo;
    bk_init(&bi,g,i);bk_init(&bo,g,o);
    const int64_t s=(pw-7)/8,ncb=ci/32,nk=ow/4;
    for(int64_t cb=0;cb<ncb;cb++)for(int32_t h=0;h<oh;h++)for(int dy=0;dy<2;dy++) {
        const int64_t H=(int64_t)ph+2*h+dy,j0=cb*4+2*dy;
        const uint8_t *r=bk_at(&bi,0,H,s,cb);
        DV A=r?dv_ld(r+128):dv_zero();
        const uint8_t *nr=bk_at(&bi,0,H,s+1,cb);
        DV nb=dv_ldz(nr),nc=dv_ldz(nr?nr+128:NULL);
        for(int64_t k=0;k<nk;k++) {
            DV B=nb,C=nc;
            if(k+1<nk) {
                nr=bk_at(&bi,0,H,k+2+s,cb);
                nb=dv_ldz(nr);nc=dv_ldz(nr?nr+128:NULL);
            }
            DV E,O;
            dv_deal32(dv_align(B,A,64),dv_align(C,B,64),&E,&O);
            uint8_t *d1=bk_at(&bo,0,h,k>>1,j0+1),*d0=bk_at(&bo,0,h,k>>1,j0);
            if(d1)dv_st(d1+128*(k&1),E);
            if(d0)dv_st(d0+128*(k&1),O);
            A=C;
        }
    }
}
/* Whole space-to-depth: -1 when not taken, else 0 or the error code the
 * generic path's checks would give. */
static int s2d_fast(G *g, const T *i, const T *o, int32_t ph, int32_t pw, int32_t oh, int32_t ow, int32_t ci, int32_t oc) {
    if(!s2d_fast_ok(g,i,o,ph,pw,oh,ow,ci,oc)) return -1;
    int32_t rd[4]={1,2*oh,2*ow,ci},ro[4]={0,ph,pw-1,0},wd[4]={1,oh,ow,oc},zero[4]={0,0,0,0};
    uint32_t ka[DM_TOUCH_CAP],kb[DM_TOUCH_CAP],na,nb;
    int e=xfer_check(g,i,rd,ro,0);
    if(!e)e=xfer_check(g,o,wd,zero,1);
    if(e) return e;
    if(!dm_touch(i,rd,ro,ka,&na) || !dm_touch(o,wd,zero,kb,&nb) || !dm_disjoint(ka,na,kb,nb)) return -1;
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
static int resize_fast(G *g, const T *xt, const T *wpt, const T *o, const int32_t *hp) {
    if(!resize_fast_ok(g,xt,wpt,o)) return -1;
    int32_t n[4]={xt->d[0],o->d[1],o->d[2],xt->d[3]};
    int e=xfer_check(g,xt,xt->d,xt->p,0);
    if(!e)e=xfer_check(g,o,n,o->p,1);
    if(e) return e;
    uint32_t ka[DM_TOUCH_CAP],kb[DM_TOUCH_CAP],na,nb;
    if(!dm_touch(xt,xt->d,xt->p,ka,&na) || !dm_touch(o,n,o->p,kb,&nb) || !dm_disjoint(ka,na,kb,nb)) return -1;
    if(g->dry) return 0;
    int32_t hi[RS_HMAX];
    for(int32_t i=0;i<n[1];i++) {
        int64_t v=(int64_t)((((uint64_t)i*(uint32_t)hp[2])+(uint32_t)hp[4])>>32)+hp[5];
        if(v<0)v+=xt->d[1];
        CHECK(v>=0 && v<xt->d[1],E_RANGE);hi[i]=(int32_t)v;
    }
    DM_COUNT(1,(uint64_t)n[0]*n[1]*n[2]*n[3]);
    BK bx,bo;
    bk_init(&bx,g,xt);bk_init(&bo,g,o);
    const int64_t ng=n[3]/32,nq=n[2]/8,xw=xt->p[2]/4,ow=o->p[2]/8;
    for(int32_t b=0;b<n[0];b++)for(int32_t h=0;h<n[1];h++)for(int64_t z=0;z<ng;z++) {
        const int64_t H=(int64_t)hi[h]+xt->p[1],B=b+(int64_t)xt->p[0],OB=b+(int64_t)o->p[0],OH=h+(int64_t)o->p[1];
        const uint8_t *s=bk_at(&bx,B,H,(xw)>>1,z+xt->p[3]/32);
        DV cur=dv_ldz(s?s+128*(xw&1):NULL);
        for(int64_t q=0;q<nq;q++) {
            DV v=cur;
            if(q+1<nq) {
                s=bk_at(&bx,B,H,(xw+q+1)>>1,z+xt->p[3]/32);
                cur=dv_ldz(s?s+128*((xw+q+1)&1):NULL);
            }
            DV lo,hi2;
            dv_dup32(v,&lo,&hi2);
            uint8_t *d=bk_at(&bo,OB,OH,ow+q,z+o->p[3]/32);
            if(d){dv_st(d,lo);dv_st(d+128,hi2);}
        }
    }
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
static int tr_hw_fast(G *g, const T *xt, const T *o) {
    if(!tr_hw_fast_ok(g,xt,o)) return -1;
    int32_t n[4]={xt->d[0],xt->d[2],xt->d[1],xt->d[3]};
    int e=xfer_check(g,xt,xt->d,xt->p,0);
    if(!e)e=xfer_check(g,o,n,o->p,1);
    if(e) return e;
    uint32_t ka[DM_TOUCH_CAP],kb[DM_TOUCH_CAP],na,nb;
    if(!dm_touch(xt,xt->d,xt->p,ka,&na) || !dm_touch(o,n,o->p,kb,&nb) || !dm_disjoint(ka,na,kb,nb)) return -1;
    if(g->dry) return 0;
    DM_COUNT(2,(uint64_t)n[0]*n[1]*n[2]*n[3]);
    BK bx,bo;
    bk_init(&bx,g,xt);bk_init(&bo,g,o);
    const int32_t H=xt->d[1],W=xt->d[2];
    for(int32_t b=0;b<n[0];b++)for(int64_t z=0;z<n[3]/32;z++)for(int32_t h=0;h<H;h+=2)for(int32_t w=0;w<W;w+=2) {
        const int64_t AH=(int64_t)h+xt->p[1],AW=(int64_t)w+xt->p[2];
        const uint8_t *s=bk_at(&bx,b+(int64_t)xt->p[0],AH,AW>>2,z+xt->p[3]/32);
        DV A=dv_zero(),B=A;
        if(s){A=dv_ld(s+128*((AW>>1)&1));B=dv_ld(s+256+128*((AW>>1)&1));}
        const int64_t OW=(int64_t)h+o->p[2],OB=b+(int64_t)o->p[0];
        uint8_t *d0=bk_at(&bo,OB,(int64_t)w+o->p[1],OW>>2,z+o->p[3]/32);
        uint8_t *d1=bk_at(&bo,OB,(int64_t)w+1+o->p[1],OW>>2,z+o->p[3]/32);
        if(d0)dv_st(d0+128*((OW>>1)&1),dv_shuffe16(B,A));
        if(d1)dv_st(d1+128*((OW>>1)&1),dv_shuffo16(B,A));
    }
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
