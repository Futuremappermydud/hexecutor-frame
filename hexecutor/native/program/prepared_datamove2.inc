/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Exact HVX fast paths, part 2: DepthToSpace (blocked u16 -> flat), SlicePad
 * (blocked u8 channel slice), Gather (flat rows), and the blocked u16 ->
 * flat f32 dequantize driver. Included by prepared_reference.c right after
 * prepared_datamove.inc (uses its DV, BK and fl_* helpers); not a standalone
 * unit. Same rules as there: the choice of path depends only on tensor
 * geometry and immutable constants (dry and wet agree), the range checks are
 * the ones the generic path makes, and a source/destination overlap the path
 * cannot order keeps the generic code. */

/* ------------------------------------------------------------- DV helpers */
#if BLK_HVX
static inline DV dv2_and(DV a, DV b) {return Q6_V_vand_VV(a,b);}
static inline DV dv2_not(DV a) {return Q6_V_vnot_V(a);}
/* Bytes whose position within their 32-byte lane is < k (0..32) are 0xff, the rest 0. */
static inline DV dv2_lanelt(uint32_t k) {
    HVX_Vector m=Q6_V_vand_QR(Q6_Q_vsetq_R((int32_t)k),0xffffffffu);
    return Q6_V_vor_VV(Q6_V_vor_VV(m,Q6_V_vror_VR(m,96)),Q6_V_vor_VV(Q6_V_vror_VR(m,64),Q6_V_vror_VR(m,32)));
}
#else
static inline DV dv2_and(DV a, DV b) {DV r;for(uint32_t i=0;i<128;i++)r.b[i]=a.b[i]&b.b[i];return r;}
static inline DV dv2_not(DV a) {DV r;for(uint32_t i=0;i<128;i++)r.b[i]=(uint8_t)~a.b[i];return r;}
static inline DV dv2_lanelt(uint32_t k) {DV r;for(uint32_t i=0;i<128;i++)r.b[i]=(i&31u)<k?0xff:0;return r;}
#endif

/* blocked_check and dm_extent in one pass over the blocks the region touches
 * (shifts instead of the 64-bit divisions by a variable block width): the same
 * range checks, and the per-pool extent of the present blocks in *x. */
static int dm2_scan(G *g, const T *t, const int32_t *dims, const int32_t *origin, int store, DMExt *x) {
    for(int i=0;i<8;i++){x->lo[i]=UINT32_MAX;x->hi[i]=0;}
    for(int i=0;i<4;i++)CHECK(origin[i]>=0 && dims[i]>=0,E_RANGE);
    if(!dims[0] || !dims[1] || !dims[2] || !dims[3]) return 0;
    CHECK(t->esize==1 || t->esize==2 || t->esize==4,E_TENSOR);
    const int cs=t->esize==1?3:t->esize==2?2:1;
    const int64_t nbh=((int64_t)t->m[1]+7)>>3,nbw=((int64_t)t->m[2]+(1<<cs)-1)>>cs,nbd=((int64_t)t->m[3]+31)>>5;
    const int64_t h1=((int64_t)origin[1]+dims[1]-1)>>3,w1=((int64_t)origin[2]+dims[2]-1)>>cs,z1=((int64_t)origin[3]+dims[3]-1)>>5;
    for(int64_t b=origin[0];b<(int64_t)origin[0]+dims[0];b++)
     for(int64_t h=origin[1]>>3;h<=h1;h++)
      for(int64_t w=origin[2]>>cs;w<=w1;w++) {
       const int64_t row=((b*nbh+h)*nbw+w)*nbd+t->slice;
       for(int64_t z=origin[3]>>5;z<=z1;z++) {
        const int64_t index=row+z;
        if(index<0 || index>=(int64_t)t->nblocks) continue;
        const uint32_t e=t->blocks[index],pool=e&127u;
        if(!pool) continue;
        CHECK(pool<7 && g->pool[pool],E_RANGE);
        const uint64_t limit=store?g->write_limit[pool]:g->pool_bytes[pool];
        const uint32_t off=e&~127u;
        CHECK((uint64_t)off+BLOCK_BYTES<=limit,E_RANGE);
        if(off<x->lo[pool])x->lo[pool]=off;
        if(off>x->hi[pool])x->hi[pool]=off;
       }
      }
    return 0;
}
/* The blocked extent x and the direct region dt share no byte (the extent test
 * strip_disjoint uses for a mixed pair). */
static int dm2_mixed_clear(const DMExt *x, const T *dt) {
    uint64_t l=((((uint64_t)(dt->p[0]+dt->d[0]-1)*dt->m[1]+dt->p[1]+dt->d[1]-1)*dt->m[2]+dt->p[2]+dt->d[2]-1)*dt->m[3]+dt->p[3]+dt->d[3]-1);
    uint64_t f=((((uint64_t)dt->p[0]*dt->m[1]+dt->p[1])*dt->m[2]+dt->p[2])*dt->m[3]+dt->p[3]);
    uint64_t d0=dt->offset+f*dt->esize,d1=dt->offset+(l+1)*dt->esize;
    uint32_t pl=dt->pool;
    return pl>=8 || x->lo[pl]>x->hi[pl] || (uint64_t)x->hi[pl]+BLOCK_BYTES<=d0 || d1<=x->lo[pl];
}

/* ------------------------------------------------------------ DepthToSpace */
/* u16 blocked [B,H,W,32] -> flat u16 [B,4H,4W,2] (block 4x4, 2 channels):
 * output (4h+i, 4w+j, c) is input (h, w, 8i+2j+c). An input block row is 4
 * pixels x 4 groups (i) of 16 bytes, an output row segment of 4 pixels is the
 * 4 j-units of one i, so across two block columns (32 sixteen-byte units, unit
 * n = 4m+i) output row i is the i-th of 4 deals of the 16-byte unit stream:
 * two stages of vdeal. Each pair of block columns yields 4 whole output
 * vectors (one per output row). */
static int d2s_fast(G *g, const T *xt, const T *o, int64_t rh, int64_t rw) {
    if(!DM_ENABLED || rh!=4 || rw!=4) return -1;
    if(!(xt->flags&F_BLOCKED) || !(xt->flags&F_SHAPE) || xt->esize!=2 || xt->dtype==DT_F32) return -1;
    if(!(o->flags&F_DIRECT) || !(o->flags&F_SHAPE) || o->esize!=2 || o->dtype==DT_F32) return -1;
    for(int k=0;k<4;k++)if(xt->d[k]<=0 || xt->p[k]<0 || o->p[k]<0) return -1;
    if(xt->d[3]!=32 || (xt->p[3]&31) || (xt->p[2]&3) || (xt->d[2]&7)) return -1;
    if(o->d[0]!=xt->d[0] || o->d[1]!=xt->d[1]*4 || o->d[2]!=xt->d[2]*4 || o->d[3]!=2 || o->m[3]!=2 || o->p[3]!=0) return -1;
    if(!dm_pools_aligned(g)) return -1;
    DMExt ext;
    int e=dm2_scan(g,xt,xt->d,xt->p,0,&ext);
    if(!e)e=xfer_check(g,o,o->d,o->p,1);
    if(e) return e;
    if(!dm2_mixed_clear(&ext,o)) return -1;
    if(g->dry) return 0;
    uint8_t *ob=g->pool[o->pool]+o->offset+fl_lin(o,o->p)*2;
    const size_t rowb=(size_t)o->m[2]*4;
    if(((uintptr_t)ob&127u) || (rowb&127u)) return -1;
    BK bx;
    bk_init(&bx,g,xt);
    const int64_t cb0=xt->p[2]/4,z=xt->p[3]/32;
    const int32_t npair=xt->d[2]/8;
    for(int32_t b=0;b<xt->d[0];b++)for(int32_t h=0;h<xt->d[1];h++) {
        const int64_t B=xt->p[0]+b,Hi=(int64_t)xt->p[1]+h;
        uint8_t *dst=ob+((size_t)b*o->m[1]*o->m[2]+(size_t)4*h*o->m[2])*4;
        for(int32_t k=0;k<npair;k++) {
            const uint8_t *a=bk_at(&bx,B,Hi,cb0+2*k,z),*c=bk_at(&bx,B,Hi,cb0+2*k+1,z);
            DV v0=a?dv_ld(a):dv_zero(),v1=a?dv_ld(a+128):dv_zero();
            DV v2=c?dv_ld(c):dv_zero(),v3=c?dv_ld(c+128):dv_zero();
            v0=dv_dealh(v0);v1=dv_dealh(v1);v2=dv_dealh(v2);v3=dv_dealh(v3); /* u16 pixel pairs are channel-interleaved */
            DV e0,o0,e1,o1,r0,r1,r2,r3;
            dv_dealk(v1,v0,4,&e0,&o0);
            dv_dealk(v3,v2,4,&e1,&o1);
            dv_dealk(e1,e0,4,&r0,&r2);
            dv_dealk(o1,o0,4,&r1,&r3);
            uint8_t *d=dst+(size_t)k*128;
            dv_st(d,r0);dv_st(d+rowb,r1);dv_st(d+2*rowb,r2);dv_st(d+3*rowb,r3);
        }
    }
    DM_COUNT(12,count4(xt->d));
    return 0;
}

/* ------------------------------------------------- SlicePad (blocked u8) */
/* A channel slice of a blocked u8 tensor (no pad, whole spatial extent, the
 * window inside the source so the fill is unused): output group z is source
 * channels a0 + 32z .. +32, where a0 is the absolute start channel. With
 * r = a0 mod 32 that is the high 32-r bytes of each 32-byte pixel lane of
 * source group A = gi0+z followed by the low r bytes of group B = A+1, lanes
 * past the slice width cleared (the generic write zero-pads to the crouton
 * boundary). One cell (batch x block row) is staged whole in the near buffer
 * before any output block is written, which is the generic read-all-then-write
 * order when the output blocks overlap the source; with several cells the
 * regions must be disjoint instead. */
static int slice_pad_blk(G *g, const T *xt, const T *o, const T *pad, const T *st, const T *sz, const T *ft) {
    if(!DM_ENABLED || !(xt->flags&F_BLOCKED) || !(xt->flags&F_SHAPE) || !(o->flags&F_BLOCKED) || !(o->flags&F_SHAPE)) return -1;
    if(xt->esize!=1 || o->esize!=1 || xt->dtype!=DT_U8 || o->dtype!=DT_U8 || !(ft->flags&F_SHAPE) || ft->dtype!=DT_U8) return -1;
    if(!(pad->flags&F_SV) || !(st->flags&F_SV) || !(sz->flags&F_SV)) return -1;
    int32_t lo[4];
    for(int k=0;k<4;k++){if(sz->sv[k]<0 || xt->d[k]<=0 || xt->p[k]<0 || o->p[k]<0) return -1;lo[k]=st->sv[k]-pad->sv[k];}
    if(lo[0]||lo[1]||lo[2]||lo[3]<0) return -1;
    for(int k=0;k<3;k++)if(sz->sv[k]!=xt->d[k] || o->d[k]!=xt->d[k]) return -1;
    const int32_t sz3=sz->sv[3];
    if(sz3<=0 || o->d[3]!=sz3 || (int64_t)lo[3]+sz3>xt->d[3]) return -1;
    int32_t dims3=o->d[3];
    if(o->crouton){dims3=(o->d[3]+o->p[3]+31)/32*32;if(dims3>o->m[3])dims3=o->m[3];dims3-=o->p[3];}
    if(dims3<sz3 || (dims3&31) || (o->p[3]&31)) return -1;
    if((xt->p[1]&7) || (o->p[1]&7) || (xt->d[1]&7) || (xt->p[2]&7) || (o->p[2]&7) || (xt->d[2]&7)) return -1;
    if(!dm_pools_aligned(g)) return -1;
    const int32_t dims[4]={xt->d[0],xt->d[1],xt->d[2],dims3};
    DMExt ext;
    int e=dm2_scan(g,xt,xt->d,xt->p,0,&ext);
    if(!e)e=dm2_scan(g,o,dims,o->p,1,&ext);
    if(!e) { /* what reading the fill scalar checks */
        if(count4(ft->d)==0) FAIL(E_TENSOR);
        e=xfer_check(g,ft,ft->d,ft->p,0);
    }
    if(e) return e;
    const int32_t nz=dims3/32,ncol=xt->d[2]/8,nhb=xt->d[1]/8;
    const size_t cell=(size_t)nz*ncol*BLOCK_BYTES;
    uint8_t *stage=NULL;size_t room=0;
    const int64_t cells=(int64_t)xt->d[0]*nhb;
    if(cells==1) {
        stage=strip_buffer(g,&room);
        if(!stage || room<cell) return -1;
        if((uintptr_t)stage&127u) return -1;
    } else if(!dm_clear(xt,xt->d,xt->p,o,dims,o->p)) return -1;
    if(g->dry) return 0;
    BK bx,bo;
    bk_init(&bx,g,xt);bk_init(&bo,g,o);
    const int64_t a0=(int64_t)xt->p[3]+lo[3],gi0=a0>>5,nbd_in=((int64_t)xt->m[3]+31)/32;
    const uint32_t r=(uint32_t)(a0&31);
    const DV mlow=dv2_lanelt(32-r),mhigh=dv2_not(mlow);
    for(int32_t b=0;b<xt->d[0];b++)for(int32_t hb=0;hb<nhb;hb++) {
        const int64_t B=xt->p[0]+b,IHb=xt->p[1]/8+hb,OB=o->p[0]+b,OHb=o->p[1]/8+hb;
        for(int phase=0;phase<(stage?2:1);phase++)
         for(int32_t z=0;z<nz;z++)for(int32_t col=0;col<ncol;col++) {
            uint8_t *ob=bk_blk(&bo,OB,OHb,o->p[2]/8+col,z+o->p[3]/32);
            if(!ob) continue;
            uint8_t *sp=stage?stage+((size_t)z*ncol+col)*BLOCK_BYTES:NULL;
            if(phase) { /* copy staged block to its output block */
                for(int k=0;k<16;k++)dv_st(ob+128*k,dv_ld(sp+128*k));
                continue;
            }
            uint8_t *dst=sp?sp:ob;
            const int64_t ga=gi0+z,gb=gi0+z+1;
            const uint8_t *ba=ga<nbd_in?bk_blk(&bx,B,IHb,xt->p[2]/8+col,ga):NULL;
            const uint8_t *bb=(r && gb<nbd_in)?bk_blk(&bx,B,IHb,xt->p[2]/8+col,gb):NULL;
            const int32_t kv=sz3-32*z;
            const DV mv=kv>=32?dv2_lanelt(32):dv2_lanelt(kv>0?(uint32_t)kv:0u);
            for(int k=0;k<16;k++) {
                DV A=ba?dv_ld(ba+128*k):dv_zero(),res=A;
                if(r) {
                    DV Bv=bb?dv_ld(bb+128*k):dv_zero();
                    res=dv_or(dv2_and(dv_ror(A,r),mlow),dv2_and(dv_ror(Bv,96+r),mhigh));
                }
                if(kv<32)res=dv2_and(res,mv);
                dv_st(dst+128*k,res);
            }
        }
    }
    DM_COUNT(13,(uint64_t)xt->d[0]*xt->d[1]*xt->d[2]*(uint64_t)sz3);
    return 0;
}

/* ------------------------------------------------------------ Gather (flat) */
/* gather_fast with per-row address arithmetic hoisted and rows of up to 512
 * bytes moved by vectors (unaligned load, masked store). */
static void gt_row(uint8_t *d, const uint8_t *s, size_t n) {
    if(n>=512){qd_vcopy(d,s,n);return;}
    size_t k=0;
    for(;k+128<=n;k+=128)dv_stup(d+k,dv_ldu(s+k),0,128);
    if(k<n)dv_stup(d+k,dv_ldn(s+k,(uint32_t)(n-k)),0,(uint32_t)(n-k));
}
static int gather_fast2(G *g, const T *xt, const T *axt, const T *o, const int32_t *idx, size_t ni) {
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
    const int64_t dn=xt->d[axis];
    for(int32_t i0=0;i0<(axis>0?xt->d[0]:1);i0++)for(int32_t i1=0;i1<(axis>1?xt->d[1]:1);i1++)for(int32_t i2=0;i2<(axis>2?xt->d[2]:1);i2++) {
        int32_t so[4]={xt->p[0]+(axis>0?i0:0),xt->p[1]+(axis>1?i1:0),xt->p[2]+(axis>2?i2:0),xt->p[3]};
        int32_t dorg[4]={o->p[0]+(axis>0?i0:0),o->p[1]+(axis>1?i1:0),o->p[2]+(axis>2?i2:0),o->p[3]};
        so[axis]=xt->p[axis];dorg[axis]=o->p[axis];
        const uint8_t *s0=sb+fl_lin(xt,so)*xt->esize;
        uint8_t *d0=db+fl_lin(o,dorg)*o->esize;
        for(size_t j=0;j<ni;j++) {
            int64_t k=idx[j];
            if(k<0)k+=dn;
            gt_row(d0+j*inner,s0+(size_t)k*inner,inner);
        }
    }
    DM_COUNT(14,count4(n));
    return 0;
}

/* ---------------------------------------------- Dequantize (blocked u16) */
/* Blocked u16 -> flat f32, same shape: a block row is 4 pixels x 32 channels,
 * so each vector holds two pixels' channel groups (interleaved by halfword: vdeal), which vunpack widens to one
 * 32-lane vector per pixel, dq_hvx32 converts, and one whole output vector
 * (32 f32 = one pixel's channel group) is stored. No strip staging. Items are
 * (batch, row, block column); a rejected lane (never seen in the models) sends
 * the whole call to the strip path. */
#if BLK_HVX
typedef struct { const G *g; const T *it, *ot; const DqCtx *c; int64_t items; int32_t nw, nz; uint8_t *ob; size_t pxs; volatile uint32_t bad[8]; } Dq2Job;
static void dq2_part(void *ctx, uint32_t part, uint32_t parts) {
    Dq2Job *J=ctx;
    const T *it=J->it,*ot=J->ot;const DqCtx *c=J->c;
    BK bi;
    bk_init(&bi,J->g,it);
    int64_t a,b;
    dm_range(J->items,part,parts,&a,&b);
    const uint32_t mh=c->m24>>12,ml=c->m24&4095u;
    const HVX_Vector vzp=Q6_V_vsplat_R(c->zp),vex0=Q6_V_vsplat_R(c->e+126);
    HVX_Vector acc=Q6_V_vzero();
    const int64_t col0=it->p[2]/4,z0=it->p[3]/32;
    for(int64_t k=a;k<b;k++) {
        const int32_t cb=(int32_t)(k%J->nw);
        const int64_t q=k/J->nw;
        const int32_t h=(int32_t)(q%it->d[1]),bb=(int32_t)(q/it->d[1]);
        const int64_t B=it->p[0]+bb,Hi=(int64_t)it->p[1]+h;
        uint8_t *od=J->ob+(size_t)(((size_t)(ot->p[0]+bb)*ot->m[1]+(size_t)(ot->p[1]+h))*ot->m[2]+(size_t)(ot->p[2]+4*cb))*ot->m[3]*4+(size_t)ot->p[3]*4;
        for(int32_t z=0;z<J->nz;z++) {
            const uint8_t *blk=bk_at(&bi,B,Hi,col0+cb,z0+z);
            for(int q2=0;q2<2;q2++) {
                HVX_Vector v=blk?*(const HVX_Vector *)(blk+128*q2):Q6_V_vzero();
                HVX_VectorPair w=Q6_Wuw_vunpack_Vuh(Q6_Vh_vdeal_Vh(v)); /* pixel pair is channel-interleaved */
                HVX_VectorPred b0,b1;
                HVX_Vector r0=dq_hvx32(Q6_V_lo_W(w),vzp,mh|mh<<16,ml|ml<<16,vex0,&b0);
                HVX_Vector r1=dq_hvx32(Q6_V_hi_W(w),vzp,mh|mh<<16,ml|ml<<16,vex0,&b1);
                acc=Q6_V_vor_VV(acc,Q6_V_vand_QR(Q6_Q_or_QQ(b0,b1),0x01010101u));
                uint8_t *d=od+(size_t)(2*q2)*J->pxs+128*(size_t)z;
                *(HVX_Vector *)d=r0;
                *(HVX_Vector *)(d+J->pxs)=r1;
            }
        }
    }
    HVX_Vector flag __attribute__((aligned(128)))=acc;
    const uint32_t *fw=(const uint32_t *)&flag;uint32_t any=0;
    for(int k2=0;k2<32;k2++) any|=fw[k2];
    J->bad[part&7]=any;
}
#endif
static int dq_blk_fast(G *g, const T *it, const T *ot, const DqCtx *c) {
#if BLK_HVX
    if(!DM_ENABLED || !c->fast || c->in_es!=2 || it->esize!=2 || it->dtype==DT_F32) return -1;
    if(!(it->flags&F_BLOCKED) || !(it->flags&F_SHAPE) || !(ot->flags&F_DIRECT) || !(ot->flags&F_SHAPE) || ot->esize!=4 || ot->dtype!=DT_F32) return -1;
    for(int k=0;k<4;k++)if(it->d[k]!=ot->d[k] || it->d[k]<=0 || it->p[k]<0 || ot->p[k]<0) return -1;
    if((it->d[3]&31) || (it->p[3]&31) || (it->p[2]&3) || (it->d[2]&3) || (ot->m[3]&31) || (ot->p[3]&31)) return -1;
    if(!dm_pools_aligned(g)) return -1;
    DMExt ext;
    int e=dm2_scan(g,it,it->d,it->p,0,&ext);
    if(!e)e=xfer_check(g,ot,ot->d,ot->p,1);
    if(e) return e;
    if(!dm2_mixed_clear(&ext,ot)) return -1;
    size_t room;
    if(!strip_buffer(g,&room)) return -1; /* the strip path's scratch, for the rare fallback */
    if(g->dry) return 0;
    uint8_t *ob=g->pool[ot->pool]+ot->offset;
    if((uintptr_t)ob&127u) return -1;
    Dq2Job J={g,it,ot,c,0,it->d[2]/4,it->d[3]/32,ob,(size_t)ot->m[3]*4,{0}};
    J.items=(int64_t)it->d[0]*it->d[1]*J.nw;
    const uint64_t elems=(uint64_t)it->d[0]*it->d[1]*it->d[2]*(uint64_t)it->d[3];
    uint32_t parts=dm_parts(g,elems,J.items);
    if(parts>1) nprg2_parallel_for(g->call->parallel,NPRG2_PAR_DM,parts,dq2_part,&J);
    else dq2_part(&J,0,1);
    for(int k=0;k<8;k++)if(J.bad[k]) return -1;
    DM_COUNT(15,elems);
    return 0;
#else
    (void)g;(void)it;(void)ot;(void)c;
    return -1;
#endif
}


/* -------------------------------------------------------------- Decimate */
/* u8 blocked in and out, column stride 2, depth stride 1, any batch/row
 * stride. Output pixel (b,h,w) is input pixel (b*sv0, h*sv1, 2w). One item =
 * (batch, depth group, output block row, output block column): an output
 * block row (8 pixels, two vectors) is the even pixels of two input block
 * rows' 16-pixel run (4 vectors, vdeal). The two input blocks and the output
 * block are looked up once per item (input again only when the row map moves
 * to another input block row). */
static int dec_fast_ok(const G *g, const T *it, const T *ot, const int32_t *sv, int32_t *n) {
    if(!DM_ENABLED || !dm_blocked_u8(it) || !dm_blocked_u8(ot) || !dm_pools_aligned(g)) return 0;
    if(sv[0]<1 || sv[1]<1 || sv[2]!=2 || sv[3]!=1) return 0;
    for(int i=0;i<4;i++){n[i]=(it->d[i]+sv[i]-1)/sv[i];if(it->d[i]<=0 || n[i]!=ot->d[i]) return 0;}
    if((n[3]&31) || (it->p[3]&31) || (ot->p[3]&31) || (n[2]&7) || (it->p[2]&7) || (ot->p[2]&7)) return 0;
    return dm_write_dims_match(ot,n);
}
typedef struct { const G *g; const T *it, *ot; const int32_t *sv; int32_t n[4]; int64_t nohb, ohb0; } DecJob;
static void dec_part(void *ctx, uint32_t part, uint32_t parts) {
    const DecJob *J=ctx;const T *it=J->it,*ot=J->ot;const int32_t *n=J->n,*sv=J->sv;
    BK bi,bo;
    bk_init(&bi,J->g,it);bk_init(&bo,J->g,ot);
    const int64_t ng=n[3]/32,nq=n[2]/8,nohb=J->nohb;
    const int64_t irow=bi.nbw*bi.nbd,orow=bo.nbw*bo.nbd;
    const int64_t zi=bi.slice+it->p[3]/32,zo=bo.slice+ot->p[3]/32;
    const int64_t ic0=it->p[2]/8,oc0=ot->p[2]/8;
    int64_t i0,i1;
    dm_range((int64_t)n[0]*ng*nohb*nq,part,parts,&i0,&i1);
    for(int64_t k=i0;k<i1;k++) {
        const int64_t q=k%nq,ohb=J->ohb0+(k/nq)%nohb,z=(k/nq/nohb)%ng,b=k/nq/nohb/ng;
        const int64_t B=it->p[0]+b*sv[0],OB=b+(int64_t)ot->p[0];
        int32_t h0=(int32_t)(8*ohb-ot->p[1]),h1=h0+8;
        if(h0<0)h0=0;
        if(h1>n[1])h1=n[1];
        uint8_t *d=bk_blk_or(&bo,(OB*bo.nbh+ohb)*orow+zo+z+(oc0+q)*bo.nbd,NULL);
        if(!d) continue;
        const int64_t icol=(ic0+2*q)*bi.nbd;
        int64_t cH=-1;const uint8_t *s0=NULL,*s1=NULL;
        for(int32_t h=h0;h<h1;h++) {
            const int64_t H=(int64_t)it->p[1]+(int64_t)h*sv[1];
            if((H>>3)!=cH) {
                cH=H>>3;
                const int64_t base=(B*bi.nbh+cH)*irow+zi+z+icol;
                s0=bk_blk_or(&bi,base,(uint8_t *)dm_zblk);
                s1=bk_blk_or(&bi,base+bi.nbd,(uint8_t *)dm_zblk);
            }
            const uint32_t ro=256u*(uint32_t)(H&7);
            DV a0=dv_ld(s0+ro),a1=dv_ld(s0+ro+128),b0=dv_ld(s1+ro),b1=dv_ld(s1+ro+128);
            DV e0,o0,e1,o1;
            dv_deal32(a0,a1,&e0,&o0);dv_deal32(b0,b1,&e1,&o1);
            const uint32_t oro=256u*(uint32_t)((h+(int64_t)ot->p[1])&7);
            dv_st(d+oro,e0);dv_st(d+oro+128,e1);
        }
    }
}
static int dec_fast(G *g, const T *it, const T *ot, const int32_t *sv) {
    int32_t n[4];
    if(!dec_fast_ok(g,it,ot,sv,n)) return -1;
    int e=xfer_check(g,it,it->d,it->p,0);
    if(!e)e=xfer_check(g,ot,n,ot->p,1);
    if(e) return e;
    if(!dm_clear(it,it->d,it->p,ot,n,ot->p)) return -1;
    if(g->dry) return 0;
    DecJob J={g,it,ot,sv,{n[0],n[1],n[2],n[3]},0,0};
    J.ohb0=ot->p[1]>>3;J.nohb=((ot->p[1]+n[1]-1)>>3)-J.ohb0+1;
    const int64_t items=(int64_t)n[0]*(n[3]/32)*J.nohb*(n[2]/8);
    uint32_t parts=dm_parts(g,(uint64_t)n[0]*n[1]*n[2]*n[3],items);
    if(parts>1) nprg2_parallel_for(g->call->parallel,NPRG2_PAR_DM,parts,dec_part,&J);
    else dec_part(&J,0,1);
    DM_COUNT(10,(uint64_t)n[0]*n[1]*n[2]*n[3]);
    return 0;
}
