/* Exact HMX u8 convolution for the fast conv path (included by
 * prepared_reference.c after near_compute; see EXECUTOR.md "HMX conv").
 *
 * Numerics are the v6 hybrid's (qd_hmx_kernel.h, 54/54 exact on the Frame):
 * u8 activations sit in the high byte of halfword lanes of the integer A
 * tile, s8 weights are weight.b tiles, and the HMX conversion with the per
 * channel word pair fi = {0x400000|(16-pre)<<10, off - round(pre)} yields
 *     x = clamp((acc + off) >> pre, 0, 65535)
 * which HVX finishes as min(255, (min(x,4096)*M + 8192) >> 14). With
 * post == 14 (every u8 conv: requant_params) that equals requant_value:
 * clip(((acc+off)>>pre)*M + 2^13 >> 14, 0, 255), off = bias - zp*wsum.
 *
 * Tiles. M: 32 output pixels, pixel P = y*ow + x over the command's oh x ow
 * region, so a quad of 4 consecutive P (ow % 4 == 0) is 4 consecutive x of
 * one row and one 128-byte load of a u8 block row (or two, across a block
 * edge). K: one 1024-byte weight block of 8 groups x 32 outputs x 4 inputs.
 * The weights of output block j are KB contiguous K blocks in the pool:
 *   unpacked (reversed 3x3/1x1):  k = (cb*H + h)*W + x, kx = reverse ? W-1-x : x
 *   packed 1x1 ("Xpb", I%32==0):  k = (h*W + x)*cbs + cb
 * so B needs no copy; A tiles follow the same k order. A, C, fi, the finish
 * record and the completion tile live in the (VTCM) workspace, M tiles in
 * batches that fit. Prep (A) and quant (C to the output blocks) run in parts
 * on the parallel hook; HMX runs on the calling thread, the HMX owner.
 *
 * Off Hexagon the same tiling, k order and addressing run with a scalar model:
 * A tiles hold the raw quads (32 pixels x 32 u8 per tile), C holds x as u16
 * [pixel][channel]; the host full-model test checks it against the emulator. */
#define HMX_TERMS_MAX 2304 /* largest K the v6 cases proved exact: 3x3x256 */
typedef struct {
    G *g; const T *xt, *o;
    int32_t oh, ow, KB, NB, total, m0, mt; /* tiles m0..m0+mt-1 of total */
    uint8_t *A, *C;
    const uint8_t *B;
    const uint32_t *ktab; /* ky | kx<<8 | cb<<16 per k */
    const int32_t *corr, *pre, *Mq;
    const uint8_t *multv; /* NB vectors: M of channel n in halfwords 2n, 2n+1 */
    uint8_t *const *xtab; /* input row pointers [r][wb-wb0][cb], or NULL: hmx_row per quad */
    int32_t xwb0, xnwb, cbs;
    /* Output row pointers [y][wb-owb0][j] of o, and of the epilogue's output
     * and other operand (same geometry), or NULL: hmx_row per quad. */
    uint8_t *const *otab, *const *etab, *const *btab;
    int32_t owb0, onwb;
    const Fuse *fx; /* fused epilogue (same geometry as o, proved by fuse_decode), or NULL */
    int64_t xnbw, xnbd, onbw, onbd;
} HmxJob;
/* Start of row h's 256 bytes in block column wb, channel block z of the u8
 * blocked tensor t (batch 0), or NULL (absent, or past the tensor's width). */
static inline uint8_t *hmx_row(uint8_t *const *pools, const T *t, int64_t h, int64_t wb, int64_t nbw, int64_t nbd,
                               int64_t z) {
    if(h<0 || wb<0 || wb>=nbw) return NULL;
    int64_t index=((h>>3)*nbw+wb)*nbd+t->slice+z;
    uint32_t e=(index>=0 && index<(int64_t)t->nblocks)?t->blocks[index]:0,pool=e&127;
    return pool && pools[pool]?pools[pool]+(e&~127u)+(size_t)(h&7)*256:NULL;
}
/* Row pointers tab[(r*nwb + w)*nz + z] = hmx_row(t, row0+r, wb0+w, z) for
 * r < nrows, w < nwb, z < nz, in 32-bit arithmetic (the tables are small). */
static void hmx_tab(uint8_t **tab, uint8_t *const *pools, const T *t, int32_t row0, int32_t nrows, int32_t wb0,
                    int32_t nwb, int32_t nz) {
    const int32_t nbw=(t->m[2]+7)/8,nbd=(t->m[3]+31)/32;
    const uint32_t nb=t->nblocks;
    for(int32_t r=0;r<nrows;r++) {
        int32_t h=row0+r;
        uint8_t **row=tab+(size_t)r*(size_t)nwb*(size_t)nz;
        for(int32_t w=0;w<nwb;w++) {
            int32_t wb=wb0+w;
            int ok=h>=0 && wb>=0 && wb<nbw;
            uint32_t idx=ok?(uint32_t)(((h>>3)*nbw+wb)*nbd)+t->slice:0;
            for(int32_t z=0;z<nz;z++) {
                uint32_t e=ok && idx+(uint32_t)z<nb?t->blocks[idx+(uint32_t)z]:0,pool=e&127;
                row[w*nz+z]=pool && pools[pool]?pools[pool]+(e&~127u)+(size_t)(h&7)*256:NULL;
            }
        }
    }
}
/* Output pixel P0 -> (y, x), without a hardware divide per quad. */
static inline void hmx_pix(int32_t P0, int32_t ow, int32_t *y, int32_t *x) {*y=P0/ow;*x=P0-*y*ow;}
#if BLK_HVX
/* Pixels col..col+3 of row, channels 32cb..32cb+31: 128 bytes, pixel-major. */
static inline HVX_Vector hmx_quad(uint8_t *const *pools, const T *t, int64_t row, int64_t col, int64_t cb,
                                  int64_t nbw, int64_t nbd) {
    int64_t wb=col>>3;int s=(int)(col&7);
    const uint8_t *p0=hmx_row(pools,t,row,wb,nbw,nbd,cb);
    if(s<=4) return p0?load_part(p0+32*s,128):Q6_V_vzero();
    uint32_t n1=(uint32_t)(8-s)*32;
    const uint8_t *p1=hmx_row(pools,t,row,wb+1,nbw,nbd,cb);
    HVX_Vector v1=p0?load_part(p0+32*s,n1):Q6_V_vzero(),v2=p1?load_part(p1,128-n1):Q6_V_vzero();
    return Q6_V_vmux_QVV(Q6_Q_vsetq_R((int32_t)n1),v1,Q6_V_vror_VR(v2,(int32_t)(128-n1)));
}
#endif
#if BLK_HVX
/* hmx_quad over a row pointer table (hmx_tab; r relative to its first row). */
static inline HVX_Vector hmx_quad_tab(uint8_t *const *tab, int32_t wb0, int32_t nwb, int32_t nz, int64_t r, int64_t col,
                                      int64_t z) {
    int64_t wb=(col>>3)-wb0;int s=(int)(col&7);
    uint8_t *const *row=tab+((size_t)r*(size_t)nwb+(size_t)wb)*(size_t)nz+(size_t)z;
    const uint8_t *p0=row[0];
    if(s<=4) return p0?load_part(p0+32*s,128):Q6_V_vzero();
    uint32_t n1=(uint32_t)(8-s)*32;
    const uint8_t *p1=row[nz];
    HVX_Vector v1=p0?load_part(p0+32*s,n1):Q6_V_vzero(),v2=p1?load_part(p1,128-n1):Q6_V_vzero();
    return Q6_V_vmux_QVV(Q6_Q_vsetq_R((int32_t)n1),v1,Q6_V_vror_VR(v2,(int32_t)(128-n1)));
}
#endif
/* A tile (local t, k): 8 quads, each 2 tile rows (hmx_prepare_x32). */
static void hmx_tile(const HmxJob *J, int32_t t, int32_t k) {
    G *g=J->g;const T *xt=J->xt;
    uint32_t e=J->ktab[k];
    int64_t ky=e&255,kx=(e>>8)&255,cb=e>>16;
    int32_t P=32*(J->m0+t),y,x,total=J->oh*J->ow;
    hmx_pix(P,J->ow,&y,&x);
    uint8_t *dst=J->A+((size_t)t*J->KB+(size_t)k)*2048;
    for(int q=0;q<8;q++,P+=4) {
        int valid=P<total;
        int64_t row=xt->p[1]+y+ky,col=xt->p[2]+x+kx;
#if BLK_HVX
        HVX_Vector v=!valid?Q6_V_vzero():J->xtab?hmx_quad_tab(J->xtab,J->xwb0,J->xnwb,J->cbs,y+ky,col,cb):hmx_quad(g->pool,xt,row,col,cb,J->xnbw,J->xnbd);
        HVX_VectorPair w=Q6_W_vshuff_VVR(v,Q6_V_vzero(),-1);
        ((HVX_Vector *)dst)[2*q]=Q6_Vh_vshuff_Vh(Q6_V_lo_W(w));
        ((HVX_Vector *)dst)[2*q+1]=Q6_Vh_vshuff_Vh(Q6_V_hi_W(w));
#else
        uint8_t *d=dst+128*q;
        for(int r=0;r<4;r++) {
            const uint8_t *s=!valid?NULL:J->xtab?J->xtab[((size_t)(y+ky)*(size_t)J->xnwb+(size_t)(((col+r)>>3)-J->xwb0))*
                                                         (size_t)J->cbs+(size_t)cb]:
                             hmx_row(g->pool,xt,row,(col+r)>>3,J->xnbw,J->xnbd,cb);
            if(s) memcpy(d+32*r,s+32*((col+r)&7),32); else memset(d+32*r,0,32);
        }
#endif
        if(valid && (x+=4)==J->ow){x=0;y++;}
    }
}
static void hmx_prep_part(void *ctx, uint32_t part, uint32_t parts) {
    const HmxJob *J=(const HmxJob *)ctx;
    uint32_t n=(uint32_t)J->mt*(uint32_t)J->KB,t=0,k=0;
    for(uint32_t i=0;i<n;i++) {
        if(i%parts==part) hmx_tile(J,(int32_t)t,(int32_t)k);
        if(++k==(uint32_t)J->KB){k=0;t++;}
    }
}
/* Quad (4 pixels x 32 channels of output block j, 128 bytes v) to pixels
 * (y, x..x+3) of the output blocks. */
#if BLK_HVX
static inline void hmx_put(const HmxJob *J, const T *o, uint8_t *const *tab, int32_t y, int32_t x, int32_t j,
                           HVX_Vector v) {
    int64_t oy=o->p[1]+y,ox=o->p[2]+x;int s=(int)(ox&7);
    uint32_t n1=(uint32_t)(s<=4?4:8-s)*32;
    uint8_t *const *row=tab?tab+((size_t)y*(size_t)J->onwb+(size_t)((ox>>3)-J->owb0))*(size_t)J->NB+(size_t)j:NULL;
    uint8_t *d=row?row[0]:hmx_row(J->g->pool,o,oy,ox>>3,J->onbw,J->onbd,j);
    if(d) store_part(d+32*s,v,0,n1);
    if(n1<128 && (d=row?row[J->NB]:hmx_row(J->g->pool,o,oy,(ox>>3)+1,J->onbw,J->onbd,j))!=NULL) store_part(d,v,n1,128-n1);
}
static HVX_Vector hmx_quant_pair(HVX_Vector pre, HVX_Vector mult) {
    pre=Q6_Vuh_vmin_VuhVuh(pre,Q6_V_vsplat_R(4096|(4096<<16)));
    HVX_VectorPair product=Q6_Ww_vmpy_VhVh(pre,mult);
    HVX_Vector a=Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_V_lo_W(product),Q6_V_vsplat_R(8192)),14);
    HVX_Vector b=Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_V_hi_W(product),Q6_V_vsplat_R(8192)),14);
    return Q6_Vh_vpack_VwVw_sat(b,a);
}
#endif
static void hmx_quant_tile(const HmxJob *J, int32_t t, int32_t j) {
    int32_t P=32*(J->m0+t),y,x,total=J->oh*J->ow;
    hmx_pix(P,J->ow,&y,&x);
    const uint8_t *src=J->C+((size_t)t*J->NB+(size_t)j)*2048;
#if BLK_HVX
    HVX_Vector mult=((const HVX_Vector *)J->multv)[j];
    for(int i=0;i<8 && P<total;i++,P+=4) {
        HVX_Vector a=hmx_quant_pair(((const HVX_Vector *)src)[2*i],mult),b=hmx_quant_pair(((const HVX_Vector *)src)[2*i+1],mult);
        HVX_Vector v=Q6_Vub_vpack_VhVh_sat(b,a);
        const Fuse *f=J->fx;
        if(!f) hmx_put(J,J->o,J->otab,y,x,j,v);
        else {
            /* e = min((q*mc + [b*mo] + rd) >> sh, 255) per byte, as rq_emit_fx. */
            if(!f->dead) hmx_put(J,J->o,J->otab,y,x,j,v);
            HVX_Vector rdv=Q6_V_vsplat_R((int32_t)f->rd),mcv=Q6_V_vsplat_R((int32_t)f->mc),
                       mov=Q6_V_vsplat_R((int32_t)f->mo),shv=Q6_V_vsplat_R((int32_t)f->sh),c255=Q6_V_vsplat_R(255);
            HVX_VectorPair w=Q6_Wuh_vunpack_Vub(v);
            HVX_VectorPair w0=Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(w)),w1=Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(w));
            HVX_Vector q[4]={Q6_V_lo_W(w0),Q6_V_hi_W(w0),Q6_V_lo_W(w1),Q6_V_hi_W(w1)},u[4],r[4];
            if(f->kind==2) {
                HVX_Vector bv=J->btab?hmx_quad_tab(J->btab,J->owb0,J->onwb,J->NB,y,(int64_t)f->ob.p[2]+x,j):
                              hmx_quad(J->g->pool,&f->ob,(int64_t)f->ob.p[1]+y,(int64_t)f->ob.p[2]+x,j,J->onbw,J->onbd);
                HVX_VectorPair z=Q6_Wuh_vunpack_Vub(bv);
                HVX_VectorPair z0=Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(z)),z1=Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(z));
                u[0]=Q6_V_lo_W(z0);u[1]=Q6_V_hi_W(z0);u[2]=Q6_V_lo_W(z1);u[3]=Q6_V_hi_W(z1);
            }
            for(int k=0;k<4;k++) {
                HVX_Vector t=Q6_Vw_vmpyieacc_VwVwVuh(rdv,q[k],mcv);
                if(f->kind==2) t=Q6_Vw_vmpyieacc_VwVwVuh(t,u[k],mov);
                r[k]=Q6_Vw_vmin_VwVw(Q6_Vw_vlsr_VwVw(t,shv),c255);
            }
            hmx_put(J,&f->oe,J->etab,y,x,j,rq_pack8(r[0],r[1],r[2],r[3]));
        }
        if((x+=4)==J->ow){x=0;y++;}
    }
#else
    const T *o=J->o;
    for(int i=0;i<8 && P<total;i++,P+=4) {
        for(int r=0;r<4;r++) {
            int64_t ox=o->p[2]+x+r;
            const Fuse *f=J->fx;
            size_t ti=((size_t)y*(size_t)J->onwb+(size_t)((ox>>3)-J->owb0))*(size_t)J->NB+(size_t)j;
            uint8_t *d=f && f->dead?NULL:J->otab?J->otab[ti]:hmx_row(J->g->pool,o,o->p[1]+y,ox>>3,J->onbw,J->onbd,j);
            uint8_t *de=!f?NULL:J->etab?J->etab[ti]:
                        hmx_row(J->g->pool,&f->oe,(int64_t)f->oe.p[1]+y,((int64_t)f->oe.p[2]+x+r)>>3,J->onbw,J->onbd,j);
            const uint8_t *db=!f || f->kind!=2?NULL:J->btab?J->btab[ti]:
                              hmx_row(J->g->pool,&f->ob,(int64_t)f->ob.p[1]+y,((int64_t)f->ob.p[2]+x+r)>>3,J->onbw,J->onbd,j);
            for(int n=0;n<32;n++) {
                uint16_t pre;memcpy(&pre,src+((size_t)(4*i+r)*32+(size_t)n)*2,2);
                uint32_t v=((uint32_t)(pre<4096?pre:4096)*(uint32_t)J->Mq[32*j+n]+8192u)>>14;
                v=v>255?255:v;
                if(d) d[32*(ox&7)+n]=(uint8_t)v;
                if(de) {
                    uint32_t t=v*f->mc+f->rd;
                    if(f->kind==2) t+=(uint32_t)(db?db[32*((f->ob.p[2]+x+r)&7)+n]:0)*f->mo;
                    t>>=f->sh;
                    de[32*((f->oe.p[2]+x+r)&7)+n]=(uint8_t)(t>255?255:t);
                }
            }
        }
        if((x+=4)==J->ow){x=0;y++;}
    }
#endif
}
static void hmx_quant_part(void *ctx, uint32_t part, uint32_t parts) {
    const HmxJob *J=(const HmxJob *)ctx;
    uint32_t n=(uint32_t)J->mt*(uint32_t)J->NB,t=0,j=0;
    for(uint32_t i=0;i<n;i++) {
        if(i%parts==part) hmx_quant_tile(J,(int32_t)t,(int32_t)j);
        if(++j==(uint32_t)J->NB){j=0;t++;}
    }
}
/* C = conversion(A x B) for the batch. fi/zero/completion: VTCM records as
 * hmx_calculate builds them. Returns 0, or nonzero on a completion timeout. */
static int hmx_mac(const HmxJob *J, int32_t *fi, int32_t *zero, uint8_t *completion) {
    const Nprg2HmxOps *h=J->g->conv->hmx;
#if BLK_HVX
    HVX_Vector sentinel=Q6_V_vsplat_R(-1);
    for(uint32_t i=0;i<2048;i+=128)*(HVX_Vector *)(completion+i)=sentinel;
    __asm__ volatile("barrier":::"memory");
    if(h->wide_n) for(int32_t j=0;j<J->NB;j++)
        h->wide_n(J->A,(void *)(J->B+(size_t)j*J->KB*1024),fi+j*64,J->C+(size_t)j*2048,
                  (uint32_t)J->KB|(uint32_t)J->NB<<16,(uint32_t)J->mt);
    else for(int32_t t=0;t<J->mt;t++)for(int32_t j=0;j<J->NB;j++)
        h->wide(J->A+(size_t)t*J->KB*2048,(void *)(J->B+(size_t)j*J->KB*1024),fi+j*64,
                J->C+((size_t)t*J->NB+(size_t)j)*2048,(uint32_t)J->KB);
    h->finish(zero,completion);
    uint64_t end=h->clock?h->clock()+1000000000ull:0;
    for(uint64_t spin=0;;spin++) {
        typedef uint32_t HmxWords __attribute__((vector_size(128)));
        HmxWords v=*(volatile HmxWords *)completion;
        if(v[0]==0) break;
        if(h->clock?h->clock()>end:spin>100000000ull) return 1;
    }
    __asm__ volatile("barrier":::"memory");
    return 0;
#else
    (void)h;(void)fi;(void)zero;(void)completion;
    for(int32_t t=0;t<J->mt;t++)for(int32_t j=0;j<J->NB;j++) {
        uint16_t *c=(uint16_t *)(J->C+((size_t)t*J->NB+(size_t)j)*2048);
        for(int p=0;p<32;p++)for(int n=0;n<32;n++) {
            int64_t acc=0;
            for(int32_t k=0;k<J->KB;k++) {
                const uint8_t *a=J->A+((size_t)t*J->KB+(size_t)k)*2048+(size_t)p*32;
                const int8_t *b=(const int8_t *)J->B+((size_t)j*J->KB+(size_t)k)*1024;
                for(int c4=0;c4<32;c4++) acc+=(int64_t)a[c4]*b[((c4/4)*32+n)*4+c4%4];
            }
            int64_t v=(acc+J->corr[32*j+n])>>J->pre[32*j+n];
            c[p*32+n]=(uint16_t)(v<0?0:v>65535?65535:v);
        }
    }
    return 0;
#endif
}
/* 1: the command ran (or would, dry) on HMX; 0: not eligible (use the HVX
 * path); -1: error in *err. */
static int hmx_conv(G *g, const T *xt, const T *wt, const T *ft, const T *o, int packed, int reverse, int32_t co,
                    int32_t oh, int32_t ow, int32_t *err) {
    const Nprg2ReferenceConv *c=g->conv;
    uint64_t (*clock)(void)=c->phase && !g->dry?c->clock:NULL,t0=clock?clock():0,t1;
    if(!c->hmx || !c->near || ((uintptr_t)c->near&127u)) return 0;
    /* A fused epilogue must cover exactly the region the conv writes. */
    if(g->fx.kind && (oh!=o->d[1] || ow!=o->d[2] || co!=o->d[3])) return 0;
    int32_t H=wt->m[0],W=wt->m[1],I=wt->m[2];
    if(I%32 || xt->m[3]<I || H>3 || W>3 || ow%4 || co%32) return 0;
    if(!(xt->flags&F_BLOCKED) || xt->esize!=1 || !(o->flags&F_BLOCKED) || o->esize!=1) return 0;
    if(xt->p[0] || o->p[0] || xt->m[0]!=1 || o->m[0]!=1 || wt->pool!=1) return 0;
    const int32_t cbs=I/32,KB=H*W*cbs,NB=co/32;
    if(KB*32>HMX_TERMS_MAX || KB>255) return 0;
    const int8_t *wbase=(const int8_t *)g->pool[wt->pool]+wt->offset;
    if((uintptr_t)wbase&1023u) return 0;
    int32_t dims[4]={1,oh+H-1,ow+W-1,I},origin[4]={0,xt->p[1],xt->p[2],0};
    int32_t on[4]={1,oh,ow,co},oorigin[4]={0,o->p[1],o->p[2],0};
    if(xfer_check(g,xt,dims,origin,0) || xfer_check(g,o,on,oorigin,1) ||
       blocked_overlap(xt,dims,origin,o,on,oorigin)) return 0;
    /* Scalar tables in near: ktab, corr, pre, M, then the multiplier vectors. */
    size_t at=0;
    size_t o_ktab=nbump(&at,(size_t)KB*4),o_corr=nbump(&at,(size_t)co*4),o_pre=nbump(&at,(size_t)co*4),
           o_M=nbump(&at,(size_t)co*4),o_mult=nbump(&at,(size_t)co*4),o_bias=nbump(&at,(size_t)co*4),
           o_post=nbump(&at,(size_t)co*4),o_wsum=nbump(&at,(size_t)co*8),o_multv=nbump(&at,(size_t)NB*128),
           o_rows=nbump(&at,(size_t)NB*256);
    /* Input row pointers for every (row, block column, channel block) the
     * quads read: rows p1..p1+oh+H-2, columns p2..p2+ow+W-2 (a straddling quad
     * reads one block further). */
    const int32_t xwb0=xt->p[2]>>3,xnwb=((xt->p[2]+ow+W-2)>>3)-xwb0+1,xrows=oh+H-1;
    size_t ntab=(size_t)xrows*(size_t)xnwb*(size_t)cbs,o_xtab=nbump(&at,ntab*sizeof(uint8_t *));
    int use_tab=at<=c->near_bytes;
    if(!use_tab) at=o_xtab;
    /* Output tables: rows oh, block columns p2..p2+ow-1 (a straddling quad
     * stays inside them), NB channel blocks; one per written tensor. */
    const int32_t owb0=o->p[2]>>3,onwb=((o->p[2]+ow-1)>>3)-owb0+1,ntabs=1+(g->fx.kind?1:0)+(g->fx.kind==2?1:0);
    size_t otn=(size_t)oh*(size_t)onwb*(size_t)NB,o_otab=nbump(&at,(size_t)ntabs*otn*sizeof(uint8_t *));
    int use_otab=at<=c->near_bytes && (!g->fx.kind || (g->fx.oe.p[2]==o->p[2] && (g->fx.kind!=2 || g->fx.ob.p[2]==o->p[2])));
    if(!use_otab) at=o_otab;
    if(at>c->near_bytes) return 0;
    uint8_t *n=c->near;
    uint32_t *ktab=(uint32_t *)(n+o_ktab);
    int32_t *corr=(int32_t *)(n+o_corr),*pre=(int32_t *)(n+o_pre),*Mq=(int32_t *)(n+o_M),*mult=(int32_t *)(n+o_mult),
            *bias=(int32_t *)(n+o_bias),*post=(int32_t *)(n+o_post);
    int64_t *wsum=(int64_t *)(n+o_wsum);
    /* The filter rows (per 32 channels: 32 mults, 32 biases) are in VTCM:
     * one vector copy to near instead of scalar VTCM loads. */
    {
        const uint8_t *rows=g->pool[ft->pool]+ft->offset;
        uint8_t *nr=n+o_rows;
        near_copy(nr,rows,(size_t)NB*256);
        for(int32_t k=0;k<co;k++){memcpy(mult+k,nr+(size_t)(k/32)*256+(k%32)*4,4);memcpy(bias+k,nr+(size_t)(k/32)*256+128+(k%32)*4,4);}
    }
    requant_params(mult,co,0,Mq,pre,post);
    memset(wsum,0,(size_t)co*8);
    if(xt->zp) for(int32_t j=0;j<NB;j++) {
        const int8_t *b=wbase+(size_t)j*KB*1024;
#if BLK_HVX
        HVX_Vector ones=Q6_V_vsplat_R(0x01010101),a=Q6_V_vzero();
        for(int32_t r=0;r<KB*8;r++) a=Q6_Vw_vrmpyacc_VwVbVb(a,((const HVX_Vector *)b)[r],ones);
        int32_t tmp[32] __attribute__((aligned(128)));
        *(HVX_Vector *)tmp=a;
        for(int q=0;q<32;q++) wsum[32*j+q]=tmp[q];
#else
        for(int32_t r=0;r<KB*8;r++)for(int q=0;q<128;q++) wsum[32*j+q/4]+=b[128*r+q];
#endif
    }
    const int64_t bound=(int64_t)KB*32*255*128;
    for(int32_t k=0;k<co;k++) {
        int64_t off=(int64_t)bias[k]-(int64_t)xt->zp*wsum[k];
        if(post[k]!=14 || pre[k]<0 || pre[k]>15 || Mq[k]<1024 || Mq[k]>2047 || (off<0?-off:off)+bound>INT32_MAX) return 0;
        corr[k]=(int32_t)off;
    }
    for(int32_t k=0;k<KB;k++) {
        int32_t h,x,cb;
        if(packed){cb=k%cbs;h=(k/cbs)/W;x=(k/cbs)%W;} else {x=k%W;h=(k/W)%H;cb=k/(W*H);}
        int32_t kx=reverse?W-1-x:x;
        ktab[k]=(uint32_t)h|(uint32_t)kx<<8|(uint32_t)cb<<16;
    }
    /* Workspace: fi, finish record, completion, then A and C for mt tiles. */
    const int32_t total=(oh*ow+31)/32;
    const size_t fixed=((size_t)NB*256+256+2048*2+2047)&~(size_t)2047,per=(size_t)(KB+NB)*2048;
    if(c->workspace_bytes<fixed+per+2048) return 0;
    int32_t mt=(int32_t)((c->workspace_bytes-fixed-2048)/per);
    if(mt>total) mt=total;
    g->wused=0;
    uint8_t *ws=walloc(g,fixed+(size_t)mt*per+2048);
    if(!ws){*err=E_MEMORY;return -1;}
    conv_path=3;
    if(g->fx.kind) g->fx_ok=1;
#ifdef CONV_SHAPES
    if(!g->dry) fprintf(stderr,"HMXCONV var=%u k=%dx%d ci=%d co=%d oh=%d ow=%d KB=%d NB=%d tiles=%d mt=%d packed=%d\n",
                        g->variant,H,W,I,co,oh,ow,KB,NB,total,mt,packed);
#endif
    if(g->dry) return 1;
    uint8_t *base=(uint8_t *)(((uintptr_t)ws+2047)&~(uintptr_t)2047);
    int32_t *fi=(int32_t *)base,*zero=(int32_t *)(base+(size_t)NB*256);
    uint8_t *completion=base+fixed-2048,*A=base+fixed,*C=A+(size_t)mt*KB*2048;
    for(int32_t k=0;k<co;k++) {
        uint32_t p=(uint32_t)pre[k];
        fi[(k>>5)*64+(k&31)]=(int32_t)(0x400000u|((16u-p)<<10));
        fi[(k>>5)*64+32+(k&31)]=corr[k]-(p?(int32_t)(1u<<(p-1)):0);
    }
    for(int k=0;k<64;k++) zero[k]=k<32?0x404000:0;
    uint16_t *mv=(uint16_t *)(n+o_multv);
    for(int32_t k=0;k<co;k++) mv[(k>>5)*64+2*(k&31)]=mv[(k>>5)*64+2*(k&31)+1]=(uint16_t)Mq[k];
    uint8_t **xtab=use_tab?(uint8_t **)(n+o_xtab):NULL;
    if(xtab) hmx_tab(xtab,g->pool,xt,xt->p[1],xrows,xwb0,xnwb,cbs);
    uint8_t **otab=NULL,**etab=NULL,**btab=NULL;
    if(use_otab) {
        otab=(uint8_t **)(n+o_otab);hmx_tab(otab,g->pool,o,o->p[1],oh,owb0,onwb,NB);
        if(g->fx.kind){etab=otab+otn;hmx_tab(etab,g->pool,&g->fx.oe,g->fx.oe.p[1],oh,owb0,onwb,NB);}
        if(g->fx.kind==2){btab=etab+otn;hmx_tab(btab,g->pool,&g->fx.ob,g->fx.ob.p[1],oh,owb0,onwb,NB);}
    }
    HmxJob J={.g=g,.xtab=xtab,.xwb0=xwb0,.xnwb=xnwb,.cbs=cbs,.otab=otab,.etab=etab,.btab=btab,.owb0=owb0,.onwb=onwb,.xt=xt,.o=o,.oh=oh,.ow=ow,.KB=KB,.NB=NB,.total=total,.A=A,.C=C,.B=(const uint8_t *)wbase,
              .ktab=ktab,.corr=corr,.pre=pre,.Mq=Mq,.multv=(const uint8_t *)mv,.fx=g->fx.kind?&g->fx:NULL,
              .xnbw=((int64_t)xt->m[2]+7)/8,.xnbd=((int64_t)xt->m[3]+31)/32,
              .onbw=((int64_t)o->m[2]+7)/8,.onbd=((int64_t)o->m[3]+31)/32};
    uint32_t parts=nprg2_parallel_parts(g->call->parallel,3);
    uint32_t saved=c->hmx->enter?c->hmx->enter():0;
    int e=0;
    for(int32_t m0=0;m0<total && !e;m0+=mt) {
        J.m0=m0;J.mt=total-m0<mt?total-m0:mt;
        if(clock){t1=clock();c->phase[2]+=t1-t0;t0=t1;}
        if(parts>1) nprg2_parallel_for(g->call->parallel,NPRG2_PAR_CONV,parts,hmx_prep_part,&J);
        else hmx_prep_part(&J,0,1);
        if(clock){t1=clock();c->phase[0]+=t1-t0;t0=t1;}
        e=hmx_mac(&J,fi,zero,completion);
        if(clock){t1=clock();c->phase[1]+=t1-t0;t0=t1;}
        if(e) break;
        if(parts>1) nprg2_parallel_for(g->call->parallel,NPRG2_PAR_CONV,parts,hmx_quant_part,&J);
        else hmx_quant_part(&J,0,1);
        if(clock){t1=clock();c->phase[3]+=t1-t0;t0=t1;}
    }
    if(c->hmx->leave) c->hmx->leave(saved);
    if(e){*err=E_RANGE;return -1;}
    return 1;
}
