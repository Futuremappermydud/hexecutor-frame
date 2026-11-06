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
#if defined(NAT_CHECK) || defined(SLIDE_TRACE)
#include <stdio.h>
#endif
#define HMX_TERMS_MAX 2304 /* largest K the v6 cases proved exact: 3x3x256 */
/* NPRG2_PLANES: emulator-exact u8 convs at the native int8 rate. One accumulator
 * tile, two conversions (QNN hmx_v75_convbbh1x1_stride1's scheme; selector bit 0
 * keeps the accumulator, bit 1 wraps, bits 12..13 pick the bias bank): the low
 * byte of q = floor(x/2^pre) and floor(q/256)+16 clipped to a byte. HVX then
 * computes clip((clamp(q,0,4096)*M + 8192) >> 14): 2048 bytes of each plane per
 * block, 4096 per output block. */
#ifndef NPRG2_PLANES
#define NPRG2_PLANES 0
#endif
/* Fewer output blocks than this: merged on the calling thread, no helper dispatch. */
#ifndef PLANE_SERIAL_BELOW
#define PLANE_SERIAL_BELOW 0
#endif
#if NPRG2_PLANES && NPRG2_HW_ROUNDING
#error NPRG2_PLANES is the emulator-rounding path
#endif
#define HMX_NAT_BLOCK (NPRG2_PLANES?4096:2048) /* conversion bytes per output block */
#define HMX_FI_BYTES (NPRG2_PLANES?512:256) /* Fi bytes per output block */
#ifndef HMX_SETUP_SLOT
#define HMX_SETUP_SLOT 2 /* phase slot that HMX per-command setup is charged to */
#endif
/* Profiling builds: HMX_PROF=s charges setup sub-step s (1 bounds checks,
 * 2 filter rows/requant/wsum, 3 fi/multv records, 4 row pointer tables; native: 5 k_hmx_conv
 * entry to hmx_conv, 6 Fi build + B copy, 7 row table, 8 per-batch output table) to
 * the output slot (3) instead, so it is the difference against a normal build. */
#ifndef HMX_PROF
#define HMX_PROF 0
#endif
#define HMX_MARK(step) do { if(clock){t1=clock();c->phase[HMX_PROF==(step)?3:HMX_SETUP_SLOT]+=t1-t0;t0=t1;} } while(0)
/* Fi words (per 32 channels: 32 requant words, 32 offsets; then a block of
 * 32 x 0x404000 and 32 zeros) into the workspace at base. fipre, when given,
 * holds the first NB*256 bytes already built (HVX vectors, 128-aligned);
 * otherwise they are built in the near buffer nf from corr/pre/Mq. */
static void hmx_fi_put(uint8_t *base, int32_t NB, int32_t co, const uint8_t *fipre, int32_t *nf, const int32_t *corr,
                       const int32_t *pre, const int32_t *Mq) {
#if BLK_HVX
    if(fipre) {
        const HVX_Vector *s=(const HVX_Vector *)fipre;HVX_Vector *d=(HVX_Vector *)base;
        int32_t n=NB*2,i=0;
        for(;i+4<=n;i+=4) {
            HVX_Vector a0=s[i],a1=s[i+1],a2=s[i+2],a3=s[i+3];
            d[i]=a0;d[i+1]=a1;d[i+2]=a2;d[i+3]=a3;
        }
        for(;i<n;i++) d[i]=s[i];
        d[n]=Q6_V_vsplat_R(0x404000);d[n+1]=Q6_V_vzero();
        return;
    }
#else
    (void)fipre;
#endif
    int32_t *nz=nf+NB*64;
    for(int32_t k=0;k<co;k++) {
        nf[(k>>5)*64+(k&31)]=(int32_t)(0x400000u|((uint32_t)(20-pre[k])<<10)|(uint32_t)(Mq[k]-1024));
        nf[(k>>5)*64+32+(k&31)]=corr[k];
    }
    for(int k=0;k<64;k++) nz[k]=k<32?0x404000:0;
    near_copy(base,(const uint8_t *)nf,(size_t)NB*256+256);
}
typedef struct {
    G *g; const T *xt, *o;
    int32_t oh, ow, KB, NB, total, m0, mt; /* tiles m0..m0+mt-1 of total */
    uint8_t *A, *C;
    const uint8_t *B;
    const uint32_t *ktab; /* ky | kx<<8 | cb<<16 per k */
    const int32_t *corr, *pre, *Mq;
    const uint8_t *multv; /* NB vectors: M of channel n in halfwords 2n, 2n+1 */
    const uint8_t *rndv;  /* NB vectors: rounding word of channel n in word n */
    uint8_t *const *xtab; /* input row pointers [r][wb-wb0][cb], or NULL: hmx_row per quad */
    int32_t xwb0, xnwb, cbs;
    /* Output row pointers [y][wb-owb0][j] of o, and of the epilogue's output
     * and other operand (same geometry), or NULL: hmx_row per quad. */
    uint8_t *const *otab, *const *etab, *const *btab;
    int32_t owb0, onwb;
    int32_t *tyx; /* per local tile t: first pixel's y, x (no divide per tile) */
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
    const int32_t width=nwb*nz;
    for(int32_t r=0;r<nrows;r++) {
        int32_t h=row0+r;
        uint8_t **row=tab+(size_t)r*(size_t)width;
        /* Within a block row the entries are the previous row's plus 256
         * (absent stays absent): no block-table lookups. */
        if(r && h>0 && (h&7)) {
            uint8_t *const *prev=row-width;
            int32_t i=0;
#if BLK_HVX
            if(sizeof(uint8_t *)==4) {
                HVX_Vector z=Q6_V_vzero(),k=Q6_V_vsplat_R(256);
                for(;i+32<=width;i+=32) {
                    HVX_Vector v=*(const HVX_UVector *)(prev+i);
                    HVX_VectorPred nz0=Q6_Q_vcmp_eq_VwVw(v,z);
                    *(HVX_UVector *)(row+i)=Q6_V_vmux_QVV(nz0,z,Q6_Vw_vadd_VwVw(v,k));
                }
            }
#endif
            for(;i<width;i++) row[i]=prev[i]?prev[i]+256:NULL;
            continue;
        }
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
static inline HVX_Vector hmx_quad_tab(uint8_t *const *tab, int32_t wb0, int32_t nwb, int32_t nz, int32_t r, int32_t col,
                                      int32_t z) {
    int32_t wb=(col>>3)-wb0;int s=col&7;
    uint8_t *const *row=tab+(uint32_t)((r*nwb+wb)*nz+z);
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
    y=J->tyx[2*t];x=J->tyx[2*t+1];
    uint8_t *dst=J->A+((size_t)t*J->KB+(size_t)k)*2048;
#if BLK_HVX
    /* Fast path: all 32 pixels in one output row, so the tile reads one input
     * row's columns col0..col0+31: 4 or 5 blocks, 10 aligned vectors. Quad q
     * starts at stream byte 32*(col0&7) + 128q: vector q + s/4, shift 32*(s%4). */
    if(J->xtab && x+32<=J->ow && P+32<=total) {
        int32_t col0=xt->p[2]+x+(int32_t)kx,s=col0&7;
        uint8_t *const *row=J->xtab+(uint32_t)((((y+(int32_t)ky)*J->xnwb)+(col0>>3)-J->xwb0)*J->cbs+(int32_t)cb);
        /* Stream vectors: dynamic indexing of V[10] materialized a 1280-byte
         * array on the stack in the V75 build. Keep only adjacent vectors. */
        HVX_Vector zero=Q6_V_vzero();
        const int first=s>>2,sh=(s&3)*32;
        const uint8_t *bp=row[(first>>1)*J->cbs];
        HVX_Vector cur=bp?((const HVX_Vector *)bp)[first&1]:zero;
        for(int q=0;q<8;q++) {
            HVX_Vector v=cur;
            if(sh || q<7) {
                int i=first+q+1;
                const uint8_t *next=row[(i>>1)*J->cbs];
                HVX_Vector hi=next?((const HVX_Vector *)next)[i&1]:zero;
                if(sh) v=Q6_V_valign_VVR(hi,cur,sh);
                cur=hi;
            }
            HVX_VectorPair w=Q6_W_vshuff_VVR(v,zero,-1);
            ((HVX_Vector *)dst)[2*q]=Q6_Vh_vshuff_Vh(Q6_V_lo_W(w));
            ((HVX_Vector *)dst)[2*q+1]=Q6_Vh_vshuff_Vh(Q6_V_hi_W(w));
        }
        return;
    }
#endif
    for(int q=0;q<8;q++,P+=4) {
        int valid=P<total;
        int64_t row=xt->p[1]+y+ky,col=xt->p[2]+x+kx;
#if BLK_HVX
        HVX_Vector v=!valid?Q6_V_vzero():J->xtab?hmx_quad_tab(J->xtab,J->xwb0,J->xnwb,J->cbs,y+(int32_t)ky,(int32_t)col,(int32_t)cb):hmx_quad(g->pool,xt,row,col,cb,J->xnbw,J->xnbd);
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
/* Tiles i = t*KB + k with i % parts == part, stepped without a divide
 * (Hexagon has none: a % per tile was a library call per tile per part). */
static void hmx_prep_part(void *ctx, uint32_t part, uint32_t parts) {
    const HmxJob *J=(const HmxJob *)ctx;
    uint32_t KB=(uint32_t)J->KB,t=0,k=part;
    while(k>=KB){k-=KB;t++;}
    while(t<(uint32_t)J->mt) {
        hmx_tile(J,(int32_t)t,(int32_t)k);
        k+=parts;
        while(k>=KB){k-=KB;t++;}
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
/* rnd: per-channel rounding word (lane n = channel n; both halves of a
 * product pair are the same channel), rq_round(0, M, 14). */
static HVX_Vector hmx_quant_pair(HVX_Vector pre, HVX_Vector mult, HVX_Vector rnd) {
    pre=Q6_Vuh_vmin_VuhVuh(pre,Q6_V_vsplat_R(4096|(4096<<16)));
    HVX_VectorPair product=Q6_Ww_vmpy_VhVh(pre,mult);
    HVX_Vector a=Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_V_lo_W(product),rnd),14);
    HVX_Vector b=Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_V_hi_W(product),rnd),14);
    return Q6_Vh_vpack_VwVw_sat(b,a);
}
#endif
static void hmx_quant_tile(const HmxJob *J, int32_t t, int32_t j) {
    int32_t P=32*(J->m0+t),y,x,total=J->oh*J->ow;
    y=J->tyx[2*t];x=J->tyx[2*t+1];
    const uint8_t *src=J->C+((size_t)t*J->NB+(size_t)j)*2048;
#if BLK_HVX
    HVX_Vector mult=((const HVX_Vector *)J->multv)[j],rnd=((const HVX_Vector *)J->rndv)[j];
    for(int i=0;i<8 && P<total;i++,P+=4) {
        HVX_Vector a=hmx_quant_pair(((const HVX_Vector *)src)[2*i],mult,rnd),b=hmx_quant_pair(((const HVX_Vector *)src)[2*i+1],mult,rnd);
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
                HVX_Vector bv=J->btab?hmx_quad_tab(J->btab,J->owb0,J->onwb,J->NB,y,f->ob.p[2]+x,j):
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
                uint32_t v=((uint32_t)(pre<4096?pre:4096)*(uint32_t)J->Mq[32*j+n]+(uint32_t)rq_round(0,J->Mq[32*j+n],14))>>14;
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
    uint32_t NB=(uint32_t)J->NB,t=0,j=part;
    while(j>=NB){j-=NB;t++;}
    while(t<(uint32_t)J->mt) {
        hmx_quant_tile(J,(int32_t)t,(int32_t)j);
        j+=parts;
        while(j>=NB){j-=NB;t++;}
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

#if NPRG2_HW_ROUNDING || NPRG2_PLANES
#if NPRG2_PLANES
static size_t slide_fixed(int32_t NB) {return ((size_t)NB*512+256+2048*2+2047)&~(size_t)2047;}
/* Fi of both banks: bank 0 low byte, scale (24-pre)<<10 (M 1024, no rounding bit:
 * floor), bias off; bank 1 scale (16-pre)<<10, bias off + 2^(12+pre); then the
 * finish record. mh: per output block M of channel n in halfwords n and 32+n. */
static void plane_fi_put(uint8_t *base, int32_t NB, const int32_t *corr, const int32_t *pre, const int32_t *Mq, uint8_t *mh) {
#if BLK_HVX
    const HVX_Vector c24=Q6_V_vsplat_R(24),c16=Q6_V_vsplat_R(16),c12=Q6_V_vsplat_R(1<<12);
    HVX_Vector *d=(HVX_Vector *)base;
    for(int32_t j=0;j<NB;j++,d+=4) {
        const HVX_Vector pv=*(const HVX_Vector *)(pre+32*j),off=*(const HVX_Vector *)(corr+32*j),Mv=*(const HVX_Vector *)(Mq+32*j);
        d[0]=Q6_Vw_vasl_VwR(Q6_Vw_vsub_VwVw(c24,pv),10);d[1]=off;
#ifdef PLANE_SAT
        /* High plane floor(q/256)+239: clips exactly at q>=4096, so the merge is one saturating
         * subtract (and the multiplier is stored doubled instead of doubling q). */
        d[2]=Q6_Vw_vasl_VwR(Q6_Vw_vsub_VwVw(c16,pv),10);d[3]=Q6_Vw_vadd_VwVw(off,Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(239<<8),pv));
        ((HVX_Vector *)mh)[j]=Q6_Vh_vadd_VhVh(Q6_Vh_vpacke_VwVw(Mv,Mv),Q6_Vh_vpacke_VwVw(Mv,Mv));
#else
        d[2]=Q6_Vw_vasl_VwR(Q6_Vw_vsub_VwVw(c16,pv),10);d[3]=Q6_Vw_vadd_VwVw(off,Q6_Vw_vasl_VwVw(c12,pv));
        ((HVX_Vector *)mh)[j]=Q6_Vh_vpacke_VwVw(Mv,Mv);
#endif
    }
    d[0]=Q6_V_vsplat_R(0x404000);d[1]=Q6_V_vzero();
#else
    (void)base;(void)NB;(void)corr;(void)pre;(void)Mq;(void)mh;
#endif
}
/* Every channel representable: pre 0..7 (hi bias offset at most 2^19 on top of acc + off). */
static int plane_params_ok(int32_t co, const int32_t *corr, const int32_t *pre, int64_t bound) {
#ifdef PLANE_SAT
    const int64_t lim=(int64_t)INT32_MAX-bound-(239LL<<15);
#else
    const int64_t lim=(int64_t)INT32_MAX-bound-(1<<19);
#endif
#if BLK_HVX
    if(lim<=0 || lim>INT32_MAX) return 0;
    const HVX_Vector c7=Q6_V_vsplat_R(7),zero=Q6_V_vzero(),l=Q6_V_vsplat_R((int32_t)lim);
    HVX_VectorPred bad=Q6_Q_vcmp_gt_VwVw(zero,zero);
    for(int32_t k=0;k<co;k+=32) {
        const HVX_Vector pv=*(const HVX_Vector *)(pre+k),off=*(const HVX_Vector *)(corr+k);
        bad=Q6_Q_or_QQ(bad,Q6_Q_or_QQ(Q6_Q_vcmp_gt_VwVw(zero,pv),Q6_Q_vcmp_gt_VwVw(pv,c7)));
        bad=Q6_Q_or_QQ(bad,Q6_Q_vcmp_gt_VwVw(Q6_Vw_vabs_Vw_sat(off),l));
    }
    HVX_Vector flag __attribute__((aligned(128)))=Q6_V_vand_QR(bad,0x01010101u);
    const uint32_t *fw=(const uint32_t *)&flag;uint32_t any=0;
    for(int q=0;q<32;q++) any|=fw[q];
    return !any;
#else
    for(int32_t k=0;k<co;k++) if(pre[k]<0 || pre[k]>7 || (corr[k]<0?-(int64_t)corr[k]:corr[k])>lim) return 0;
    return 1;
#endif
}
#if BLK_HVX
static inline __attribute__((always_inline)) HVX_Vector plane_rq(HVX_Vector lo, HVX_Vector hi, HVX_Vector mv) {
#ifdef PLANE_SAT
    /* q = saturating(v - 239*256), at most 4351, whole-range exact against the clamped rule
     * for M 1024..2047 (bench/lanes/plane_sat_sim.c). mv holds 2M: (2*q*2M + 2^15)>>16. */
    const HVX_Vector off=Q6_Vh_vsplat_R(239<<8);
    HVX_VectorPair ww=Q6_W_vshuff_VVR(hi,lo,-1);
    return Q6_Vub_vpack_VhVh_sat(Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_Vuh_vsub_VuhVuh_sat(Q6_V_hi_W(ww),off),mv),
                                 Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_Vuh_vsub_VuhVuh_sat(Q6_V_lo_W(ww),off),mv));
#endif
    const HVX_Vector c4096=Q6_Vh_vsplat_R(4096),c8192=Q6_Vh_vsplat_R(8192);
    HVX_VectorPair w=Q6_W_vshuff_VVR(hi,lo,-1);
    HVX_Vector a=Q6_Vh_vsub_VhVh(Q6_Vuh_vmin_VuhVuh(Q6_Vuh_vmax_VuhVuh(Q6_V_lo_W(w),c4096),c8192),c4096),
               b=Q6_Vh_vsub_VhVh(Q6_Vuh_vmin_VuhVuh(Q6_Vuh_vmax_VuhVuh(Q6_V_hi_W(w),c4096),c8192),c4096);
    /* (2q*M*2 + 2^15) >> 16 = (q*M + 2^13) >> 14 */
    a=Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_Vh_vadd_VhVh(a,a),mv);
    b=Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_Vh_vadd_VhVh(b,b),mv);
    return Q6_Vub_vpack_VhVh_sat(b,a);
}
#endif
/* Merge job. Whole tiles: out[(t*NB)+j] from C + (t*NB+j)*4096. shift (slide):
 * tile (t, j) + its next column's tile, realigned by one pixel. */
typedef struct {
    const uint8_t *C; uint8_t *const *out; const uint8_t *mh; const int32_t *Mq;
    int32_t mt, NB, bw, ncol, shift, tile0; /* shift: output block bx starts in column bx+tile0 */
    int32_t rows, partial_from; /* tiles t >= partial_from write only their first rows rows (oh%8) */
    /* shift: output blocks read here, in the parts, not ahead of the job. */
    uint8_t *const *pool; const uint32_t *oblk; uint32_t ocount, oslice, onbw, onbd; int32_t oR, oC;
} PlaneJob;
static inline void plane_block(const PlaneJob *J, uint8_t *o, const uint8_t *c, const uint8_t *next, int32_t j, int32_t rows) {
#if BLK_HVX
    const HVX_Vector mv=((const HVX_Vector *)J->mh)[j];
    const HVX_Vector *lo=(const HVX_Vector *)c,*hi=(const HVX_Vector *)(c+2048);
    HVX_Vector *d=(HVX_Vector *)o;
    if(!next) {
        for(int i=0;i<2*rows;i++) d[i]=plane_rq(lo[i],hi[i],mv);
        return;
    }
    const HVX_Vector *lo2=(const HVX_Vector *)next,*hi2=(const HVX_Vector *)(next+2048);
    for(int y=0;y<rows;y++) {
        HVX_Vector l0=lo[2*y],l1=lo[2*y+1],l2=lo2[2*y],h0=hi[2*y],h1=hi[2*y+1],h2=hi2[2*y];
        d[2*y]=plane_rq(Q6_V_valign_VVR(l1,l0,32),Q6_V_valign_VVR(h1,h0,32),mv);
        d[2*y+1]=plane_rq(Q6_V_valign_VVR(l2,l1,32),Q6_V_valign_VVR(h2,h1,32),mv);
    }
#else
    const int32_t *M=J->Mq+32*j;
    for(int p=0;p<rows*8;p++)for(int n=0;n<32;n++) {
        const size_t i=(size_t)p*32+n;
        int32_t v=(int32_t)c[i]|(int32_t)c[2048+i]<<8;
#ifdef PLANE_SAT
        v=v<(239<<8)?0:v-(239<<8);
#else
        v=(v<4096?4096:v>8192?8192:v)-4096;
#endif
        v=(v*M[n]+8192)>>14;
        o[i]=(uint8_t)(v>255?255:v);
    }
    (void)next;
#endif
}
static void plane_part(void *ctx, uint32_t part, uint32_t parts) {
    const PlaneJob *J=(const PlaneJob *)ctx;
    uint32_t u=0,next=part;
    if(!J->shift) {
        for(int32_t t=0;t<J->mt;t++)for(int32_t j=0;j<J->NB;j++,u++) {
            if(u!=next) continue;
            next+=parts;
            if(J->out[u]) plane_block(J,J->out[u],J->C+(size_t)u*4096,NULL,j,t>=J->partial_from?J->rows:8);
        }
        return;
    }
    /* mt = strips*bw tiles in output order; C has ncol columns per strip */
    int32_t bx=0,by=0;size_t row=0;
    for(int32_t t=0;t<J->mt;t++) {
        const size_t col=(row+(size_t)bx+(size_t)J->tile0)*(size_t)J->NB;
        const uint32_t idx=(((uint32_t)(J->oR+by))*J->onbw+(uint32_t)(J->oC+bx))*J->onbd+J->oslice;
        for(int32_t j=0;j<J->NB;j++,u++) {
            if(u!=next) continue;
            next+=parts;
            const uint32_t ix=idx+(uint32_t)j,ee=ix<J->ocount?J->oblk[ix]:0,pl=ee&127u;
            if(pl && J->pool[pl]) {
                const uint8_t *c=J->C+(col+(size_t)j)*4096;
                plane_block(J,J->pool[pl]+(ee&~127u),c,c+(size_t)J->NB*4096,j,t>=J->partial_from?J->rows:8);
            }
        }
        if(++bx==J->bw){bx=0;by++;row+=(size_t)J->ncol;}
    }
}
#endif
/* ---------------------------------------------------------- native int8 HMX
 * HW-rounding builds only (the conversion is silicon's: requant_value with
 * NPRG2_HW_ROUNDING=1 equals cvt.ub exactly, HMX_NATIVE_ROUNDING.md). A tile
 * is one 8x8 output crouton block (64 px): its A operand per K block (tap ky,
 * kx and channel block cb) is the 8x8 input window shifted by (ky, kx),
 * gathered as 8 rows of 256 bytes (crouton :cm layout, byte p*32+c, p=y*8+x);
 * B is v6's (weight.b blocks); Fi = {0x400000|(20-pre)<<10|(M-1024), off}.
 * The conversion is stored :cm straight into the output block, or into a
 * workspace C block that an HVX pass turns into the fused epilogue. */
typedef struct {
    G *g; uint8_t *const *pool; const Nprg2ReferenceConv *conv; const T *xt, *o; const Fuse *fx;
    int32_t KB, NB, cbs, bw, mt, m0, xwb0, xnwb;
    const uint32_t *ktab; uint8_t *const *xtab;
    uint8_t *A, *C;
    const int32_t *tyx; /* per local tile: block row, block column */
    const int32_t *pre, *Mq, *corr; /* host model */
    const uint8_t *B;
    int one; int32_t R0, C0; int64_t xnbw, xnbd; /* one: 1x1, origin block aligned: A tile = the input block */
} NatJob;
#ifndef NAT_SERIAL_BELOW
#define NAT_SERIAL_BELOW 16 /* fewer 1x1 A tiles than this: staged serially, no thread dispatch */
#endif
#ifndef NAT_1X1
#define NAT_1X1 1
#endif
static void nat_tile(const NatJob *J, int32_t t, int32_t k) {
    if(J->one) {
        /* A tile (by, bx, cb) of a 1x1 conv with block-aligned origin is the
         * input block itself: one block lookup, whole-block copy (staged:
         * reading the input in place hangs silicon). */
        uint32_t e=J->ktab[k];
        const uint8_t *p=hmx_row(J->pool,J->xt,(int64_t)(J->R0+J->tyx[2*t])*8,J->C0+J->tyx[2*t+1],J->xnbw,J->xnbd,(int64_t)(e>>16));
        uint8_t *dst=J->A+((size_t)t*J->KB+(size_t)k)*2048;
#if BLK_HVX
        HVX_Vector *d=(HVX_Vector *)dst;
        if(p) {
            const HVX_Vector *sv=(const HVX_Vector *)p;
            HVX_Vector v0=sv[0],v1=sv[1],v2=sv[2],v3=sv[3],v4=sv[4],v5=sv[5],v6=sv[6],v7=sv[7];
            HVX_Vector v8=sv[8],v9=sv[9],v10=sv[10],v11=sv[11],v12=sv[12],v13=sv[13],v14=sv[14],v15=sv[15];
            d[0]=v0;d[1]=v1;d[2]=v2;d[3]=v3;d[4]=v4;d[5]=v5;d[6]=v6;d[7]=v7;
            d[8]=v8;d[9]=v9;d[10]=v10;d[11]=v11;d[12]=v12;d[13]=v13;d[14]=v14;d[15]=v15;
        } else {
            HVX_Vector z=Q6_V_vzero();
            for(int i=0;i<16;i++) d[i]=z;
        }
#else
        if(p) memcpy(dst,p,2048); else memset(dst,0,2048);
#endif
        return;
    }
    uint32_t e=J->ktab[k];
    int32_t ky=(int32_t)(e&255),kx=(int32_t)((e>>8)&255),cb=(int32_t)(e>>16);
    int32_t by=J->tyx[2*t],bx=J->tyx[2*t+1];
    int32_t col0=J->xt->p[2]+bx*8+kx,s=col0&7,wb=(col0>>3)-J->xwb0;
    uint8_t *dst=J->A+((size_t)t*J->KB+(size_t)k)*2048;
    for(int32_t r=0;r<8;r++) {
        uint8_t *const *row=J->xtab+(uint32_t)((((by*8+r+ky)*J->xnwb)+wb)*J->cbs+cb);
        const uint8_t *p0=row[0],*p1=s?row[J->cbs]:NULL;
#if BLK_HVX
        /* Stream V = p0's 256 bytes then p1's; the row is bytes 32s..32s+255:
         * vector index s>>2, shift (s&3)*32 (valign uses its shift mod 128). */
        /* only the 2-3 vectors this window uses: stream vectors i, i+1 (and i+2) */
        HVX_Vector z=Q6_V_vzero();
        int32_t i=s>>2,sh=(s&3)*32;
#define NAT_V(k) ((k)<2?(p0?((const HVX_Vector *)p0)[(k)]:z):(p1?((const HVX_Vector *)p1)[(k)-2]:z))
        HVX_Vector a0=NAT_V(i),a1=NAT_V(i+1);
        HVX_Vector o0=a0,o1=a1;
        if(sh) {HVX_Vector a2=NAT_V(i+2);o0=Q6_V_valign_VVR(a1,a0,sh);o1=Q6_V_valign_VVR(a2,a1,sh);}
#undef NAT_V
        ((HVX_Vector *)dst)[2*r]=o0;((HVX_Vector *)dst)[2*r+1]=o1;
#else
        uint8_t *d=dst+256*r;
        uint32_t n0=256u-32u*(uint32_t)s;
        if(p0) memcpy(d,p0+32*s,n0); else memset(d,0,n0);
        if(s) {if(p1) memcpy(d+n0,p1,256-n0); else memset(d+n0,0,256-n0);}
#endif
    }
}
static void nat_prep_part(void *ctx, uint32_t part, uint32_t parts) {
    const NatJob *J=(const NatJob *)ctx;
    uint32_t KB=(uint32_t)J->KB,t=0,k=part;
    while(k>=KB){k-=KB;t++;}
    while(t<(uint32_t)J->mt) {
        nat_tile(J,(int32_t)t,(int32_t)k);
        k+=parts;
        while(k>=KB){k-=KB;t++;}
    }
}
/* Fused epilogue from the C blocks: e = min((q*mc + [b*mo] + rd) >> sh, 255)
 * into oe's block, q into o's block unless dead (as rq_emit_fx). */
static void nat_fx_block(const NatJob *J, int32_t t, int32_t j) {
    const Fuse *f=J->fx;G *g=J->g;
    int32_t by=J->tyx[2*t],bx=J->tyx[2*t+1];
    const uint8_t *c=J->C+((size_t)t*J->NB+(size_t)j)*2048;
    int64_t onbw=((int64_t)J->o->m[2]+7)/8,onbd=((int64_t)J->o->m[3]+31)/32;
    if(!f) { /* staged (an output block is not 2048-aligned for :cm): copy */
        uint8_t *ob=hmx_row(g->pool,J->o,(int64_t)J->o->p[1]+by*8,((int64_t)J->o->p[2]>>3)+bx,onbw,onbd,j);
        if(ob) {
#if BLK_HVX
            for(int i=0;i<16;i++) ((HVX_Vector *)ob)[i]=((const HVX_Vector *)c)[i];
#else
            memcpy(ob,c,2048);
#endif
        }
        return;
    }
    uint8_t *ob=f->dead?NULL:hmx_row(g->pool,J->o,(int64_t)J->o->p[1]+by*8,((int64_t)J->o->p[2]>>3)+bx,onbw,onbd,j);
    uint8_t *eb=hmx_row(g->pool,&f->oe,(int64_t)f->oe.p[1]+by*8,((int64_t)f->oe.p[2]>>3)+bx,onbw,onbd,j);
    const uint8_t *bb=f->kind==2?hmx_row(g->pool,&f->ob,(int64_t)f->ob.p[1]+by*8,((int64_t)f->ob.p[2]>>3)+bx,onbw,onbd,j):NULL;
#if BLK_HVX
    HVX_Vector rdv=Q6_V_vsplat_R((int32_t)f->rd),mcv=Q6_V_vsplat_R((int32_t)f->mc),
               mov=Q6_V_vsplat_R((int32_t)f->mo),shv=Q6_V_vsplat_R((int32_t)f->sh),c255=Q6_V_vsplat_R(255);
    for(int i=0;i<16;i++) {
        HVX_Vector v=((const HVX_Vector *)c)[i];
        if(ob) ((HVX_Vector *)ob)[i]=v;
        if(!eb) continue;
        HVX_VectorPair w=Q6_Wuh_vunpack_Vub(v);
        HVX_VectorPair w0=Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(w)),w1=Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(w));
        HVX_Vector q[4]={Q6_V_lo_W(w0),Q6_V_hi_W(w0),Q6_V_lo_W(w1),Q6_V_hi_W(w1)},u[4],r[4];
        if(f->kind==2) {
            HVX_Vector bv=bb?((const HVX_Vector *)bb)[i]:Q6_V_vzero();
            HVX_VectorPair zz=Q6_Wuh_vunpack_Vub(bv);
            HVX_VectorPair z0=Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(zz)),z1=Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(zz));
            u[0]=Q6_V_lo_W(z0);u[1]=Q6_V_hi_W(z0);u[2]=Q6_V_lo_W(z1);u[3]=Q6_V_hi_W(z1);
        }
        for(int k=0;k<4;k++) {
            HVX_Vector tt=Q6_Vw_vmpyieacc_VwVwVuh(rdv,q[k],mcv);
            if(f->kind==2) tt=Q6_Vw_vmpyieacc_VwVwVuh(tt,u[k],mov);
            r[k]=Q6_Vw_vmin_VwVw(Q6_Vw_vlsr_VwVw(tt,shv),c255);
        }
        ((HVX_Vector *)eb)[i]=rq_pack8(r[0],r[1],r[2],r[3]);
    }
#else
    for(int i=0;i<2048;i++) {
        uint32_t q=c[i];
        if(ob) ob[i]=(uint8_t)q;
        if(!eb) continue;
        uint32_t tt=q*f->mc+f->rd;
        if(f->kind==2) tt+=(uint32_t)(bb?bb[i]:0)*f->mo;
        tt>>=f->sh;
        eb[i]=(uint8_t)(tt>255?255:tt);
    }
#endif
}
static void nat_fx_part(void *ctx, uint32_t part, uint32_t parts) {
    const NatJob *J=(const NatJob *)ctx;
    uint32_t NB=(uint32_t)J->NB,t=0,j=part;
    while(j>=NB){j-=NB;t++;}
    while(t<(uint32_t)J->mt) {
        nat_fx_block(J,(int32_t)t,(int32_t)j);
        j+=parts;
        while(j>=NB){j-=NB;t++;}
    }
}
/* MAC + conversion for the batch: native_n per N block, then the finish
 * record drains HMX. Off Hexagon: the exact scalar model of the same tiles. */
static int nat_mac(const NatJob *J, int32_t *fi, int32_t *zero, uint8_t *completion, uint8_t *const *outp) {
    const Nprg2HmxOps *h=J->conv->hmx;
#if BLK_HVX
    HVX_Vector sentinel=Q6_V_vsplat_R(-1);
    for(uint32_t i=0;i<2048;i+=128)*(HVX_Vector *)(completion+i)=sentinel;
    __asm__ volatile("barrier":::"memory");
#if NPRG2_PLANES
    for(int32_t j=0;j<J->NB;j++)
        h->nativep_n(J->A,(void *)(J->B+(size_t)j*J->KB*1024),fi+j*128,(void *const *)(outp+(size_t)j*J->mt),
                     (uint32_t)J->KB,(uint32_t)J->mt);
#else
    for(int32_t j=0;j<J->NB;j++)
        h->native_n(J->A,(void *)(J->B+(size_t)j*J->KB*1024),fi+j*64,(void *const *)(outp+(size_t)j*J->mt),
                    (uint32_t)J->KB,(uint32_t)J->mt);
#endif
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
    for(int32_t j=0;j<J->NB;j++)for(int32_t t=0;t<J->mt;t++) {
        uint8_t *d=outp[(size_t)j*J->mt+(size_t)t];
        for(int p=0;p<64;p++)for(int n=0;n<32;n++) {
            int64_t acc=0;
            for(int32_t k=0;k<J->KB;k++) {
                const uint8_t *a=J->A+((size_t)t*J->KB+(size_t)k)*2048+(size_t)p*32;
                const int8_t *b=(const int8_t *)J->B+((size_t)j*J->KB+(size_t)k)*1024;
                for(int c4=0;c4<32;c4++) acc+=(int64_t)a[c4]*b[((c4/4)*32+n)*4+c4%4];
            }
            int32_t ch=32*j+n;
            int64_t q=(acc+J->corr[ch])>>J->pre[ch];
#if NPRG2_PLANES
            d[p*32+n]=(uint8_t)(q&255);
#ifdef PLANE_SAT
            {int64_t v=(q>>8)+239;d[2048+p*32+n]=(uint8_t)(v<0?0:v>255?255:v);}
#else
            {int64_t v=(q>>8)+16;d[2048+p*32+n]=(uint8_t)(v<0?0:v>255?255:v);}
#endif
#else
            int64_t v=((2*q+1)*J->Mq[ch]+((int64_t)1<<14))>>15;
            d[p*32+n]=(uint8_t)(v<0?0:v>255?255:v);
#endif
        }
    }
    return 0;
#endif
}
/* Silicon (2026-11-02): reading the input blocks in place (pool-wide Rt
 * deltas, a workspace zero block) wedges the CDSP on the first call, and the
 * ISS accepts it. Staging each strip's 3 block rows into one contiguous window
 * (small positive deltas, as QNN's probe) and skipping the first two stores
 * runs exact. SLIDE_INPLACE keeps the hanging form for experiments. */
#if defined(NPRG2_SLIDE3) && NPRG2_SLIDE3 && !defined(SLIDE_INPLACE)
#ifndef SLIDE_STAGE
#define SLIDE_STAGE 1
#endif
#ifndef SLIDE_NODUMMY
#define SLIDE_NODUMMY 1
#endif
#endif
#ifdef SLIDE_LIMIT
static uint32_t slide_used;
#endif
#if BLK_HVX
/* Staging-free 3x3 stride 1 (qd_hmx_slide3_n): the window of output block
 * (by, bx) is input rows 8by+7.., columns 8bx+7.. of the command's input view
 * (origin p ≡ 7 mod 8), read straight from the input blocks: per strip, input
 * block rows R, R+1, R+2 and columns C0..C0+bw+1. Weights: the reversed
 * unpacked 3x3 order (ktab = cb*9 + ky*3 + (2-kx)), i.e. the hardware's tap
 * order. Output blocks are written in place (2048-aligned) or into C blocks
 * for the HVX epilogue. Same arithmetic as the native path (Fi, cvt.ub).
 * 1 ran (or would, dry), 0 not eligible (no side effects), -1 error. */
#if NPRG2_PLANES
/* The slide on the planes. Everything in a ConvPlan is fixed for a bound program
 * and pool table; only the weights and filter rows are read at run time, so a run
 * can be repeated from the plan alone (nprg2_exec_reference_plan_run). The window:
 * per (column, channel block) group i, blocks 2i, 2i+1, 2i+2 of S. The taps read
 * only the last row of the first and the first row of the third block, so a group's
 * third block is the next group's first, and every Rt delta is the small positive
 * 2048 that silicon needs (reading the input blocks in place hangs it). */
typedef struct {
    const uint32_t *xblk, *oblk;
    const uint8_t *ft; const int8_t *wbase;
    uint32_t xcount, xslice, xnbw, xnbd, ocount, oslice, onbw, onbd;
    int32_t R0, C0, oR, oC, bw, bh, oh, NB, cbs, KB, ncol, zp;
    uint32_t valid, inplace, s2; /* s2: stride 2, cbs = output groups, four input channel blocks each */
} ConvPlan;
static size_t slide_ws(const ConvPlan *P) {
    return slide_fixed(P->NB)+(size_t)P->bh*P->ncol*P->NB*4096+((size_t)2*P->ncol*P->cbs+1)*2048+2048;
}
static size_t slide_near(const ConvPlan *P, size_t at) {
    (void)nbump(&at,(size_t)P->ncol*P->cbs*(P->s2?32:16));(void)nbump(&at,(size_t)P->ncol*8);
    return at+512; /* the vector record build may store up to one batch past its end */
}
#ifndef SLIDE_STAGE_SERIAL_BELOW
#define SLIDE_STAGE_SERIAL_BELOW 16 /* fewer (column, channel block) groups: staged on the HMX thread */
#endif
typedef struct {const ConvPlan *P; uint8_t *const *pool; uint8_t *S; int32_t Rb, groups;} SlideStage;
/* Groups i with i % parts == part; adjacent groups touch different rows of their shared block. */
static void slide_stage_part(void *ctx, uint32_t part, uint32_t parts) {
    const SlideStage *J=(const SlideStage *)ctx;const ConvPlan *P=J->P;uint8_t *const *pool=J->pool;
    const uint32_t dr=P->xnbw*P->xnbd;
    const HVX_Vector zv=Q6_V_vzero();
    const int32_t Rb=J->Rb,cbs=P->cbs;
    int32_t cc=0,cb=0;
    for(uint32_t i=0;i<part;i++){if(++cb==cbs){cb=0;cc++;}}
    for(uint32_t i=part;i<(uint32_t)J->groups;i+=parts) {
        HVX_Vector *dv=(HVX_Vector *)(J->S+(size_t)i*4096);
        const int32_t wb=P->C0+cc;
        const int live=Rb>=0 && wb>=0 && wb<(int32_t)P->xnbw;
        const uint32_t ia=(live?(((uint32_t)Rb*P->xnbw+(uint32_t)wb)*P->xnbd)+P->xslice:0)+(uint32_t)cb,ib=ia+dr,id=ib+dr;
        const uint32_t ea=live&&ia<P->xcount?P->xblk[ia]:0,eb=live&&ib<P->xcount?P->xblk[ib]:0,ed=live&&id<P->xcount?P->xblk[id]:0;
        const uint32_t pa=ea&127u,pb=eb&127u,pd=ed&127u;
        const HVX_Vector *sa=pa&&pool[pa]?(const HVX_Vector *)(pool[pa]+(ea&~127u)):NULL,
                         *sb=pb&&pool[pb]?(const HVX_Vector *)(pool[pb]+(eb&~127u)):NULL,
                         *sd=pd&&pool[pd]?(const HVX_Vector *)(pool[pd]+(ed&~127u)):NULL;
        if(sa){HVX_Vector v0=sa[14],v1=sa[15];dv[14]=v0;dv[15]=v1;} else {dv[14]=zv;dv[15]=zv;}
        if(sb) {for(int k=0;k<16;k++) dv[16+k]=sb[k];} else for(int k=0;k<16;k++) dv[16+k]=zv;
        if(sd){HVX_Vector v0=sd[0],v1=sd[1];dv[32]=v0;dv[33]=v1;} else {dv[32]=zv;dv[33]=zv;}
        for(uint32_t k=0;k<parts;k++){if(++cb==cbs){cb=0;cc++;}}
    }
    /* This thread's stores complete before the join, and so before HMX reads them. */
    __asm__ volatile("barrier":::"memory");
    if(part<(uint32_t)J->groups) {
        typedef uint32_t W32S __attribute__((vector_size(128)));
        W32S v=*(volatile W32S *)(J->S+(size_t)part*4096+16*128);
        __asm__ volatile(""::"r"(v[0]):"memory");
    }
}
/* base: 2048-aligned workspace with Fi and the finish record in place; at: first
 * free near offset, slide_near() checked; mh as plane_fi_put. 1 done, -1 timeout. */
static int plane_slide_run(const ConvPlan *P, const Nprg2ReferenceConv *c, uint8_t *const *pool, Nprg2Parallel *par,
                           uint8_t *base, const uint8_t *mh, size_t at, int32_t *err) {
    const int32_t NB=P->NB,bw=P->bw,bh=P->bh,cbs=P->cbs,ncol=P->ncol,KB=P->KB,total=bh*bw,groups=ncol*cbs;
    const size_t fixed=slide_fixed(NB),cbytes=(size_t)bh*ncol*NB*4096,sbytes=((size_t)2*groups+1)*2048;
    uint64_t (*clock)(void)=c->phase?c->clock:NULL,t0=clock?clock():0,t1;
    int32_t *fi=(int32_t *)base,*zero=(int32_t *)(base+(size_t)NB*512);
    uint8_t *completion=base+fixed-2048,*C=base+fixed,*S=C+cbytes,*n=c->near;
    size_t o_rec=nbump(&at,(size_t)groups*(P->s2?32:16)),o_st=nbump(&at,(size_t)ncol*8);
    uint32_t *rec=(uint32_t *)(n+o_rec),*st=(uint32_t *)(n+o_st);
    typedef uint32_t W32 __attribute__((vector_size(128)));
    /* HVX stores must land before HMX reads them: on silicon a stale Fi can leave
     * HMX never completing (an HVX VTCM read after the stores forces them). */
    __asm__ volatile("barrier":::"memory");
    {W32 v=*(volatile W32 *)base;__asm__ volatile(""::"r"(v[0]):"memory");}
#ifdef SLIDE_TRIM
    /* The in-place walks below write their own records. */
    if(!P->s2 && !P->inplace)
#endif
    {
        uint32_t *r=rec;
        for(int32_t i=0;i<groups;i++,r+=4) {
            const uint32_t a=(uint32_t)(uintptr_t)S+(uint32_t)i*4096u;
            r[0]=a|0x740u;r[1]=2048u|0x71fu;r[2]=(a+2048u)|0x140u;r[3]=2048u|0x71fu;
        }
    }
    if(clock){t1=clock();c->phase[HMX_SETUP_SLOT]+=t1-t0;t0=t1;}
    uint32_t saved=c->hmx->enter?c->hmx->enter():0;
    uint32_t sparts=groups>=SLIDE_STAGE_SERIAL_BELOW?nprg2_parallel_parts(par,NPRG2_PAR_MAX_PARTS):1u;
#ifdef SLIDE_PROF
#define SP4(slot) do { if(clock){uint64_t x_=clock();c->phase[slot]+=x_-sp_t;sp_t=x_;} } while(0)
    uint64_t sp_t=clock?clock():0;
#else
#define SP4(slot) do {} while(0)
#endif
    for(int32_t by=0;by<bh;by++) {
#ifdef SLIDE_INPLACE_ALL
        if(P->s2) {
            /* one window per input channel block: tap rows 0 and 1, the block below or zero */
            static const uint16_t rl[4]={0x120,0x100,0x020,0x000};
            const uint32_t drr=P->xnbw*P->xnbd,zb=(uint32_t)(uintptr_t)S,pb=(uint32_t)(uintptr_t)pool[1];uint32_t *r=rec;
            const int32_t Rb=P->R0+by;
            if(P->inplace==2u && !by) {const HVX_Vector zv=Q6_V_vzero();for(int i=0;i<16;i++)((HVX_Vector *)S)[i]=zv;}
            for(int32_t cc=0;cc<ncol;cc++) {
                const int32_t wb=P->C0+cc;
                const int live=Rb>=0 && wb>=0 && wb<(int32_t)P->xnbw;
                const uint32_t i0=live?(((uint32_t)Rb*P->xnbw+(uint32_t)wb)*P->xnbd)+P->xslice:0;
                for(uint32_t q=0;q<(uint32_t)cbs*4u;q++,r+=2) {
                    const uint32_t ia=i0+q,ib=ia+drr;
                    const uint32_t ea=live&&ia<P->xcount?P->xblk[ia]:0,eb=live&&ib<P->xcount?P->xblk[ib]:0;
                    const uint32_t a=ea&127u?pb+(ea&~127u):zb,bb=eb&127u?pb+(eb&~127u):zb;
                    r[0]=a|rl[q&3u];r[1]=(bb-a)|0x71fu;
                }
            }
            __asm__ volatile("barrier":::"memory");
            {W32 v=*(volatile W32 *)(uintptr_t)(rec[0]&~2047u);__asm__ volatile(""::"r"(v[0]):"memory");}
        } else
        if(P->inplace) {
            /* HMX reads the input blocks where they are (needs the single VTCM page:
             * QDS_VTCM_SINGLE_PAGE). Absent blocks read a zero block at S. */
            const uint32_t drr=P->xnbw*P->xnbd,zb=(uint32_t)(uintptr_t)S,pb=(uint32_t)(uintptr_t)pool[1];
            const int32_t Rb=P->R0+by;
            if(P->inplace==2u && !by) {const HVX_Vector zv=Q6_V_vzero();for(int i=0;i<16;i++)((HVX_Vector *)S)[i]=zv;}
            const uint32_t first=((uint32_t)Rb*P->xnbw+(uint32_t)P->C0)*P->xnbd+P->xslice;
            if(Rb>=0 && P->C0>=0 && P->C0+ncol<=(int32_t)P->xnbw && (int32_t)P->xnbd==cbs && first+2u*drr+(uint32_t)groups<=P->xcount) {
                /* The window's rows are contiguous runs of the block table: 32 groups at a time. */
                const HVX_Vector m127=Q6_V_vsplat_R(127),mhi=Q6_V_vsplat_R(~127),pbv=Q6_V_vsplat_R((int32_t)pb),zbv=Q6_V_vsplat_R((int32_t)zb),zero=Q6_V_vzero(),
                                 ra=Q6_V_vsplat_R(0x740),rt=Q6_V_vsplat_R(0x71f),rc=Q6_V_vsplat_R(0x140);
                const uint32_t *ta=P->xblk+first,*tb=ta+drr,*td=tb+drr;
                HVX_Vector *rv=(HVX_Vector *)rec;
#define SLIDE_ADDR(e) Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV((e),m127),zero),zbv,Q6_Vw_vadd_VwVw(pbv,Q6_V_vand_VV((e),mhi)))
                for(int32_t i=0;i<groups;i+=32,rv+=4) {
                    const HVX_Vector ea=*(const HVX_UVector *)(ta+i),eb=*(const HVX_UVector *)(tb+i),ed=*(const HVX_UVector *)(td+i);
                    const HVX_Vector a=SLIDE_ADDR(ea),bb=SLIDE_ADDR(eb),d=SLIDE_ADDR(ed);
                    const HVX_Vector r0=Q6_V_vor_VV(a,ra),r1=Q6_V_vor_VV(Q6_Vw_vsub_VwVw(bb,a),rt),r2=Q6_V_vor_VV(bb,rc),r3=Q6_V_vor_VV(Q6_Vw_vsub_VwVw(d,bb),rt);
                    const HVX_VectorPair w01=Q6_W_vshuff_VVR(r1,r0,-4),w23=Q6_W_vshuff_VVR(r3,r2,-4);
                    const HVX_VectorPair lo=Q6_W_vshuff_VVR(Q6_V_lo_W(w23),Q6_V_lo_W(w01),-8),hi=Q6_W_vshuff_VVR(Q6_V_hi_W(w23),Q6_V_hi_W(w01),-8);
                    rv[0]=Q6_V_lo_W(lo);rv[1]=Q6_V_hi_W(lo);rv[2]=Q6_V_lo_W(hi);rv[3]=Q6_V_hi_W(hi);
                }
#undef SLIDE_ADDR
            } else {
                uint32_t *r=rec;
                for(int32_t cc=0;cc<ncol;cc++) {
                    const int32_t wb=P->C0+cc;
                    const int live=Rb>=0 && wb>=0 && wb<(int32_t)P->xnbw;
                    const uint32_t i0=live?(((uint32_t)Rb*P->xnbw+(uint32_t)wb)*P->xnbd)+P->xslice:0;
                    for(int32_t cb=0;cb<cbs;cb++,r+=4) {
                        const uint32_t ia=i0+(uint32_t)cb,ib=ia+drr,id=ib+drr;
                        const uint32_t ea=live&&ia<P->xcount?P->xblk[ia]:0,eb=live&&ib<P->xcount?P->xblk[ib]:0,ed=live&&id<P->xcount?P->xblk[id]:0;
                        const uint32_t a=ea&127u?pb+(ea&~127u):zb,bb=eb&127u?pb+(eb&~127u):zb,d=ed&127u?pb+(ed&~127u):zb;
                        r[0]=a|0x740u;r[1]=(bb-a)|0x71fu;r[2]=bb|0x140u;r[3]=(d-bb)|0x71fu;
                    }
                }
            }
            __asm__ volatile("barrier":::"memory");
            {W32 v=*(volatile W32 *)(uintptr_t)(rec[0]&~2047u);__asm__ volatile(""::"r"(v[0]):"memory");}
        } else
#endif
        {
        SlideStage SJ={.P=P,.pool=pool,.S=S,.Rb=P->R0+by,.groups=groups};
        if(sparts>1) nprg2_parallel_for(par,NPRG2_PAR_CONV,sparts,slide_stage_part,&SJ);
        else slide_stage_part(&SJ,0,1);
        }
        __asm__ volatile("barrier":::"memory");
        {W32 v=*(volatile W32 *)(S+sbytes-2048);__asm__ volatile(""::"r"(v[0]):"memory");}
        SP4(4);
        if(clock){t1=clock();c->phase[0]+=t1-t0;t0=t1;}
        for(int32_t j=0;j<NB;j++) {
            /* whole tiles, realigned by plane_part: column 0 holds no output pixel */
            st[0]=0;st[1]=0;
            for(int32_t cc=P->s2?0:1;cc<ncol;cc++) {
                const uint32_t d=(uint32_t)(uintptr_t)(C+(((size_t)by*ncol+cc)*NB+(size_t)j)*4096);
                st[2*cc]=d;st[2*cc+1]=d+2048u;
            }
#ifdef SLIDE_INPLACE_ALL
            if(P->s2) c->hmx->slide2p_n(rec,P->wbase+(size_t)j*KB*1024,fi+j*128,st,(uint32_t)cbs,(uint32_t)ncol);
            else
#endif
            c->hmx->slide3p_n(rec,P->wbase+(size_t)j*KB*1024,fi+j*128,st,(uint32_t)cbs,(uint32_t)ncol);
        }
        if(clock){t1=clock();c->phase[1]+=t1-t0;t0=t1;}
    }
    int e=0;
    {
        HVX_Vector sentinel=Q6_V_vsplat_R(-1);
        for(uint32_t i=0;i<2048;i+=128)*(HVX_Vector *)(completion+i)=sentinel;
        __asm__ volatile("barrier":::"memory");
        c->hmx->finish(zero,completion);
        uint64_t end=c->hmx->clock?c->hmx->clock()+1000000000ull:0;
        for(uint64_t spin=0;;spin++) {
            W32 v=*(volatile W32 *)completion;
            if(v[0]==0) break;
            if(c->hmx->clock?c->hmx->clock()>end:spin>100000000ull){e=1;break;}
        }
        __asm__ volatile("barrier":::"memory");
    }
    if(c->hmx->leave) c->hmx->leave(saved);
    SP4(5);
    if(clock){t1=clock();c->phase[1]+=t1-t0;t0=t1;}
    if(e){*err=E_RANGE;return -1;}
    SP4(7);
    PlaneJob J={.C=C,.mh=mh,.mt=total,.NB=NB,.bw=bw,.ncol=ncol,.shift=1,.tile0=P->s2?0:1,
                .rows=P->oh&7,.partial_from=(P->oh&7)?(bh-1)*bw:total,
                .pool=pool,.oblk=P->oblk,.ocount=P->ocount,.oslice=P->oslice,.onbw=P->onbw,.onbd=P->onbd,.oR=P->oR,.oC=P->oC};
    uint32_t parts=nprg2_parallel_parts(par,NPRG2_PAR_MAX_PARTS);
    if(parts>1) nprg2_parallel_for(par,NPRG2_PAR_CONV,parts,plane_part,&J);
    else plane_part(&J,0,1);
    if(clock){t1=clock();c->phase[3]+=t1-t0;t0=t1;}
#ifdef SLIDE_PROF
    if(clock){uint64_t x_=clock();c->phase[2]+=x_-sp_t;}
#endif
    return 1;
}
/* Plans of the bound program, by command index (lane 0, the conv list). */
static ConvPlan *conv_plans;
static const Nprg2ReferenceConv *plan_conv;
static const void *plan_program;
static uint8_t *plan_pool[7];
static uint32_t plan_pool_bytes[7];
static const uint8_t *plan_commands;
static int hmx_slide_planes(G *g, const T *xt, const T *o, int32_t KB, int32_t NB, int32_t cbs, int32_t co, int32_t oh,
                            int32_t ow, const uint32_t *ktab, const int32_t *corr, const int32_t *pre, const int32_t *Mq,
                            const int8_t *wbase, const T *ft, const uint8_t *fipre, size_t at, int32_t *err, int s2) {
    const Nprg2ReferenceConv *c=g->conv;
    (void)co;
#ifdef SLIDE_INPLACE_ALL
    if(s2) {
        /* raw weights in place (s2ip), input origin on a block corner */
        if(!c->hmx->slide2p_n || cbs%4 || KB!=9*(cbs/4) || (xt->p[1]&7) || (xt->p[2]&7)) return 0;
        cbs/=4;
    } else
#else
    if(s2) return 0;
#endif
    {
    if(!c->hmx->slide3p_n || KB!=9*cbs || cbs<1 || cbs>8) return 0;
    if((xt->p[1]&7)!=7 || (xt->p[2]&7)!=7) return 0;
    for(int32_t cb=0;cb<cbs;cb++)for(int32_t ky=0;ky<3;ky++)for(int32_t kp=0;kp<3;kp++)
        if(ktab[cb*9+ky*3+kp]!=((uint32_t)ky|(uint32_t)(2-kp)<<8|(uint32_t)cb<<16)) return 0;
    }
    if((uintptr_t)wbase&1023u) return 0;
    const int32_t bw=ow/8,bh=(oh+7)/8;
    ConvPlan P={.xblk=xt->blocks,.oblk=o->blocks,.ft=g->pool[ft->pool]+ft->offset,.wbase=wbase,
                .xcount=xt->nblocks,.xslice=xt->slice,.xnbw=(uint32_t)(xt->m[2]+7)/8u,.xnbd=(uint32_t)(xt->m[3]+31)/32u,
                .ocount=o->nblocks,.oslice=o->slice,.onbw=(uint32_t)(o->m[2]+7)/8u,.onbd=(uint32_t)(o->m[3]+31)/32u,
                .R0=xt->p[1]>>3,.C0=xt->p[2]>>3,.oR=o->p[1]>>3,.oC=o->p[2]>>3,.bw=bw,.bh=bh,.oh=oh,.NB=NB,.cbs=cbs,.KB=KB,
                .ncol=s2?bw+1:bw+2,.zp=xt->zp,.valid=1,.s2=(uint32_t)s2};
#ifdef SLIDE_INPLACE_ALL
    if(!g->dry) {
        /* every present block of the windows in pool 1 and 2048-aligned; 2: some absent */
        uint32_t ok=1u;const uint32_t drr=P.xnbw*P.xnbd;
        const int32_t nrow=s2?2:3,nq=s2?cbs*4:cbs;
        for(int32_t by=0;by<bh && ok;by++)for(int32_t cc=0;cc<P.ncol && ok;cc++) {
            const int32_t wb=P.C0+cc,Rb=P.R0+by;
            const int live=Rb>=0 && wb>=0 && wb<(int32_t)P.xnbw;
            const uint32_t i0=live?(((uint32_t)Rb*P.xnbw+(uint32_t)wb)*P.xnbd)+P.xslice:0;
            for(int32_t cb=0;cb<nq && ok;cb++)for(int32_t rr=0;rr<nrow;rr++) {
                const uint32_t ix=i0+(uint32_t)cb+(uint32_t)rr*drr,ee=live&&ix<P.xcount?P.xblk[ix]:0,pl=ee&127u;
                if(!pl){ok=2u;continue;}
                if(pl!=1u || !g->pool[1] || (ee&(~127u)&2047u)){ok=0;break;}
            }
        }
        P.inplace=ok;
    } else P.inplace=1u;
    if(s2 && !P.inplace) return 0; /* stride 2 has no staged form */
#endif
    size_t o_mh=nbump(&at,(size_t)NB*128);
    if(slide_near(&P,at)>c->near_bytes || c->workspace_bytes<slide_ws(&P)) return 0;
    g->wused=0;
    uint8_t *ws=walloc(g,slide_ws(&P));
    if(!ws){*err=E_MEMORY;return -1;}
    conv_path[g->call->execution_lane<MEMO_LANES?g->call->execution_lane:0u]=5;
    if(g->dry) return 1;
    uint8_t *base=(uint8_t *)(((uintptr_t)ws+2047)&~(uintptr_t)2047);
    plane_fi_put(base,NB,corr,pre,Mq,c->near+o_mh);
    int r=plane_slide_run(&P,c,g->pool,g->call->parallel,base,c->near+o_mh,at,err);
#ifndef NO_CONV_PLANS
    /* Worth repeating only with the vector parameters the plan also uses. */
    if(r==1 && fipre && conv_plans && plan_conv==c && (g->variant==33 || g->variant==36) && g->call->execution_lane==0 &&
       g->call->program==plan_program && g->call->command_index<MEMO_COMMANDS && !((uintptr_t)P.ft&127u))
        conv_plans[g->call->command_index]=P;
#else
    (void)fipre;
#endif
    return r;
}
#endif
static int hmx_slide(G *g, const T *xt, const T *o, int32_t KB, int32_t NB, int32_t cbs, int32_t co, int32_t oh,
                     int32_t ow, const uint32_t *ktab, const int32_t *corr, const int32_t *pre, const int32_t *Mq,
                     const int8_t *wbase, int staged, const uint8_t *fipre, size_t at, int32_t *err) {
    const Nprg2ReferenceConv *c=g->conv;
    if(!c->hmx->slide3_n) return 0;
    if(KB!=9*cbs || cbs<1 || cbs>8) return 0;
#ifdef SLIDE_LIMIT
    if(!g->dry && slide_used>=SLIDE_LIMIT) return 0;
#endif
#ifdef SLIDE_ONLY_CBS
    if(cbs!=SLIDE_ONLY_CBS) return 0;
#endif
    if((xt->p[1]&7)!=7 || (xt->p[2]&7)!=7) return 0;
    for(int32_t cb=0;cb<cbs;cb++)for(int32_t ky=0;ky<3;ky++)for(int32_t kp=0;kp<3;kp++)
        if(ktab[cb*9+ky*3+kp]!=((uint32_t)ky|(uint32_t)(2-kp)<<8|(uint32_t)cb<<16)) return 0;
    if((uintptr_t)wbase&1023u) return 0;
    const int32_t bw=ow/8,bh=(oh+7)/8,total=bh*bw,ncol=bw+2;
    const int64_t xnbw=((int64_t)xt->m[2]+7)/8,xnbd=((int64_t)xt->m[3]+31)/32;
    const int32_t R0=xt->p[1]>>3,C0=xt->p[2]>>3;
    /* every present input block of the windows 2048-aligned (else not eligible):
     * only when HMX reads them in place, staged windows take any block */
    {
        Memo *mm=g->dry?NULL:gmemo(g);
        uint8_t *mb=mm?&mm->slide[g->call->command_index]:NULL;
        if(!mb || !*mb) {
            int ok=1;
            for(int32_t r=0;r<bh+2 && ok;r++)for(int32_t cc=0;cc<ncol && ok;cc++)for(int32_t cb=0;cb<cbs;cb++) {
                const uint8_t *p=hmx_row(g->pool,xt,(int64_t)(R0+r)*8,C0+cc,xnbw,xnbd,cb);
                if(p && ((uintptr_t)p&2047u)){ok=0;break;}
#ifdef SLIDE_POSDELTA
                if(r) {
                    const uint8_t *q=hmx_row(g->pool,xt,(int64_t)(R0+r-1)*8,C0+cc,xnbw,xnbd,cb);
                    if(p && q && p<q){ok=0;break;}
                }
#endif
            }
            if(mb && !g->dry) *mb=ok?1:2;
            if(!ok) return 0;
        } else if(*mb==2) return 0;
    }
    size_t o_rec=nbump(&at,(size_t)ncol*cbs*16),o_st=nbump(&at,(size_t)ncol*8),o_tyx=nbump(&at,(size_t)total*8);
    if(at>c->near_bytes) return 0;
    const size_t fixed=((size_t)NB*256+256+2048*2+2047)&~(size_t)2047,cbytes=staged?(size_t)total*NB*2048:0;
#ifdef SLIDE_STAGE
    /* bisect: copy each strip's 3 input block rows into one contiguous
     * workspace window (pads zero-filled), so every Rt delta is the small
     * positive row stride, as in the silicon QNN probe (xwb1022). */
    const size_t sbytes=(size_t)3*ncol*cbs*2048;
#else
    const size_t sbytes=0;
#endif
    const size_t need=fixed+3*2048+cbytes+sbytes+2048;
    if(c->workspace_bytes<need) return 0;
    g->wused=0;
    uint8_t *ws=walloc(g,need);
    if(!ws){*err=E_MEMORY;return -1;}
    conv_path[g->call->execution_lane<MEMO_LANES?g->call->execution_lane:0u]=5;
    if(g->fx.kind) g->fx_ok=1;
    if(g->dry) return 1;
#ifdef SLIDE_LIMIT
    slide_used++;
#endif
#ifdef SLIDE_TRACE
    printf("SLIDE cmd %u cbs=%d NB=%d oh=%d ow=%d staged=%d\n",(unsigned)g->call->command_index,(int)cbs,(int)NB,(int)oh,(int)ow,staged);
#endif
    uint64_t (*clock)(void)=c->phase?c->clock:NULL,t0=clock?clock():0,t1;
    uint8_t *base=(uint8_t *)(((uintptr_t)ws+2047)&~(uintptr_t)2047);
    int32_t *fi=(int32_t *)base,*zero=(int32_t *)(base+(size_t)NB*HMX_FI_BYTES);
    uint8_t *completion=base+fixed-2048,*dummy=base+fixed,*zblk=dummy+4096,*C=zblk+2048,*S=C+cbytes;
    (void)S;
    uint8_t *n=c->near;
    {
        int32_t *nf=NULL;
        if(!fipre) {
            nf=(int32_t *)(n+nbump(&at,(size_t)NB*256+256));
            if(at>c->near_bytes) return 0;
        }
        hmx_fi_put(base,NB,co,fipre,nf,corr,pre,Mq);
    }
    {HVX_Vector z=Q6_V_vzero();for(int i=0;i<16;i++)((HVX_Vector *)zblk)[i]=z;}
    /* HVX stores (Fi, zero block) must land before HMX reads them: on silicon
     * a stale Fi can leave HMX never completing (the ISS orders them anyway). */
    __asm__ volatile("barrier":::"memory");
    { /* an HVX VTCM read after the stores forces their completion (HMX.md) */
        typedef uint32_t W32 __attribute__((vector_size(128)));
        W32 a=*(volatile W32 *)base,b=*(volatile W32 *)zblk;
        __asm__ volatile(""::"r"(a[0]),"r"(b[0]):"memory");
    }
    uint32_t *rec=(uint32_t *)(n+o_rec),*st=(uint32_t *)(n+o_st);
    int32_t *tyx=(int32_t *)(n+o_tyx);
    const uint32_t onbw=(uint32_t)(o->m[2]+7)/8u,onbd=(uint32_t)(o->m[3]+31)/32u,ocount=o->nblocks;
    const uint32_t *oblk=o->blocks;
    if(clock){t1=clock();c->phase[HMX_SETUP_SLOT]+=t1-t0;t0=t1;}
    uint32_t saved=c->hmx->enter?c->hmx->enter():0;
    for(int32_t by=0;by<bh;by++) {
        for(int32_t cc=0;cc<ncol;cc++)for(int32_t cb=0;cb<cbs;cb++) {
            uint32_t *r=rec+((size_t)cc*cbs+cb)*4;
            const uint8_t *a=hmx_row(g->pool,xt,(int64_t)(R0+by)*8,C0+cc,xnbw,xnbd,cb),
                          *b=hmx_row(g->pool,xt,(int64_t)(R0+by+1)*8,C0+cc,xnbw,xnbd,cb),
                          *d=hmx_row(g->pool,xt,(int64_t)(R0+by+2)*8,C0+cc,xnbw,xnbd,cb);
#ifdef SLIDE_STAGE
            {
                const uint8_t *src[3]={a,b,d};
                for(int rr=0;rr<3;rr++) {
                    HVX_Vector *dv=(HVX_Vector *)(S+(((size_t)rr*ncol+cc)*cbs+cb)*2048);
                    const int i0=0,i1=16;
                    if(src[rr]) {const HVX_Vector *sv=(const HVX_Vector *)src[rr];for(int i=i0;i<i1;i++)dv[i]=sv[i];}
                    else {HVX_Vector z=Q6_V_vzero();for(int i=i0;i<i1;i++)dv[i]=z;}
                }
                a=S+(((size_t)0*ncol+cc)*cbs+cb)*2048;b=S+(((size_t)1*ncol+cc)*cbs+cb)*2048;d=S+(((size_t)2*ncol+cc)*cbs+cb)*2048;
            }
#endif
            if(!a)a=zblk;if(!b)b=zblk;if(!d)d=zblk;
            r[0]=(uint32_t)(uintptr_t)a|0x740u;r[1]=((uint32_t)(uintptr_t)b-(uint32_t)(uintptr_t)a)|0x71fu;
            r[2]=(uint32_t)(uintptr_t)b|0x140u;r[3]=((uint32_t)(uintptr_t)d-(uint32_t)(uintptr_t)b)|0x71fu;
        }
#ifdef SLIDE_STAGE
        __asm__ volatile("barrier":::"memory");
        {typedef uint32_t W32 __attribute__((vector_size(128)));W32 v=*(volatile W32 *)(S+sbytes-2048);__asm__ volatile(""::"r"(v[0]):"memory");}
#endif
        for(int32_t j=0;j<NB;j++) {
            /* after input column cc the accumulator holds virtual tile cc; the
             * store (QNN's: Rs = block|0xe0, Rt = 0x700) writes output block
             * cc-2 = previous tile's pixels 1..7 and this tile's pixel 0. */
            for(int32_t cc=0;cc<ncol;cc++) {
                int32_t bx=cc-2;uint8_t *d=dummy;
                if(bx>=0 && bx<bw) {
                    if(staged) d=C+((size_t)(by*bw+bx)*NB+(size_t)j)*2048;
                    else {
                        uint32_t idx=(((uint32_t)(o->p[1]>>3)+(uint32_t)by)*onbw+(uint32_t)(o->p[2]>>3)+(uint32_t)bx)*onbd+o->slice+(uint32_t)j;
                        if(idx<ocount){uint32_t e=oblk[idx],pool=e&127u;if(pool && g->pool[pool]) d=g->pool[pool]+(e&~127u);}
                    }
                }
#ifdef SLIDE_NODUMMY
                /* QNN skips the first stores (r28/r31) instead of storing to a dummy */
                if(bx<0) {st[2*cc]=0;st[2*cc+1]=0;continue;}
#endif
                st[2*cc]=(uint32_t)(uintptr_t)d|0xe0u;st[2*cc+1]=0x700u;
            }
            c->hmx->slide3_n(rec,wbase+(size_t)j*KB*1024,fi+j*64,st,(uint32_t)cbs,(uint32_t)ncol);
        }
    }
    int e=0;
    {
        HVX_Vector sentinel=Q6_V_vsplat_R(-1);
        for(uint32_t i=0;i<2048;i+=128)*(HVX_Vector *)(completion+i)=sentinel;
        __asm__ volatile("barrier":::"memory");
        c->hmx->finish(zero,completion);
        uint64_t end=c->hmx->clock?c->hmx->clock()+1000000000ull:0;
        for(uint64_t spin=0;;spin++) {
            typedef uint32_t HmxWords __attribute__((vector_size(128)));
            HmxWords v=*(volatile HmxWords *)completion;
            if(v[0]==0) break;
            if(c->hmx->clock?c->hmx->clock()>end:spin>100000000ull){e=1;break;}
        }
        __asm__ volatile("barrier":::"memory");
    }
    if(c->hmx->leave) c->hmx->leave(saved);
    if(clock){t1=clock();c->phase[1]+=t1-t0;t0=t1;}
    if(e){*err=E_RANGE;return -1;}
    if(staged) {
        int32_t y=0,x=0;
        for(int32_t t=0;t<total;t++){tyx[2*t]=y;tyx[2*t+1]=x;if(++x==bw){x=0;y++;}}
        NatJob J={.g=g,.pool=g->pool,.conv=c,.xt=xt,.o=o,.fx=g->fx.kind?&g->fx:NULL,.KB=KB,.NB=NB,.cbs=cbs,.bw=bw,.mt=total,.C=C,.tyx=tyx};
        uint32_t parts=nprg2_parallel_parts(g->call->parallel,NPRG2_PAR_MAX_PARTS);
        if(parts>1) nprg2_parallel_for(g->call->parallel,NPRG2_PAR_CONV,parts,nat_fx_part,&J);
        else nat_fx_part(&J,0,1);
        if(clock){t1=clock();c->phase[3]+=t1-t0;t0=t1;}
    }
    return 1;
}
#endif
/* 1 ran (or would, dry), 0 not eligible, -1 error. Called by hmx_conv with
 * its validated geometry, tables and parameters. */
#endif
/* Stride-2 conv weights. The raw 3x3 tensor holds, per output block j, the
 * 9*cbO tiles (cb, h, x) in raw order, each a contiguous 1024-byte K block,
 * and every one of them feeds exactly one (ky, kx, cb2) of the 2x2 conv over
 * the space-to-depth channels. Taking K in raw order (k = cb*9 + h*3 + x) makes
 * B the raw tensor itself (block j at j*KB*1024): no copy, no scans. The MAC
 * sums are order independent. */
#ifdef S2_COUNT
static uint32_t s2_ip_taken,s2_bc_taken;
#endif
#ifndef S2_FORCE_BCOPY
#define S2_FORCE_BCOPY 0
#endif
static void s2_ktab(uint32_t *ktab, int32_t cbO) {
    uint32_t t9[9];
    for(int r=0;r<9;r++) {
        int y3=S2_TAPS[r][0],x3=S2_TAPS[r][1];
        t9[r]=(uint32_t)(y3>>1)|(uint32_t)((x3+1)>>1)<<8|(uint32_t)((y3&1)*2+(x3&1))<<16;
    }
    for(int32_t cb=0;cb<cbO;cb++)for(int r=0;r<9;r++) ktab[cb*9+r]=t9[r]+((uint32_t)(4*cb)<<16);
}
/* Fallback (weights not 1024-aligned): the present K blocks' raw offsets in
 * 2x2 order, copied per command. Returns KB or -1. */
static int32_t s2_scan(const T *wt, int32_t cbs, uint32_t *s2off) {
    int32_t KB=0;
    for(int32_t ky=0;ky<2;ky++)for(int32_t kx=0;kx<2;kx++)for(int32_t cb=0;cb<cbs;cb++) {
        uint32_t t=weight_tile(wt,1,0,0,ky,kx,cb*8,0);
        if(t==NO_TILE) continue;
        for(int32_t q=1;q<8;q++) if(weight_tile(wt,1,0,0,ky,kx,cb*8+q,0)!=t+(uint32_t)q*128u) return -1;
        if(KB>=255) return -1;
        s2off[KB++]=t;
    }
    return KB;
}
#if NPRG2_HW_ROUNDING || NPRG2_PLANES
#if NPRG2_PLANES
/* The non-slide planes path (1x1, stride 2, odd alignments), repeatable from a plan
 * as the slide is: geometry and tables are fixed, Fi and weights read at run time. */
typedef struct {
    T xt;
    const uint32_t *oblk;
    const uint8_t *ft; const int8_t *wbase;
    uint32_t ocount, oslice, onbw, onbd, wstride, wrows;
    int32_t oR, oC, bw, total, oh, NB, KB, cbs, mt, xrows, xwb0, xnwb, kvec, js;
    uint8_t one, bcopy, kkind, valid; /* kkind: 0 packed, 1 unpacked, 2 unpacked reversed, 3 stride 2 in place */
} NativePlan;
static void plan_ktab(uint32_t *ktab, const NativePlan *Q) {
    const int32_t cbs=Q->cbs,KB=Q->KB;
    if(Q->kkind==3) {s2_ktab(ktab,KB/9);return;}
    if(Q->one) {for(int32_t k=0;k<KB;k++) ktab[k]=(uint32_t)k<<16;return;}
    const int32_t W=KB/cbs==9?3:KB/cbs==4?2:1,H=W;
    int32_t k=0;
    if(Q->kkind==0) {for(int32_t h=0;h<H;h++)for(int32_t x=0;x<W;x++)for(int32_t cb=0;cb<cbs;cb++,k++)
        ktab[k]=(uint32_t)h|(uint32_t)x<<8|(uint32_t)cb<<16;}
    else for(int32_t cb=0;cb<cbs;cb++)for(int32_t h=0;h<H;h++)for(int32_t x=0;x<W;x++,k++)
        ktab[k]=(uint32_t)h|(uint32_t)(Q->kkind==2?W-1-x:x)<<8|(uint32_t)cb<<16;
}
static size_t plane_native_near(const NativePlan *Q, size_t at, size_t *o_xtab, size_t *o_tyx, size_t *o_out, size_t *o_pout) {
    *o_xtab=nbump(&at,(size_t)Q->xrows*(size_t)Q->xnwb*(size_t)Q->cbs*sizeof(uint8_t *));
    *o_tyx=nbump(&at,(size_t)Q->mt*8);
    *o_out=nbump(&at,(size_t)Q->NB*Q->mt*sizeof(uint8_t *));
    *o_pout=nbump(&at,(size_t)Q->NB*Q->mt*sizeof(uint8_t *));
    return at;
}
static size_t plane_native_ws(const NativePlan *Q) {
    const size_t bsz=Q->bcopy?((size_t)Q->KB*Q->NB*1024+2047)&~(size_t)2047:0;
    return slide_fixed(Q->NB)+bsz+2048+(size_t)Q->mt*((size_t)Q->KB*2048+(size_t)Q->NB*4096)+2048;
}
/* base: Fi and the finish record in place; ktab filled; xtab/tyx/outp/pout free near
 * space. s2off: raw offsets of a stride 2 B copy, or NULL (all 0). corr/pre/Mq: host
 * scalar model only. */
static int plane_native_run(const NativePlan *Q, const Nprg2ReferenceConv *c, uint8_t *const *pool, Nprg2Parallel *par,
                            uint8_t *base, const uint8_t *mh, const uint32_t *ktab, uint8_t **xtab, int32_t *tyx, uint8_t **outp,
                            uint8_t **pout, const uint32_t *s2off, const int32_t *corr, const int32_t *pre, const int32_t *Mq, int32_t *err) {
    const int32_t NB=Q->NB,KB=Q->KB,bw=Q->bw,total=Q->total,mt=Q->mt;
    const T *xt=&Q->xt;
    uint64_t (*clock)(void)=c->phase?c->clock:NULL,t0=clock?clock():0,t1;
    const size_t fixed=slide_fixed(NB),bsz=Q->bcopy?((size_t)KB*NB*1024+2047)&~(size_t)2047:0;
    int32_t *fi=(int32_t *)base,*zero=(int32_t *)(base+(size_t)NB*512);
    uint8_t *completion=base+fixed-2048,*Bc=base+fixed,*A=Bc+bsz+2048,*C=A+(size_t)mt*KB*2048;
    const int8_t *wbase=Q->wbase;
    if(Q->bcopy) {
        for(int32_t j=0;j<NB;j++)for(int32_t k=0;k<KB;k++) {
            uint8_t *d=Bc+((size_t)j*KB+(size_t)k)*1024;
            qd_vcopy(d,wbase+(s2off?s2off[k]:0u)+(size_t)j*(size_t)Q->js,(size_t)Q->kvec*128);
            if(Q->kvec<8) memset(d+(size_t)Q->kvec*128,0,(size_t)(8-Q->kvec)*128);
        }
        wbase=(const int8_t *)Bc;
    }
    if(!Q->one) hmx_tab(xtab,pool,xt,xt->p[1],Q->xrows,Q->xwb0,Q->xnwb,Q->cbs);
    NatJob J={.pool=pool,.conv=c,.xt=xt,.KB=KB,.NB=NB,.cbs=Q->cbs,.bw=bw,.xwb0=Q->xwb0,.xnwb=Q->xnwb,
              .ktab=ktab,.xtab=xtab,.A=A,.C=C,.tyx=tyx,.pre=pre,.Mq=Mq,.corr=corr,.B=(const uint8_t *)wbase,
              .one=Q->one,.R0=xt->p[1]>>3,.C0=xt->p[2]>>3,.xnbw=((int64_t)xt->m[2]+7)/8,.xnbd=((int64_t)xt->m[3]+31)/32};
    if(clock){t1=clock();c->phase[HMX_SETUP_SLOT]+=t1-t0;t0=t1;}
    uint32_t parts=nprg2_parallel_parts(par,NPRG2_PAR_MAX_PARTS);
    uint32_t saved=c->hmx->enter?c->hmx->enter():0;
    int e=0;
    int32_t by=0,bx=0;
    for(int32_t m0=0;m0<total && !e;m0+=mt) {
        J.m0=m0;J.mt=total-m0<mt?total-m0:mt;
        for(int32_t t=0;t<J.mt;t++) {
            tyx[2*t]=by;tyx[2*t+1]=bx;
            const uint32_t idx=(((uint32_t)Q->oR+(uint32_t)by)*Q->onbw+(uint32_t)Q->oC+(uint32_t)bx)*Q->onbd+Q->oslice;
            for(int32_t j=0;j<NB;j++) {
                uint8_t *ob=NULL;
                if(idx+(uint32_t)j<Q->ocount) {
                    const uint32_t ee=Q->oblk[idx+(uint32_t)j],pl=ee&127u;
                    if(pl && pool[pl]) ob=pool[pl]+(ee&~127u);
                }
                pout[(size_t)t*NB+(size_t)j]=ob;
                outp[(size_t)j*J.mt+(size_t)t]=C+((size_t)t*NB+(size_t)j)*4096;
            }
            if(++bx==bw){bx=0;by++;}
        }
#if BLK_HVX && defined(ONE_INPLACE)
        /* 1x1 over present, 2048-aligned input blocks: HMX reads them where they are. */
        const uint8_t **ap=(const uint8_t **)A;
        int inplace=Q->one && xtab && c->hmx->nativep1_n;
        if(inplace) {
            const uint32_t nbw=(uint32_t)J.xnbw,nbd=(uint32_t)J.xnbd,kn=(uint32_t)KB;
            ap=(const uint8_t **)xtab; /* the 1x1 path has no row table: this near space is free */
            for(int32_t t=0;t<J.mt && inplace;t++) {
                const uint32_t i0=(((uint32_t)(J.R0+tyx[2*t]))*nbw+(uint32_t)(J.C0+tyx[2*t+1]))*nbd+xt->slice;
                for(uint32_t k=0;k<kn;k++) {
                    const uint32_t ix=i0+(ktab[k]>>16);
                    const uint32_t ee=ix<xt->nblocks?xt->blocks[ix]:0,pl=ee&127u;
                    const uint8_t *b=pl && pool[pl]?pool[pl]+(ee&~127u):NULL;
                    if(!b || ((uintptr_t)b&2047u)){inplace=0;break;}
                    ap[(size_t)t*kn+k]=b;
                }
            }
        }
        if(inplace) {
            typedef uint32_t W32I __attribute__((vector_size(128)));
            __asm__ volatile("barrier":::"memory");
            {W32I v=*(volatile W32I *)base;__asm__ volatile(""::"r"(v[0]):"memory");}
            HVX_Vector sentinel=Q6_V_vsplat_R(-1);
            for(uint32_t i=0;i<2048;i+=128)*(HVX_Vector *)(completion+i)=sentinel;
            __asm__ volatile("barrier":::"memory");
            if(clock){t1=clock();c->phase[0]+=t1-t0;t0=t1;}
            for(int32_t j=0;j<NB;j++)
                c->hmx->nativep1_n((const void *const *)ap,(void *)(wbase+(size_t)j*KB*1024),fi+j*128,(void *const *)(outp+(size_t)j*J.mt),(uint32_t)KB,(uint32_t)J.mt);
            c->hmx->finish(zero,completion);
            uint64_t end=c->hmx->clock?c->hmx->clock()+1000000000ull:0;
            for(uint64_t spin=0;;spin++) {
                W32I v=*(volatile W32I *)completion;
                if(v[0]==0) break;
                if(c->hmx->clock?c->hmx->clock()>end:spin>100000000ull){e=1;break;}
            }
            __asm__ volatile("barrier":::"memory");
            if(clock){t1=clock();c->phase[1]+=t1-t0;t0=t1;}
        } else
#endif
        {
        if(parts>1 && !(Q->one && (int64_t)J.mt*KB<NAT_SERIAL_BELOW)) nprg2_parallel_for(par,NPRG2_PAR_CONV,parts,nat_prep_part,&J);
        else nat_prep_part(&J,0,1);
        if(clock){t1=clock();c->phase[0]+=t1-t0;t0=t1;}
        e=nat_mac(&J,fi,zero,completion,outp);
        if(clock){t1=clock();c->phase[1]+=t1-t0;t0=t1;}
        }
        if(e) break;
        PlaneJob P={.C=C,.out=pout,.mh=mh,.Mq=Mq,.mt=J.mt,.NB=NB,.shift=0,.rows=Q->oh&7,
                    .partial_from=(Q->oh&7)?((Q->oh+7)/8-1)*bw-m0:J.mt};
        if(parts>1 && J.mt*NB>=PLANE_SERIAL_BELOW) nprg2_parallel_for(par,NPRG2_PAR_CONV,parts,plane_part,&P);
        else plane_part(&P,0,1);
        if(clock){t1=clock();c->phase[3]+=t1-t0;t0=t1;}
    }
    if(c->hmx->leave) c->hmx->leave(saved);
    if(e){*err=E_RANGE;return -1;}
    return 1;
}
#if BLK_HVX
static NativePlan *native_plans;
#endif
static int hmx_native(G *g, const T *xt, const T *o, int32_t KB, int32_t NB, int32_t cbs, int32_t co, int32_t oh,
                      int32_t ow, const uint32_t *ktab, const int32_t *corr, const int32_t *pre, const int32_t *Mq,
                      const int8_t *wbase, const T *ft, int bcopy, int32_t kvec, const uint32_t *s2off, uint32_t js, int s2ip, int32_t xrows, int32_t xwb0,
                      int32_t xnwb, uint8_t **xtab, const uint8_t *fipre, int hit, size_t at, int32_t *err, int s2, int packed, int reverse) {
    const Nprg2ReferenceConv *c=g->conv;
    (void)hit;
    /* 1x1 conv (ktab k = (0,0,k)), input origin on a block corner: no row table */
    int one=0;
#if NAT_1X1
    if(KB==cbs && (xt->p[1]&7)==0 && (xt->p[2]&7)==0) {
        one=1;
        for(int32_t k=0;k<KB;k++) if(ktab[k]!=((uint32_t)k<<16)) {one=0;break;}
    }
#endif
    if(g->fx.kind || (BLK_HVX && !c->hmx->nativep_n)) return 0;
    if(!plane_params_ok(co,corr,pre,(int64_t)KB*32*255*128)) return 0;
    if(!xtab && !one) return 0;
    if(ow%8 || o->p[1]%8 || o->p[2]%8) return 0; /* oh%8: full tiles computed, rows written */
#if BLK_HVX && defined(NPRG2_SLIDE3) && NPRG2_SLIDE3
    if(!bcopy && !s2ip) {
        int r=hmx_slide_planes(g,xt,o,KB,NB,cbs,co,oh,ow,ktab,corr,pre,Mq,wbase,ft,fipre,at,err,0);
        if(r) return r;
    }
#ifdef SLIDE_INPLACE_ALL
    if(s2ip) {
        int r=hmx_slide_planes(g,xt,o,KB,NB,cbs,co,oh,ow,ktab,corr,pre,Mq,wbase,ft,fipre,at,err,1);
        if(r) return r;
    }
#endif
#endif
    const int32_t bw=ow/8,total=((oh+7)/8)*bw;
    const size_t bsz=bcopy?((size_t)KB*NB*1024+2047)&~(size_t)2047:0,per=(size_t)KB*2048+(size_t)NB*4096,fixed=slide_fixed(NB);
    if(c->workspace_bytes<fixed+bsz+2048+per+2048) return 0;
    int32_t mt=(int32_t)((c->workspace_bytes-fixed-bsz-2048-2048)/per);
    if(mt>total) mt=total;
    NativePlan Q={.xt=*xt,.oblk=o->blocks,.ft=g->pool[ft->pool]+ft->offset,.wbase=wbase,.ocount=o->nblocks,.oslice=o->slice,
                  .onbw=(uint32_t)(o->m[2]+7)/8u,.onbd=(uint32_t)(o->m[3]+31)/32u,
                  .wstride=bcopy?js:(uint32_t)KB*1024u,.wrows=bcopy?(uint32_t)kvec:(uint32_t)KB*8u,
                  .oR=o->p[1]>>3,.oC=o->p[2]>>3,.bw=bw,.total=total,.oh=oh,.NB=NB,.KB=KB,.cbs=cbs,.mt=mt,.xrows=xrows,.xwb0=xwb0,.xnwb=xnwb,
                  .kvec=kvec,.js=(int32_t)js,.one=(uint8_t)one,.bcopy=(uint8_t)bcopy,
                  .kkind=(uint8_t)(s2ip?3:packed?0:reverse?2:1),.valid=1};
    size_t o_mh=nbump(&at,(size_t)NB*128),o_xtab_unused=0,o_tyx,o_out,o_pout;
    if(plane_native_near(&Q,at,&o_xtab_unused,&o_tyx,&o_out,&o_pout)>c->near_bytes) return 0;
    g->wused=0;
    uint8_t *ws=walloc(g,plane_native_ws(&Q));
    if(!ws){*err=E_MEMORY;return -1;}
    conv_path[g->call->execution_lane<MEMO_LANES?g->call->execution_lane:0u]=4;
#ifdef SLIDE_GEOM
    if(!g->dry && !bcopy && !s2ip && KB==9*cbs && (xt->p[1]&7)==7 && (xt->p[2]&7)==7) {
        const int32_t Rg=xt->p[1]>>3,Cg=xt->p[2]>>3,nbwg=(xt->m[2]+7)/8,nbdg=(xt->m[3]+31)/32,ncg=bw+2;
        long dmin=1L<<40,dmax=-(1L<<40),absent=0,nonv=0,unal=0,total_=0,negd=0;uint32_t pools=0;
        for(int32_t rr=0;rr<3;rr++)for(int32_t cc=0;cc<ncg;cc++)for(int32_t cb=0;cb<cbs;cb++) {
            uint32_t ix=(uint32_t)(((Rg+rr)*nbwg+Cg+cc)*nbdg)+xt->slice+(uint32_t)cb;
            uint32_t e=ix<xt->nblocks&&Cg+cc<nbwg?xt->blocks[ix]:0;total_++;
            if(!(e&127u)){absent++;continue;}
            pools|=1u<<(e&127u);if((e&127u)!=1u)nonv++;if((e&~127u)&2047u)unal++;
            if(rr<2){uint32_t ix2=(uint32_t)(((Rg+rr+1)*nbwg+Cg+cc)*nbdg)+xt->slice+(uint32_t)cb;uint32_t e2=ix2<xt->nblocks?xt->blocks[ix2]:0;
                if(e2&127u){long d=(long)(e2&~127u)-(long)(e&~127u);if(d<dmin)dmin=d;if(d>dmax)dmax=d;if(d<0)negd++;}}
        }
        {/* block regularity: per row, per channel block, address = a0 + cc*sx? */
         long runs=0;long ent=0,breaks=0;
         for(int32_t rr=0;rr<3;rr++){uint32_t prev=0;long pd=0;int have=0;
           for(int32_t cc=0;cc<ncg;cc++)for(int32_t cb=0;cb<cbs;cb++){uint32_t ix=(uint32_t)(((Rg+rr)*nbwg+Cg+cc)*nbdg)+xt->slice+(uint32_t)cb;uint32_t e=ix<xt->nblocks&&Cg+cc<nbwg?xt->blocks[ix]:0;ent++;
              if(have){long dd=(long)(e&~127u)-(long)(prev&~127u);if(have>1&&dd!=pd)breaks++;pd=dd;}prev=e;have++;}
           runs++;}
         /* output */
         long obreaks=0,oent=0;{uint32_t onw=(uint32_t)(o->m[2]+7)/8u,ond=(uint32_t)(o->m[3]+31)/32u;uint32_t prev=0;long pd=0;int have=0;
           for(int32_t bx=0;bx<bw;bx++)for(int32_t j=0;j<NB;j++){uint32_t ix=((uint32_t)(o->p[1]>>3)*onw+(uint32_t)(o->p[2]>>3)+(uint32_t)bx)*ond+o->slice+(uint32_t)j;uint32_t e=ix<o->nblocks?o->blocks[ix]:0;oent++;
              if(have){long dd=(long)(e&~127u)-(long)(prev&~127u);if(have>1&&dd!=pd)obreaks++;pd=dd;}prev=e;have++;}}
         fprintf(stderr,"SLIDEREG cmd=%u ent=%ld breaks=%ld oent=%ld obreaks=%ld\n",(unsigned)g->call->command_index,ent,breaks,oent,obreaks);}
        {long gt=0,gcross=0,gmix=0,g64=0;
         for(int32_t cc=0;cc<ncg;cc++)for(int32_t cb=0;cb<cbs;cb++){
            uint32_t ee[3];int pr=0;
            for(int32_t rr=0;rr<3;rr++){uint32_t ix=(uint32_t)(((Rg+rr)*nbwg+Cg+cc)*nbdg)+xt->slice+(uint32_t)cb;ee[rr]=ix<xt->nblocks&&Cg+cc<nbwg?xt->blocks[ix]:0;if(ee[rr]&127u)pr++;}
            gt++;
            if(pr<3){gmix++;continue;}
            uint32_t a=ee[0]&~127u,b=ee[1]&~127u,d=ee[2]&~127u;
            if((a>>22)!=(b>>22)||(b>>22)!=(d>>22))gcross++;
            long d1=(long)b-(long)a,d2=(long)d-(long)b;
            if(d1<0)d1=-d1;if(d2<0)d2=-d2;if(d1<65536&&d2<65536)g64++;
         }
         fprintf(stderr,"SLIDEPAGE cmd=%u groups=%ld cross=%ld absent=%ld small=%ld\n",(unsigned)g->call->command_index,gt,gcross,gmix,g64);}
        fprintf(stderr,"SLIDEGEOM cmd=%u cbs=%d bw=%d total=%ld absent=%ld nonvtcm=%ld unaligned=%ld neg=%ld dmin=%ld dmax=%ld pools=%x\n",(unsigned)g->call->command_index,(int)cbs,(int)bw,total_,absent,nonv,unal,negd,dmin,dmax,(unsigned)pools);
    }
#endif
#ifdef CONV_SHAPES
    if(!g->dry) fprintf(stderr,"NATCONV var=%u KB=%d cbs=%d NB=%d oh=%d ow=%d px=%d py=%d one=%d bcopy=%d s2ip=%d\n",g->variant,KB,cbs,NB,oh,ow,xt->p[2]&7,xt->p[1]&7,one,bcopy,s2ip);
#endif
    if(g->dry) return 1;
    uint8_t *base=(uint8_t *)(((uintptr_t)ws+2047)&~(uintptr_t)2047),*n=c->near;
    plane_fi_put(base,NB,corr,pre,Mq,n+o_mh);
    int r=plane_native_run(&Q,c,g->pool,g->call->parallel,base,n+o_mh,ktab,xtab,(int32_t *)(n+o_tyx),(uint8_t **)(n+o_out),(uint8_t **)(n+o_pout),
                           bcopy?s2off:NULL,corr,pre,Mq,err);
#if BLK_HVX && !defined(NO_CONV_PLANS)
    /* A stride 2 B copy (raw offsets per tile) stays on the full path. */
    if(r==1 && fipre && native_plans && plan_conv==c && !(s2 && bcopy) && g->call->execution_lane==0 &&
       g->call->program==plan_program && g->call->command_index<MEMO_COMMANDS && !((uintptr_t)Q.ft&127u))
        native_plans[g->call->command_index]=Q;
#else
    (void)s2;(void)fipre;
#endif
    return r;
}
#else
static int hmx_native(G *g, const T *xt, const T *o, int32_t KB, int32_t NB, int32_t cbs, int32_t co, int32_t oh,
                      int32_t ow, const uint32_t *ktab, const int32_t *corr, const int32_t *pre, const int32_t *Mq,
                      const int8_t *wbase, const T *ft, int bcopy, int32_t kvec, const uint32_t *s2off, uint32_t js, int s2ip, int32_t xrows, int32_t xwb0,
                      int32_t xnwb, uint8_t **xtab, const uint8_t *fipre, int hit, size_t at, int32_t *err, int s2, int packed, int reverse) {
    (void)s2;(void)packed;(void)reverse;(void)ft;
    const Nprg2ReferenceConv *c=g->conv;
    /* 1x1 conv (ktab k = (0,0,k)), input origin on a block corner: no row table */
    int one=0;
#if NAT_1X1
    if(KB==cbs && (xt->p[1]&7)==0 && (xt->p[2]&7)==0) {
        one=1;
        for(int32_t k=0;k<KB;k++) if(ktab[k]!=((uint32_t)k<<16)) {one=0;break;}
    }
#endif
    if((!xtab && !one) || (BLK_HVX && !c->hmx->native_n)) return 0;
    if(oh%8 || ow%8 || o->p[1]%8 || o->p[2]%8) return 0;
    if(g->fx.kind && (g->fx.oe.p[1]%8 || g->fx.oe.p[2]%8 || (g->fx.kind==2 && (g->fx.ob.p[1]%8 || g->fx.ob.p[2]%8)))) return 0;
    if(!hit) for(int32_t k=0;k<co;k++) if(pre[k]<0 || pre[k]>20) return 0;
    uint64_t (*clock)(void)=c->phase && !g->dry?c->clock:NULL,t0=clock?clock():0,t1;
    const int32_t bw=ow/8,total=(oh/8)*bw;
    /* :cm stores need 2048-aligned blocks; pool blocks are only 128-aligned.
     * Fused convs, or any misaligned output block, go through the C blocks
     * (2048-aligned workspace) and an HVX pass. Memoised per command (bit 6/7
     * of conv_checked are free for convs). */
    int fused=g->fx.kind!=0,staged=fused;
    if(!staged) {
        Memo *mm=g->dry?NULL:gmemo(g);
        uint8_t *mb=mm?&mm->conv_checked[g->call->command_index]:NULL;
        if(mb && (*mb&64u)) staged=!!(*mb&128u);
        else {
            int64_t nbw=((int64_t)o->m[2]+7)/8,nbd=((int64_t)o->m[3]+31)/32;
            for(int32_t by=0;by<oh/8 && !staged;by++)for(int32_t bx=0;bx<bw && !staged;bx++)for(int32_t j=0;j<NB;j++) {
                uint8_t *d=hmx_row(g->pool,o,(int64_t)o->p[1]+by*8,((int64_t)o->p[2]>>3)+bx,nbw,nbd,j);
                if(d && ((uintptr_t)d&2047u)){staged=1;break;}
            }
            if(mb && !g->dry) *mb|=(uint8_t)(64u|(staged?128u:0u));
        }
    }
#if BLK_HVX && defined(NPRG2_SLIDE3) && NPRG2_SLIDE3
    if(!bcopy && !s2ip) {
        int r=hmx_slide(g,xt,o,KB,NB,cbs,co,oh,ow,ktab,corr,pre,Mq,wbase,staged,fipre,at,err);
        if(r) return r;
    }
#endif
    const size_t fixed=((size_t)NB*256+256+2048*2+2047)&~(size_t)2047,
                 bsz=bcopy?((size_t)KB*NB*1024+2047)&~(size_t)2047:0,per=(size_t)KB*2048+(staged?(size_t)NB*2048:0);
    if(c->workspace_bytes<fixed+bsz+2048+per+2048) return 0;
    int32_t mt=(int32_t)((c->workspace_bytes-fixed-bsz-2048-2048)/per);
    if(mt>total) mt=total;
    uint8_t *n=c->near;
    size_t o_fi=fipre?0:nbump(&at,(size_t)NB*256+256),o_tyx=nbump(&at,(size_t)mt*8),o_out=nbump(&at,(size_t)NB*mt*sizeof(uint8_t *));
    if(at>c->near_bytes) return 0;
    g->wused=0;
    uint8_t *ws=walloc(g,fixed+bsz+2048+(size_t)mt*per+2048);
    if(!ws){*err=E_MEMORY;return -1;}
    conv_path[g->call->execution_lane<MEMO_LANES?g->call->execution_lane:0u]=4;
    if(fused) g->fx_ok=1;
#ifdef SLIDE_GEOM
    if(!g->dry && !bcopy && !s2ip && KB==9*cbs && (xt->p[1]&7)==7 && (xt->p[2]&7)==7) {
        const int32_t Rg=xt->p[1]>>3,Cg=xt->p[2]>>3,nbwg=(xt->m[2]+7)/8,nbdg=(xt->m[3]+31)/32,ncg=bw+2;
        long dmin=1L<<40,dmax=-(1L<<40),absent=0,nonv=0,unal=0,total_=0,negd=0;uint32_t pools=0;
        for(int32_t rr=0;rr<3;rr++)for(int32_t cc=0;cc<ncg;cc++)for(int32_t cb=0;cb<cbs;cb++) {
            uint32_t ix=(uint32_t)(((Rg+rr)*nbwg+Cg+cc)*nbdg)+xt->slice+(uint32_t)cb;
            uint32_t e=ix<xt->nblocks&&Cg+cc<nbwg?xt->blocks[ix]:0;total_++;
            if(!(e&127u)){absent++;continue;}
            pools|=1u<<(e&127u);if((e&127u)!=1u)nonv++;if((e&~127u)&2047u)unal++;
            if(rr<2){uint32_t ix2=(uint32_t)(((Rg+rr+1)*nbwg+Cg+cc)*nbdg)+xt->slice+(uint32_t)cb;uint32_t e2=ix2<xt->nblocks?xt->blocks[ix2]:0;
                if(e2&127u){long d=(long)(e2&~127u)-(long)(e&~127u);if(d<dmin)dmin=d;if(d>dmax)dmax=d;if(d<0)negd++;}}
        }
        {/* block regularity: per row, per channel block, address = a0 + cc*sx? */
         long runs=0;long ent=0,breaks=0;
         for(int32_t rr=0;rr<3;rr++){uint32_t prev=0;long pd=0;int have=0;
           for(int32_t cc=0;cc<ncg;cc++)for(int32_t cb=0;cb<cbs;cb++){uint32_t ix=(uint32_t)(((Rg+rr)*nbwg+Cg+cc)*nbdg)+xt->slice+(uint32_t)cb;uint32_t e=ix<xt->nblocks&&Cg+cc<nbwg?xt->blocks[ix]:0;ent++;
              if(have){long dd=(long)(e&~127u)-(long)(prev&~127u);if(have>1&&dd!=pd)breaks++;pd=dd;}prev=e;have++;}
           runs++;}
         /* output */
         long obreaks=0,oent=0;{uint32_t onw=(uint32_t)(o->m[2]+7)/8u,ond=(uint32_t)(o->m[3]+31)/32u;uint32_t prev=0;long pd=0;int have=0;
           for(int32_t bx=0;bx<bw;bx++)for(int32_t j=0;j<NB;j++){uint32_t ix=((uint32_t)(o->p[1]>>3)*onw+(uint32_t)(o->p[2]>>3)+(uint32_t)bx)*ond+o->slice+(uint32_t)j;uint32_t e=ix<o->nblocks?o->blocks[ix]:0;oent++;
              if(have){long dd=(long)(e&~127u)-(long)(prev&~127u);if(have>1&&dd!=pd)obreaks++;pd=dd;}prev=e;have++;}}
         fprintf(stderr,"SLIDEREG cmd=%u ent=%ld breaks=%ld oent=%ld obreaks=%ld\n",(unsigned)g->call->command_index,ent,breaks,oent,obreaks);}
        {long gt=0,gcross=0,gmix=0,g64=0;
         for(int32_t cc=0;cc<ncg;cc++)for(int32_t cb=0;cb<cbs;cb++){
            uint32_t ee[3];int pr=0;
            for(int32_t rr=0;rr<3;rr++){uint32_t ix=(uint32_t)(((Rg+rr)*nbwg+Cg+cc)*nbdg)+xt->slice+(uint32_t)cb;ee[rr]=ix<xt->nblocks&&Cg+cc<nbwg?xt->blocks[ix]:0;if(ee[rr]&127u)pr++;}
            gt++;
            if(pr<3){gmix++;continue;}
            uint32_t a=ee[0]&~127u,b=ee[1]&~127u,d=ee[2]&~127u;
            if((a>>22)!=(b>>22)||(b>>22)!=(d>>22))gcross++;
            long d1=(long)b-(long)a,d2=(long)d-(long)b;
            if(d1<0)d1=-d1;if(d2<0)d2=-d2;if(d1<65536&&d2<65536)g64++;
         }
         fprintf(stderr,"SLIDEPAGE cmd=%u groups=%ld cross=%ld absent=%ld small=%ld\n",(unsigned)g->call->command_index,gt,gcross,gmix,g64);}
        fprintf(stderr,"SLIDEGEOM cmd=%u cbs=%d bw=%d total=%ld absent=%ld nonvtcm=%ld unaligned=%ld neg=%ld dmin=%ld dmax=%ld pools=%x\n",(unsigned)g->call->command_index,(int)cbs,(int)bw,total_,absent,nonv,unal,negd,dmin,dmax,(unsigned)pools);
    }
#endif
#ifdef CONV_SHAPES
    if(!g->dry) fprintf(stderr,"NATCONV var=%u KB=%d cbs=%d NB=%d oh=%d ow=%d px=%d py=%d ox=%d oy=%d one=%d bcopy=%d s2ip=%d\n",g->variant,KB,cbs,NB,oh,ow,xt->p[2]&7,xt->p[1]&7,o->p[2]&7,o->p[1]&7,one,bcopy,s2ip);
#endif
    if(g->dry) return 1;
    uint8_t *base=(uint8_t *)(((uintptr_t)ws+2047)&~(uintptr_t)2047);
    int32_t *fi=(int32_t *)base,*zero=(int32_t *)(base+(size_t)NB*256);
    uint8_t *completion=base+fixed-2048,*Bc=base+fixed,*dummy=Bc+bsz,*A=dummy+2048,*C=A+(size_t)mt*KB*2048;
    if(bcopy) {
        for(int32_t j=0;j<NB;j++)for(int32_t k=0;k<KB;k++) {
            uint8_t *d=Bc+((size_t)j*KB+(size_t)k)*1024;
            qd_vcopy(d,wbase+s2off[k]+(size_t)j*js,(size_t)kvec*128);
            if(kvec<8) memset(d+(size_t)kvec*128,0,(size_t)(8-kvec)*128);
        }
        wbase=(const int8_t *)Bc;
    }
    hmx_fi_put(base,NB,co,fipre,fipre?NULL:(int32_t *)(n+o_fi),corr,pre,Mq);
    HMX_MARK(6);
    if(!one) hmx_tab(xtab,g->pool,xt,xt->p[1],xrows,xwb0,xnwb,cbs);
    HMX_MARK(7);
    int32_t *tyx=(int32_t *)(n+o_tyx);
    uint8_t **outp=(uint8_t **)(n+o_out);
    /* Output block of tile (by, bx), channel block j: table entry
     * ((oy0/8 + by)*nbw + owb0 + bx)*nbd + slice + j, 32-bit (rows aligned). */
    const uint32_t onbw=(uint32_t)(o->m[2]+7)/8u,onbd=(uint32_t)(o->m[3]+31)/32u,ocount=o->nblocks;
    const uint32_t *oblk=o->blocks;
    NatJob J={.g=g,.pool=g->pool,.conv=c,.xt=xt,.o=o,.fx=fused?&g->fx:NULL,.KB=KB,.NB=NB,.cbs=cbs,.bw=bw,.xwb0=xwb0,.xnwb=xnwb,
              .ktab=ktab,.xtab=xtab,.A=A,.C=staged?C:NULL,.tyx=tyx,.pre=pre,.Mq=Mq,.corr=corr,.B=(const uint8_t *)wbase,
              .one=one,.R0=xt->p[1]>>3,.C0=xt->p[2]>>3,.xnbw=((int64_t)xt->m[2]+7)/8,.xnbd=((int64_t)xt->m[3]+31)/32};
    uint32_t parts=nprg2_parallel_parts(g->call->parallel,NPRG2_PAR_MAX_PARTS);
    uint32_t saved=c->hmx->enter?c->hmx->enter():0;
    int e=0;
    int32_t by=0,bx=0;
    for(int32_t m0=0;m0<total && !e;m0+=mt) {
        J.m0=m0;J.mt=total-m0<mt?total-m0:mt;
        for(int32_t t=0;t<J.mt;t++) {
            tyx[2*t]=by;tyx[2*t+1]=bx;
            uint32_t idx=(((uint32_t)(o->p[1]>>3)+(uint32_t)by)*onbw+(uint32_t)(o->p[2]>>3)+(uint32_t)bx)*onbd+o->slice;
            for(int32_t j=0;j<NB;j++) {
                uint8_t *d=dummy;
                if(staged) d=C+((size_t)t*NB+(size_t)j)*2048;
                else if(idx+(uint32_t)j<ocount) {
                    uint32_t e=oblk[idx+(uint32_t)j],pool=e&127u;
                    if(pool && g->pool[pool]) d=g->pool[pool]+(e&~127u);
                }
                outp[(size_t)j*J.mt+(size_t)t]=d;
            }
            if(++bx==bw){bx=0;by++;}
        }
        HMX_MARK(8);
        if(parts>1 && !(one && (int64_t)J.mt*KB<NAT_SERIAL_BELOW)) nprg2_parallel_for(g->call->parallel,NPRG2_PAR_CONV,parts,nat_prep_part,&J);
        else nat_prep_part(&J,0,1);
        if(clock){t1=clock();c->phase[0]+=t1-t0;t0=t1;}
        e=nat_mac(&J,fi,zero,completion,outp);
        if(clock){t1=clock();c->phase[1]+=t1-t0;t0=t1;}
        if(e) break;
        if(staged) {
            if(parts>1) nprg2_parallel_for(g->call->parallel,NPRG2_PAR_CONV,parts,nat_fx_part,&J);
            else nat_fx_part(&J,0,1);
            if(clock){t1=clock();c->phase[3]+=t1-t0;t0=t1;}
        }
    }
    if(c->hmx->leave) c->hmx->leave(saved);
    if(e){*err=E_RANGE;return -1;}
    return 1;
}
#endif
#endif
static int hmx_conv(G *g, const T *xt, const T *wt, const T *ft, const T *o, int packed, int reverse, int s2, int32_t co,
                    int32_t oh, int32_t ow, int32_t *err) {
    const Nprg2ReferenceConv *c=g->conv;
    uint64_t (*clock)(void)=c->phase && !g->dry?c->clock:NULL,t0=clock?clock():0,t1;
#if HMX_PROF==5
    uint32_t lane=g->call->execution_lane<MEMO_LANES?g->call->execution_lane:0u;
    if(clock && hmx_t_entry[lane] && hmx_t_entry[lane]<=t0) c->phase[3]+=t0-hmx_t_entry[lane];
#endif
    if(!c->hmx || !c->near || ((uintptr_t)c->near&127u)) return 0;
    /* A fused epilogue must cover exactly the region the conv writes. */
    if(g->fx.kind && (oh!=o->d[1] || ow!=o->d[2] || co!=o->d[3])) return 0;
    int32_t H=wt->m[0],W=wt->m[1],I=wt->m[2];
    /* Stride 2 (3x3 raw weights): a 2x2 conv over the space-to-depth input's
     * 4I channels (fast_conv's regrouping). A 32-channel K block is either
     * all NO_TILE or one contiguous 1024-byte raw weight block, so B is the
     * present blocks, copied per command (s2off: raw offset of K block k for
     * output block 0; block j adds js). */
    uint32_t s2off[256],js=0;
    /* bcopy: B copied per command into the workspace, K block k from
     * s2off[k] (+ j*js for output block j), kvec valid 128-byte vectors per
     * block, zeros after. Stride 2 (kvec 8), or packed 1x1 with I < 32
     * (6/32: I = 12, 20): K padded to 32 with zero weights (the input
     * crouton's lanes >= I then multiply zeros). */
    int bcopy=s2;int32_t kvec=8,Ireal=I;
    if(s2) {
        if(H!=3 || W!=3 || I%32 || packed || reverse) return 0;
        js=(uint32_t)H*(uint32_t)W*(uint32_t)I*32u;
        H=2;W=2;I*=4;Ireal=I;
    } else if(packed && H==1 && W==1 && I>0 && I<32 && I%4==0) {
        bcopy=1;kvec=I/4;js=(uint32_t)I*32u;I=32;
    }
    if(I%32 || xt->m[3]<Ireal || H>3 || W>3 || ow%4 || co%32) return 0;
    if(!(xt->flags&F_BLOCKED) || xt->esize!=1 || !(o->flags&F_BLOCKED) || o->esize!=1) return 0;
    if(xt->p[0] || o->p[0] || xt->m[0]!=1 || o->m[0]!=1 || wt->pool!=1) return 0;
    const int32_t cbs=I/32,NB=co/32;
    int32_t KB=H*W*cbs;
    const int8_t *wbase=(const int8_t *)g->pool[wt->pool]+wt->offset;
    int s2ip=0; /* stride 2 with the raw weights used in place (see s2_ktab) */
    if(s2) {
        if(!((uintptr_t)wbase&1023u) && !S2_FORCE_BCOPY) {s2ip=1;bcopy=0;KB=9*(cbs/4);}
        else KB=s2_scan(wt,cbs,s2off);
#ifdef S2_COUNT
        if(!g->dry){if(s2ip)s2_ip_taken++;else s2_bc_taken++;}
#endif
        if(KB<0) return 0;
    }
    if(bcopy && !s2) s2off[0]=0; /* KB == 1 */
    if(!KB || KB*32>HMX_TERMS_MAX || KB>255) return 0;
    if(!bcopy && ((uintptr_t)wbase&1023u)) return 0;
    int32_t dims[4]={1,oh+H-1,ow+W-1,Ireal},origin[4]={0,xt->p[1],xt->p[2],0};
    int32_t on[4]={1,oh,ow,co},oorigin[4]={0,o->p[1],o->p[2],0};
    {
        /* bits 0..2: checks passed with epilogue kind 0..2 (bits 3..5 are fast_conv's) */
        Memo *mm=g->dry || g->fx.kind<0 || g->fx.kind>2?NULL:gmemo(g);
        uint8_t bit=(uint8_t)(1u<<(g->fx.kind&3));
        if(!mm || !(mm->conv_checked[g->call->command_index]&bit)) {
            if(xfer_check(g,xt,dims,origin,0) || xfer_check(g,o,on,oorigin,1) ||
               blocked_overlap(xt,dims,origin,o,on,oorigin)) return 0;
            if(mm) mm->conv_checked[g->call->command_index]|=bit;
        }
    }
    HMX_MARK(1);
    /* Scalar tables in near: ktab, corr, pre, M, then the multiplier vectors. */
    size_t at=0;
    size_t o_ktab=nbump(&at,(size_t)KB*4),o_corr=nbump(&at,(size_t)co*4),o_pre=nbump(&at,(size_t)co*4),
           o_M=nbump(&at,(size_t)co*4),o_mult=nbump(&at,(size_t)co*4),o_bias=nbump(&at,(size_t)co*4),
           o_post=nbump(&at,(size_t)co*4),o_wsum=nbump(&at,(size_t)co*8),o_multv=nbump(&at,(size_t)NB*128),o_rndv=nbump(&at,(size_t)NB*128),
           o_rows=nbump(&at,(size_t)NB*256),o_fi=nbump(&at,(size_t)NB*256+256);
    /* Input row pointers for every (row, block column, channel block) the
     * quads read: rows p1..p1+oh+H-2, columns p2..p2+ow+W-2 (a straddling quad
     * reads one block further). */
#if NPRG2_PLANES
    const int32_t xrows=((oh+7)&~7)+H-1; /* full tiles: rows past the view feed only discarded output rows */
#else
    const int32_t xrows=oh+H-1;
#endif
    const int32_t xwb0=xt->p[2]>>3,xnwb=((xt->p[2]+ow+W-2)>>3)-xwb0+1;
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
    const int64_t bound=(int64_t)KB*32*255*128;
    /* The filter rows (per 32 channels: 32 mults, 32 biases) are in VTCM:
     * one vector copy to near instead of scalar VTCM loads. */
    int vec_done=0;
    const uint8_t *fipre=NULL;
#if BLK_HVX
    /* Vector form of everything up to HMX_MARK(2): per block, M = 1024+(mult&1023),
     * pre = 20-((mult>>10)&31) (post is 14 by construction), wsum by vrmpy,
     * corr = bias - zp*wsum in int32 (|wsum| <= KB*4096 and 0 <= zp <= 255,
     * |bias| < 2^30 checked). The native Fi word is 0x400000|(mult&0x7fff).
     * Any lane outside the scalar checks: the scalar code decides. */
    if(!vec_done && xt->zp>=0 && xt->zp<=255 && bound<(1ll<<30)) {
        uint8_t *nr=n+o_rows;
        near_copy(nr,g->pool[ft->pool]+ft->offset,(size_t)NB*256);
        const HVX_Vector ones=Q6_V_vsplat_R(0x01010101),m1023=Q6_V_vsplat_R(1023),m31=Q6_V_vsplat_R(31),
                         c1024=Q6_V_vsplat_R(1024),c20=Q6_V_vsplat_R(20),zero=Q6_V_vzero(),c15=Q6_V_vsplat_R(15),
                         lim=Q6_V_vsplat_R((int32_t)(INT32_MAX-bound)),b30=Q6_V_vsplat_R(1<<30),
                         m7fff=Q6_V_vsplat_R(0x7fff),f40=Q6_V_vsplat_R(0x400000);
        HVX_VectorPred badq=Q6_Q_vcmp_gt_VwVw(zero,zero);
        HVX_Vector *fiq=(HVX_Vector *)(n+o_fi);
        for(int32_t j=0;j<NB;j++) {
            HVX_Vector mv=*(const HVX_Vector *)(nr+(size_t)j*256),bv=*(const HVX_Vector *)(nr+(size_t)j*256+128);
            HVX_Vector Mv=Q6_Vw_vadd_VwVw(c1024,Q6_V_vand_VV(mv,m1023));
            HVX_Vector pv=Q6_Vw_vsub_VwVw(c20,Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(mv,10),m31));
            HVX_Vector a=zero;
            if(xt->zp) {
                if(bcopy) {
                    for(int32_t k=0;k<KB;k++) {
                        const int8_t *b=wbase+s2off[k]+(size_t)j*js;
                        for(int32_t r=0;r<kvec;r++) a=Q6_Vw_vrmpyacc_VwVbVb(a,*(const HVX_UVector *)(b+128*r),ones);
                    }
                } else {
                    const int8_t *b=wbase+(size_t)j*KB*1024;
                    for(int32_t r=0;r<KB*8;r++) a=Q6_Vw_vrmpyacc_VwVbVb(a,((const HVX_Vector *)b)[r],ones);
                }
            }
            HVX_Vector off=Q6_Vw_vsub_VwVw(bv,Q6_Vw_vmpyi_VwRh(a,(int32_t)((uint32_t)xt->zp|(uint32_t)xt->zp<<16)));
            badq=Q6_Q_or_QQ(badq,Q6_Q_or_QQ(Q6_Q_vcmp_gt_VwVw(zero,pv),Q6_Q_vcmp_gt_VwVw(pv,c15)));
            badq=Q6_Q_or_QQ(badq,Q6_Q_vcmp_gt_VwVw(Q6_Vw_vabs_Vw_sat(bv),b30));
            badq=Q6_Q_or_QQ(badq,Q6_Q_vcmp_gt_VwVw(Q6_Vw_vabs_Vw_sat(off),lim));
            *(HVX_Vector *)(Mq+32*j)=Mv;*(HVX_Vector *)(pre+32*j)=pv;*(HVX_Vector *)(corr+32*j)=off;
            fiq[2*j]=Q6_V_vor_VV(f40,Q6_V_vand_VV(mv,m7fff));fiq[2*j+1]=off;
        }
        HVX_Vector flag __attribute__((aligned(128)))=Q6_V_vand_QR(badq,0x01010101u);
        const uint32_t *fw=(const uint32_t *)&flag;uint32_t any=0;
        for(int q=0;q<32;q++) any|=fw[q];
        vec_done=!any;
        if(vec_done) fipre=(const uint8_t *)fiq;
    }
#endif
    if(!vec_done) {
        const uint8_t *rows=g->pool[ft->pool]+ft->offset;
        uint8_t *nr=n+o_rows;
        near_copy(nr,rows,(size_t)NB*256);
        for(int32_t k=0;k<co;k++){memcpy(mult+k,nr+(size_t)(k/32)*256+(k%32)*4,4);memcpy(bias+k,nr+(size_t)(k/32)*256+128+(k%32)*4,4);}
    requant_params(mult,co,0,Mq,pre,post);
    memset(wsum,0,(size_t)co*8);
    if(xt->zp && bcopy) for(int32_t j=0;j<NB;j++) {
        /* the present raw blocks of output block j (NO_TILE blocks are zero) */
#if BLK_HVX
        HVX_Vector ones=Q6_V_vsplat_R(0x01010101),a=Q6_V_vzero();
        for(int32_t k=0;k<KB;k++) {
            const int8_t *b=wbase+s2off[k]+(size_t)j*js;
            for(int32_t r=0;r<kvec;r++) a=Q6_Vw_vrmpyacc_VwVbVb(a,*(const HVX_UVector *)(b+128*r),ones);
        }
        int32_t tmp[32] __attribute__((aligned(128)));
        *(HVX_Vector *)tmp=a;
        for(int q=0;q<32;q++) wsum[32*j+q]=tmp[q];
#else
        for(int32_t k=0;k<KB;k++) {
            const int8_t *b=wbase+s2off[k]+(size_t)j*js;
            for(int q=0;q<kvec*128;q++) wsum[32*j+(q/4)%32]+=b[q];
        }
#endif
    } else if(xt->zp) for(int32_t j=0;j<NB;j++) {
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
    for(int32_t k=0;k<co;k++) {
        int64_t off=(int64_t)bias[k]-(int64_t)xt->zp*wsum[k];
        if(post[k]!=14 || pre[k]<0 || pre[k]>15 || Mq[k]<1024 || Mq[k]>2047 || (off<0?-off:off)+bound>INT32_MAX) return 0;
        corr[k]=(int32_t)off;
    }
    }
    HMX_MARK(2);
    /* k order of the weight layout, enumerated without divides. */
    if(s2) {
        if(s2ip) s2_ktab(ktab,cbs/4);
        else {
            int32_t k=0;
            for(int32_t ky=0;ky<2;ky++)for(int32_t kx=0;kx<2;kx++)for(int32_t cb=0;cb<cbs;cb++)
                if(weight_tile(wt,1,0,0,ky,kx,cb*8,0)!=NO_TILE) ktab[k++]=(uint32_t)ky|(uint32_t)kx<<8|(uint32_t)cb<<16;
        }
    } else {
        int32_t k=0;
        if(packed) {for(int32_t h=0;h<H;h++)for(int32_t x=0;x<W;x++)for(int32_t cb=0;cb<cbs;cb++,k++)
            ktab[k]=(uint32_t)h|(uint32_t)x<<8|(uint32_t)cb<<16;}
        else for(int32_t cb=0;cb<cbs;cb++)for(int32_t h=0;h<H;h++)for(int32_t x=0;x<W;x++,k++)
            ktab[k]=(uint32_t)h|(uint32_t)(reverse?W-1-x:x)<<8|(uint32_t)cb<<16;
    }
#if NPRG2_HW_ROUNDING || NPRG2_PLANES
#ifdef NAT_CHECK
    /* Debug: run native, keep its output, then run v6 (also HW-rounded) over it and compare. */
    uint8_t *nat_snap=NULL;int32_t nat_dims[4]={1,oh,ow,co},nat_org[4]={0,o->p[1],o->p[2],0};
#endif
    {
        int r=hmx_native(g,xt,o,KB,NB,cbs,co,oh,ow,ktab,corr,pre,Mq,wbase,ft,bcopy,kvec,s2off,js,s2ip,xrows,xwb0,xnwb,
                         use_tab?(uint8_t **)(n+o_xtab):NULL,vec_done?fipre:NULL,0,at,err,s2,packed,reverse);
#ifdef NAT_CHECK
        if(r==1 && !g->dry && !g->fx.kind) {
            nat_snap=malloc((size_t)oh*ow*co);
            if(nat_snap) xfer_copy(g,o,nat_snap,nat_dims,nat_org,0);
            r=0;
        }
#endif
        if(r) return r;
    }
#endif
    /* Workspace: fi, finish record, completion, then A and C for mt tiles. */
    const int32_t total=(oh*ow+31)/32;
    const size_t fixed=((size_t)NB*256+256+2048*2+2047)&~(size_t)2047,per=(size_t)(KB+NB)*2048,
                 bsz=bcopy?((size_t)KB*NB*1024+2047)&~(size_t)2047:0;
    if(c->workspace_bytes<fixed+bsz+per+2048) return 0;
    int32_t mt=(int32_t)((c->workspace_bytes-fixed-bsz-2048)/per);
    if(mt>total) mt=total;
    size_t o_tyx=nbump(&at,(size_t)mt*8);
    if(at>c->near_bytes) return 0;
    g->wused=0;
    uint8_t *ws=walloc(g,fixed+bsz+(size_t)mt*per+2048);
    if(!ws){*err=E_MEMORY;return -1;}
    conv_path[g->call->execution_lane<MEMO_LANES?g->call->execution_lane:0u]=3;
    if(g->fx.kind) g->fx_ok=1;
#ifdef CONV_SHAPES
    if(!g->dry) fprintf(stderr,"HMXCONV var=%u k=%dx%d ci=%d co=%d oh=%d ow=%d KB=%d NB=%d tiles=%d mt=%d packed=%d\n",
                        g->variant,H,W,I,co,oh,ow,KB,NB,total,mt,packed);
#endif
    if(g->dry) return 1;
    uint8_t *base=(uint8_t *)(((uintptr_t)ws+2047)&~(uintptr_t)2047);
    int32_t *fi=(int32_t *)base,*zero=(int32_t *)(base+(size_t)NB*256);
    uint8_t *completion=base+fixed-2048,*Bc=base+fixed,*A=Bc+bsz,*C=A+(size_t)mt*KB*2048;
    if(bcopy) {
        for(int32_t j=0;j<NB;j++)for(int32_t k=0;k<KB;k++) {
            uint8_t *d=Bc+((size_t)j*KB+(size_t)k)*1024;
            qd_vcopy(d,wbase+s2off[k]+(size_t)j*js,(size_t)kvec*128);
            if(kvec<8) memset(d+(size_t)kvec*128,0,(size_t)(8-kvec)*128);
        }
        wbase=(const int8_t *)Bc;
    }
    {
        /* Built in near with scalar stores, then one vector copy to VTCM. */
        int32_t *nf=(int32_t *)(n+o_fi),*nz=nf+NB*64;
        for(int32_t k=0;k<co;k++) {
            uint32_t p=(uint32_t)pre[k];
            nf[(k>>5)*64+(k&31)]=(int32_t)(0x400000u|((16u-p)<<10));
            nf[(k>>5)*64+32+(k&31)]=corr[k]-(p?(int32_t)(1u<<(p-1)):0);
        }
        for(int k=0;k<64;k++) nz[k]=k<32?0x404000:0;
        near_copy(base,(const uint8_t *)nf,(size_t)NB*256+256);
    }
    uint16_t *mv=(uint16_t *)(n+o_multv);
    for(int32_t k=0;k<co;k++) mv[(k>>5)*64+2*(k&31)]=mv[(k>>5)*64+2*(k&31)+1]=(uint16_t)Mq[k];
    int32_t *rv=(int32_t *)(n+o_rndv);
    for(int32_t k=0;k<co;k++) rv[k]=(int32_t)rq_round(0,Mq[k],14);
    HMX_MARK(3);
    uint8_t **xtab=use_tab?(uint8_t **)(n+o_xtab):NULL;
    if(xtab) hmx_tab(xtab,g->pool,xt,xt->p[1],xrows,xwb0,xnwb,cbs);
    uint8_t **otab=NULL,**etab=NULL,**btab=NULL;
    if(use_otab) {
        otab=(uint8_t **)(n+o_otab);hmx_tab(otab,g->pool,o,o->p[1],oh,owb0,onwb,NB);
        if(g->fx.kind){etab=otab+otn;hmx_tab(etab,g->pool,&g->fx.oe,g->fx.oe.p[1],oh,owb0,onwb,NB);}
        if(g->fx.kind==2){btab=etab+otn;hmx_tab(btab,g->pool,&g->fx.ob,g->fx.ob.p[1],oh,owb0,onwb,NB);}
    }
#ifdef CONV_SHAPES
    if(!g->dry) fprintf(stderr,"HMXTAB xtab=%lu otab=%lu use=%d/%d\n",(unsigned long)(use_tab?ntab*sizeof(uint8_t *):0),
                        (unsigned long)(use_otab?(size_t)ntabs*otn*sizeof(uint8_t *):0),use_tab,use_otab);
#endif
    HMX_MARK(4);
    HmxJob J={.g=g,.xtab=xtab,.xwb0=xwb0,.xnwb=xnwb,.cbs=cbs,.otab=otab,.etab=etab,.btab=btab,.owb0=owb0,.onwb=onwb,.xt=xt,.o=o,.oh=oh,.ow=ow,.KB=KB,.NB=NB,.total=total,.A=A,.C=C,.B=(const uint8_t *)wbase,
              .ktab=ktab,.corr=corr,.pre=pre,.Mq=Mq,.multv=(const uint8_t *)mv,.rndv=(const uint8_t *)rv,.fx=g->fx.kind?&g->fx:NULL,
              .xnbw=((int64_t)xt->m[2]+7)/8,.xnbd=((int64_t)xt->m[3]+31)/32,
              .onbw=((int64_t)o->m[2]+7)/8,.onbd=((int64_t)o->m[3]+31)/32};
    uint32_t parts=nprg2_parallel_parts(g->call->parallel,NPRG2_PAR_MAX_PARTS);
    uint32_t saved=c->hmx->enter?c->hmx->enter():0;
    int e=0;
    J.tyx=(int32_t *)(n+o_tyx);
    for(int32_t m0=0;m0<total && !e;m0+=mt) {
        J.m0=m0;J.mt=total-m0<mt?total-m0:mt;
        {
            int32_t P=32*m0,y=P/ow,x=P-y*ow;
            for(int32_t t=0;t<J.mt;t++) {
                J.tyx[2*t]=y;J.tyx[2*t+1]=x;
                for(x+=32;x>=ow;x-=ow) y++;
            }
        }
        if(clock){t1=clock();c->phase[HMX_SETUP_SLOT]+=t1-t0;t0=t1;}
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
#if NPRG2_HW_ROUNDING && defined(NAT_CHECK)
    if(nat_snap) {
        uint8_t *v6=malloc((size_t)oh*ow*co);
        if(v6) {
            xfer_copy(g,o,v6,nat_dims,nat_org,0);
            size_t nbad=0,first=0;
            for(size_t i=0;i<(size_t)oh*ow*co;i++) if(v6[i]!=nat_snap[i]){if(!nbad)first=i;nbad++;}
            if(nbad) {
                size_t px=first/(size_t)co;int32_t ch=(int32_t)(first%(size_t)co),yy=(int32_t)(px/(size_t)ow),xx=(int32_t)(px%(size_t)ow);
                printf("NATCHECK cmd %u KB=%d NB=%d oh=%d ow=%d co=%d s2=%d: %lu bytes differ, first y%d x%d c%d native=%u v6=%u (xp=%d,%d op=%d,%d)\n",
                       (unsigned)g->call->command_index,(int)KB,(int)NB,(int)oh,(int)ow,(int)co,s2,(unsigned long)nbad,(int)yy,(int)xx,(int)ch,
                       (unsigned)nat_snap[first],(unsigned)v6[first],(int)xt->p[1],(int)xt->p[2],(int)o->p[1],(int)o->p[2]);
            }
            free(v6);
        }
        free(nat_snap);
    }
#endif
    return 1;
}

#if NPRG2_HW_ROUNDING_U16 && BLK_HVX
/* ---------------------------------------------------- native u16 x int8 HMX
 * u16 convs (zp 0, stride 1) on HMX with cvt.uh:2x1, exact against the
 * silicon-rounding reference (HMX_NATIVE_ROUNDING.md: x = (acc>>8) + bias,
 * q = floor(x*2^(e-8)), out = ((2q+1)M + 2^18) >> 19; sim-verified for KB up
 * to 9 with full-range data, e 8..19). An A tile is a u16 crouton block: 8
 * rows x 4 pixels (two pixel pairs of 128 bytes per row) x 32 channels; a
 * tap's column shift kx moves whole pairs, or for odd shifts recombines the
 * halfwords of neighbouring pairs. The C tile of cvt.uh is the u16 crouton
 * output block. */
typedef struct {
    uint8_t *const *pool; const T *xt, *o; int32_t KB, NB, bw, m0, mt; const uint32_t *ktab; uint8_t *A, *C;
    int64_t xnbw, xnbd, onbw, onbd;
    int32_t H, W, cbs; uint8_t kinv[256]; /* (cb*3+ky)*3+kx -> k */
    /* N16_PT: input block pointer table ptab[(R*ncol+w)*cbs+cb] (R from
     * p[1]>>3, w from p[2]>>2; absent blocks point at n16_zero), per-row
     * block row / byte offset for the 10 rows of a 3x3 tile, and p[2]&3. */
    const uint8_t *const *ptab; int32_t ncol, cph; uint32_t mi[10], ro[10];
    /* N16_DIRECT_OUT: HMX writes 2048-aligned output blocks in place; outp
     * (NB*mt) is filled during staging, nc[t] marks tiles with a C copy. */
    uint8_t **outp; uint8_t *nc; int32_t direct; uint32_t anyc[4];
    void (*stage)(void *, uint32_t, uint32_t);
} N16Job;
#ifndef N16_FUSED
#define N16_FUSED 1
#endif
/* Unrolled 3x3 staging from the block pointer table (-DN16_PT=1, default 1). */
#ifndef N16_PT
#define N16_PT 1
#endif
/* HMX stores straight into aligned output blocks, no C copy pass (default 1). */
#ifndef N16_DIRECT_OUT
#define N16_DIRECT_OUT 1
#endif
/* Double-buffered A/C: stage chunk i+1 while HMX runs chunk i (default 0). */
#ifndef N16_OVERLAP
#define N16_OVERLAP 0
#endif
static const uint8_t n16_zero[2048] __attribute__((aligned(2048)));
#define N16_SH(a,b) Q6_V_vor_VV(Q6_Vuw_vlsr_VuwR((a),16),Q6_Vw_vasl_VwR((b),16))
/* One input row i (constant) of a 3x3 tile: build the three column shifts
 * from the pixel pairs and store them into every tap tile using the row. */
static inline __attribute__((always_inline)) void n16_row3(const N16Job *J, const uint8_t *const B[3][3],
                                                           HVX_Vector *const D[3][3], const int i, const int c) {
    const uint32_t m=J->mi[i],ro=J->ro[i];
    const HVX_Vector *b0=(const HVX_Vector *)(B[m][0]+ro),*b1=(const HVX_Vector *)(B[m][1]+ro),
                     *b2=(const HVX_Vector *)(B[m][2]+ro);
    HVX_Vector P0,P1,P2,P3;
    if(c&2){P0=b0[1];P1=b1[0];P2=b1[1];P3=b2[0];}
    else {P0=b0[0];P1=b0[1];P2=b1[0];P3=b1[1];}
    HVX_Vector v0[3],v1[3];
    if(!(c&1)){v0[0]=P0;v1[0]=P1;v0[1]=N16_SH(P0,P1);v1[1]=N16_SH(P1,P2);v0[2]=P1;v1[2]=P2;}
    else {v0[0]=N16_SH(P0,P1);v1[0]=N16_SH(P1,P2);v0[1]=P1;v1[1]=P2;v0[2]=N16_SH(P1,P2);v1[2]=N16_SH(P2,P3);}
    const int lo=i>7?i-7:0,hi=i<2?i:2;
#pragma unroll
    for(int kx=0;kx<3;kx++) {
#pragma unroll
        for(int ky=lo;ky<=hi;ky++) {HVX_Vector *d=D[ky][kx]+2*(i-ky);d[0]=v0[kx];d[1]=v1[kx];}
    }
}
static inline __attribute__((always_inline)) void n16_tile3(const N16Job *J, int32_t t, int32_t cb, const int c) {
    const int32_t tt=J->m0+t,by=tt/J->bw,bx=tt-by*J->bw;
    const size_t cbs=(size_t)J->cbs,rs=(size_t)J->ncol*cbs;
    const uint8_t *const *pr=J->ptab+((size_t)by*J->ncol+(size_t)bx)*cbs+(size_t)cb;
    const uint8_t *B[3][3];
    for(int m=0;m<3;m++)for(int b=0;b<3;b++)B[m][b]=pr[(size_t)m*rs+(size_t)b*cbs];
    HVX_Vector *D[3][3];
    for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++)
        D[ky][kx]=(HVX_Vector *)(J->A+((size_t)t*J->KB+J->kinv[(cb*3+ky)*3+kx])*2048);
    n16_row3(J,B,D,0,c);n16_row3(J,B,D,1,c);n16_row3(J,B,D,2,c);n16_row3(J,B,D,3,c);n16_row3(J,B,D,4,c);
    n16_row3(J,B,D,5,c);n16_row3(J,B,D,6,c);n16_row3(J,B,D,7,c);n16_row3(J,B,D,8,c);n16_row3(J,B,D,9,c);
}
static void n16_prep3_part(void *ctx, uint32_t part, uint32_t parts) {
    const N16Job *J=(const N16Job *)ctx;
    const uint32_t units=(uint32_t)J->mt*(uint32_t)J->cbs;
    const uint32_t cbs=(uint32_t)J->cbs;
    switch(J->cph) {
#define N16_CASE(C) case C: for(uint32_t u=part;u<units;u+=parts) n16_tile3(J,(int32_t)(u/cbs),(int32_t)(u%cbs),C); break;
    N16_CASE(0) N16_CASE(1) N16_CASE(2) N16_CASE(3)
#undef N16_CASE
    }
}
/* Output block pointers for this part's tiles (N16_DIRECT_OUT). */
static void n16_outp_part(const N16Job *J, uint32_t part, uint32_t parts) {
    const T *o=J->o;uint32_t any=0;
    for(int32_t t=(int32_t)part;t<J->mt;t+=(int32_t)parts) {
        int32_t tt=J->m0+t,by=tt/J->bw,bx=tt-by*J->bw;
        uint32_t nc=0;
        for(int32_t j=0;j<J->NB;j++) {
            uint8_t *ob=NULL;
            {   /* hmx_row(o, p[1]+by*8, (p[2]>>2)+bx, j) block start, in 32-bit arithmetic */
                const int32_t ry=(o->p[1]>>3)+by,wb=(o->p[2]>>2)+bx;
                if(ry>=0 && wb>=0 && wb<(int32_t)J->onbw) {
                    const uint32_t idx=(uint32_t)(((int32_t)(ry*(int32_t)J->onbw)+wb)*(int32_t)J->onbd)+o->slice+(uint32_t)j;
                    const uint32_t e=idx<o->nblocks?o->blocks[idx]:0,pool=e&127u;
                    if(pool && J->pool[pool]) ob=J->pool[pool]+(e&~127u);
                }
            }
            if(ob && !((uintptr_t)ob&2047u)) J->outp[(size_t)j*J->mt+t]=ob;
            else {J->outp[(size_t)j*J->mt+t]=J->C+((size_t)t*J->NB+(size_t)j)*2048;nc|=ob!=NULL;}
        }
        J->nc[t]=(uint8_t)nc;any|=nc;
    }
    ((N16Job *)J)->anyc[part]=any;
}
static void n16_stage_part(void *ctx, uint32_t part, uint32_t parts) {
    const N16Job *J=(const N16Job *)ctx;
    if(J->direct) n16_outp_part(J,part,parts);
    J->stage(ctx,part,parts);
}
/* All H*W taps of tile t, channel block cb, in one pass: each input row
 * (8+H-1 of them) is loaded once as up to 4 pixel pairs, the W column
 * shifts are built from them and stored into every tap tile using that row.
 * Same bytes as n16_tile for every (t, k). */
static void n16_tile_all(const N16Job *J, int32_t t, int32_t cb) {
    const T *xt=J->xt;
    int32_t tt=J->m0+t,by=tt/J->bw,bx=tt-by*J->bw;
    const HVX_Vector z=Q6_V_vzero();
    const int32_t X0=xt->p[2]+bx*4,a0=X0>>1,wbA=a0>>1;
    int32_t ra[3],od[3];
    for(int32_t kx=0;kx<J->W;kx++){ra[kx]=((X0+kx)>>1)-a0;od[kx]=(X0+kx)&1;}
    HVX_Vector *tile[3][3];
    for(int32_t ky=0;ky<J->H;ky++)for(int32_t kx=0;kx<J->W;kx++)
        tile[ky][kx]=(HVX_Vector *)(J->A+((size_t)t*J->KB+J->kinv[(cb*3+ky)*3+kx])*2048);
    const int32_t Ybase=xt->p[1]+by*8,nrows=8+J->H-1;
    int32_t brow=INT32_MIN;const uint8_t *blk[3]={NULL,NULL,NULL};
    for(int32_t i=0;i<nrows;i++) {
        const int32_t Y=Ybase+i;
        if((Y>>3)!=brow) {
            brow=Y>>3;
            for(int q=0;q<3;q++){const uint8_t *r=hmx_row(J->pool,xt,(int64_t)brow*8,wbA+q,J->xnbw,J->xnbd,cb);blk[q]=r;}
        }
        const size_t ro=(size_t)(Y&7)*256;
        HVX_Vector P[5];
        for(int q=0;q<5;q++) {
            const int32_t a=a0+q,b=(a>>1)-wbA;
            P[q]=(q<4 && blk[b])?((const HVX_Vector *)(blk[b]+ro))[a&1]:z;
        }
        for(int32_t kx=0;kx<J->W;kx++) {
            const int32_t r0=ra[kx];HVX_Vector v0,v1;
            if(!od[kx]){v0=P[r0];v1=P[r0+1];}
            else {
                v0=Q6_V_vor_VV(Q6_Vuw_vlsr_VuwR(P[r0],16),Q6_Vw_vasl_VwR(P[r0+1],16));
                v1=Q6_V_vor_VV(Q6_Vuw_vlsr_VuwR(P[r0+1],16),Q6_Vw_vasl_VwR(P[r0+2],16));
            }
            for(int32_t ky=0;ky<J->H;ky++) {
                const int32_t r=i-ky;
                if(r<0 || r>7) continue;
                HVX_Vector *d=tile[ky][kx]+2*r;d[0]=v0;d[1]=v1;
            }
        }
    }
}
static void n16_prep_all_part(void *ctx, uint32_t part, uint32_t parts) {
    const N16Job *J=(const N16Job *)ctx;
    const uint32_t units=(uint32_t)J->mt*(uint32_t)J->cbs;
    for(uint32_t u=part;u<units;u+=parts) n16_tile_all(J,(int32_t)(u/(uint32_t)J->cbs),(int32_t)(u%(uint32_t)J->cbs));
}
static void n16_tile(const N16Job *J, int32_t t, int32_t k) {
    const T *xt=J->xt;
    uint32_t e=J->ktab[k];
    int32_t ky=(int32_t)(e&255),kx=(int32_t)((e>>8)&255),cb=(int32_t)(e>>16);
    int32_t tt=J->m0+t,by=tt/J->bw,bx=tt-by*J->bw;
    HVX_Vector *dst=(HVX_Vector *)(J->A+((size_t)t*J->KB+(size_t)k)*2048);
    const HVX_Vector z=Q6_V_vzero();
    int64_t X0=(int64_t)xt->p[2]+bx*4+kx,a=X0>>1;
    int odd=(int)(X0&1);
    int64_t wb0=a>>1,wb1=(a+2)>>1;
    int64_t brow=-1;uint8_t *b0=NULL,*b1=NULL;
    for(int32_t r=0;r<8;r++) {
        int64_t Y=(int64_t)xt->p[1]+by*8+r+ky;
        if((Y>>3)!=brow) {
            brow=Y>>3;
            b0=hmx_row(J->pool,xt,brow*8,wb0,J->xnbw,J->xnbd,cb);
            b1=wb1!=wb0?hmx_row(J->pool,xt,brow*8,wb1,J->xnbw,J->xnbd,cb):b0;
        }
        size_t ro=(size_t)(Y&7)*256;
#define N16P(q) ((((q)>>1)==wb0?b0:b1)?((const HVX_Vector *)((((q)>>1)==wb0?b0:b1)+ro))[(q)&1]:z)
        if(!odd) {dst[2*r]=N16P(a);dst[2*r+1]=N16P(a+1);}
        else {
            HVX_Vector p0=N16P(a),p1=N16P(a+1),p2=N16P(a+2);
            dst[2*r]=Q6_V_vor_VV(Q6_Vuw_vlsr_VuwR(p0,16),Q6_Vw_vasl_VwR(p1,16));
            dst[2*r+1]=Q6_V_vor_VV(Q6_Vuw_vlsr_VuwR(p1,16),Q6_Vw_vasl_VwR(p2,16));
        }
#undef N16P
    }
}
static void n16_prep_part(void *ctx, uint32_t part, uint32_t parts) {
    const N16Job *J=(const N16Job *)ctx;
    uint32_t KB=(uint32_t)J->KB,t=0,k=part;
    while(k>=KB){k-=KB;t++;}
    while(t<(uint32_t)J->mt) {n16_tile(J,(int32_t)t,(int32_t)k);k+=parts;while(k>=KB){k-=KB;t++;}}
}
static void n16_out_part(void *ctx, uint32_t part, uint32_t parts) {
    const N16Job *J=(const N16Job *)ctx;const T *o=J->o;
    for(int32_t t=(int32_t)part;t<J->mt;t+=(int32_t)parts) {
        if(J->direct && !J->nc[t]) continue;
        int32_t tt=J->m0+t,by=tt/J->bw,bx=tt-by*J->bw;
        for(int32_t j=0;j<J->NB;j++) {
            if(J->direct && (uint8_t *)J->outp[(size_t)j*J->mt+t]!=J->C+((size_t)t*J->NB+(size_t)j)*2048) continue;
            uint8_t *ob=hmx_row(J->pool,o,(int64_t)o->p[1]+by*8,((int64_t)o->p[2]>>2)+bx,J->onbw,J->onbd,j);
            if(!ob) continue;
            const HVX_Vector *c=(const HVX_Vector *)(J->C+((size_t)t*J->NB+(size_t)j)*2048);
            for(int i=0;i<16;i++) ((HVX_Vector *)ob)[i]=c[i];
        }
    }
}
/* ptab[(R*ncol+w)*cbs+cb] = hmx_row(xt, (p[1]>>3)+R) block start for block column (p[2]>>2)+w, or n16_zero.
 * Whole runs of the (contiguous) block table are converted with HVX when every
 * entry is absent or in the tensor's pool; otherwise (or with vec=0) per entry. */
static void n16_build_ptab(const uint8_t **pt, uint8_t *const *pool, const T *xt, int32_t nR, int32_t ncol, int32_t cbs,
                           int64_t xnbw, int64_t xnbd, int vec) {
    const int32_t lo=xt->p[1]>>3,wlo=xt->p[2]>>2;
    const uint32_t Pk=xt->pool;
    vec=vec && xnbd==cbs && Pk>0 && Pk<7 && pool[Pk];
    HVX_Vector acc=Q6_V_vzero();uint32_t badrow=0;
    const HVX_Vector zv=Q6_V_vsplat_R((int32_t)(uintptr_t)n16_zero),m127=Q6_V_vsplat_R(127),mhi=Q6_V_vsplat_R(~127),
                     pk=Q6_V_vsplat_R((int32_t)Pk),pb=Q6_V_vsplat_R((int32_t)(uintptr_t)pool[Pk]),one=Q6_V_vsplat_R(1);
    for(int32_t R=0;R<nR;R++) {
        const uint8_t **row=pt+(size_t)R*(size_t)ncol*(size_t)cbs;
        const int64_t h=(int64_t)(lo+R)*8;
        int32_t w0=wlo<0?-wlo:0,w1=ncol;
        if((int64_t)wlo+w1>xnbw) w1=(int32_t)(xnbw-wlo);
        if(h<0 || w1<=w0) {w0=0;w1=0;}
        for(int32_t i=0;i<w0*cbs;i++) row[i]=n16_zero;
        for(int32_t i=w1*cbs;i<ncol*cbs;i++) row[i]=n16_zero;
        if(w1<=w0) continue;
        const int32_t n=(w1-w0)*cbs;
        const int64_t start=((int64_t)(lo+R)*xnbw+wlo+w0)*xnbd+xt->slice;
        int32_t i=0;
        if(vec && start>=0 && start+n<=(int64_t)xt->nblocks) {
            const uint32_t *src=xt->blocks+start;
            for(;i+32<=n;i+=32) {
                HVX_Vector e=*(const HVX_UVector *)(src+i);
                HVX_Vector pf=Q6_V_vand_VV(e,m127);
                HVX_VectorPred okq=Q6_Q_vcmp_eq_VwVw(pf,pk),abq=Q6_Q_vcmp_eq_VwVw(pf,Q6_V_vzero());
                acc=Q6_V_vor_VV(acc,Q6_V_vmux_QVV(Q6_Q_or_QQ(okq,abq),Q6_V_vzero(),one));
                HVX_Vector ptr=Q6_V_vmux_QVV(okq,Q6_Vw_vadd_VwVw(pb,Q6_V_vand_VV(e,mhi)),zv);
                *(HVX_UVector *)(row+w0*cbs+i)=ptr;
            }
            /* 32-bit tail, no hmx_row (64-bit index math): absent or in the tensor's pool */
            const uint8_t *pbase=pool[Pk];uint32_t sbad=0;
            for(;i<n;i++) {
                const uint32_t e=src[i],pf=e&127u;
                row[w0*cbs+i]=pf==Pk?pbase+(e&~127u):n16_zero;
                sbad|=pf!=Pk && pf!=0;
            }
            if(sbad) badrow=1;
        }
        for(;i<n;i++) {
            const uint8_t *b=hmx_row(pool,xt,h,wlo+w0+i/cbs,xnbw,xnbd,i%cbs);
            row[w0*cbs+i]=b?b:n16_zero;
        }
    }
    if(vec) {
        uint32_t tmp[32] __attribute__((aligned(128)));
        *(HVX_Vector *)tmp=acc;
        uint32_t bad=badrow;for(int k=0;k<32;k++) bad|=tmp[k];
        if(bad) n16_build_ptab(pt,pool,xt,nR,ncol,cbs,xnbw,xnbd,0);
    }
}
/* Stage, MAC and output of a prepared u16 conv: J (nbuf jobs) ready, Fi and the
 * finish record at fi/zero. 1 done, -1 timeout. */
static int n16_exec(const Nprg2ReferenceConv *c, N16Job *J, int32_t nbuf, Nprg2Parallel *par, int32_t KB, int32_t NB, int32_t mt, int32_t total,
                    const int8_t *wbase, int32_t *fi, int32_t *zero, uint8_t *completion, int32_t *err) {
    const Nprg2HmxOps *h=c->hmx;
#ifdef SLIDE_PROF
    uint64_t (*clock)(void)=NULL,t0=0,t1;
#else
    uint64_t (*clock)(void)=c->phase?c->clock:NULL,t0=clock?clock():0,t1;
#endif
    uint32_t parts=nprg2_parallel_parts(par,NPRG2_PAR_MAX_PARTS);
    uint32_t saved=h->enter?h->enter():0;
    int e=0;
    const int32_t nch=(total+mt-1)/mt;
    /* stage chunk ci into buffer ci%nbuf */
#define N16_STAGE(ci) do { \
        N16Job *Jc=&J[(ci)%nbuf];Jc->m0=(ci)*mt;Jc->mt=total-Jc->m0<mt?total-Jc->m0:mt; \
        if(!Jc->direct) for(int32_t j=0;j<NB;j++)for(int32_t t=0;t<Jc->mt;t++) \
            Jc->outp[(size_t)j*Jc->mt+t]=Jc->C+((size_t)t*NB+(size_t)j)*2048; \
        for(int k=0;k<4;k++) Jc->anyc[k]=0; \
        if(parts>1) nprg2_parallel_for(par,NPRG2_PAR_CONV,parts,n16_stage_part,Jc); \
        else n16_stage_part(Jc,0,1); \
    } while(0)
    if(clock){t1=clock();c->phase[6]+=t1-t0;t0=t1;}
    N16_STAGE(0);
    if(clock){t1=clock();c->phase[4]+=t1-t0;t0=t1;}
    for(int32_t ci=0;ci<nch && !e;ci++) {
        N16Job *Jc=&J[ci%nbuf];
        HVX_Vector sentinel=Q6_V_vsplat_R(-1);
        for(uint32_t i=0;i<2048;i+=128)*(HVX_Vector *)(completion+i)=sentinel;
        __asm__ volatile("barrier":::"memory");
        for(int32_t j=0;j<NB;j++)
            h->native16_n(Jc->A,(void *)(wbase+(size_t)j*KB*1024),fi+j*64,(void *const *)(Jc->outp+(size_t)j*Jc->mt),(uint32_t)KB,(uint32_t)Jc->mt);
        h->finish(zero,completion);
        if(N16_OVERLAP && ci+1<nch) {
            if(clock){t1=clock();c->phase[5]+=t1-t0;t0=t1;}
            N16_STAGE(ci+1);
            if(clock){t1=clock();c->phase[4]+=t1-t0;t0=t1;}
        }
        uint64_t end=h->clock?h->clock()+1000000000ull:0;
        for(uint64_t spin=0;;spin++) {
            typedef uint32_t HmxWords __attribute__((vector_size(128)));
            HmxWords v=*(volatile HmxWords *)completion;
            if(v[0]==0) break;
            if(h->clock?h->clock()>end:spin>100000000ull){e=1;break;}
        }
        __asm__ volatile("barrier":::"memory");
        if(clock){t1=clock();c->phase[5]+=t1-t0;t0=t1;}
        if(e) break;
        if(!Jc->direct || (Jc->anyc[0]|Jc->anyc[1]|Jc->anyc[2]|Jc->anyc[3])) {
            if(parts>1) nprg2_parallel_for(par,NPRG2_PAR_CONV,parts,n16_out_part,Jc);
            else n16_out_part(Jc,0,1);
        }
        if(clock){t1=clock();c->phase[7]+=t1-t0;t0=t1;}
        if(!N16_OVERLAP && ci+1<nch) {
            N16_STAGE(ci+1);
            if(clock){t1=clock();c->phase[4]+=t1-t0;t0=t1;}
        }
    }
#undef N16_STAGE
    if(h->leave) h->leave(saved);
    if(e){*err=E_RANGE;return -1;}
    return 1;
}
#ifndef NO_CONV_PLANS
/* u16 conv plans, as ConvPlan: geometry and the job are fixed, Fi read at run time. */
typedef struct {
    T xt, o; N16Job J;
    const int8_t *wbase; const uint8_t *ft;
    int32_t KB, NB, mt, total;
} N16Plan;
enum { N16_PLANS=160, N16_ARENA=0x40000 };
static N16Plan *n16_plan_pool;
/* The plans' block pointer tables and tap tables: fixed data, so they live with the plan. */
static uint8_t *n16_arena;
static uint32_t n16_arena_used;
static uint16_t *n16_plan_ix; /* by command index: pool index + 1 */
static uint32_t n16_plan_used;
#endif
/* 1 ran (or would, dry), 0 not eligible, -1 error. */
static int hmx_native16(G *g, const T *xt, const T *wt, const T *ft, const T *o, int packed, int reverse, int32_t co,
                        int32_t oh, int32_t ow, int32_t *err) {
    const Nprg2ReferenceConv *c=g->conv;
    if(!c->hmx || !c->hmx->native16_n || !c->near || ((uintptr_t)c->near&127u) || g->fx.kind) return 0;
    int32_t H=wt->m[0],W=wt->m[1],I=wt->m[2];
    if(I%32 || xt->m[3]<I || H>3 || W>3 || co%32 || oh%8 || ow%4 || o->p[1]%8 || o->p[2]%4) return 0;
    if(!(xt->flags&F_BLOCKED) || xt->esize!=2 || !(o->flags&F_BLOCKED) || o->esize!=2 || xt->zp) return 0;
    if(xt->p[0] || o->p[0] || xt->m[0]!=1 || o->m[0]!=1 || wt->pool!=1) return 0;
    const int32_t cbs=I/32,NB=co/32,KB=H*W*cbs;
    if(!KB || KB*32>HMX_TERMS_MAX || KB>255) return 0;
    const int8_t *wbase=(const int8_t *)g->pool[wt->pool]+wt->offset;
    if((uintptr_t)wbase&1023u) return 0;
    int32_t dims[4]={1,oh+H-1,ow+W-1,I},origin[4]={0,xt->p[1],xt->p[2],0};
    int32_t on[4]={1,oh,ow,co},oorigin[4]={0,o->p[1],o->p[2],0};
    {
        /* bit 0 of conv_checked (the u8 HMX bits are free for u16 convs) */
        Memo *mm=g->dry?NULL:gmemo(g);
        if(!mm || !(mm->conv_checked[g->call->command_index]&1u)) {
            if(xfer_check(g,xt,dims,origin,0) || xfer_check(g,o,on,oorigin,1) ||
               blocked_overlap(xt,dims,origin,o,on,oorigin)) return 0;
            if(mm) mm->conv_checked[g->call->command_index]|=1u;
        }
    }
    size_t at=0;
    size_t o_ktab=nbump(&at,(size_t)KB*4),o_rows=nbump(&at,(size_t)NB*256),o_fi=nbump(&at,(size_t)NB*256+256),
           o_out=0,o_nc=0,o_pt=0;
    const int32_t bw=ow/4,total=(oh/8)*bw;
    const int32_t nbuf=N16_OVERLAP?2:1;
    const size_t fixed=((size_t)NB*256+256+2048*2+2047)&~(size_t)2047,per=(size_t)KB*2048+(size_t)NB*2048;
    if(c->workspace_bytes<fixed+per*(size_t)nbuf+2048) return 0;
    int32_t mt=(int32_t)((c->workspace_bytes-fixed-2048)/(per*(size_t)nbuf));
    if(mt>total) mt=total;
    o_out=nbump(&at,(size_t)nbuf*NB*mt*sizeof(uint8_t *));
    o_nc=nbump(&at,(size_t)nbuf*((mt+3)&~3));
    const int32_t ncol=bw+3,nR=oh/8+2;
    int use_pt=N16_PT && H==3 && W==3;
#if N16_PT
    o_pt=nbump(&at,(size_t)nR*ncol*cbs*sizeof(uint8_t *));
    if(at>c->near_bytes) {use_pt=0;at=o_pt;}
#endif
    if(at>c->near_bytes) return 0;
    uint8_t *n=c->near;
    /* Fi words straight from the filter rows: scale 0x400000|e<<10|(M-1024), bias. */
    int32_t *nf=(int32_t *)(n+o_fi),*nz=nf+NB*64;
    /* Fi rows are run-time data (the dry pass may see stale VTCM): checked on
     * real runs only; a rejection there falls back to the HVX path. */
    if(!g->dry) near_copy(n+o_rows,g->pool[ft->pool]+ft->offset,(size_t)NB*256);
    for(int32_t k=0;k<co && !g->dry;k++) {
        uint32_t m=*(const uint32_t *)(n+o_rows+(size_t)(k>>5)*256+(size_t)(k&31)*4)&65535u,e=(m>>10)&31u;
        int32_t b=*(const int32_t *)(n+o_rows+(size_t)(k>>5)*256+128+(size_t)(k&31)*4);
        if(e<8 || e>19 || b<-(1<<23) || b>(1<<23)) return 0;
        nf[(k>>5)*64+(k&31)]=(int32_t)(0x400000u|(e<<10)|(m&1023u));
        nf[(k>>5)*64+32+(k&31)]=b;
    }
    for(int k=0;k<64;k++) nz[k]=k<32?0x404000:0;
    uint32_t *ktab=(uint32_t *)(n+o_ktab);
    {
        int32_t k=0;
        if(packed) {for(int32_t h=0;h<H;h++)for(int32_t x=0;x<W;x++)for(int32_t cb=0;cb<cbs;cb++,k++)
            ktab[k]=(uint32_t)h|(uint32_t)x<<8|(uint32_t)cb<<16;}
        else for(int32_t cb=0;cb<cbs;cb++)for(int32_t h=0;h<H;h++)for(int32_t x=0;x<W;x++,k++)
            ktab[k]=(uint32_t)h|(uint32_t)(reverse?W-1-x:x)<<8|(uint32_t)cb<<16;
    }
    g->wused=0;
    uint8_t *ws=walloc(g,fixed+(size_t)nbuf*mt*per+2048);
    if(!ws){*err=E_MEMORY;return -1;}
    conv_path[g->call->execution_lane<MEMO_LANES?g->call->execution_lane:0u]=5;
    if(g->dry) return 1;
    uint8_t *base=(uint8_t *)(((uintptr_t)ws+2047)&~(uintptr_t)2047);
    int32_t *fi=(int32_t *)base,*zero=(int32_t *)(base+(size_t)NB*256);
    uint8_t *completion=base+fixed-2048;
    near_copy(base,(const uint8_t *)nf,(size_t)NB*256+256);
    const Nprg2HmxOps *h=c->hmx;
    N16Job J[2];
    for(int b=0;b<nbuf;b++) {
        J[b]=(N16Job){.pool=g->pool,.xt=xt,.o=o,.KB=KB,.NB=NB,.bw=bw,.ktab=ktab,
              .A=base+fixed+(size_t)b*mt*per,.C=base+fixed+(size_t)b*mt*per+(size_t)mt*KB*2048,
              .xnbw=((int64_t)xt->m[2]+3)/4,.xnbd=((int64_t)xt->m[3]+31)/32,
              .onbw=((int64_t)o->m[2]+3)/4,.onbd=((int64_t)o->m[3]+31)/32};
        J[b].outp=(uint8_t **)(n+o_out)+(size_t)b*NB*mt;J[b].nc=n+o_nc+(size_t)b*((mt+3)&~3);
        J[b].direct=N16_DIRECT_OUT;J[b].H=H;J[b].W=W;J[b].cbs=0;J[b].stage=NULL;
    }
#if N16_FUSED
    /* the fused staging needs every (cb, ky, kx) tap exactly once */
    if(H*W*cbs==KB) {
        memset(J[0].kinv,255,sizeof J[0].kinv);int ok=cbs*9<=256;
        for(int32_t k=0;k<KB && ok;k++) {
            uint32_t e=ktab[k];uint32_t ky=e&255u,kx=(e>>8)&255u,cb=e>>16;
            uint32_t ix=(cb*3u+ky)*3u+kx;
            if(ky>=(uint32_t)H || kx>=(uint32_t)W || cb>=(uint32_t)cbs || J[0].kinv[ix]!=255) ok=0;
            else J[0].kinv[ix]=(uint8_t)k;
        }
        if(ok) for(int b=0;b<nbuf;b++) {J[b].cbs=cbs;if(b) memcpy(J[b].kinv,J[0].kinv,sizeof J[0].kinv);}
    }
#endif
    void (*stage)(void *, uint32_t, uint32_t)=J[0].cbs?n16_prep_all_part:n16_prep_part;
#if N16_PT
    if(use_pt && J[0].cbs) {
        const uint8_t **pt=(const uint8_t **)(n+o_pt);
        n16_build_ptab(pt,g->pool,xt,nR,ncol,cbs,J[0].xnbw,J[0].xnbd,1);
        stage=n16_prep3_part;
        for(int b=0;b<nbuf;b++) {
            J[b].ptab=pt;J[b].ncol=ncol;J[b].cph=xt->p[2]&3;
            const int32_t s=xt->p[1]&7;
            for(int i=0;i<10;i++){J[b].mi[i]=(uint32_t)((s+i)>>3);J[b].ro[i]=(uint32_t)((s+i)&7)*256u;}
        }
    }
#endif
    for(int b=0;b<nbuf;b++) J[b].stage=stage;
#ifndef NO_CONV_PLANS
    int r16=n16_exec(c,J,nbuf,g->call->parallel,KB,NB,mt,total,wbase,fi,zero,completion,err);
    /* The same near and workspace layout on every run, so the job's near pointers stay
     * valid: outp and nc are rebuilt by every stage. ktab and the block table are fixed
     * but near is not preserved between commands, so the plan keeps its own copies. */
    if(r16==1 && n16_plan_pool && nbuf==1 && g->call->execution_lane==0 && g->call->program==plan_program && g->conv==plan_conv &&
       g->call->command_index<MEMO_COMMANDS && !n16_plan_ix[g->call->command_index] && n16_plan_used<N16_PLANS &&
       !((uintptr_t)(g->pool[ft->pool]+ft->offset)&127u)) {
#if N16_PT
        const size_t ptb=J[0].ptab?(size_t)nR*ncol*cbs*sizeof(uint8_t *):0;
#else
        const size_t ptb=0;
#endif
        const size_t kb=(size_t)KB*4,need=ptb+kb;
        if(n16_arena_used+need<=N16_ARENA) {
            N16Plan *P=&n16_plan_pool[n16_plan_used];
            uint8_t *mem=n16_arena+n16_arena_used;
            memcpy(mem,ktab,kb);
            P->xt=*xt;P->o=*o;P->J=J[0];P->J.xt=&P->xt;P->J.o=&P->o;P->J.pool=plan_pool;P->J.ktab=(const uint32_t *)mem;
#if N16_PT
            if(ptb) {memcpy(mem+kb,J[0].ptab,ptb);P->J.ptab=(const uint8_t *const *)(mem+kb);}
#endif
            P->wbase=wbase;P->ft=g->pool[ft->pool]+ft->offset;
            P->KB=KB;P->NB=NB;P->mt=mt;P->total=total;
            n16_arena_used+=(uint32_t)need;
            n16_plan_ix[g->call->command_index]=(uint16_t)(++n16_plan_used);
        }
    }
    return r16;
#else
    return n16_exec(c,J,nbuf,g->call->parallel,KB,NB,mt,total,wbase,fi,zero,completion,err);
#endif
}
#endif
