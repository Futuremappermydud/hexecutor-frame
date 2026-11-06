/* Native ARM executor for QDRX images; see qd_graph.h. Each kernel mirrors the
 * like-named function of tools/htp_emu.py (and fast_depth.py for convolution),
 * including NumPy's broadcasting, padding and float rounding. Anything outside
 * the shapes and arguments those kernels accept is rejected, never guessed. */
#define _GNU_SOURCE
#include "qd_graph.h"
#include "qd_resident.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef QDG_DSP
#include <pthread.h>
#include <sched.h>
#endif
#include <stdatomic.h>
#if defined(QDG_DSP) && QDS_BLOCKED_GLUE
#include "kernels/blocked/linearclip_u16.h"
#include "kernels/blocked/linearclip_u8_u16.h"
#include "kernels/blocked/sign_blocked.h"
#include "kernels/blocked/add_u8.h"
#include "kernels/layout/input_slice_pad_direct.h"
#include "kernels/layout/im2col_u8_crouton_direct.h"
#include "kernels/layout/space_to_depth_u8_direct.h"
#endif

enum { DT_U8=1, DT_I8=2, DT_U16=3, DT_I16=4, DT_I32=5, DT_F32=6, DT_Q32=7 };
#define FAIL(code) do { g->error_detail=__LINE__; return (code); } while(0)
#define CHECK(cond,code) do { if(!(cond)) FAIL(code); } while(0)
#define TRY(expr) do { int e_=(expr); if(e_) return e_; } while(0)

typedef struct {
    uint32_t id, dtype, flags, esize, crouton, cls;
    int32_t d[4], m[4], p[4];
    uint32_t pool, offset, nblocks, slice;
    const uint32_t *blocks;
    int32_t zp, sv[4];
    uint32_t scalar, unique_blocks; /* 0 unknown, 1 unique, 2 aliased */
    double scale;
} T;

typedef struct {
    uint32_t seq, op, variant, io, nin, nout, nfv, ncalls;
    const uint32_t *in, *out, *fv;
    const uint8_t *calls;
} C;

typedef struct { int32_t n[4]; uint32_t dt, scalar; void *p; } A;

typedef struct {
    int used, s2, co;
    uint32_t source, key;     /* key: globally unique, 0 for dynamic weights */
    int8_t *w;               /* dense HWIO, :co */
    int32_t kh, kw, ci;
    int64_t *wsum;           /* per output channel */
    int64_t *wpos;           /* per output channel: sum of positive weights */
    uint8_t *packed[2];      /* DSP layouts: [0] vrmpy, [1] HMX */
    uint32_t packed_bytes[2], packed_ci[2];
} Weights;

typedef struct {
    int ready, quant_ok, static_q, block_unique;
    int32_t *params;         /* [co][3] correction, multiplier, preshift */
    int hmx;
} ConvState;

typedef struct { uint32_t h,width,c,g,wp,kh,kw,n,oh,ow,owp,xo,wo,co,xb,wb,cb,pb,kb,quantized,hmx,u16,plane,vb,keep,key,x32,out_h,out_w; } Layout;
typedef struct {
    int busy;
    uint32_t cmd, kind, pending, planes;
    int32_t oh, ow, co, origin[4], zp, sixteen;
    uint8_t *r; size_t r_cap;
    int64_t *acc[2]; size_t acc_cap[2];
    int32_t *mult, *bias; size_t mult_cap, bias_cap;
    int64_t *wsum; size_t wsum_cap;
} Job;
typedef struct { int active; uint32_t seq, plane, slot; Layout L; Job *job; } Req;

/* Helper threads that split bulk copies out of the uncached DSP arena: one ARM
 * core reads it at ~2 GB/s, several in parallel approach 9 GB/s. */
#define MAX_HELPERS 6
#ifdef QDG_DSP
typedef struct { int started; } Pool;
#else
typedef struct {
    pthread_t thread[MAX_HELPERS];
    int count, started;
    pthread_mutex_t lock;
    pthread_cond_t wake;
    atomic_uint generation, done;
    atomic_int active, quit;
    void (*job)(void *ctx, unsigned part, unsigned parts);
    void *ctx;
} Pool;
#endif

struct QdgGraph {
    uint8_t *image;
    QdrImage img;
    uint32_t ntensors, ncommands;
    T *t;
    C *c;
    uint8_t *pool[QDG_POOLS];
    uint64_t pool_bytes[QDG_POOLS];
    uint32_t *static_source;  /* tensor index -> constant-pool source index or NONE */
    uint8_t *static_fi;
    Weights *weights; uint32_t nweights, capweights;
    ConvState *conv;
    uint8_t *scratch; size_t scratch_used, scratch_bytes;
#ifdef QDG_DSP
    uint8_t *dsp_scratch; size_t dsp_scratch_bytes, dsp_scratch_peak;
    size_t dsp_l2_used,dsp_l2_peak;uint32_t dsp_l2_active;
    const T *dsp_direct_tensor;uint32_t dsp_direct_h,dsp_direct_w,dsp_direct_calls,dsp_direct_enabled;
    int32_t dsp_direct_origin[4];
    uint64_t dsp_read_cycles,dsp_write_cycles,dsp_pack_cycles[3],dsp_weights_cycles,dsp_quant_cycles,dsp_weights_miss_cycles,dsp_weights_misses;
    uint64_t dsp_conv_phase[8]; /* FI/metadata, input read, params, output collect/write; u8/u16 pairs */
    uint64_t dsp_dynamic_tcm_calls;
    uint64_t dsp_fallback_phase[4],dsp_fallback_tcm_calls,dsp_fallback_l2_calls;
#if QDS_NATIVE_1X1
    uint64_t dsp_native_1x1_diag[8];
#endif
#if QDS_NATIVE_3X3
    uint64_t dsp_native_3x3_diag[8];
#endif
#if QDS_BLOCKED_GLUE
    QdBlockedU16Linearclip **blocked_lc;
    QdBlockedLinearclipU8U16 **blocked_lc_u8;
    QdBlockedSignU16 **blocked_sign;
    QdBlockedAddU8 **blocked_add;
    uint64_t dsp_blocked_lc_calls,dsp_blocked_lc_prepare,dsp_blocked_lc_apply,dsp_blocked_lc_u8_calls;
    uint64_t dsp_blocked_sign_calls,dsp_blocked_sign_prepare,dsp_blocked_sign_apply;
    uint64_t dsp_blocked_add_calls,dsp_blocked_add_prepare,dsp_blocked_add_apply;
    uint64_t dsp_direct_input_pad_calls;
    uint64_t dsp_direct_im2col_calls;
    uint64_t dsp_direct_space_to_depth_calls;
#if QDS_BLOCKED_S2D
    QdLayoutS2dPlan **blocked_s2d;
    uint64_t dsp_s2d_diag[6]; /* prepare, apply, proof, gather, scatter, cache misses */
#endif
#endif
#endif
    const void *const *inputs; const uint64_t *input_bytes;
    void *const *outputs; const uint64_t *output_bytes;
    int backend, profile, hoist, block_output, native_hmx;
    /* DSP */
    uint8_t *arena;
    const volatile uint32_t *callback_word;
    uint32_t callback_value, sequence, persistent, raise_clock, use_hmx;
    double timeout;
    QdgWake wake;
    uint8_t *staging;
    Pool helpers;
    uint32_t resident_key, resident_b, resident_wb;
    Job jobs[2];
    Req inflight;
    uint64_t block_base[QDG_POOLS];
    uint32_t fp_words;
    uint64_t *fp_all, **fp_conv_in, **fp_conv_out;
    uint8_t *hoisted;
    QdgStats stats;
    uint32_t error_detail, current;
};
#ifdef QDG_DSP
static int resident_cancelled(void);
static const T *dsp_tensor(QdgGraph *,uint32_t);
static void dsp_command_begin(QdgGraph *,uint32_t,C *);
#endif

static uint64_t now_ns(void) {
#ifdef QDG_DSP
    uint64_t c;__asm__ volatile("%0=upcycle":"=r"(c));return c;
#else
    struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);
    return (uint64_t)ts.tv_sec*1000000000u+(uint64_t)ts.tv_nsec;
#endif
}

/* ------------------------------------------------------------------ scratch */
static void *salloc(QdgGraph *g, size_t bytes) {
#ifdef QDG_DSP
    /* Between convolutions the first 512 KB of L2-TCM contains no live
     * weights/tables. Scalar graph kernels are much faster there than VTCM.
     * No pointer survives the command; overflow allocations use VTCM. */
    if(g->dsp_l2_active) {
        size_t l2=(g->dsp_l2_used+127)&~(size_t)127;
        if(l2<=0x80000u && bytes<=0x80000u-l2) {
            g->dsp_l2_used=l2+bytes;
            if(g->dsp_l2_used>g->dsp_l2_peak)g->dsp_l2_peak=g->dsp_l2_used;
            return (void *)(0x30000000u+l2);
        }
    }
#endif
    size_t at=(g->scratch_used+63)&~(size_t)63;
    if(at+bytes>g->scratch_bytes) return NULL;
    g->scratch_used=at+bytes;
#ifdef QDG_DSP
    if(g->scratch_used>g->dsp_scratch_peak)g->dsp_scratch_peak=g->scratch_used;
#endif
    return g->scratch+at;
}
static size_t count4(const int32_t *n) {return (size_t)n[0]*n[1]*n[2]*n[3];}
static uint32_t dsize(uint32_t dt) {return dt==DT_U8||dt==DT_I8?1:dt==DT_U16||dt==DT_I16?2:4;}
static int anew(QdgGraph *g, A *a, const int32_t *n, uint32_t dt) {
    for(int i=0;i<4;i++){if(n[i]<0)return QDG_ERR_SHAPE;a->n[i]=n[i];}
    a->dt=dt;a->scalar=0;
    a->p=salloc(g,count4(n)*dsize(dt)+64);
    return a->p?0:QDG_ERR_MEMORY;
}
static int64_t aget(const A *a, size_t i) {
    switch(a->dt) {
    case DT_U8: return ((const uint8_t *)a->p)[i];
    case DT_I8: return ((const int8_t *)a->p)[i];
    case DT_U16: return ((const uint16_t *)a->p)[i];
    case DT_I16: return ((const int16_t *)a->p)[i];
    default: return ((const int32_t *)a->p)[i];
    }
}
/* Integer values of an integer array, widened once per array. */
static int to_i32(QdgGraph *g, const A *a, int32_t **out) {
    size_t n=count4(a->n);
    int32_t *r=salloc(g,n*4+64);
    if(!r) return QDG_ERR_MEMORY;
    if(a->dt==DT_F32) return QDG_ERR_UNSUPPORTED;
    if(a->dt==DT_U8){const uint8_t *s=a->p;for(size_t i=0;i<n;i++)r[i]=s[i];}
    else if(a->dt==DT_U16){const uint16_t *s=a->p;for(size_t i=0;i<n;i++)r[i]=s[i];}
    else for(size_t i=0;i<n;i++)r[i]=(int32_t)aget(a,i);
    *out=r;return 0;
}
/* Store already range-checked values (NumPy's astype after an explicit clip). */
static void from_i64(A *a, size_t i, int64_t v) {
    switch(a->dt) {
    case DT_U8: ((uint8_t *)a->p)[i]=(uint8_t)v;break;
    case DT_I8: ((int8_t *)a->p)[i]=(int8_t)v;break;
    case DT_U16: ((uint16_t *)a->p)[i]=(uint16_t)v;break;
    case DT_I16: ((int16_t *)a->p)[i]=(int16_t)v;break;
    default: ((int32_t *)a->p)[i]=(int32_t)v;break;
    }
}
static int64_t clip64(int64_t v, int64_t lo, int64_t hi) {return v<lo?lo:v>hi?hi:v;}
static int64_t dtmax(uint32_t dt) {return dt==DT_U8?255:dt==DT_U16?65535:dt==DT_I8?127:dt==DT_I16?32767:2147483647;}
static int64_t s32(uint64_t v) {return (int64_t)(int32_t)(uint32_t)v;}

/* ------------------------------------------------------------ tensor access */
static void parallel(QdgGraph *g, void (*job)(void *, unsigned, unsigned), void *ctx);
#ifdef QDG_DSP
static void dsp_cache_begin(uint8_t *const *pools);
static uint8_t *dsp_cache_block(uint32_t encoded,int store);
static void dsp_cache_end(void);
#endif
/* Croutons: 2 KB blocks of [8 rows][8|4|2 columns][32 lanes]; u16 lanes interleave
 * column pairs. Element size is a compile-time constant in each instance. */
#ifdef QDG_DSP
typedef int32_t Xindex; /* setup bounds the complete logical tensor to 512 MB */
#else
typedef int64_t Xindex;
#endif
static inline __attribute__((always_inline)) void blocked_impl(uint8_t *const *pools, const T *t, uint8_t *values,
        const int32_t *dims, const int32_t *origin, const uint32_t size, const int store) {
    const Xindex cw=size==1?8:size==2?4:2;
    Xindex maxh=t->m[1],maxw=t->m[2],maxd=t->m[3],slice=t->slice,count=t->nblocks;
    Xindex nbh=(maxh+7)/8,nbw=(maxw+cw-1)/cw,nbd=(maxd+31)/32;
    const uint32_t *blocks=t->blocks;
    size_t pos=0;
    for(Xindex b=0;b<dims[0];b++)for(Xindex h=0;h<dims[1];h++) {
        Xindex B=b+origin[0],H=h+origin[1];
        Xindex hrow=(B*nbh+H/8)*nbw;
        size_t hin=size==1?(H%8)*256:size==2?(H%8)*128:(H%8)*64;
        for(Xindex w=0;w<dims[2];w++) {
            Xindex W=w+origin[2];
            Xindex wrow=(hrow+W/cw)*nbd+slice;
            size_t win=hin+(size==1?(W%8)*32:size==2?((W%4)/2)*64+W%2:(W%2)*32);
            for(Xindex d=0;d<dims[3];) {
                Xindex D=d+origin[3],n=32-D%32;
                if(n>dims[3]-d)n=dims[3]-d;
                Xindex index=wrow+D/32;
                uint32_t e=(index>=0 && index<count)?blocks[index]:0,pool=e&127;
                uint8_t *base=pool?pools[pool]:NULL;
                if(base) {
                    size_t within=win+(size==2?(D%32)*2:D%32);
                    uint8_t *ptr=
#ifdef QDG_DSP
                        dsp_cache_block(e,store)+within*size;
#else
                        base+(e&~127u)+within*size;
#endif
                    if(size==1 && n==32) {
#ifdef QDG_DSP
                        if(store)dsp_copy_small(ptr,values+pos,32);else dsp_copy_small(values+pos,ptr,32);
#else
                        if(store)__builtin_memcpy(ptr,values+pos,32);else __builtin_memcpy(values+pos,ptr,32);
#endif
                    } else if(size!=2) {
#ifdef QDG_DSP
                        if(store)dsp_copy_small(ptr,values+pos,n*size);else dsp_copy_small(values+pos,ptr,n*size);
#else
                        if(store)memcpy(ptr,values+pos,n*size);else memcpy(values+pos,ptr,n*size);
#endif
                    } else {
                        uint16_t *v=(uint16_t *)(values+pos),*q=(uint16_t *)ptr;
                        if(store)for(Xindex c=0;c<n;c++)q[2*c]=v[c];
                        else for(Xindex c=0;c<n;c++)v[c]=q[2*c];
                    }
                } else if(!store)memset(values+pos,0,n*size);
                pos+=n*size;d+=n;
            }
        }
    }
}
static void blocked_dispatch(uint8_t *const *p, const T *t, uint8_t *values, const int32_t *dims,
                             const int32_t *origin, int store) {
    switch(t->esize*2+!!store) {
    case 2: blocked_impl(p,t,values,dims,origin,1,0);break;
    case 3: blocked_impl(p,t,values,dims,origin,1,1);break;
    case 4: blocked_impl(p,t,values,dims,origin,2,0);break;
    case 5: blocked_impl(p,t,values,dims,origin,2,1);break;
    case 8: blocked_impl(p,t,values,dims,origin,4,0);break;
    default: blocked_impl(p,t,values,dims,origin,4,1);break;
    }
}
typedef struct { QdgGraph *g; const T *t; uint8_t *values; const int32_t *dims, *origin; int store; } XferJob;
static void xfer_job(void *ctx, unsigned part, unsigned parts) {
    XferJob *j=ctx;
    int32_t rows=j->dims[1],per=(rows+(int32_t)parts-1)/(int32_t)parts,h0=(int32_t)part*per;
    if(h0>=rows) return;
    int32_t n=rows-h0<per?rows-h0:per;
    int32_t d[4]={1,n,j->dims[2],j->dims[3]},o[4]={j->origin[0],j->origin[1]+h0,j->origin[2],j->origin[3]};
    size_t row=(size_t)j->dims[2]*j->dims[3]*j->t->esize;
    blocked_dispatch(j->g->pool,j->t,j->values+(size_t)h0*row,d,o,j->store);
}
static int blocked_xfer(QdgGraph *g, const T *t, uint8_t *values, const int32_t *dims,
                        const int32_t *origin, int store) {
#ifdef QDG_DSP
    g->resident_key=0; /* block cache reuses HMX's VTCM scratch */
    if(t->esize==1 && !(dims[3]&31) && !(origin[3]&31)) {
        dsp_cache_begin(g->pool);
        dsp_block_u8(g->pool,t->blocks,t->nblocks,t->slice,t->m,dims,origin,values,store);
        dsp_cache_end();return 0;
    }
    if(t->esize==2 && !(dims[3]&31) && !(origin[3]&31)) {
        dsp_cache_begin(g->pool);
        dsp_block_u16(g->pool,t->blocks,t->nblocks,t->slice,t->m,dims,origin,values,store);
        dsp_cache_end();return 0;
    }
    dsp_cache_begin(g->pool);
    blocked_dispatch(g->pool,t,values,dims,origin,store);
    dsp_cache_end();return 0;
#endif
    size_t bytes=count4(dims)*t->esize;
    if(dims[0]==1 && dims[1]>=4 && bytes>=(48u<<10) && g->helpers.started) {
        XferJob j={g,t,values,dims,origin,store};
        parallel(g,xfer_job,&j);
    } else blocked_dispatch(g->pool,t,values,dims,origin,store);
    return 0;
}
static int direct_xfer(QdgGraph *g, const T *t, uint8_t *values, const int32_t *dims,
                       const int32_t *origin, int store) {
    for(int i=0;i<4;i++)CHECK(origin[i]>=0 && dims[i]>=0 && origin[i]+dims[i]<=t->m[i],QDG_ERR_RANGE);
    uint8_t *base=g->pool[t->pool];
    CHECK(base,QDG_ERR_TENSOR);
    size_t es=t->esize,row=(size_t)dims[3]*es,pos=0;
#ifdef QDG_DSP
    if(dims[3]==t->m[3] && origin[3]==0) {
        size_t rows=(size_t)dims[2]*row;
        for(int32_t b=0;b<dims[0];b++)for(int32_t h=0;h<dims[1];h++) {
            size_t lin=(((size_t)(b+origin[0])*t->m[1]+h+origin[1])*t->m[2]+origin[2])*t->m[3];
            uint8_t *ptr=base+t->offset+lin*es;
            if(store)memcpy(ptr,values+pos,rows);else memcpy(values+pos,ptr,rows);
            pos+=rows;
        }
        return 0;
    }
#endif
    for(int32_t b=0;b<dims[0];b++)for(int32_t h=0;h<dims[1];h++)for(int32_t w=0;w<dims[2];w++) {
        size_t lin=(((size_t)(b+origin[0])*t->m[1]+h+origin[1])*t->m[2]+w+origin[2])*t->m[3]+origin[3];
        uint8_t *ptr=base+t->offset+lin*es;
#ifdef QDG_DSP
        if(store)dsp_copy_small(ptr,values+pos,row);else dsp_copy_small(values+pos,ptr,row);
#else
        if(store)memcpy(ptr,values+pos,row);else memcpy(values+pos,ptr,row);
#endif
        pos+=row;
    }
    return 0;
}
static int xfer(QdgGraph *g, const T *t, void *values, const int32_t *dims, const int32_t *origin, int store) {
#ifdef QDG_DSP
    uint64_t start=now_ns();int e;
    if(t->flags&QDR_DIRECT)e=direct_xfer(g,t,values,dims,origin,store);
    else if(t->flags&QDR_BLOCKED)e=blocked_xfer(g,t,values,dims,origin,store);
    else FAIL(QDG_ERR_TENSOR);
    if(store)g->dsp_write_cycles+=now_ns()-start;else g->dsp_read_cycles+=now_ns()-start;
    return e;
#endif
    if(t->flags&QDR_DIRECT) return direct_xfer(g,t,values,dims,origin,store);
    if(t->flags&QDR_BLOCKED) return blocked_xfer(g,t,values,dims,origin,store);
    FAIL(QDG_ERR_TENSOR);
}
static int t_read(QdgGraph *g, const T *t, A *a, int full) {
    static const int32_t zero[4]={0,0,0,0};
    CHECK(t->flags&QDR_HAS_SHAPE,QDG_ERR_TENSOR);
    TRY(anew(g,a,full?t->m:t->d,t->dtype));
    return xfer(g,t,a->p,a->n,full?zero:t->p,0);
}
static int t_read_region(QdgGraph *g, const T *t, A *a, const int32_t *dims, const int32_t *origin) {
    TRY(anew(g,a,dims,t->dtype));
    return xfer(g,t,a->p,a->n,origin,0);
}
/* Memory.write: crouton depth extends to its 32-lane boundary, a narrower 4-D
 * value is zero padded in depth, then broadcast. Value dtype must already be
 * the tensor's (each kernel performs NumPy's conversion explicitly). */
static int t_write(QdgGraph *g, const T *t, const A *v) {
    int32_t dims[4];
    CHECK(t->flags&QDR_HAS_SHAPE,QDG_ERR_TENSOR);
    CHECK(v->dt==t->dtype || (dsize(v->dt)==t->esize && v->dt!=DT_F32 && t->dtype!=DT_F32),QDG_ERR_TENSOR);
    memcpy(dims,t->d,sizeof dims);
    if(t->crouton) {
        int32_t e=(t->d[3]+t->p[3]+31)/32*32;
        if(e>t->m[3])e=t->m[3];
        dims[3]=e-t->p[3];
    }
    int32_t vn[4];memcpy(vn,v->n,sizeof vn);
    int pad=!v->scalar && vn[3]<dims[3];
    int same=!pad;
    for(int i=0;i<4;i++) {
        int32_t have=(pad && i==3)?dims[3]:vn[i];
        if(have!=dims[i] && have!=1) FAIL(QDG_ERR_SHAPE);
        if(have!=dims[i]) same=0;
    }
    if(same) return xfer(g,t,v->p,dims,t->p,1);
    A full;TRY(anew(g,&full,dims,t->dtype));
    size_t es=t->esize,row=(size_t)dims[3]*es;
    uint8_t *dst=full.p;
    for(int32_t b=0;b<dims[0];b++)for(int32_t h=0;h<dims[1];h++)for(int32_t w=0;w<dims[2];w++,dst+=row) {
        int32_t ib=vn[0]==1?0:b,ih=vn[1]==1?0:h,iw=vn[2]==1?0:w;
        const uint8_t *src=(const uint8_t *)v->p+((((size_t)ib*vn[1]+ih)*vn[2]+iw)*vn[3])*es;
        if(pad){memcpy(dst,src,(size_t)vn[3]*es);memset(dst+(size_t)vn[3]*es,0,row-(size_t)vn[3]*es);}
        else if(vn[3]==dims[3])memcpy(dst,src,row);
        else for(int32_t d=0;d<dims[3];d++)memcpy(dst+d*es,src,es);
    }
    return xfer(g,t,full.p,dims,t->p,1);
}
static int t_write_region(QdgGraph *g, const T *t, const A *v, const int32_t *origin) {
    CHECK(dsize(v->dt)==t->esize,QDG_ERR_TENSOR);
    return xfer(g,t,v->p,v->n,origin,1);
}
static int read_scalar(QdgGraph *g, const T *t, int64_t *value) {
    A a;TRY(t_read(g,t,&a,0));
    CHECK(count4(a.n)>0 && a.dt!=DT_F32,QDG_ERR_TENSOR);
    *value=aget(&a,0);return 0;
}
static int read_i32(QdgGraph *g, const T *t, int32_t **values, size_t *n) {
    A a;TRY(t_read(g,t,&a,0));
    CHECK(a.dt==DT_I32 || a.dt==DT_Q32,QDG_ERR_TENSOR);
    *values=a.p;*n=count4(a.n);return 0;
}
static int sv_or_scalar(const T *t, int64_t *v) {
    if(t->flags&QDR_SCALAR){*v=t->dtype==DT_F32?0:(int32_t)t->scalar;return t->dtype==DT_F32;}
    return 1;
}

/* ---------------------------------------------------------------- arguments */
static const T *in(QdgGraph *g, const C *c, uint32_t i) {
#ifdef QDG_DSP
    return i<c->nin?dsp_tensor(g,c->in[i]):NULL;
#else
    return i<c->nin?&g->t[c->in[i]]:NULL;
#endif
}
static const T *out(QdgGraph *g, const C *c, uint32_t i) {
#ifdef QDG_DSP
    return i<c->nout?dsp_tensor(g,c->out[i]):NULL;
#else
    return i<c->nout?&g->t[c->out[i]]:NULL;
#endif
}
/* call_args: exactly one call to `func`; masked (pointer) arguments are invalid. */
static int call_args(QdgGraph *g, const C *c, uint32_t func, int64_t *args, uint32_t used) {
    const uint8_t *found=NULL;
    for(uint32_t i=0;i<c->ncalls;i++) {
        const uint8_t *r=c->calls+i*88;
        if(qdr_u32(r)==func){CHECK(!found,QDG_ERR_ARGS);found=r;}
    }
    CHECK(found,QDG_ERR_ARGS);
    uint32_t mask=qdr_u32(found+4);
    for(uint32_t j=0;j<10;j++) {
        uint64_t v=qdr_u64(found+8+8*j);
        if(used&(1u<<j)) CHECK(!(mask&(1u<<j)),QDG_ERR_ARGS);
        args[j]=(int64_t)v;
    }
    return 0;
}
static int shift_ok(int64_t s) {return s>=0 && s<=48;}
/* Arguments used as true integers (not modulo 2**32) must be small. */
static int small(const int64_t *r, uint32_t mask) {
    for(int j=0;j<10;j++)if((mask&(1u<<j)) && (uint64_t)r[j]>=(1ull<<40))return 0;
    return 1;
}

/* ------------------------------------------------------------------ kernels */
static int k_input_slice_pad(QdgGraph *g, const C *c) {
    CHECK(c->nin==7 && c->nout==1 && c->io<2,QDG_ERR_ARGS);
    const T *pb=in(g,c,0),*st=in(g,c,1),*os=in(g,c,2),*fl=in(g,c,3),*is=in(g,c,6),*o=out(g,c,0);
    CHECK((pb->flags&st->flags&os->flags&is->flags&QDR_SHAPE_VALUE),QDG_ERR_ARGS);
    int64_t fill=0;
    if(fl->flags&QDR_SCALAR)CHECK(!sv_or_scalar(fl,&fill),QDG_ERR_ARGS);
    else CHECK(fl->flags&QDR_SHAPE_VALUE && !fl->sv[0] && !fl->sv[1] && !fl->sv[2] && !fl->sv[3],QDG_ERR_ARGS);
    const uint8_t *img=g->inputs[c->io];
    CHECK(o->dtype==DT_U8 && g->input_bytes[c->io]==count4(is->sv),QDG_ERR_ARGS);
#if defined(QDG_DSP) && QDS_BLOCKED_GLUE
    /* Contiguous direct output needs no dense temporary or t_write pass.
     * Keep broadcasting, cropped views and blocked outputs on the old path. */
    if((o->flags&QDR_DIRECT) && !o->crouton && o->pool<QDG_POOLS && g->pool[o->pool] &&
       !memcmp(o->d,o->m,16) && !memcmp(o->d,os->sv,16) &&
       !o->p[0] && !o->p[1] && !o->p[2] && !o->p[3]) {
        size_t bytes=count4(o->d);
        if(o->offset<=g->pool_bytes[o->pool] && bytes<=g->pool_bytes[o->pool]-o->offset &&
           qd_layout_input_slice_pad_direct_u8(g->pool[o->pool]+o->offset,bytes,
               img,g->input_bytes[c->io],is->sv,os->sv,st->sv,pb->sv,(uint8_t)fill)) {
            g->dsp_direct_input_pad_calls++;return 0;
        }
    }
#endif
    A a;TRY(anew(g,&a,os->sv,DT_U8));
    memset(a.p,(uint8_t)fill,count4(a.n));
    int32_t lo[4],s0[4],s1[4];
    for(int i=0;i<4;i++) {
        lo[i]=st->sv[i]-pb->sv[i];
        s0[i]=lo[i]>0?lo[i]:0;
        s1[i]=lo[i]+os->sv[i]<is->sv[i]?lo[i]+os->sv[i]:is->sv[i];
        if(s1[i]<s0[i])s1[i]=s0[i];
    }
    uint8_t *dst=a.p;
    for(int32_t b=s0[0];b<s1[0];b++)for(int32_t h=s0[1];h<s1[1];h++)for(int32_t w=s0[2];w<s1[2];w++) {
        size_t si=(((size_t)b*is->sv[1]+h)*is->sv[2]+w)*is->sv[3];
        size_t di=(((size_t)(b-lo[0])*a.n[1]+(h-lo[1]))*a.n[2]+(w-lo[2]))*a.n[3];
        memcpy(dst+di+(s0[3]-lo[3]),img+si+s0[3],s1[3]-s0[3]);
    }
    return t_write(g,o,&a);
}
static int k_input_slice(QdgGraph *g, const C *c) {
    CHECK(c->nout==1 && (c->io==2 || c->io==3),QDG_ERR_ARGS);
    const T *o=out(g,c,0);
    CHECK(o->dtype==DT_F32 && g->input_bytes[c->io]==count4(o->d)*4,QDG_ERR_ARGS);
    A a={{o->d[0],o->d[1],o->d[2],o->d[3]},DT_F32,0,(void *)g->inputs[c->io]};
    return t_write(g,o,&a);
}
static int k_weights_to_vtcm(QdgGraph *g, const C *c) {
    CHECK(c->nin==1 && c->nout==1,QDG_ERR_ARGS);
    const T *s=in(g,c,0),*d=out(g,c,0);
    CHECK((s->flags&d->flags&QDR_DIRECT) && g->pool[s->pool] && g->pool[d->pool],QDG_ERR_TENSOR);
    size_t ns=count4(s->m)*s->esize,nd=count4(d->m)*d->esize,n=ns<nd?ns:nd;
    CHECK(s->offset+n<=g->pool_bytes[s->pool] && d->offset+n<=g->pool_bytes[d->pool],QDG_ERR_RANGE);
    memmove(g->pool[d->pool]+d->offset,g->pool[s->pool]+s->offset,n);
    return 0;
}
#if defined(QDG_DSP) && QDS_BLOCKED_GLUE
static uint8_t *layout_cached_block(uint32_t encoded,int store,void *ctx) {
    (void)ctx;return dsp_cache_block(encoded,store);
}
#endif
static int k_im2col(QdgGraph *g, const C *c) {
    CHECK(c->nin==3 && c->nout==1,QDG_ERR_ARGS);
    const T *i=in(g,c,0),*k=in(g,c,1),*s=in(g,c,2),*o=out(g,c,0);
    CHECK(k->flags&s->flags&QDR_SHAPE_VALUE,QDG_ERR_ARGS);
    A x;TRY(t_read(g,i,&x,0));
    CHECK(x.n[0]==1,QDG_ERR_SHAPE);
    int32_t kh=k->sv[0],kw=k->sv[1],cin=k->sv[2],sh=s->sv[1],sw=s->sv[2],oh=o->d[1],ow=o->d[2];
    CHECK(sh>0 && sw>0 && cin<=x.n[3] && kh>0 && kw>0,QDG_ERR_ARGS);
    CHECK((kh-1)+sh*(oh-1)<x.n[1] && (kw-1)+sw*(ow-1)<x.n[2],QDG_ERR_SHAPE);
#if defined(QDG_DSP) && QDS_BLOCKED_GLUE
    if(x.dt==DT_U8 && o->dtype==DT_U8 && (o->flags&QDR_BLOCKED) && o->crouton &&
       !o->p[0] && !o->p[1] && !o->p[2] && !o->p[3]) {
        size_t sizes[QDG_POOLS];int sizes_ok=1;
        for(int p=0;p<QDG_POOLS;p++) {
            if(g->pool_bytes[p]>SIZE_MAX)sizes_ok=0;
            sizes[p]=(size_t)g->pool_bytes[p];
        }
        if(sizes_ok) {
            /* Same cache ownership rule as blocked_xfer: its VTCM storage
             * invalidates any resident packed-HMX scratch key. */
            g->resident_key=0;dsp_cache_begin(g->pool);
            int handled=qd_layout_im2col_u8_crouton_direct(x.p,count4(x.n),x.n,o->d,o->m,
                k->sv,s->sv,o->blocks,o->nblocks,o->slice,g->pool,sizes,layout_cached_block,NULL);
            dsp_cache_end();
            if(handled){g->dsp_direct_im2col_calls++;return 0;}
        }
    }
#endif
    int32_t depth=kh*kw*cin,keep=o->d[3]<depth?o->d[3]:depth;
    int32_t n[4]={1,oh,ow,keep};
    A r;TRY(anew(g,&r,n,x.dt));
    size_t es=dsize(x.dt);
    for(int32_t h=0;h<oh;h++)for(int32_t w=0;w<ow;w++) {
        uint8_t *dst=(uint8_t *)r.p+((size_t)h*ow+w)*keep*es;
        int32_t col=0;
        for(int32_t y=0;y<kh && col<keep;y++)for(int32_t xx=0;xx<kw && col<keep;xx++) {
            int32_t n=cin<keep-col?cin:keep-col;
            size_t si=((size_t)(y+sh*h)*x.n[2]+(xx+sw*w))*x.n[3];
            if(es==1){const uint8_t *src=(const uint8_t *)x.p+si;for(int32_t ch=0;ch<n;ch++)dst[col+ch]=src[ch];}
            else memcpy(dst+col*es,(const uint8_t *)x.p+si*es,(size_t)n*es);
            col+=n;
        }
    }
    return t_write(g,o,&r);
}
static const int S2_TAPS[9][2]={{0,2},{0,0},{2,2},{2,0},{0,1},{2,1},{1,2},{1,0},{1,1}};
#if defined(QDG_DSP) && QDS_BLOCKED_GLUE && QDS_BLOCKED_S2D
static uint64_t layout_s2d_clock(void *ctx) {(void)ctx;return now_ns();}
static int blocked_s2d_try(QdgGraph *g,const C *c,const T *i,const T *o) {
    if(o->dtype!=DT_U8 || !(i->flags&o->flags&QDR_BLOCKED))return 0;
    size_t sizes[QDG_POOLS];
    for(int p=0;p<QDG_POOLS;p++) {
        if(g->pool_bytes[p]>SIZE_MAX)return 0;
        sizes[p]=(size_t)g->pool_bytes[p];
    }
    QdLayoutS2dPlan *cached=g->blocked_s2d[g->current];
    if(cached==(void *)1)return 0;
    if(!cached) {
        uint64_t started=now_ns();QdLayoutS2dPlan prepared;QdLayoutS2dStats stats;
        const T *a=&g->t[c->in[0]],*b=&g->t[c->out[0]];
        if(!qd_layout_s2d_prepare_profiled(a->m,a->p,b->d,b->m,b->p,
            a->blocks,a->nblocks,a->slice,b->blocks,b->nblocks,b->slice,g->pool,sizes,
            &prepared,layout_s2d_clock,NULL,&stats))cached=(void *)1;
        else {cached=malloc(sizeof *cached);if(cached)memcpy(cached,&prepared,sizeof *cached);}
        g->blocked_s2d[g->current]=cached;
        g->dsp_s2d_diag[0]+=now_ns()-started;g->dsp_s2d_diag[2]+=stats.proof_ticks;
        g->dsp_s2d_diag[5]++;
    }
    if((uintptr_t)cached<=1)return 0;
    uint8_t *tile=salloc(g,QD_LAYOUT_S2D_TILE_BYTES);
    if(!tile)return 0;
    uint64_t started=now_ns();QdLayoutS2dPlan hot;QdLayoutS2dStats stats;
    /* Copy prepared metadata to coherent L2. The plan stores immutable encoded
     * block IDs, never pointers into the per-command metadata or block cache. */
    memcpy(&hot,cached,sizeof hot);g->resident_key=0;dsp_cache_begin(g->pool);
    int handled=qd_layout_s2d_apply_profiled(&hot,g->pool,sizes,tile,QD_LAYOUT_S2D_TILE_BYTES,
        layout_cached_block,NULL,layout_s2d_clock,NULL,&stats);
    dsp_cache_end();g->dsp_s2d_diag[1]+=now_ns()-started;
    g->dsp_s2d_diag[3]+=stats.gather_ticks;g->dsp_s2d_diag[4]+=stats.scatter_ticks;
    if(handled)g->dsp_direct_space_to_depth_calls++;
    return handled;
}
#endif
static int k_space_to_depth(QdgGraph *g, const C *c) {
    CHECK(c->nin>=1 && c->nout==1,QDG_ERR_ARGS);
    const T *i=in(g,c,0),*o=out(g,c,0);
    /* Python reads the full padded extent; only rows ph..ph+2oh and columns
     * pw-1..pw+2ow-2 of it are used. */
    int32_t ph=i->p[1],pw=i->p[2],oh=o->d[1],oc=o->d[3],ci=i->m[3];
    int32_t ow=(i->m[2]-pw)/2;
    if(o->m[2]<ow)ow=o->m[2];
    CHECK(i->dtype==DT_U8 && i->m[0]>=1 && oh>0 && ow>0 && ph+2*(oh-1)+1<i->m[1] && pw+2*(ow-1)<i->m[2] && pw>=1,QDG_ERR_SHAPE);
#if defined(QDG_DSP) && QDS_BLOCKED_GLUE && QDS_BLOCKED_S2D
    if(blocked_s2d_try(g,c,i,o))return 0;
#endif
    int32_t rd[4]={1,2*oh,2*ow,ci},ro[4]={0,ph,pw-1,0};
    A xf;TRY(t_read_region(g,i,&xf,rd,ro));
    ph=0;pw=1;
    int32_t n[4]={1,oh,ow,oc};
    A r;TRY(anew(g,&r,n,DT_U8));
    memset(r.p,0,count4(n));
    const uint8_t *s=xf.p;uint8_t *d=r.p;
    for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++)for(int32_t cb=0;cb<(ci+31)/32;cb++) {
        int32_t j=cb*4+2*dy+dx;
        if(32*j>=oc)continue;
        int32_t width=32;
        if(32*j+width>oc)width=oc-32*j;
        int32_t src_width=ci-32*cb<32?ci-32*cb:32;
        CHECK(width==src_width,QDG_ERR_SHAPE);
        for(int32_t h=0;h<oh;h++)for(int32_t w=0;w<ow;w++)
            memcpy(d+((size_t)h*ow+w)*oc+32*j,s+((size_t)(ph+2*h+dy)*xf.n[2]+(pw+2*w-dx))*ci+32*cb,width);
    }
    static const int32_t zero[4]={0,0,0,0};
    return t_write_region(g,o,&r,zero);
}

/* Elementwise helpers: binary broadcasting as NumPy for two 4-D arrays. */
static int bshape(const A *a, const A *b, int32_t *n) {
    for(int i=0;i<4;i++) {
        if(a->n[i]==b->n[i] || b->n[i]==1)n[i]=a->n[i];
        else if(a->n[i]==1)n[i]=b->n[i];
        else return QDG_ERR_SHAPE;
    }
    return 0;
}
static size_t bindex(const int32_t *have, int32_t b, int32_t h, int32_t w, int32_t d) {
    return (((size_t)(have[0]==1?0:b)*have[1]+(have[1]==1?0:h))*have[2]+(have[2]==1?0:w))*have[3]+(have[3]==1?0:d);
}
static int64_t f_add_u8(int64_t a,int64_t b,const int64_t *k){return (a*k[0]+b*k[1]+((int64_t)1<<(k[2]-1)))>>k[2];}
static int64_t f_add_u16(int64_t a,int64_t b,const int64_t *k){return (a*k[0]+b*k[1]+((int64_t)1<<k[2]))>>(k[2]+1);}
static int64_t f_addsub_u16(int64_t a,int64_t b,const int64_t *k){return (a*k[0]-b*k[1]+k[3])>>k[2];}
static int64_t f_add1111(int64_t x,int64_t s,const int64_t *k){return (x*k[0]+s*k[1]+k[3])>>k[2];}
static inline int64_t mul_u16(int64_t a, int64_t b, int64_t P1, int64_t zb, int64_t za, int64_t K, int64_t P2) {
    int64_t D=s32((uint64_t)(a*b)+(uint64_t)K-(uint64_t)(b*(zb&0xffff)+a*(za&0xffff)));
    int64_t lo=P1&0xffff,hi=s32((uint64_t)P1)>>16;
    int64_t e=(D*lo)>>16;
    int64_t q=clip64((2*e+2*D*hi)>>16,-2147483648LL,2147483647LL);
    int64_t s=clip64(q+s32((uint64_t)P2),-2147483648LL,2147483647LL);
    return clip64((s+4)>>3,0,65535);
}
static int64_t f_mul_u16(int64_t a,int64_t b,const int64_t *k){return mul_u16(a,b,k[0],k[1],k[2],k[3],k[4]);}
static int64_t f_addsub_flat(int64_t a,int64_t b,const int64_t *k){return (a*k[0]+k[4]*b*k[1]+2*k[3]+((int64_t)1<<k[2]))>>(k[2]+1);}
/* Each instance inlines its arithmetic; the dtype switch is outside the loop. */
#define BINARY_BODY(F) do { \
    int32_t n[4];TRY(bshape(a,b,n)); \
    int32_t *ia,*ib;TRY(to_i32(g,a,&ia));TRY(to_i32(g,b,&ib)); \
    A r;TRY(anew(g,&r,n,odt)); \
    int64_t hi=dtmax(odt); \
    size_t total=count4(n); \
    int same=!memcmp(a->n,n,16) && !memcmp(b->n,n,16),scalar=!memcmp(a->n,n,16) && count4(b->n)==1; \
    if(same||scalar) { \
        if(odt==DT_U16){uint16_t *o=r.p;for(size_t k2=0;k2<total;k2++)o[k2]=(uint16_t)clip64(F(ia[k2],ib[scalar?0:k2],k),0,hi);} \
        else if(odt==DT_U8){uint8_t *o=r.p;for(size_t k2=0;k2<total;k2++)o[k2]=(uint8_t)clip64(F(ia[k2],ib[scalar?0:k2],k),0,hi);} \
        else for(size_t k2=0;k2<total;k2++)from_i64(&r,k2,clip64(F(ia[k2],ib[scalar?0:k2],k),0,hi)); \
    } else { \
        size_t k2=0; \
        for(int32_t i0=0;i0<n[0];i0++)for(int32_t i1=0;i1<n[1];i1++)for(int32_t i2=0;i2<n[2];i2++)for(int32_t i3=0;i3<n[3];i3++,k2++) \
            from_i64(&r,k2,clip64(F(ia[bindex(a->n,i0,i1,i2,i3)],ib[bindex(b->n,i0,i1,i2,i3)],k),0,hi)); \
    } \
    return t_write(g,o,&r); } while(0)
#define DEFINE_BINARY(NAME,F) static int NAME(QdgGraph *g, const A *a, const A *b, const int64_t *k, uint32_t odt, const T *o) { BINARY_BODY(F); }
DEFINE_BINARY(b_add_u8,f_add_u8)
DEFINE_BINARY(b_add_u16,f_add_u16)
DEFINE_BINARY(b_addsub_u16,f_addsub_u16)
DEFINE_BINARY(b_add1111,f_add1111)
DEFINE_BINARY(b_mul_u16,f_mul_u16)
DEFINE_BINARY(b_addsub_flat,f_addsub_flat)
#define binary(g,c,a,b,f,k,odt,o) b_##f(g,a,b,k,odt,o)
#define b_f_add_u8 b_add_u8
#define b_f_add_u16 b_add_u16
#define b_f_addsub_u16 b_addsub_u16
#define b_f_add1111 b_add1111
#define b_f_mul_u16 b_mul_u16
#define b_f_addsub_flat b_addsub_flat

static int read2(QdgGraph *g, const C *c, A *a, A *b) {
    TRY(t_read(g,in(g,c,0),a,0));
    return t_read(g,in(g,c,1),b,0);
}
#if defined(QDG_DSP) && QDS_BLOCKED_GLUE
static void blocked_add_tensor(QdgGraph *g,const T *t,QdBlockedU8 *v) {
    v->pools=g->pool;v->blocks=t->blocks;v->block_count=t->nblocks;v->slice=t->slice;
    memcpy(v->meta,t->m,16);memcpy(v->dims,t->d,16);memcpy(v->origin,t->p,16);
}
static int blocked_add_try(QdgGraph *g,const C *c,const int64_t *k) {
    const T *a=in(g,c,0),*b=in(g,c,1),*o=out(g,c,0);
    if(a->dtype!=DT_U8 || b->dtype!=DT_U8 || o->dtype!=DT_U8 ||
       !(a->flags&b->flags&o->flags&QDR_BLOCKED) || k[0]<0 || k[1]<0 || k[2]>31 ||
       ((uint64_t)k[0]+(uint64_t)k[1])*255+(1ull<<(k[2]-1))>0xffffffffull)return 0;
    int32_t depth=(o->d[3]+o->p[3]+31)/32*32;
    if(depth>o->m[3])depth=o->m[3];
    if(o->crouton && depth-o->p[3]!=a->d[3])return 0;
    QdBlockedAddU8 *cached=g->blocked_add[g->current];
    if(cached==(void *)1)return 0;
    if(!cached) {
        uint64_t started=now_ns();QdBlockedU8 x,y,z;QdBlockedAddU8 prepared;
        blocked_add_tensor(g,&g->t[c->in[0]],&x);blocked_add_tensor(g,&g->t[c->in[1]],&y);
        blocked_add_tensor(g,&g->t[c->out[0]],&z);
        if(!qd_blocked_add_u8_prepare(&x,&y,&z,&prepared))cached=(void *)1;
        else {cached=malloc(sizeof *cached);if(cached)memcpy(cached,&prepared,sizeof *cached);}
        g->blocked_add[g->current]=cached;g->dsp_blocked_add_prepare+=now_ns()-started;
    }
    if((uintptr_t)cached<=1)return 0;
    QdAddU8Params params={(uint32_t)k[0],(uint32_t)k[1],(uint32_t)k[2]};
    uint64_t started=now_ns();QdBlockedAddU8 hot;memcpy(&hot,cached,sizeof hot);
    hot.a.blocks=a->blocks;hot.b.blocks=b->blocks;hot.out.blocks=o->blocks;
    int applied=qd_blocked_add_u8_apply(&hot,&params);g->dsp_blocked_add_apply+=now_ns()-started;
    if(applied)g->dsp_blocked_add_calls++;
    return applied;
}
#endif
static int k_add_u8(QdgGraph *g, const C *c) {
    int64_t r[10],k[4];
    TRY(call_args(g,c,QDR_CALL_ELEMENTWISE_ADD_CROUTON_U8,r,0x1f0));
    CHECK(r[7]==0 && small(r,0x70),QDG_ERR_UNSUPPORTED);
    k[0]=r[4];k[1]=r[5];k[2]=r[6]+8;
    CHECK(shift_ok(k[2]) && k[2]>=1,QDG_ERR_ARGS);
#if defined(QDG_DSP) && QDS_BLOCKED_GLUE
    if(blocked_add_try(g,c,k))return 0;
#endif
    A a,b;TRY(read2(g,c,&a,&b));
    /* Native fast path for the common equal-shape u8 case. */
    if(!memcmp(a.n,b.n,16) && a.dt==DT_U8 && b.dt==DT_U8 && k[0]>=0 && k[1]>=0 && k[2]<=31 &&
       ((uint64_t)k[0]+(uint64_t)k[1])*255+(1ull<<(k[2]-1))<=0xffffffffull) {
        A o;TRY(anew(g,&o,a.n,DT_U8));
        uint32_t ma=(uint32_t)k[0],mb=(uint32_t)k[1],rd=1u<<(k[2]-1),s=(uint32_t)k[2];
        const uint8_t *pa=a.p,*pb=b.p;uint8_t *po=o.p;
#ifdef QDG_DSP
        (void)rd;dsp_add_u8(po,pa,pb,count4(a.n),ma,mb,s);
#else
        for(size_t i=0,n=count4(a.n);i<n;i++){uint32_t v=(pa[i]*ma+pb[i]*mb+rd)>>s;po[i]=(uint8_t)(v>255?255:v);}
#endif
        return t_write(g,out(g,c,0),&o);
    }
    return binary(g,c,&a,&b,f_add_u8,k,DT_U8,out(g,c,0));
}
static int k_add_u16(QdgGraph *g, const C *c) {
    int64_t r[10],k[3];
    TRY(call_args(g,c,QDR_CALL_ELEMENTWISE_ADD_CROUTON_U16,r,0x1f0));
    CHECK(r[7]==0 && small(r,0x70) && shift_ok(r[6]),QDG_ERR_ARGS);
    k[0]=r[4];k[1]=r[5];k[2]=r[6];
    A a,b;TRY(read2(g,c,&a,&b));
#ifdef QDG_DSP
    if(a.dt==DT_U16 && b.dt==DT_U16 && !memcmp(a.n,b.n,16) && k[0]>=0 && k[1]>=0 && k[2]<31 &&
       ((uint64_t)k[0]+(uint64_t)k[1])*65535+(1ull<<k[2])<=UINT32_MAX) {
        A v;TRY(anew(g,&v,a.n,DT_U16));
        dsp_add_u16(v.p,a.p,b.p,count4(a.n),(uint32_t)k[0],(uint32_t)k[1],(uint32_t)k[2]+1);
        return t_write(g,out(g,c,0),&v);
    }
#endif
    return binary(g,c,&a,&b,f_add_u16,k,DT_U16,out(g,c,0));
}
static int k_addsub_u16(QdgGraph *g, const C *c) {
    int64_t r[10],k[4];
    TRY(call_args(g,c,QDR_CALL_ELEMENTWISE_ADDSUB_CROUTON_U16,r,0x70));
    const T *o=out(g,c,0);
    CHECK(small(r,0x70) && shift_ok(r[6]+1),QDG_ERR_ARGS);
    k[0]=r[4];k[1]=r[5];k[2]=r[6]+1;k[3]=((int64_t)o->zp<<k[2])+((int64_t)1<<r[6])+1;
    A a,b;TRY(read2(g,c,&a,&b));
    return binary(g,c,&a,&b,f_addsub_u16,k,DT_U16,o);
}
static int k_add1111_u16(QdgGraph *g, const C *c) {
    int64_t r[10],k[4],s;
    TRY(call_args(g,c,QDR_CALL_ELEMENTWISE_ADD1111_CROUTON_U16,r,0xf0));
    CHECK(small(r,0x70) && shift_ok(r[6]+1),QDG_ERR_ARGS);
    k[0]=r[4];k[1]=r[5];k[2]=r[6]+1;k[3]=2*s32((uint64_t)r[7])+((int64_t)1<<r[6]);
    A x;TRY(t_read(g,in(g,c,0),&x,0));
    TRY(read_scalar(g,in(g,c,1),&s));
    A b={{1,1,1,1},DT_I32,1,&s};
    int32_t sv=(int32_t)s;b.p=&sv;
    return binary(g,c,&x,&b,f_add1111,k,DT_U16,out(g,c,0));
}
static int k_mul_u16(QdgGraph *g, const C *c) {
    int64_t r[10],k[5];
    TRY(call_args(g,c,QDR_CALL_ELEMENTWISE_MUL_CROUTON_U16,r,0x1f0));
    k[0]=r[4];k[1]=r[5];k[2]=r[6];k[3]=r[7];k[4]=r[8];
    A a,b;TRY(read2(g,c,&a,&b));
    return binary(g,c,&a,&b,f_mul_u16,k,DT_U16,out(g,c,0));
}
static int k_mul1111_u16(QdgGraph *g, const C *c) {
    int64_t r[10],k[5],s;
    TRY(call_args(g,c,QDR_CALL_ELEMENTWISE_MUL1111_CROUTON_U16,r,0x1f0));
    k[0]=r[4];k[1]=r[5];k[2]=r[6];k[3]=r[7];k[4]=r[8];
    A x;TRY(t_read(g,in(g,c,0),&x,0));
    TRY(read_scalar(g,in(g,c,1),&s));
    CHECK(s>=INT32_MIN && s<=INT32_MAX,QDG_ERR_ARGS);
    int32_t sv=(int32_t)s;
    A b={{1,1,1,1},DT_I32,1,&sv};
    return binary(g,c,&x,&b,f_mul_u16,k,DT_U16,out(g,c,0));
}
static int k_mul_inline(QdgGraph *g, const C *c) {
    CHECK(c->nin==3,QDG_ERR_ARGS);
    int32_t *p;size_t np;int64_t s,k[5];
    TRY(read_i32(g,in(g,c,2),&p,&np));
    CHECK(np>=6,QDG_ERR_ARGS);
    uint64_t z=(uint32_t)p[3];
    k[0]=p[2];k[1]=(int64_t)(z&65535);k[2]=(int64_t)(z>>16);k[3]=p[4];k[4]=p[5];
    A x;TRY(t_read(g,in(g,c,0),&x,0));
    TRY(read_scalar(g,in(g,c,1),&s));
    CHECK(s>=INT32_MIN && s<=INT32_MAX,QDG_ERR_ARGS);
    int32_t sv=(int32_t)s;
    A b={{1,1,1,1},DT_I32,1,&sv};
    return binary(g,c,&x,&b,f_mul_u16,k,DT_U16,out(g,c,0));
}
static int k_addsub_flat(QdgGraph *g, const C *c) {
    CHECK(c->ncalls>=1,QDG_ERR_ARGS);
    uint32_t fn=qdr_u32(c->calls);
    CHECK(fn==QDR_CALL_ELEMENTWISE_SUB_FLAT_U16 || fn==QDR_CALL_ELEMENTWISE_ADD_FLAT_U16,QDG_ERR_ARGS);
    int64_t r[10],k[5];
    uint32_t mask=qdr_u32(c->calls+4);
    for(int j=0;j<10;j++)r[j]=(int64_t)qdr_u64(c->calls+8+8*j);
    CHECK(!(mask&0xf0) && small(r,0x70) && shift_ok(r[6]+1),QDG_ERR_ARGS);
    k[0]=r[4];k[1]=r[5];k[2]=r[6];k[3]=s32((uint64_t)r[7]);k[4]=fn==QDR_CALL_ELEMENTWISE_SUB_FLAT_U16?-1:1;
    A a,b;TRY(read2(g,c,&a,&b));
    return binary(g,c,&a,&b,f_addsub_flat,k,DT_U16,out(g,c,0));
}
static int k_addsub_inline(QdgGraph *g, const C *c) {
    CHECK(c->nin==3,QDG_ERR_ARGS);
    int32_t *p;size_t np;int64_t k[4];
    TRY(read_i32(g,in(g,c,2),&p,&np));
    CHECK(np>=5,QDG_ERR_ARGS);
    const T *o=out(g,c,0);
    int64_t sh=(int64_t)p[4]>>16;
    CHECK(shift_ok(sh+1),QDG_ERR_ARGS);
    k[0]=(int64_t)p[1]>>16;k[1]=p[2];k[2]=sh+1;k[3]=((int64_t)o->zp<<(sh+1))+((int64_t)1<<sh)+1;
    A a,b;TRY(read2(g,c,&a,&b));
    return binary(g,c,&a,&b,f_addsub_u16,k,DT_U16,o);
}
static int k_requant_u8(QdgGraph *g, const C *c) {
    int64_t r[10];
    TRY(call_args(g,c,QDR_CALL_REQUANTIZE_CROUTON_U8,r,0x38));
    CHECK(r[4]==0 && r[5]==0 && small(r,0x8),QDG_ERR_UNSUPPORTED);
    int64_t m=r[3]&0xffff,s=15-(r[3]>>24);
    CHECK(s>=1 && s<=31,QDG_ERR_ARGS);
    A x;TRY(t_read(g,in(g,c,0),&x,0));
    CHECK(x.dt==DT_U8,QDG_ERR_TENSOR);
    A o;TRY(anew(g,&o,x.n,DT_U8));
    uint32_t mm=(uint32_t)m,rd=1u<<(s-1),ss=(uint32_t)s;
    const uint8_t *px=x.p;uint8_t *po=o.p;
#ifdef QDG_DSP
    (void)rd;dsp_add_u8(po,px,px,count4(x.n),mm,0,ss);
#else
    for(size_t i=0,n=count4(x.n);i<n;i++){uint32_t v=(px[i]*mm+rd)>>ss;po[i]=(uint8_t)(v>255?255:v);}
#endif
    return t_write(g,out(g,c,0),&o);
}
static int k_requant_u16(QdgGraph *g, const C *c) {
    int64_t r[10];
    TRY(call_args(g,c,QDR_CALL_REQUANTIZE_CROUTON_U16,r,0x18));
    CHECK(small(r,0x18),QDG_ERR_ARGS);
    int64_t m=r[3]&0xffff,s=15-(r[3]>>24),zin=r[4]&0xffff,zout=r[4]>>16;
    CHECK(s>=1 && s<=48,QDG_ERR_ARGS);
    A x;TRY(t_read(g,in(g,c,0),&x,0));
#ifdef QDG_DSP
    if(x.dt==DT_U16 && dsp_affine_fits(m,zin,zout,s)) {
        A o;TRY(anew(g,&o,x.n,DT_U16));
        dsp_affine_u16(o.p,x.p,count4(x.n),(int32_t)m,(int32_t)zin,(int32_t)zout,(uint32_t)s,0);
        return t_write(g,out(g,c,0),&o);
    }
#endif
    int32_t *v;TRY(to_i32(g,&x,&v));
    A o;TRY(anew(g,&o,x.n,DT_U16));
    uint16_t *po=o.p;
    for(size_t i=0,n=count4(x.n);i<n;i++)po[i]=(uint16_t)clip64((((v[i]-zin)*m+((int64_t)1<<(s-1)))>>s)+zout,0,65535);
    return t_write(g,out(g,c,0),&o);
}
#if defined(QDG_DSP) && QDS_BLOCKED_GLUE
static void blocked_lc_tensor(QdgGraph *g,const T *t,QdBlockedU16 *v) {
    v->pools=g->pool;v->blocks=t->blocks;v->block_count=t->nblocks;v->slice=t->slice;
    memcpy(v->meta,t->m,16);memcpy(v->dims,t->d,16);memcpy(v->origin,t->p,16);
}
static int blocked_lc_try(QdgGraph *g,const C *c,const T *i,const T *o,const int32_t *p,size_t np) {
    if(i->dtype!=DT_U16 || o->dtype!=DT_U16 || np<12 ||
       !(i->flags&QDR_BLOCKED) || !(o->flags&QDR_BLOCKED))return 0;
    /* t_write extends crouton depth and explicitly zero pads it. This first
     * direct path must have no such extra output lanes to preserve parity. */
    int32_t depth=(o->d[3]+o->p[3]+31)/32*32;
    if(depth>o->m[3])depth=o->m[3];
    if(o->crouton && depth-o->p[3]!=i->d[3])return 0;
    QdBlockedU16Linearclip *cached=g->blocked_lc[g->current];
    if(cached==(void *)1)return 0;
    if(!cached) {
        uint64_t started=now_ns();
        QdBlockedU16 a,b;
        /* Persist original immutable tables; current command's L2 copies
         * are reused by the very next command and cannot be cached here. */
        blocked_lc_tensor(g,&g->t[c->in[0]],&a);blocked_lc_tensor(g,&g->t[c->out[0]],&b);
        QdLinearclipParams identity={2,0,0,1,0};
        QdBlockedU16Linearclip prepared;
        if(!qd_blocked_linearclip_u16_prepare(&a,&b,&identity,&prepared)) {
            g->blocked_lc[g->current]=(void *)1;
            g->dsp_blocked_lc_prepare+=now_ns()-started;return 0;
        }
        cached=malloc(sizeof *cached);
        if(!cached)return 0;
        memcpy(cached,&prepared,sizeof *cached);g->blocked_lc[g->current]=cached;
        g->dsp_blocked_lc_prepare+=now_ns()-started;
    }
    int64_t P=(uint32_t)p[10],zp=(uint32_t)p[11],shift=15-(P>>24);
    if(shift<1 || shift>31)return 0;
    QdLinearclipParams params={(int32_t)(P&65535),(int32_t)(zp&65535),(int32_t)(zp>>16),(uint32_t)shift,0};
    if(p[6]<0)params.multiplier=-params.multiplier;
    uint64_t started=now_ns();
    QdBlockedU16Linearclip hot;memcpy(&hot,cached,sizeof hot);
    /* Only the immutable geometry proof is cached. Use the current command's
     * TCM metadata copies and re-prove the current runtime coefficients. */
    hot.in.blocks=i->blocks;hot.out.blocks=o->blocks;
    int applied=qd_blocked_linearclip_u16_apply_params(&hot,&params);
    g->dsp_blocked_lc_apply+=now_ns()-started;
    if(applied)g->dsp_blocked_lc_calls++;
    return applied;
}
static int blocked_lc_u8_try(QdgGraph *g,const C *c,const T *i,const T *o,const int32_t *p,size_t np) {
    if(i->dtype!=DT_U8 || o->dtype!=DT_U16 || np<2 ||
       !(i->flags&QDR_BLOCKED) || !(o->flags&QDR_BLOCKED))return 0;
    int32_t depth=(o->d[3]+o->p[3]+31)/32*32;
    if(depth>o->m[3])depth=o->m[3];
    if(o->crouton && depth-o->p[3]!=i->d[3])return 0;
    QdBlockedLinearclipU8U16 *cached=g->blocked_lc_u8[g->current];
    if(cached==(void *)1)return 0;
    if(!cached) {
        uint64_t started=now_ns();
        const T *it=&g->t[c->in[0]],*ot=&g->t[c->out[0]];
        QdBlockedU8 a={.pools=g->pool,.blocks=it->blocks,.block_count=it->nblocks,.slice=it->slice};
        QdBlockedU16Out b={.pools=g->pool,.blocks=ot->blocks,.block_count=ot->nblocks,.slice=ot->slice};
        memcpy(a.meta,it->m,16);memcpy(a.dims,it->d,16);memcpy(a.origin,it->p,16);
        memcpy(b.meta,ot->m,16);memcpy(b.dims,ot->d,16);memcpy(b.origin,ot->p,16);
        QdBlockedLinearclipU8U16 prepared;
        if(!qd_blocked_linearclip_u8_u16_prepare(&a,&b,&prepared)) {
            g->blocked_lc_u8[g->current]=(void *)1;
            g->dsp_blocked_lc_prepare+=now_ns()-started;return 0;
        }
        cached=malloc(sizeof *cached);if(!cached)return 0;
        memcpy(cached,&prepared,sizeof *cached);g->blocked_lc_u8[g->current]=cached;
        g->dsp_blocked_lc_prepare+=now_ns()-started;
    }
    int64_t shift=7-((int64_t)p[1]>>16);
    if(shift<1 || shift>31)return 0;
    QdLinearclipU8U16Params params={(uint32_t)p[1]&65535,(uint32_t)shift};
    uint64_t started=now_ns();
    QdBlockedLinearclipU8U16 hot;memcpy(&hot,cached,sizeof hot);
    hot.in.blocks=i->blocks;hot.out.blocks=o->blocks;
    int applied=qd_blocked_linearclip_u8_u16_apply(&hot,&params);
    g->dsp_blocked_lc_apply+=now_ns()-started;
    if(applied){g->dsp_blocked_lc_calls++;g->dsp_blocked_lc_u8_calls++;}
    return applied;
}
#endif
static int k_linearclip(QdgGraph *g, const C *c) {
    CHECK(c->nin==2,QDG_ERR_ARGS);
    const T *i=in(g,c,0),*o=out(g,c,0);
    int32_t *p;size_t np;
    TRY(read_i32(g,in(g,c,1),&p,&np));
#if defined(QDG_DSP) && QDS_BLOCKED_GLUE
    if(blocked_lc_u8_try(g,c,i,o,p,np) || blocked_lc_try(g,c,i,o,p,np))return 0;
#endif
    A x;TRY(t_read(g,i,&x,0));
#ifdef QDG_DSP
    if(i->dtype==DT_U16 && (o->dtype==DT_U16 || o->dtype==DT_U8) && np>=12) {
        int64_t P=(uint32_t)p[10],zp=(uint32_t)p[11],m=P&65535,sh=15-(P>>24),zin=zp&65535,zout=zp>>16;
        if(p[6]<0)m=-m;
        if(dsp_affine_fits(m,zin,zout,sh)) {
            A r;TRY(anew(g,&r,x.n,o->dtype));
            dsp_affine_u16(r.p,x.p,count4(x.n),(int32_t)m,(int32_t)zin,(int32_t)zout,(uint32_t)sh,o->dtype==DT_U8);
            return t_write(g,o,&r);
        }
    }
#endif
    int32_t *v;TRY(to_i32(g,&x,&v));
    A r;TRY(anew(g,&r,x.n,o->dtype));
    int64_t hi=dtmax(o->dtype);
    size_t n=count4(x.n);
    if(i->dtype==DT_U8 && o->dtype==DT_U16) {
        CHECK(np>=2,QDG_ERR_ARGS);
        int64_t m=p[1]&0xffff,sh=7-((int64_t)p[1]>>16);
        CHECK(sh>=1 && sh<=48,QDG_ERR_ARGS);
        for(size_t j=0;j<n;j++)from_i64(&r,j,clip64((v[j]*m+((int64_t)1<<(sh-1)))>>sh,0,hi));
    } else {
        CHECK(np>=12,QDG_ERR_ARGS);
        int64_t P=(uint32_t)p[10],zp=(uint32_t)p[11],m=P&0xffff,sh=15-(P>>24);
        if(p[6]<0)m=-m;
        if(i->dtype==DT_U8) {
            CHECK(sh==14 && zp==0 && m>=0,QDG_ERR_UNSUPPORTED);
            for(size_t j=0;j<n;j++)from_i64(&r,j,clip64((((v[j]*m+128)>>8)+32)>>6,0,hi));
        } else {
            CHECK(sh>=1 && sh<=48,QDG_ERR_ARGS);
            int64_t zin=zp&0xffff,zout=zp>>16;
            for(size_t j=0;j<n;j++)from_i64(&r,j,clip64((((v[j]-zin)*m+((int64_t)1<<(sh-1)))>>sh)+zout,0,hi));
        }
    }
    return t_write(g,o,&r);
}
static int k_copy_values(QdgGraph *g, const C *c) {
    CHECK(c->nin>=1 && c->nout==1,QDG_ERR_ARGS);
    const T *o=out(g,c,0);
    A x;TRY(t_read(g,in(g,c,0),&x,0));
    CHECK(count4(x.n)==count4(o->d),QDG_ERR_SHAPE);
    memcpy(x.n,o->d,16);
    if(x.dt!=o->dtype) {
        /* Same-width integer reinterpretation is a value conversion in NumPy;
         * only identical dtypes are supported. */
        CHECK(0,QDG_ERR_TENSOR);
    }
    return t_write(g,o,&x);
}
static int k_copy_activation(QdgGraph *g, const C *c) {
    static const int32_t zero[4]={0,0,0,0};
    const T *i=in(g,c,0),*o=out(g,c,0);
    CHECK(i->dtype==o->dtype,QDG_ERR_TENSOR);
    A x;TRY(t_read(g,i,&x,1));
    return t_write_region(g,o,&x,zero);
}
static int k_transpose(QdgGraph *g, const C *c) {
    CHECK(c->nin>=2,QDG_ERR_ARGS);
    int32_t *perm;size_t np;
    TRY(read_i32(g,in(g,c,c->nin-1),&perm,&np));
    CHECK(np==4,QDG_ERR_ARGS);
    int used=0;
    for(int i=0;i<4;i++){CHECK(perm[i]>=0 && perm[i]<4 && !(used&(1<<perm[i])),QDG_ERR_ARGS);used|=1<<perm[i];}
    A x;TRY(t_read(g,in(g,c,0),&x,0));
    int32_t n[4];for(int i=0;i<4;i++)n[i]=x.n[perm[i]];
    A r;TRY(anew(g,&r,n,x.dt));
    size_t es=dsize(x.dt),st[4]={(size_t)x.n[1]*x.n[2]*x.n[3],(size_t)x.n[2]*x.n[3],(size_t)x.n[3],1},k=0;
    size_t ps[4]={st[perm[0]],st[perm[1]],st[perm[2]],st[perm[3]]};
    for(int32_t a=0;a<n[0];a++)for(int32_t b=0;b<n[1];b++)for(int32_t d2=0;d2<n[2];d2++) {
        size_t base=a*ps[0]+b*ps[1]+d2*ps[2];
        if(es==1){const uint8_t *s=x.p;uint8_t *d=(uint8_t *)r.p+k;for(int32_t e=0;e<n[3];e++)d[e]=s[base+e*ps[3]];}
        else if(es==2){const uint16_t *s=x.p;uint16_t *d=(uint16_t *)r.p+k;for(int32_t e=0;e<n[3];e++)d[e]=s[base+e*ps[3]];}
        else {const uint32_t *s=x.p;uint32_t *d=(uint32_t *)r.p+k;for(int32_t e=0;e<n[3];e++)d[e]=s[base+e*ps[3]];}
        k+=n[3];
    }
    return t_write(g,out(g,c,0),&r);
}
static int k_concat(QdgGraph *g, const C *c) {
    CHECK(c->nin>=2,QDG_ERR_ARGS);
    int64_t axis;
    CHECK(!sv_or_scalar(in(g,c,0),&axis),QDG_ERR_ARGS);
    if(axis<0)axis+=4;
    CHECK(axis>=0 && axis<4,QDG_ERR_ARGS);
    int32_t n[4];
    A *parts=salloc(g,sizeof(A)*(c->nin-1));
    CHECK(parts,QDG_ERR_MEMORY);
    for(uint32_t i=1;i<c->nin;i++) {
        TRY(t_read(g,in(g,c,i),&parts[i-1],0));
        if(i==1){memcpy(n,parts[0].n,16);continue;}
        for(int d=0;d<4;d++)CHECK(d==axis || parts[i-1].n[d]==n[d],QDG_ERR_SHAPE);
        CHECK(parts[i-1].dt==parts[0].dt,QDG_ERR_TENSOR);
        n[axis]+=parts[i-1].n[axis];
    }
    A r;TRY(anew(g,&r,n,parts[0].dt));
    size_t es=dsize(r.dt),outer=1,inner=es;
    for(int d=0;d<axis;d++)outer*=n[d];
    for(int d=axis+1;d<4;d++)inner*=n[d];
    size_t at=0;
    for(uint32_t i=0;i<c->nin-1;i++) {
        size_t chunk=inner*parts[i].n[axis];
        for(size_t o=0;o<outer;o++)memcpy((uint8_t *)r.p+o*inner*n[axis]+at,(uint8_t *)parts[i].p+o*chunk,chunk);
        at+=chunk;
    }
    return t_write(g,out(g,c,0),&r);
}
static int k_gather(QdgGraph *g, const C *c) {
    CHECK(c->nin==4,QDG_ERR_ARGS);
    int64_t axis;TRY(read_scalar(g,in(g,c,1),&axis));
    int32_t *idx;size_t ni;TRY(read_i32(g,in(g,c,2),&idx,&ni));
    A x;TRY(t_read(g,in(g,c,0),&x,0));
    if(axis<0)axis+=4;
    CHECK(axis>=0 && axis<4,QDG_ERR_ARGS);
    /* np.take on a 4-D array with a 1-D index keeps the rank. */
    int32_t n[4];memcpy(n,x.n,16);n[axis]=(int32_t)ni;
    A r;TRY(anew(g,&r,n,x.dt));
    size_t es=dsize(x.dt),outer=1,inner=es;
    for(int d=0;d<axis;d++)outer*=n[d];
    for(int d=axis+1;d<4;d++)inner*=n[d];
    for(size_t o=0;o<outer;o++)for(size_t j=0;j<ni;j++) {
        int64_t k=idx[j];
        if(k<0)k+=x.n[axis];
        CHECK(k>=0 && k<x.n[axis],QDG_ERR_RANGE);
        memcpy((uint8_t *)r.p+(o*ni+j)*inner,(uint8_t *)x.p+(o*x.n[axis]+k)*inner,inner);
    }
    return t_write(g,out(g,c,0),&r);
}
static int k_slice(QdgGraph *g, const C *c) {
    CHECK(c->nin==3,QDG_ERR_ARGS);
    const T *s=in(g,c,1),*n=in(g,c,2);
    CHECK(s->flags&n->flags&QDR_SHAPE_VALUE,QDG_ERR_ARGS);
    const T *i=in(g,c,0);
    int32_t origin[4];
    for(int k=0;k<4;k++){CHECK(s->sv[k]>=0 && n->sv[k]>=0 && s->sv[k]+n->sv[k]<=i->d[k],QDG_ERR_RANGE);origin[k]=i->p[k]+s->sv[k];}
    A r;TRY(t_read_region(g,i,&r,n->sv,origin));
    return t_write(g,out(g,c,0),&r);
}
static int k_slice_pad(QdgGraph *g, const C *c) {
    CHECK(c->nin==5,QDG_ERR_ARGS);
    const T *pad=in(g,c,1),*st=in(g,c,2),*sz=in(g,c,3);
    CHECK(pad->flags&st->flags&sz->flags&QDR_SHAPE_VALUE,QDG_ERR_ARGS);
    int64_t fill;TRY(read_scalar(g,in(g,c,4),&fill));
    A x;TRY(t_read(g,in(g,c,0),&x,0));
    CHECK(fill>=0 && fill<=dtmax(x.dt) && x.dt!=DT_F32,QDG_ERR_ARGS); /* np.full rejects overflow */
    A r;TRY(anew(g,&r,sz->sv,x.dt));
    size_t es=dsize(x.dt),n=count4(r.n);
    if(es==1)memset(r.p,(uint8_t)fill,n);
    else for(size_t i=0;i<n;i++)from_i64(&r,i,fill);
    int32_t lo[4],s0[4],s1[4];
    for(int i=0;i<4;i++) {
        lo[i]=st->sv[i]-pad->sv[i];
        s0[i]=lo[i]>0?lo[i]:0;
        s1[i]=lo[i]+r.n[i]<x.n[i]?lo[i]+r.n[i]:x.n[i];
        if(s1[i]<s0[i])s1[i]=s0[i];
    }
    for(int32_t b=s0[0];b<s1[0];b++)for(int32_t h=s0[1];h<s1[1];h++)for(int32_t w=s0[2];w<s1[2];w++) {
        size_t si=(((size_t)b*x.n[1]+h)*x.n[2]+w)*x.n[3]+s0[3];
        size_t di=(((size_t)(b-lo[0])*r.n[1]+(h-lo[1]))*r.n[2]+(w-lo[2]))*r.n[3]+(s0[3]-lo[3]);
        memcpy((uint8_t *)r.p+di*es,(uint8_t *)x.p+si*es,(s1[3]-s0[3])*es);
    }
    return t_write(g,out(g,c,0),&r);
}
static int k_decimate(QdgGraph *g, const C *c) {
    const T *s=in(g,c,1);
    CHECK(c->nin==2 && (s->flags&QDR_SHAPE_VALUE),QDG_ERR_ARGS);
    A x;TRY(t_read(g,in(g,c,0),&x,0));
    int32_t n[4];
    for(int i=0;i<4;i++){CHECK(s->sv[i]>0,QDG_ERR_ARGS);n[i]=(x.n[i]+s->sv[i]-1)/s->sv[i];}
    A r;TRY(anew(g,&r,n,x.dt));
    size_t es=dsize(x.dt),k=0;
    for(int32_t b=0;b<n[0];b++)for(int32_t h=0;h<n[1];h++)for(int32_t w=0;w<n[2];w++)for(int32_t d=0;d<n[3];d++,k++) {
        size_t si=((((size_t)b*s->sv[0])*x.n[1]+(size_t)h*s->sv[1])*x.n[2]+(size_t)w*s->sv[2])*x.n[3]+(size_t)d*s->sv[3];
        if(es==1)((uint8_t *)r.p)[k]=((const uint8_t *)x.p)[si];
        else if(es==2)((uint16_t *)r.p)[k]=((const uint16_t *)x.p)[si];
        else memcpy((uint8_t *)r.p+k*4,(uint8_t *)x.p+si*4,4);
    }
    return t_write(g,out(g,c,0),&r);
}
static int k_resize(QdgGraph *g, const C *c) {
    CHECK(c->nin==3,QDG_ERR_ARGS);
    int32_t *hp,*wp;size_t nh,nw;
    TRY(read_i32(g,in(g,c,1),&hp,&nh));
    TRY(read_i32(g,in(g,c,2),&wp,&nw));
    CHECK(nh>=6 && nw>=6,QDG_ERR_ARGS);
    A x;TRY(t_read(g,in(g,c,0),&x,0));
    const T *o=out(g,c,0);
    int32_t n[4]={x.n[0],o->d[1],o->d[2],x.n[3]};
    int32_t *hi=salloc(g,4*(size_t)n[1]+4),*wi=salloc(g,4*(size_t)n[2]+4);
    CHECK(hi && wi,QDG_ERR_MEMORY);
    for(int32_t i=0;i<n[1];i++) {
        int64_t v=(int64_t)((((uint64_t)i*(uint32_t)hp[2])+(uint32_t)hp[4])>>32)+hp[5];
        if(v<0)v+=x.n[1];
        CHECK(v>=0 && v<x.n[1],QDG_ERR_RANGE);hi[i]=(int32_t)v;
    }
    for(int32_t i=0;i<n[2];i++) {
        int64_t v=(int64_t)((((uint64_t)i*(uint32_t)wp[2])+(uint32_t)wp[4])>>32)+wp[5];
        if(v<0)v+=x.n[2];
        CHECK(v>=0 && v<x.n[2],QDG_ERR_RANGE);wi[i]=(int32_t)v;
    }
    A r;TRY(anew(g,&r,n,x.dt));
    size_t es=dsize(x.dt),row=(size_t)n[3]*es;
    for(int32_t b=0;b<n[0];b++)for(int32_t h=0;h<n[1];h++)for(int32_t w=0;w<n[2];w++)
        memcpy((uint8_t *)r.p+(((size_t)b*n[1]+h)*n[2]+w)*row,(uint8_t *)x.p+(((size_t)b*x.n[1]+hi[h])*x.n[2]+wi[w])*row,row);
    return t_write(g,o,&r);
}
static int k_depth_to_space(QdgGraph *g, const C *c) {
    CHECK(c->nin==2,QDG_ERR_ARGS);
    int32_t *f;size_t nf;
    TRY(read_i32(g,in(g,c,1),&f,&nf));
    CHECK(nf==2 && f[0]>0 && f[1]>0,QDG_ERR_ARGS);
    A x;TRY(t_read(g,in(g,c,0),&x,0));
    int32_t rh=f[0],rw=f[1];
    CHECK(x.n[3]%(rh*rw)==0,QDG_ERR_SHAPE);
    int32_t cc=x.n[3]/(rh*rw),n[4]={x.n[0],x.n[1]*rh,x.n[2]*rw,cc};
    A r;TRY(anew(g,&r,n,x.dt));
    size_t es=dsize(x.dt);
    for(int32_t b=0;b<x.n[0];b++)for(int32_t h=0;h<x.n[1];h++)for(int32_t i=0;i<rh;i++)
        for(int32_t w=0;w<x.n[2];w++)for(int32_t j=0;j<rw;j++) {
            size_t si=((((size_t)b*x.n[1]+h)*x.n[2]+w)*x.n[3]+(size_t)(i*rw+j)*cc);
            size_t di=((((size_t)b*n[1]+h*rh+i)*n[2]+w*rw+j)*cc);
            memcpy((uint8_t *)r.p+di*es,(uint8_t *)x.p+si*es,cc*es);
        }
    return t_write(g,out(g,c,0),&r);
}
static int k_spill_fill(QdgGraph *g, const C *c, int spill) {
    CHECK(c->nfv>=4,QDG_ERR_ARGS);
    uint64_t offset=c->fv[3];
    for(uint32_t i=4;i+1<c->nfv;i+=2) {
        uint32_t p=c->fv[i],n=c->fv[i+1],pool=p&0x7ff,addr=p&~0x7ffu;
        CHECK(pool<QDG_POOLS && pool!=QDG_POOL_SPILL && g->pool[pool],QDG_ERR_RANGE);
        CHECK((uint64_t)addr+n<=g->pool_bytes[pool] && offset+n<=g->pool_bytes[QDG_POOL_SPILL],QDG_ERR_RANGE);
        uint8_t *dst=spill?g->pool[QDG_POOL_SPILL]+offset:g->pool[pool]+addr;
        const uint8_t *src=spill?g->pool[pool]+addr:g->pool[QDG_POOL_SPILL]+offset;
#if defined(QDG_DSP) && QDS_FAST_SPILL
        /* Resident ABI puts mutable pools in private DDR +16..32 MiB,
         * outside the scalar-cached immutable/heap windows. For disjoint,
         * aligned spans use our HVX memcpy instead of libc's scalar memmove.
         * Other bindings and overlapping moves retain the generic semantics. */
        uint64_t d=(uintptr_t)dst,s=(uintptr_t)src;
        if(d>=0xa1000000ull && s>=0xa1000000ull &&
           d+n<=0xa2000000ull && s+n<=0xa2000000ull &&
           !((d|s|n)&127) && (d+n<=s || s+n<=d))memcpy(dst,src,n);
        else
#endif
        memmove(dst,src,n);
        offset+=n;
    }
    return 0;
}
static int k_output_slice(QdgGraph *g, const C *c) {
    CHECK(c->nin==4 && c->io<2,QDG_ERR_ARGS);
    const T *sh=in(g,c,1),*st=in(g,c,2);
    CHECK(sh->flags&st->flags&QDR_SHAPE_VALUE,QDG_ERR_ARGS);
    A x;TRY(t_read(g,in(g,c,0),&x,0));
    size_t es=dsize(x.dt);
    CHECK(g->output_bytes[c->io]==count4(sh->sv)*es,QDG_ERR_ARGS);
    for(int i=0;i<4;i++)CHECK(st->sv[i]>=0 && st->sv[i]+x.n[i]<=sh->sv[i],QDG_ERR_RANGE);
    uint8_t *o=g->outputs[c->io];
    for(int32_t b=0;b<x.n[0];b++)for(int32_t h=0;h<x.n[1];h++)for(int32_t w=0;w<x.n[2];w++) {
        size_t di=((((size_t)b+st->sv[0])*sh->sv[1]+h+st->sv[1])*sh->sv[2]+w+st->sv[2])*sh->sv[3]+st->sv[3];
        memcpy(o+di*es,(uint8_t *)x.p+(((size_t)b*x.n[1]+h)*x.n[2]+w)*x.n[3]*es,x.n[3]*es);
    }
    return 0;
}
static int k_quantize(QdgGraph *g, const C *c) {
    const T *o=out(g,c,0);
    A x;TRY(t_read(g,in(g,c,0),&x,0));
    CHECK(x.dt==DT_F32 && o->dtype==DT_U16 && (o->flags&QDR_QUANT),QDG_ERR_TENSOR);
    double inv=(double)(float)(1.0/o->scale);
    A r;TRY(anew(g,&r,x.n,DT_U16));
    const float *s=x.p;uint16_t *d=r.p;
    for(size_t i=0,n=count4(x.n);i<n;i++) {
        double v=floor((double)s[i]*inv+0.5)+o->zp;
        d[i]=(uint16_t)(v<0?0:v>65535?65535:v);
    }
    return t_write(g,o,&r);
}
static int k_dequantize(QdgGraph *g, const C *c) {
    const T *i=in(g,c,0);
    A x;TRY(t_read(g,i,&x,0));
    int32_t *v;TRY(to_i32(g,&x,&v));
    double sc=(double)(float)i->scale;
    A r;TRY(anew(g,&r,x.n,DT_F32));
    float *d=r.p;
    for(size_t j=0,n=count4(x.n);j<n;j++) {
        double exact=((double)v[j]-i->zp)*sc;
        float rounded=(float)exact;
        float away=nextafterf(rounded,exact<0?-INFINITY:INFINITY);
        d[j]=exact==((double)rounded+(double)away)*0.5?away:rounded;
    }
    return t_write(g,out(g,c,0),&r);
}
static int k_sign(QdgGraph *g, const C *c) {
    const T *i=in(g,c,0),*o=out(g,c,0);
#if defined(QDG_DSP) && QDS_BLOCKED_GLUE
    QdSignLevels levels;
    int have_levels=qd_sign_prepare_levels(o->scale,o->zp,&levels);
    if(have_levels && i->dtype==DT_U16 && o->dtype==DT_U16 &&
       (i->flags&QDR_BLOCKED) && (o->flags&QDR_BLOCKED)) {
        int32_t depth=(o->d[3]+o->p[3]+31)/32*32;
        if(depth>o->m[3])depth=o->m[3];
        if(!o->crouton || depth-o->p[3]==i->d[3]) {
            QdBlockedSignU16 *cached=g->blocked_sign[g->current];
            if(!cached) {
                uint64_t started=now_ns();QdBlockedU16 a,b;QdBlockedSignU16 prepared;
                blocked_lc_tensor(g,&g->t[c->in[0]],&a);blocked_lc_tensor(g,&g->t[c->out[0]],&b);
                if(qd_blocked_sign_u16_prepare(&a,&b,&prepared)) {
                    cached=malloc(sizeof *cached);
                    if(cached)memcpy(cached,&prepared,sizeof *cached);
                } else cached=(void *)1;
                g->blocked_sign[g->current]=cached;
                g->dsp_blocked_sign_prepare+=now_ns()-started;
            }
            if((uintptr_t)cached>1) {
                uint64_t started=now_ns();QdBlockedSignU16 hot;memcpy(&hot,cached,sizeof hot);
                hot.in.blocks=i->blocks;hot.out.blocks=o->blocks;
                int applied=qd_blocked_sign_u16_apply(&hot,i->zp,&levels);
                g->dsp_blocked_sign_apply+=now_ns()-started;
                if(applied){g->dsp_blocked_sign_calls++;return 0;}
            }
        }
    }
    if(have_levels && i->dtype==DT_U16 && o->dtype==DT_U16) {
        A x,r;TRY(t_read(g,i,&x,0));TRY(anew(g,&r,x.n,DT_U16));
        if(qd_sign_apply_u16_u16(x.p,r.p,count4(x.n),i->zp,&levels))return t_write(g,o,&r);
    }
#endif
    A x;TRY(t_read(g,i,&x,0));
    int32_t *v;TRY(to_i32(g,&x,&v));
    A r;TRY(anew(g,&r,x.n,o->dtype));
    for(size_t j=0,n=count4(x.n);j<n;j++) {
        int64_t d=(int64_t)v[j]-i->zp,s=d>0?1:d<0?-1:0;
        double y=floor((double)s/o->scale+o->zp+0.5);
        from_i64(&r,j,y<0?0:y>65535?65535:(int64_t)y);
    }
    return t_write(g,o,&r);
}
static int k_mean(QdgGraph *g, const C *c) {
    const T *i=in(g,c,0),*o=out(g,c,0);
    A x;TRY(t_read(g,i,&x,0));
    CHECK(count4(x.n)==32 && i->zp==o->zp && i->scale==o->scale,QDG_ERR_UNSUPPORTED);
    int64_t s=0;for(size_t j=0;j<32;j++)s+=aget(&x,j);
    int32_t one[4]={1,1,1,1};
    A r;TRY(anew(g,&r,one,o->dtype));r.scalar=1;
    from_i64(&r,0,clip64((s+33)>>5,0,65535));
    return t_write(g,o,&r);
}
static float f32_from_bits(uint32_t b){float f;memcpy(&f,&b,4);return f;}
static uint32_t f32_bits(float f){uint32_t b;memcpy(&b,&f,4);return b;}
static int k_weight_scales(QdgGraph *g, const C *c) {
    const T *i=in(g,c,0),*o=out(g,c,0);
    A x;TRY(t_read(g,i,&x,0));
    CHECK(x.dt==DT_U8 && o->dtype==DT_F32,QDG_ERR_TENSOR);
    int32_t co=x.n[3];
    CHECK((size_t)co==count4(o->d),QDG_ERR_SHAPE);
    A r;TRY(anew(g,&r,o->d,DT_F32));
    size_t rows=count4(x.n)/co;
    for(int32_t k=0;k<co;k++) {
        int32_t mn=INT32_MAX,mx=INT32_MIN;
        for(size_t j=0;j<rows;j++){int32_t v=((uint8_t *)x.p)[j*co+k]-i->zp;if(v<mn)mn=v;if(v>mx)mx=v;}
        int32_t a=-127*mn>128*mx?-127*mn:128*mx;
        float s=(float)a/(float)16256;
        if(!(s>=1.0f))s=s!=s?s:1.0f;
        ((float *)r.p)[k]=f32_from_bits((f32_bits(s)+0x800u)&0xfffff000u);
    }
    return t_write(g,o,&r);
}
static int k_invscale(QdgGraph *g, const C *c) {
    const T *o=out(g,c,0);
    A x;TRY(t_read(g,in(g,c,0),&x,0));
    CHECK(x.dt==DT_F32 && (o->dtype==DT_Q32 || o->dtype==DT_I32),QDG_ERR_TENSOR);
    A r;TRY(anew(g,&r,x.n,o->dtype));
    for(size_t j=0,n=count4(x.n);j<n;j++) {
        volatile float q=(float)65536/((float *)x.p)[j];
        volatile float s=q+0.5f;
        float t=truncf(s);
        CHECK(t>-2147483648.0f && t<2147483648.0f,QDG_ERR_RANGE);
        ((int32_t *)r.p)[j]=(int32_t)t;
    }
    return t_write(g,o,&r);
}
static int k_combine_scales(QdgGraph *g, const C *c) {
    CHECK(c->nin==3,QDG_ERR_ARGS);
    const T *s=in(g,c,2);
    CHECK((s->flags&QDR_SCALAR) && s->dtype==DT_F32,QDG_ERR_ARGS);
    A a,b;TRY(read2(g,c,&a,&b));
    CHECK(a.dt==DT_F32 && b.dt==DT_F32 && !memcmp(a.n,b.n,16),QDG_ERR_SHAPE);
    float k=f32_from_bits(s->scalar);
    A r;TRY(anew(g,&r,a.n,DT_F32));
    for(size_t j=0,n=count4(a.n);j<n;j++){volatile float p=((float *)a.p)[j]*((float *)b.p)[j];((float *)r.p)[j]=p*k;}
    return t_write(g,out(g,c,0),&r);
}
static int k_convert_weights(QdgGraph *g, const C *c) {
    CHECK(c->nin==4,QDG_ERR_ARGS);
    const T *i=in(g,c,0),*o=out(g,c,0);
    A x,s;TRY(t_read(g,i,&x,0));TRY(t_read(g,in(g,c,2),&s,0));
    CHECK(x.dt==DT_U8 && (s.dt==DT_Q32 || s.dt==DT_I32) && (o->flags&QDR_DIRECT),QDG_ERR_TENSOR);
    int32_t H=x.n[0],W=x.n[1],I=x.n[2],O=x.n[3];
    CHECK(I%4==0 && O%32==0,QDG_ERR_SHAPE);
    int32_t bn[4];CHECK(!bshape(&x,&s,bn) && !memcmp(bn,x.n,16),QDG_ERR_SHAPE);
    size_t total=count4(x.n);
    CHECK(o->offset+total<=g->pool_bytes[o->pool] && g->pool[o->pool],QDG_ERR_RANGE);
    int8_t *raw=(int8_t *)g->pool[o->pool]+o->offset;
    for(int32_t h=0;h<H;h++)for(int32_t w=0;w<W;w++)for(int32_t ii=0;ii<I;ii++)for(int32_t oo=0;oo<O;oo++) {
        int64_t v=((uint8_t *)x.p)[(((size_t)h*W+w)*I+ii)*O+oo]-(int64_t)i->zp;
        int64_t sc=((int32_t *)s.p)[bindex(s.n,h,w,ii,oo)];
        int64_t q=clip64((v*sc+32768)>>16,-128,127);
        /* [O/32][H][W][I/4][O%32][I%4] */
        size_t at=(((((size_t)(oo/32)*H+h)*W+w)*(I/4)+ii/4)*32+oo%32)*4+ii%4;
        raw[at]=(int8_t)q;
    }
    return 0;
}
static int k_scale_convert(QdgGraph *g, const C *c) {
    const T *o=out(g,c,0);
    A x;TRY(t_read(g,in(g,c,0),&x,0));
    CHECK(x.dt==DT_F32 && o->esize==4 && o->dtype!=DT_F32,QDG_ERR_TENSOR);
    A r;TRY(anew(g,&r,x.n,o->dtype));
    for(size_t j=0,n=count4(x.n);j<n;j++) {
        int64_t h=((int64_t)((uint32_t *)x.p)[j]+0x1000)>>13;
        ((int32_t *)r.p)[j]=(int32_t)((h&0x3ff)+(((h&0x7c00)+0x6400)&0x7c00)+0x400000);
    }
    return t_write(g,o,&r);
}
static int k_dynamic_bias(QdgGraph *g, const C *c) {
    for(uint32_t i=3;i<c->nin;i++) {
        const T *t=in(g,c,i);int64_t v;
        CHECK(!sv_or_scalar(t,&v) && v==0,QDG_ERR_UNSUPPORTED);
    }
    const T *o=out(g,c,0);
    int32_t one[4]={1,1,1,1},zero=0;
    A r={{1,1,1,1},o->dtype,1,&zero};(void)one;
    CHECK(o->esize==4 && o->dtype!=DT_F32,QDG_ERR_TENSOR);
    return t_write(g,o,&r);
}
static int k_bias_shuffle(QdgGraph *g, const C *c) {
    CHECK(c->nin==3,QDG_ERR_ARGS);
    int64_t mode;
    CHECK(!sv_or_scalar(in(g,c,2),&mode) && mode==0,QDG_ERR_UNSUPPORTED);
    int32_t *a,*b;size_t na,nb;
    TRY(read_i32(g,in(g,c,0),&a,&na));TRY(read_i32(g,in(g,c,1),&b,&nb));
    const T *o=out(g,c,0);
    CHECK(na==nb && na%32==0 && 2*na==count4(o->d) && o->esize==4 && o->dtype!=DT_F32,QDG_ERR_SHAPE);
    A r;TRY(anew(g,&r,o->d,o->dtype));
    int32_t *d=r.p;
    for(size_t row=0;row<na/32;row++){memcpy(d+row*64,a+row*32,128);memcpy(d+row*64+32,b+row*32,128);}
    return t_write(g,o,&r);
}
static int k_abs(QdgGraph *g, const C *c) {
    const T *i=in(g,c,0),*o=out(g,c,0);
    float ratio=(float)i->scale*(float)(1.0/o->scale);
    int64_t mult=(int64_t)rintf(ratio*262144.0f);
    volatile float a=ratio*32768.0f;
    volatile float b=a+(float)o->zp;
    volatile float e=b*8.0f;
    int64_t offset=(int64_t)rintf(e);
    A x;TRY(t_read(g,i,&x,0));
    int32_t *v;TRY(to_i32(g,&x,&v));
    CHECK(i->esize==2,QDG_ERR_TENSOR);
    A r;TRY(anew(g,&r,x.n,o->dtype));
    size_t k=0;
    for(int32_t bb=0;bb<x.n[0];bb++)for(int32_t h=0;h<x.n[1];h++)for(int32_t w=0;w<x.n[2];w++)for(int32_t d=0;d<x.n[3];d++,k++) {
        int64_t lane;
        int64_t H=h+i->p[1],W=w+i->p[2],B=bb+i->p[0],D=d+i->p[3];
        if(i->flags&QDR_DIRECT) {
            int64_t lin=((B*i->m[1]+H)*i->m[2]+W)*i->m[3]+D;
            lane=((i->offset+lin*2)/2)%2==0;
        } else lane=W%2==0; /* block offsets are 128-byte aligned */
        int64_t centered=llabs((int64_t)v[k]-i->zp)-32768;
        int64_t y=((centered*mult+lane*16384)>>15)+offset;
        from_i64(&r,k,clip64((y+4)>>3,0,65535));
    }
    return t_write(g,o,&r);
}
static int k_block_zap(QdgGraph *g, const C *c) {
    CHECK(c->nfv>=2,QDG_ERR_ARGS);
    uint32_t fill=c->fv[0],header=c->fv[1],count=header&0x7fff;
    CHECK((header&0x8000) && (header&0xc00000)==0xc00000,QDG_ERR_UNSUPPORTED);
    uint32_t nzero=((header>>16)&7)*256;
    uint8_t tile[2048];
    for(uint32_t j=0;j<count;j++) {
        CHECK(2+2*j+1<c->nfv,QDG_ERR_ARGS);
        uint32_t src=c->fv[2+2*j],dst=c->fv[3+2*j];
        CHECK((uint64_t)src+2048<=g->pool_bytes[1] && (uint64_t)dst+2048<=g->pool_bytes[1],QDG_ERR_RANGE);
        memcpy(tile,g->pool[1]+src,2048);
        memset(tile+2048-nzero,fill&255,nzero);
        memcpy(g->pool[1]+dst,tile,2048);
    }
    return 0;
}

/* -------------------------------------------------------------- convolution */
static void unpack_weights(int8_t *o, const int8_t *raw, int32_t H, int32_t W, int32_t I, int32_t O, int packed) {
    size_t k=0;
    for(int32_t h=0;h<H;h++)for(int32_t w=0;w<W;w++)for(int32_t i=0;i<I;i++)for(int32_t oo=0;oo<O;oo++) {
        size_t off=packed?(((size_t)(oo/32)*H+h)*W+w)*(I/4)*128+(size_t)(i/4)*128+(oo%32)*4+i%4:
            ((((size_t)(oo/32)*(I/32)+i/32)*H+h)*W+w)*1024+(size_t)((i/4)%8)*128+(oo%32)*4+i%4;
        o[k++]=raw[off];
    }
}
static int is_class(const T *t, const char *a, const char *b) {
    uint32_t x=0,y=0;memcpy(&x,a,strlen(a));memcpy(&y,b,strlen(b));
    return t->cls==x || t->cls==y;
}
/* Returns a cached entry for static weights; dynamic weights get a fresh one in `tmp`. */
static int conv_weights(QdgGraph *g, uint32_t wi, int s2, int32_t co, Weights *tmp, Weights **result) {
    int reverse=!s2 && g->t[wi].cls==('X'|'w'<<8|'b'<<16);
    uint32_t src=g->static_source[wi];
    int cacheable=src!=QDR_NONE;
    if(!cacheable)src=wi;
    if(cacheable) for(uint32_t i=0;i<g->nweights;i++) {
        Weights *e=&g->weights[i];
        if(e->source==src && e->s2==s2 && e->co==co){*result=e;return 0;}
    }
    const T *t=&g->t[src];
    CHECK((t->flags&QDR_DIRECT) && g->pool[t->pool],QDG_ERR_TENSOR);
    int32_t H=t->m[0],W=t->m[1],I=t->m[2],O=t->m[3];
    int packed=is_class(t,"Xpb","xpb");
    size_t n=(size_t)H*W*I*O;
    CHECK(I%4==0 && O%32==0 && (packed || I%32==0),QDG_ERR_SHAPE);
    CHECK(t->offset+n<=g->pool_bytes[t->pool],QDG_ERR_RANGE);
#if defined(QDG_DSP) && QDS_DYNAMIC_TCM
    /* Runtime 1x1 Xpb bytes already have exactly the HMX B tile order:
     * [O/32][I/32][I%32/4][O%32][I%4]. Avoid the DDR -> HWIO ->
     * HMX round trip. A dense TCM copy serves the non-HMX fallback, and
     * channel sums are computed in TCM. Only this serial command owns it;
     * workers overwrite it after packing/parameter preparation is complete. */
    if(!cacheable && !s2 && !reverse && packed && H==1 && W==1 &&
       !(I&31) && co==O && n<=0x4000u && O<=256) {
        Weights *e=tmp;memset(e,0,sizeof *e);
        e->used=2;e->source=src;e->co=O;e->kh=e->kw=1;e->ci=I;
        e->w=(int8_t *)0x30000000u;e->wsum=(int64_t *)0x30004000u;e->wpos=(int64_t *)0x30004800u;
        uint8_t *raw=g->pool[t->pool]+t->offset;
        memcpy((void *)0x30008000u,raw,n);
        unpack_weights(e->w,(const int8_t *)0x30008000u,H,W,I,O,1);
        memset(e->wsum,0,O*8);memset(e->wpos,0,O*8);
        for(int32_t k=0;k<I;k++)for(int32_t o=0;o<O;o++) {
            int8_t v=e->w[k*O+o];e->wsum[o]+=v;if(v>0)e->wpos[o]+=v;
        }
        e->packed[1]=raw;e->packed_bytes[1]=n;e->packed_ci[1]=I;
        g->dsp_dynamic_tcm_calls++;*result=e;return 0;
    }
#endif
    int8_t *w=malloc(n),*final=NULL;
    CHECK(w,QDG_ERR_MEMORY);
    unpack_weights(w,(const int8_t *)g->pool[t->pool]+t->offset,H,W,I,O,packed);
    int32_t kh=H,kw=W,ci=I;
    if(s2) {
        if(H!=3 || W!=3 || I%32){free(w);FAIL(QDG_ERR_SHAPE);}
        int8_t *tr=calloc(n,1);
        if(!tr){free(w);FAIL(QDG_ERR_MEMORY);}
        for(int k=0;k<9;k++)memcpy(tr+((size_t)S2_TAPS[k][0]*3+S2_TAPS[k][1])*I*O,w+(size_t)k*I*O,(size_t)I*O);
        kh=kw=2;ci=4*I;
        final=calloc((size_t)4*ci*O,1);
        if(!final){free(w);free(tr);FAIL(QDG_ERR_MEMORY);}
        for(int a=0;a<2;a++)for(int b=0;b<2;b++)for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++) {
            int ky=2*a+dy,kx=2*b-dx;
            if(ky<3 && kx>=0 && kx<3)for(int32_t cb=0;cb<I/32;cb++) {
                int32_t j=cb*4+2*dy+dx;
                for(int32_t ch=0;ch<32;ch++)
                    memcpy(final+(((size_t)a*2+b)*ci+32*j+ch)*O,tr+(((size_t)ky*3+kx)*I+32*cb+ch)*O,O);
            }
        }
        free(tr);free(w);
    } else if(reverse) {
        final=malloc(n);
        if(!final){free(w);FAIL(QDG_ERR_MEMORY);}
        for(int32_t h=0;h<H;h++)for(int32_t x=0;x<W;x++)memcpy(final+((size_t)h*W+x)*I*O,w+((size_t)h*W+(W-1-x))*I*O,(size_t)I*O);
        free(w);
    } else final=w;
    int32_t ow=O;
    if(co>ow){free(final);FAIL(QDG_ERR_SHAPE);}
    Weights *e=tmp;
    if(cacheable) {
        if(g->nweights==g->capweights) {
            uint32_t cap=g->capweights?2*g->capweights:64;
            Weights *grown=realloc(g->weights,cap*sizeof *grown);
            if(!grown){free(final);FAIL(QDG_ERR_MEMORY);}
            g->weights=grown;g->capweights=cap;
        }
        e=&g->weights[g->nweights++];
    }
    memset(e,0,sizeof *e);
    e->used=1;e->source=src;e->s2=s2;e->co=co;e->kh=kh;e->kw=kw;e->ci=ci;
    if(cacheable){
#ifdef QDG_DSP
        static uint32_t keys;e->key=++keys;
#else
        static atomic_uint keys;e->key=atomic_fetch_add(&keys,1)+1;
#endif
    }
    size_t terms=(size_t)kh*kw*ci;
    e->w=malloc(terms*co);e->wsum=calloc(co,sizeof(int64_t));e->wpos=calloc(co,sizeof(int64_t));
    if(!e->w || !e->wsum || !e->wpos){free(final);FAIL(QDG_ERR_MEMORY);}
    for(size_t k=0;k<terms;k++)for(int32_t o=0;o<co;o++){int8_t v=final[k*ow+o];e->w[k*co+o]=v;e->wsum[o]+=v;if(v>0)e->wpos[o]+=v;}
    free(final);
    *result=e;
    return 0;
}
static void free_weights(Weights *e) {
#ifdef QDG_DSP
    if(e->used==2) { /* dense/sums are borrowed TCM, HMX packet aliases pool 1 */
        free(e->packed[0]);memset(e,0,sizeof *e);return;
    }
#endif
    free(e->w);free(e->wsum);free(e->wpos);free(e->packed[0]);free(e->packed[1]);
    memset(e,0,sizeof *e);
}
/* qdm_quant_params: proves the DSP epilogue fits signed 32 bits. */
static int quant_params(int32_t *out, const Weights *w, const int32_t *mult, const int32_t *bias, int32_t zp) {
    /* Channel sums come from conv_weights' row-major pass: a column walk of
     * w->w misses cache on every element over the DSP's L1-only DDR heap. */
    for(int32_t c=0;c<w->co;c++) {
        int64_t sum=w->wsum[c],pos=w->wpos[c],neg=sum-pos;
        int64_t correction=(int64_t)bias[c]-(int64_t)zp*sum;
        int32_t h=mult[c]&65535,pre=20-((h>>10)&31);
        if(pre<0 || pre>31 || correction+255*neg<INT32_MIN || correction+255*pos>INT32_MAX)return 1;
        out[c*3]=(int32_t)correction;out[c*3+1]=1024+(h&1023);out[c*3+2]=pre;
    }
    return 0;
}

/* CPU backend: exact integer sum x*w for x [H,W,ci] (u8 or u16). */
static int cpu_accumulate(QdgGraph *g, const A *x, const Weights *w, int64_t *acc, int32_t oh, int32_t ow) {
    int32_t co=w->co,ci=w->ci;
    int32_t *row=salloc(g,(size_t)co*4+64);
    CHECK(row,QDG_ERR_MEMORY);
    for(int32_t y=0;y<oh;y++)for(int32_t xx=0;xx<ow;xx++) {
        int64_t *a=acc+((size_t)y*ow+xx)*co;
        memset(a,0,(size_t)co*8);
        for(int32_t ky=0;ky<w->kh;ky++)for(int32_t kx=0;kx<w->kw;kx++) {
            memset(row,0,(size_t)co*4);
            const int8_t *wk=w->w+((size_t)(ky*w->kw+kx)*ci)*co;
            size_t base=((size_t)(y+ky)*x->n[2]+xx+kx)*x->n[3];
            if(x->dt==DT_U8) {
                const uint8_t *xs=(const uint8_t *)x->p+base;
                for(int32_t i=0;i<ci;i++){int32_t v=xs[i];const int8_t *wr=wk+(size_t)i*co;for(int32_t o=0;o<co;o++)row[o]+=v*wr[o];}
                for(int32_t o=0;o<co;o++)a[o]+=row[o];
            } else {
                const uint16_t *xs=(const uint16_t *)x->p+base;
                for(int32_t i=0;i<ci;i++){int64_t v=xs[i];const int8_t *wr=wk+(size_t)i*co;for(int32_t o=0;o<co;o++)a[o]+=v*wr[o];}
            }
        }
    }
    return 0;
}

/* ---------------------------------------------------------------- DSP link */
#define ARENA_BYTES 0x800000u
#define STATUS_OFFSET 128u
#define VTCM_BYTES 0x2d0000u
#define L2TCM_BYTES 0x100000u
#define EXITED 0x54495845u
static uint32_t chunked(uint32_t n) {return (n+1023)&~1023u;}
static void fence(void) {
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}
static int conv_layout_u16(Layout *L, uint32_t h, uint32_t width, uint32_t c, uint32_t kh, uint32_t kw, uint32_t n, int quantized, int hmx, int u16) {
    if(n%8 || n>1024 || kh>8 || kw>8 || h<kh || width<kw || (u16 && (quantized || hmx))) return 1;
    L->h=h;L->width=width;L->c=c;L->kh=kh;L->kw=kw;L->n=n;L->quantized=quantized;L->hmx=hmx;L->u16=u16;L->keep=0;L->x32=0;L->out_h=L->out_w=0;
    L->g=(c+3)/4;L->oh=h-kh+1;L->ow=width-kw+1;L->owp=(L->ow+31)/32*32;L->wp=L->owp+kw-1;
    if(!L->oh || !L->ow || L->oh>64 || L->owp>2048 || L->g>256) return 1;
    L->plane=h*L->g*L->wp*4;
    L->xb=L->plane*(u16?2:1);L->wb=kh*kw*L->g*n*4;L->cb=L->oh*n*L->owp*(quantized?1:u16?2:4);L->vb=0;
    L->kb=(kh*kw*L->g+7)/8;
    if(hmx) {
        if(!quantized || n%32) return 1;
        L->wb=L->kb*(n/32)*1024;L->cb=L->oh*(L->owp/32)*(n/32)*1024;
        uint32_t xa=(L->xb+128+2047)&~2047u,b=xa+L->oh*(L->owp/32)*L->kb*2048,cc=(b+L->wb+2047)&~2047u;
        L->vb=b;
        uint32_t end=((cc+3*L->cb+((n*8+127)&~127u)+256+2047)&~2047u)+2048;
        if(end>VTCM_BYTES) return 1;
    }
    if(chunked(L->xb+128)>VTCM_BYTES || chunked(L->wb)>L2TCM_BYTES) return 1;
    L->pb=quantized?chunked(n*12):u16?chunked(n*32):0;
    L->xo=4096+L->pb;L->wo=L->xo+chunked(L->xb+128);L->co=L->wo+chunked(L->wb);
    if(L->co+L->cb>ARENA_BYTES) return 1;
    return 0;
}
static int conv_layout(Layout *L, uint32_t h, uint32_t width, uint32_t c, uint32_t kh, uint32_t kw, uint32_t n, int quantized, int hmx) {
    return conv_layout_u16(L,h,width,c,kh,kw,n,quantized,hmx,0);
}
static void pack_x(uint8_t *o, const uint8_t *x, uint32_t H, uint32_t W, uint32_t C, uint32_t G, uint32_t WP, uint32_t bytes) {
    memset(o,0,bytes);
    for(uint32_t h=0;h<H;h++)for(uint32_t g=0;g<G;g++) {
        uint8_t *d=o+((size_t)h*G*WP+(size_t)g*WP)*4;
        const uint8_t *s=x+(size_t)h*W*C+g*4;
        uint32_t n=C-g*4;if(n>4)n=4;
        if(n==4)for(uint32_t w=0;w<W;w++)__builtin_memcpy(d+4*w,s+(size_t)w*C,4);
        else for(uint32_t w=0;w<W;w++)memcpy(d+4*w,s+(size_t)w*C,n);
    }
}
static uint8_t *pack_weights(const Weights *w, uint32_t C, uint32_t G, uint32_t KB, int hmx, uint32_t *bytes) {
    uint32_t KH=w->kh,KW=w->kw,N=w->co;
    size_t n=hmx?(size_t)KB*(N/32)*1024:(size_t)KH*KW*G*N*4;
    int8_t *o=calloc(n,1);
    if(!o) return NULL;
    if(hmx) {
        uint32_t cp=G*4,depth=KH*KW*cp;
        for(uint32_t nb=0;nb<N/32;nb++)for(uint32_t kb=0;kb<KB;kb++)for(uint32_t group=0;group<8;group++)
            for(uint32_t nn=0;nn<32;nn++)for(uint32_t c=0;c<4;c++) {
                uint32_t k=kb*32+group*4+c,tap=k/cp,channel=k%cp;
                o[((((size_t)nb*KB+kb)*8+group)*32+nn)*4+c]=k<depth && channel<C?w->w[((size_t)tap*C+channel)*N+nb*32+nn]:0;
            }
    } else {
        for(uint32_t k=0;k<KH*KW;k++)for(uint32_t g=0;g<G;g++)for(uint32_t nn=0;nn<N;nn++)for(uint32_t c=0;c<4;c++)
            o[(((size_t)k*G+g)*N+nn)*4+c]=g*4+c<C?w->w[((size_t)k*C+g*4+c)*N+nn]:0;
    }
    *bytes=(uint32_t)n;
    return (uint8_t *)o;
}
static void put32(uint8_t *p, uint32_t v) {*(volatile uint32_t *)p=v;}
static uint32_t get32(const uint8_t *p) {return *(const volatile uint32_t *)p;}
static int dsp_wait(QdgGraph *g, uint32_t seq) {
    const uint8_t *st=g->arena+STATUS_OFFSET;
    uint64_t start=0;
    int retried=0;
    for(uint32_t spin=0;;spin++) {
        if(get32(st)==seq && get32(st+16)==seq && get32(st+32)==seq) return 0;
        if((spin&1023)==1023) {
            uint64_t t=now_ns();
            if(!start)start=t;
            if(g->callback_word && *g->callback_word!=g->callback_value) FAIL(QDG_ERR_DSP);
            if(get32(st+12)==EXITED && get32(st+28)==EXITED && get32(st+44)==EXITED) {
                /* Workers left on their idle timeout before reading this command. */
                if(retried || !g->wake || g->wake(0) || g->wake(1)) FAIL(QDG_ERR_WAKE);
                retried=1;
                continue;
            }
            if((double)(t-start)*1e-9>g->timeout) FAIL(QDG_ERR_TIMEOUT);
        }
    }
}
static void requant(uint8_t *o, const int64_t *acc, const int32_t *mult, const int32_t *bias, size_t pixels, int32_t co, int sixteen) {
    int32_t M[1024],pre[1024],post[1024];
    for(int32_t c=0;c<co;c++) {
        uint32_t h=(uint32_t)mult[c]&65535,shift=34-((h>>10)&31);
        M[c]=1024+(h&1023);pre[c]=sixteen?8:(int32_t)shift-14;post[c]=(int32_t)shift-pre[c];
    }
    int64_t max=sixteen?65535:255;
    for(size_t p=0;p<pixels;p++)for(int32_t c=0;c<co;c++) {
        size_t j=p*co+c;
        int64_t a=acc[j]+(int64_t)bias[c]*(sixteen?256:1);
        int64_t q=(((a>>pre[c])*M[c])+((int64_t)1<<(post[c]-1)))>>post[c];
        q=clip64(q,0,max);
        if(sixteen)((uint16_t *)o)[j]=(uint16_t)q;else o[j]=(uint8_t)q;
    }
}

/* Pipelined requests: one executes on the DSP while its successor is packed
 * into the other half of the arena; a finished request's results are collected
 * while the next one runs. Commands touching an in-flight output wait first. */
#define SLOT_BYTES (((ARENA_BYTES-4096)/2)&~2047u)
static const uint32_t slot_base[3]={4096,4096+SLOT_BYTES,4096};
static const uint32_t slot_end[3]={4096+SLOT_BYTES,ARENA_BYTES,ARENA_BYTES};
static int place(Layout *L, uint32_t slot) {
    uint32_t base=slot_base[slot];
    L->xo=base+L->pb;L->wo=L->xo+chunked(L->xb+128);L->co=L->wo+chunked(L->wb);
    return L->co+L->cb<=slot_end[slot];
}
static int grow(void **p, size_t *cap, size_t bytes) {
    if(bytes<=*cap) return 0;
    void *n=realloc(*p,bytes);
    if(!n) return QDG_ERR_MEMORY;
    *p=n;*cap=bytes;return 0;
}
static int finalize(QdgGraph *g, Job *j) {
    const T *o=&g->t[g->c[j->cmd].out[0]];
    int32_t n[4]={1,j->oh,j->ow,j->co};
    size_t pixels=(size_t)j->oh*j->ow;
    if(j->kind==1) {
        int64_t *acc=j->acc[0];
        if(j->planes==2)for(size_t k=0;k<pixels*j->co;k++)acc[k]=acc[k]*256+j->acc[1][k];
        for(size_t p=0;p<pixels;p++)for(int32_t k=0;k<j->co;k++)acc[p*j->co+k]-=(int64_t)j->zp*j->wsum[k];
        requant(j->r,acc,j->mult,j->bias,pixels,j->co,j->sixteen);
    }
    A r={{n[0],n[1],n[2],n[3]},j->sixteen?DT_U16:DT_U8,0,j->r};
    j->busy=0;
    return t_write_region(g,o,&r,j->origin);
}
#ifndef QDG_DSP
typedef struct { QdgGraph *g; unsigned index; int cpu; } HelperArgs;
static void *helper_main(void *arg) {
    HelperArgs a=*(HelperArgs *)arg;free(arg);
    Pool *h=&a.g->helpers;
    if(a.cpu>=0) {
#ifdef __linux__
        cpu_set_t set;CPU_ZERO(&set);CPU_SET(a.cpu,&set);pthread_setaffinity_np(pthread_self(),sizeof set,&set);
#endif
    }
    /* Start from the creation-time generation (0): a job published before this
     * thread first runs must still be seen as new. */
    unsigned seen=0;
    for(;;) {
        unsigned gen;
        uint32_t spins=0;
        while((gen=atomic_load_explicit(&h->generation,memory_order_acquire))==seen) {
            if(atomic_load(&h->quit)) return NULL;
            if(!atomic_load(&h->active) || ++spins>4000000u) {
                pthread_mutex_lock(&h->lock);
                while(atomic_load(&h->generation)==seen && !atomic_load(&h->quit) && (!atomic_load(&h->active) || spins))
                    {pthread_cond_wait(&h->wake,&h->lock);spins=0;}
                pthread_mutex_unlock(&h->lock);
                spins=0;
            }
        }
        seen=gen;
        h->job(h->ctx,a.index+1,(unsigned)h->started+1);
        atomic_fetch_add_explicit(&h->done,1,memory_order_release);
    }
}
/* Runs job(ctx, part, parts) for every part, part 0 on the calling thread. */
static void parallel(QdgGraph *g, void (*job)(void *, unsigned, unsigned), void *ctx) {
    Pool *h=&g->helpers;
    if(h->started<1 || !atomic_load(&h->active)) {job(ctx,0,1);return;}
    h->job=job;h->ctx=ctx;
    atomic_store_explicit(&h->done,0,memory_order_relaxed);
    atomic_fetch_add_explicit(&h->generation,1,memory_order_release);
    /* Helpers idle for a while sleep on the condition variable. */
    pthread_mutex_lock(&h->lock);pthread_cond_broadcast(&h->wake);pthread_mutex_unlock(&h->lock);
    job(ctx,0,(unsigned)h->started+1);
    while(atomic_load_explicit(&h->done,memory_order_acquire)<(unsigned)h->started) {}
}
typedef struct { uint8_t *dst; const uint8_t *src; size_t bytes; } CopyJob;
static void copy_job(void *ctx, unsigned part, unsigned parts) {
    CopyJob *j=ctx;
    size_t chunk=((j->bytes+parts-1)/parts+4095)&~(size_t)4095,off=part*chunk;
    if(off<j->bytes)memcpy(j->dst+off,j->src+off,j->bytes-off<chunk?j->bytes-off:chunk);
}
static void parallel_copy(QdgGraph *g, uint8_t *dst, const uint8_t *src, size_t bytes) {
    if(bytes<(64u<<10)){memcpy(dst,src,bytes);return;}
    CopyJob j={dst,src,bytes};
    parallel(g,copy_job,&j);
}
static void helpers_active(QdgGraph *g, int on) {
    Pool *h=&g->helpers;
    if(!h->started) return;
    atomic_store(&h->active,on);
    if(on){pthread_mutex_lock(&h->lock);pthread_cond_broadcast(&h->wake);pthread_mutex_unlock(&h->lock);}
}
static void stop_helpers(QdgGraph *g) {
    Pool *h=&g->helpers;
    if(!h->started) return;
    atomic_store(&h->quit,1);
    pthread_mutex_lock(&h->lock);pthread_cond_broadcast(&h->wake);pthread_mutex_unlock(&h->lock);
    for(int i=0;i<h->started;i++)pthread_join(h->thread[i],NULL);
    h->started=0;
    pthread_mutex_destroy(&h->lock);pthread_cond_destroy(&h->wake);
}
int qdg_set_helpers(QdgGraph *g, int count, const int *cpus) {
    stop_helpers(g);
    Pool *h=&g->helpers;
    memset(h,0,sizeof *h);
    if(count<=0) return 0;
    if(count>MAX_HELPERS) count=MAX_HELPERS;
    pthread_mutex_init(&h->lock,NULL);pthread_cond_init(&h->wake,NULL);
    for(int i=0;i<count;i++) {
        HelperArgs *a=malloc(sizeof *a);
        if(!a) break;
        a->g=g;a->index=(unsigned)i;a->cpu=cpus?cpus[i]:-1;
        if(pthread_create(&h->thread[i],NULL,helper_main,a)){free(a);break;}
        h->started++;
    }
    return h->started;
}
#else
static void parallel(QdgGraph *g,void (*job)(void *,unsigned,unsigned),void *ctx){(void)g;job(ctx,0,1);}
static void helpers_active(QdgGraph *g,int on){(void)g;(void)on;}
static void stop_helpers(QdgGraph *g){(void)g;}
int qdg_set_helpers(QdgGraph *g,int count,const int *cpus){(void)g;(void)count;(void)cpus;return 0;}
static void parallel_copy(QdgGraph *g,uint8_t *d,const uint8_t *s,size_t n){(void)g;memcpy(d,s,n);}
#endif
/* HMX emits [32 pixels][32 lanes] tiles. Up to eight adjacent pixels are
 * already a contiguous row of a u8 crouton: copy those bytes directly from the
 * arena, preserving block-table holes, slices and untouched padding. The old
 * path copied to staging, unpacked to dense NHWC, then blocked the same bytes.
 * This writes exactly the t_write_region box, so footprints are unchanged. */
typedef struct { QdgGraph *g; const Req *q; T *t; } HmxCollect;
static int compare_block(const void *a,const void *b) {
    uint32_t x=*(const uint32_t *)a,y=*(const uint32_t *)b;
    return (x>y)-(x<y);
}
/* Shared block tables may alias physical storage. Parallel copies of duplicate
 * blocks would lose the serial write order; conservatively keep those serial. */
static int blocks_unique(T *t) {
    if(t->unique_blocks)return t->unique_blocks==1;
    uint32_t *sorted=malloc((size_t)t->nblocks*4);
    if(!sorted)return 0;
    memcpy(sorted,t->blocks,(size_t)t->nblocks*4);
    qsort(sorted,t->nblocks,4,compare_block);
    t->unique_blocks=1;
    for(uint32_t i=1;i<t->nblocks;i++)if((sorted[i]&127) && sorted[i]==sorted[i-1]) {
        t->unique_blocks=2;break;
    }
    free(sorted);return t->unique_blocks==1;
}
static uint32_t hmx_block_entry(const HmxCollect *a,uint32_t i) {
    const Job *j=a->q->job;
    const Layout *l=&a->q->L;
    const T *t=a->t;
    uint32_t nb=l->n/32,wb=l->out_w/8,h=i/(wb*nb),w=(i/nb)%wb,d=i%nb;
    int64_t nbh=(t->m[1]+7)/8,nbw=(t->m[2]+7)/8,nbd=(t->m[3]+31)/32;
    int64_t index=(((int64_t)j->origin[0]*nbh+j->origin[1]/8+h)*nbw+j->origin[2]/8+w)*nbd+
                  t->slice+j->origin[3]/32+d;
    return (index>=0 && index<t->nblocks)?t->blocks[index]:0;
}
static int hmx_blocks_unique(const HmxCollect *a) {
    /* A tensor's shared table can repeat blocks outside this command's output
     * box. Only the entries actually written determine copy independence. The
     * decoded command's shape, origin and block table are immutable. */
    ConvState *cs=&a->g->conv[a->q->job->cmd];
    if(cs->block_unique)return cs->block_unique==1;
    const Layout *l=&a->q->L;
    uint32_t count=(l->out_h/8)*(l->out_w/8)*(l->n/32);
    uint32_t *sorted=malloc((size_t)count*4);
    if(!sorted)return 0;
    for(uint32_t i=0;i<count;i++)sorted[i]=hmx_block_entry(a,i);
    qsort(sorted,count,4,compare_block);
    cs->block_unique=1;
    for(uint32_t i=1;i<count;i++)if((sorted[i]&127) && sorted[i]==sorted[i-1]) {
        cs->block_unique=2;break;
    }
    free(sorted);return cs->block_unique==1;
}
static void hmx_collect_blocks(void *ctx, unsigned part, unsigned parts) {
    HmxCollect *a=ctx;
    const Layout *l=&a->q->L;
    uint32_t nb=l->n/32,wb=l->out_w/8,count=(l->out_h/8)*wb*nb;
    uint32_t per=(count+parts-1)/parts,start=part*per,end=start+per;
    if(end>count)end=count;
    for(uint32_t i=start;i<end;i++) {
        uint32_t e=hmx_block_entry(a,i),pool=e&127;
        if(pool && a->g->pool[pool])
            memcpy(a->g->pool[pool]+(e&~127u),a->g->arena+l->co+(size_t)i*2048,2048);
    }
}
static void hmx_collect_job(void *ctx, unsigned part, unsigned parts) {
    HmxCollect *a=ctx;
    QdgGraph *g=a->g;
    const Req *q=a->q;
    const Layout *l=&q->L;
    const Job *j=q->job;
    const T *t=a->t;
    const uint8_t *raw=g->arena+l->co;
    int32_t per=(j->oh+(int32_t)parts-1)/(int32_t)parts,h0=(int32_t)part*per;
    int32_t end=h0+per;if(end>j->oh)end=j->oh;
    int64_t nbh=(t->m[1]+7)/8,nbw=(t->m[2]+7)/8,nbd=(t->m[3]+31)/32;
    uint32_t src_stride=l->oh==1?(uint32_t)j->ow:l->owp;
    for(int32_t h=h0;h<end;h++) {
        int64_t H=h+j->origin[1];
        int64_t row=((int64_t)j->origin[0]*nbh+H/8)*nbw;
        for(int32_t w=0;w<j->ow;) {
            int64_t W=w+j->origin[2];
            uint32_t p=(uint32_t)h*src_stride+(uint32_t)w;
            int32_t n=8-(int32_t)(W%8);
            if(n>32-(int32_t)(p%32))n=32-(int32_t)(p%32);
            if(n>j->ow-w)n=j->ow-w;
            int64_t block=(row+W/8)*nbd+t->slice+j->origin[3]/32;
            size_t within=(size_t)(H%8)*256+(size_t)(W%8)*32;
            for(uint32_t nb=0;nb<l->n/32;nb++) {
                int64_t index=block+nb;
                uint32_t e=(index>=0 && index<t->nblocks)?t->blocks[index]:0,pool=e&127;
                if(pool && g->pool[pool]) {
                    const uint8_t *src=raw+(((size_t)(p/32)*(l->n/32)+nb)*32+p%32)*32;
                    memcpy(g->pool[pool]+(e&~127u)+within,src,(size_t)n*32);
                }
            }
            w+=n;
        }
    }
}
static int hmx_collect_direct(QdgGraph *g, const Req *q) {
    const Job *j=q->job;
    T *t=&g->t[g->c[j->cmd].out[0]];
    if(!q->L.hmx || j->kind || j->sixteen || j->pending!=1 ||
       !(t->flags&QDR_BLOCKED) || (t->flags&QDR_DIRECT) || t->esize!=1 ||
       j->co!=(int32_t)q->L.n || (j->origin[3]&31) ||
       !((q->L.oh==(uint32_t)j->oh && q->L.ow==(uint32_t)j->ow) ||
         (q->L.oh==1 && q->L.ow==(uint32_t)j->oh*(uint32_t)j->ow)))return 0;
    for(int i=0;i<4;i++)if(j->origin[i]<0)return 0;
    HmxCollect a={g,q,t};
    if(q->L.cb>=(64u<<10) && j->oh>=4 && blocks_unique(t))parallel(g,hmx_collect_job,&a);
    else hmx_collect_job(&a,0,1);
    return 1;
}
static int collect(QdgGraph *g, Req *q) {
    uint64_t t0=now_ns();
    const Layout *L=&q->L;
    const uint8_t *raw=g->arena+L->co;
    Job *j=q->job;
    uint32_t H=L->oh,W=L->ow,N=L->n,WP=L->owp;
    if(L->out_h || (g->block_output && hmx_collect_direct(g,q))) {
        if(L->out_h) {
            g->stats.hmx_block_calls++;
            HmxCollect a={g,q,&g->t[g->c[j->cmd].out[0]]};
            if((uint64_t)L->out_h*L->out_w*L->n>=(16u<<10) && hmx_blocks_unique(&a))parallel(g,hmx_collect_blocks,&a);
            else hmx_collect_blocks(&a,0,1);
        }
        q->active=0;j->pending=0;j->busy=0;
        g->stats.collect_ns+=now_ns()-t0;
        return 0;
    }
    parallel_copy(g,g->staging,raw,L->cb);
    raw=g->staging;
    if(L->hmx) {
        uint8_t *q8=j->r;
        for(uint32_t h=0;h<H;h++)for(uint32_t ww=0;ww<W;ww++) {
            uint32_t p=h*WP+ww;
            for(uint32_t nb=0;nb<N/32;nb++)
                memcpy(q8+((size_t)h*W+ww)*N+nb*32,raw+((((size_t)(p/32)*(N/32)+nb)*32+p%32)*32),32);
        }
    } else if(L->u16) {
        uint16_t *o=(uint16_t *)j->r;
        const uint16_t *r=(const uint16_t *)raw;
        for(uint32_t h=0;h<H;h++)for(uint32_t n=0;n<N;n++)for(uint32_t ww=0;ww<W;ww++)
            o[((size_t)h*W+ww)*N+n]=r[(((size_t)h*(N/2)+n/2)*(WP/32)+ww/32)*64+(n%2)*32+ww%32];
    } else if(L->quantized) {
        uint8_t *q8=j->r;
        for(uint32_t h=0;h<H;h++)for(uint32_t n=0;n<N;n++)for(uint32_t ww=0;ww<W;ww++)
            q8[((size_t)h*W+ww)*N+n]=raw[(((size_t)h*(N/4)+n/4)*(WP/32)+ww/32)*128+(n%4)*32+ww%32];
    } else {
        const int32_t *r=(const int32_t *)raw;
        int64_t *acc=j->acc[q->plane];
        for(uint32_t h=0;h<H;h++)for(uint32_t n=0;n<N;n++)for(uint32_t ww=0;ww<W;ww++)
            acc[((size_t)h*W+ww)*N+n]=r[((size_t)h*N+n)*WP+ww];
    }
    q->active=0;
    g->stats.collect_ns+=now_ns()-t0;
    if(--j->pending==0) return finalize(g,j);
    return 0;
}
static int check_done(QdgGraph *g, Req *q) {
    const uint8_t *a=g->arena,*st=a+STATUS_OFFSET;
    uint32_t units=q->L.oh*((q->L.owp+63)/64)*(q->L.n/8),total=0;
    for(int t=0;t<3;t++) {
        CHECK(get32(st+16*t)==q->seq && get32(st+16*t+4)==0,QDG_ERR_DSP);
        total+=get32(st+16*t+8);
    }
    CHECK(total==units,QDG_ERR_DSP);
    if(q->L.hmx) {
        for(int p=0;p<4;p++){uint32_t m=0;for(int t=0;t<3;t++){uint32_t v=get32(a+2048+4*(p+4*t));if(v>m)m=v;}g->stats.hmx_phase_cycles[p]+=m;}
        g->stats.hmx_calls++;
    } else for(int p=0;p<3;p++){uint32_t m=0;for(int t=0;t<3;t++){uint32_t v=get32(a+2048+4*(p+4*t));if(v>m)m=v;}g->stats.hvx_phase_cycles[q->L.u16][p]+=m;}
    return 0;
}
static int wait_inflight(QdgGraph *g) {
    uint64_t t0=now_ns();
    TRY(dsp_wait(g,g->inflight.seq));
    fence();
    g->stats.wait_ns+=now_ns()-t0;
    return check_done(g,&g->inflight);
}
static int dsp_complete(QdgGraph *g) {
    if(!g->inflight.active) return 0;
    TRY(wait_inflight(g));
    return collect(g,&g->inflight);
}
static int publish(QdgGraph *g, const Layout *L, uint32_t seq) {
    uint8_t *a=g->arena;
    uint32_t param=(L->quantized||L->u16)?L->xo-L->pb:0;
    memset(a+STATUS_OFFSET,0,48);
    uint32_t fields[20]={0x32564451u,(g->native_hmx && L->hmx)?6u+(uint32_t)g->native_hmx:6u,0,1,L->h,L->wp,L->g,L->kh,L->kw,L->n,L->oh,L->owp,L->xo,L->wo,L->co,ARENA_BYTES,20000,
        (g->raise_clock?0u:1u)|(L->quantized?2u:0u)|(L->hmx?4u:0u)|(L->u16?8u:0u)|(L->keep?16u:0u)|(L->x32?32u:0u)|
        (L->out_h?64u|(L->out_h<<8)|(L->out_w<<16):0u),param,L->hmx?L->key:0};
    for(int i=0;i<20;i++)if(i!=2)put32(a+4*i,fields[i]);
    fence();
    put32(a+8,seq);
    fence();
    if(!g->persistent) {
        CHECK(g->wake && !g->wake(1),QDG_ERR_WAKE);
        g->persistent=1;
    }
    return 0;
}
/* Pack one request (x: u8 [1,h,w,c]); returns once it is published to the DSP. */
static int dsp_submit(QdgGraph *g, Job *job, uint32_t plane, const A *x, Weights *w, const int32_t *params, int hmx) {
    int u16=x->dt==DT_U16,quantized=params!=NULL && !u16;
    Layout L;
    CHECK((x->dt==DT_U8 || (u16 && params && !hmx)) && x->n[0]==1,QDG_ERR_TENSOR);
    CHECK(!conv_layout_u16(&L,x->n[1],x->n[2],x->n[3],w->kh,w->kw,w->co,quantized,hmx,u16),QDG_ERR_SHAPE);
    CHECK((uint32_t)w->ci==L.c,QDG_ERR_SHAPE);
    const T *out_t=&g->t[g->c[job->cmd].out[0]];
    if(g->block_output && hmx && !job->kind && !job->sixteen && job->pending==1 &&
       (out_t->flags&QDR_BLOCKED) && !(out_t->flags&QDR_DIRECT) && out_t->esize==1 &&
       job->oh>0 && job->oh<=255 && job->ow>0 && job->ow<=65535 &&
       !(job->oh&7) && !(job->ow&7) &&
       job->origin[0]>=0 && job->origin[1]>=0 && job->origin[2]>=0 && job->origin[3]>=0 &&
       !(job->origin[1]&7) && !(job->origin[2]&7) && !(job->origin[3]&31)) {
        L.out_h=(uint32_t)job->oh;L.out_w=(uint32_t)job->ow;
    }
    Req *prev=&g->inflight;
    uint32_t slot=0;
    if(prev->active && prev->slot!=2 && place(&L,1-prev->slot)) slot=1-prev->slot;
    else {
        TRY(dsp_complete(g));
        slot=place(&L,0)?0:2;
        CHECK(place(&L,slot),QDG_ERR_SHAPE);
    }
    uint64_t t0=now_ns();
    uint8_t *a=g->arena;
    const uint8_t *st=a+STATUS_OFFSET;
    if(u16) {
        /* Low then high byte plane of the u16 activation, each in the u8 layout. */
        size_t n=count4(x->n);
        uint8_t *lo=salloc(g,n),*hi=salloc(g,n);
        CHECK(lo && hi,QDG_ERR_MEMORY);
        const uint16_t *v=x->p;
        for(size_t k=0;k<n;k++){lo[k]=(uint8_t)v[k];hi[k]=(uint8_t)(v[k]>>8);}
        memset(a+L.xo,0,chunked(L.xb+128));
        pack_x(a+L.xo,lo,L.h,L.width,L.c,L.g,L.wp,L.plane);
        pack_x(a+L.xo+L.plane,hi,L.h,L.width,L.c,L.g,L.wp,L.plane);
    } else if(hmx && L.c%32==0) {
        /* [h][c/32][wp][32]: each pixel's 32-channel block is one contiguous copy. */
        L.x32=1;
        uint32_t cbs=L.c/32,rowb=L.wp*32;
        uint8_t *d=a+L.xo;
        const uint8_t *src=x->p;
        for(uint32_t h=0;h<L.h;h++)for(uint32_t cb=0;cb<cbs;cb++,d+=rowb) {
            const uint8_t *s=src+(size_t)h*L.width*L.c+cb*32;
            for(uint32_t w=0;w<L.width;w++)memcpy(d+32*w,s+(size_t)w*L.c,32);
            memset(d+32*L.width,0,rowb-32*L.width);
        }
        memset(a+L.xo+L.xb,0,chunked(L.xb+128)-L.xb);
    } else pack_x(a+L.xo,x->p,L.h,L.width,L.c,L.g,L.wp,chunked(L.xb+128));
    int ws=hmx?1:0;
    /* HMX weights left in VTCM by the previous command can be reused. */
    if(g->native_hmx && hmx) {
        uint32_t native_mb=(L.oh*(L.owp/32)+1)/2;
        L.vb=((L.xb+128+2047)&~2047u)+native_mb*L.kb*2048;
    }
    L.key=hmx?w->key:0;
    L.keep=hmx && L.key && g->persistent && g->resident_key==L.key && g->resident_b==L.vb && g->resident_wb==L.wb;
    uint8_t *packed=w->packed[ws];
    if(!packed || w->packed_ci[ws]!=L.c) {
        free(w->packed[ws]);
        w->packed[ws]=packed=pack_weights(w,L.c,L.g,L.kb,hmx,&w->packed_bytes[ws]);
        CHECK(packed,QDG_ERR_MEMORY);
        w->packed_ci[ws]=L.c;
    }
    CHECK(w->packed_bytes[ws]==L.wb,QDG_ERR_SHAPE);
    if(!L.keep)memcpy(a+L.wo,packed,L.wb);
    else g->stats.weights_reused++;
    if(quantized || u16) {
        memset(a+L.xo-L.pb,0,L.pb);
        memcpy(a+L.xo-L.pb,params,(size_t)L.n*(u16?32:12));
    }
    g->resident_key=L.key;g->resident_b=L.vb;g->resident_wb=L.wb;
    g->stats.prepare_ns+=now_ns()-t0;
    g->stats.calls++;
    g->stats.bytes_uploaded+=L.xb+L.wb+(quantized?(uint64_t)L.n*12:0);
    g->stats.bytes_x[hmx]+=L.xb;g->stats.bytes_w[hmx]+=L.keep?0:L.wb;
    g->stats.bytes_out[hmx]+=L.out_h?(uint64_t)L.out_h*L.out_w*L.n:L.cb;
    CHECK(g->sequence<0xffffffffu,QDG_ERR_DSP);
    Req next={1,++g->sequence,plane,slot,L,job};
    if(prev->active) {
        /* The successor starts as soon as the predecessor is verified. */
        TRY(wait_inflight(g));
        Req done=*prev;
        TRY(publish(g,&L,next.seq));
        *prev=next;
        return collect(g,&done);
    }
    if(g->persistent && get32(st+12)==EXITED && get32(st+28)==EXITED && get32(st+44)==EXITED) {
        CHECK(g->wake && !g->wake(0),QDG_ERR_WAKE);
        g->persistent=0;
    }
    TRY(publish(g,&L,next.seq));
    *prev=next;
    return 0;
}
static Job *new_job(QdgGraph *g) {
    for(int i=0;i<2;i++)if(!g->jobs[i].busy){g->jobs[i].busy=1;return &g->jobs[i];}
    return NULL;
}
#ifdef QDG_DSP
#define QDG_DSP_DEFER 2001
static int resident_conv(QdgGraph *,const C *,uint32_t,const A *,Weights *,const int32_t *,const int32_t *,int32_t,int32_t,int32_t,int,const int32_t *);
#endif
#if defined(QDG_DSP) && QDS_NATIVE_1X1
static int native_1x1_try(QdgGraph *,const C *,uint32_t);
#endif
#if defined(QDG_DSP) && QDS_NATIVE_3X3
static int native_3x3_try(QdgGraph *,const C *,uint32_t);
#endif
static int k_hmx_conv(QdgGraph *g, const C *c, uint32_t index) {
    CHECK(c->nin>=3 && c->nout==1,QDG_ERR_ARGS);
#if defined(QDG_DSP) && QDS_NATIVE_1X1
    int native_status=native_1x1_try(g,c,index);
    if(native_status!=QDG_DSP_DEFER)return native_status;
#endif
#if defined(QDG_DSP) && QDS_NATIVE_3X3
    int spatial_status=native_3x3_try(g,c,index);
    if(spatial_status!=QDG_DSP_DEFER)return spatial_status;
#endif
    uint64_t start=now_ns();
    uint32_t wi=c->in[1],fi=c->in[2];
    const T *xt=in(g,c,0),*ft=in(g,c,2),*o=out(g,c,0);
    int s2=c->variant==36; /* ConvLayer_s2.opt@CB*2.Xwb.Fi */
    /* read_fi: rows of 64 int32 = 32 multiplier words then 32 biases. */
    CHECK((ft->flags&QDR_DIRECT) && ft->dtype==DT_I32,QDG_ERR_TENSOR);
    size_t nfi=count4(ft->m);
    CHECK(nfi%64==0 && ft->offset+nfi*4<=g->pool_bytes[ft->pool],QDG_ERR_RANGE);
    const int32_t *rows=(const int32_t *)(g->pool[ft->pool]+ft->offset);
    size_t nm=nfi/2;
#ifdef QDG_DSP
    CHECK(nm<=1024,QDG_ERR_SHAPE);
    /* Survive worker execution in the master's private L2 stack. Shared L2
     * scratch is overwritten by convolution weights/prepare tables. */
    int32_t mult_local[1024],bias_local[1024];
    int32_t *mult=mult_local,*bias=bias_local;
#else
    int32_t *mult=salloc(g,nm*4),*bias=salloc(g,nm*4);
#endif
    CHECK(mult && bias,QDG_ERR_MEMORY);
    for(size_t r=0;r<nfi/64;r++){memcpy(mult+r*32,rows+r*64,128);memcpy(bias+r*32,rows+r*64+32,128);}
    int sixteen=xt->cls==('C'|'H'<<8) || xt->cls==('c'|'H'<<8);
#ifdef QDG_DSP
    g->dsp_conv_phase[sixteen]+=now_ns()-start;
#endif
    int32_t oh=o->d[1],ow=o->d[2],co=(o->d[3]+31)/32*32;
    const T *wt=&g->t[g->static_source[wi]!=QDR_NONE?g->static_source[wi]:wi];
    int32_t wco=wt->m[3];
    if(co>wco)co=wco;
    if((size_t)co>nm)co=(int32_t)nm;
    Weights tmp,*w;
#ifdef QDG_DSP
    uint64_t weights_start=now_ns();uint32_t weights_before=g->nweights;
#endif
    TRY(conv_weights(g,wi,s2,co,&tmp,&w));
#ifdef QDG_DSP
    uint64_t weights_elapsed=now_ns()-weights_start;g->dsp_weights_cycles+=weights_elapsed;
    if(w==&tmp || g->nweights!=weights_before){g->dsp_weights_miss_cycles+=weights_elapsed;g->dsp_weights_misses++;}
#endif
    int dynamic=w==&tmp;
    int rc=0;
    int32_t xH=xt->m[1]-xt->p[1],xW=xt->m[2]-xt->p[2];
    CHECK(w->ci<=xt->m[3],QDG_ERR_SHAPE);
    int32_t cw=sixteen?4:8;
    oh=(oh+7)/8*8;if(oh>o->m[1])oh=o->m[1];if(oh>xH-w->kh+1)oh=xH-w->kh+1;
    ow=(ow+cw-1)/cw*cw;if(ow>o->m[2])ow=o->m[2];if(ow>xW-w->kw+1)ow=xW-w->kw+1;
    CHECK(oh>0 && ow>0,QDG_ERR_SHAPE);
    int32_t dims[4]={1,oh+w->kh-1,ow+w->kw-1,w->ci},origin[4]={0,xt->p[1],xt->p[2],0};
    int32_t on[4]={1,oh,ow,co},oorigin[4]={0,o->p[1],o->p[2],0};
    size_t pixels=(size_t)oh*ow,values=pixels*co;
    A x;
#ifdef QDG_DSP
    Weights fast;memcpy(&fast,w,sizeof fast);
    if(fast.used==2) {
        /* A fallback epilogue reads sums after workers overwrite lower L2. */
        int64_t *hot=__builtin_alloca((size_t)co*8);
        memcpy(hot,fast.wsum,(size_t)co*8);fast.wsum=hot;
    }
    if(g->dsp_direct_enabled && !sixteen && w->key && !(w->ci&31) && !(co&31) &&
       !(oh&7) && !(ow&7) && (xt->flags&QDR_BLOCKED) && !(xt->flags&QDR_DIRECT) &&
       xt->p[1]>=0 && xt->p[2]>=0 && (o->flags&QDR_BLOCKED) && !(o->flags&QDR_DIRECT) &&
       !(oorigin[1]&7) && !(oorigin[2]&7) && !(oorigin[3]&31)) {
        A direct;memcpy(direct.n,dims,16);direct.dt=DT_U8;direct.scalar=0;direct.p=NULL;
        g->dsp_direct_tensor=xt;g->dsp_direct_h=oh;g->dsp_direct_w=ow;
        memcpy(g->dsp_direct_origin,origin,16);
        rc=resident_conv(g,c,index,&direct,&fast,mult,bias,oh,ow,co,0,oorigin);
        g->dsp_direct_tensor=NULL;
        if(rc!=QDG_DSP_DEFER){memcpy(w,&fast,sizeof fast);goto done;}
    }
    uint8_t *saved_scratch=g->scratch;size_t saved_used=g->scratch_used,saved_bytes=g->scratch_bytes;
    g->scratch=(uint8_t *)0x31000000u;g->scratch_used=0;g->scratch_bytes=0x250000u;
    uint64_t read_started=now_ns();
    rc=t_read_region(g,xt,&x,dims,origin);
    g->dsp_conv_phase[2+sixteen]+=now_ns()-read_started;
    g->scratch=saved_scratch;g->scratch_used=saved_used;g->scratch_bytes=saved_bytes;
    if(rc)goto done;
#else
    if((rc=t_read_region(g,xt,&x,dims,origin)))goto done;
#endif
#ifdef QDG_DSP
    rc=resident_conv(g,c,index,&x,&fast,mult,bias,oh,ow,co,sixteen,oorigin);
    memcpy(w,&fast,sizeof fast);
    goto done;
#endif
    if(g->backend==QDG_BACKEND_DSP) {
        A xx=x;
        if(w->kh==1 && w->kw==1){xx.n[1]=1;xx.n[2]=x.n[1]*x.n[2];}
        Job *j=new_job(g);
        if(!j){rc=QDG_ERR_MEMORY;goto done;}
        j->cmd=index;j->oh=oh;j->ow=ow;j->co=co;j->zp=xt->zp;j->sixteen=sixteen;
        memcpy(j->origin,oorigin,16);
        if((rc=grow((void **)&j->r,&j->r_cap,values*(sixteen?2:1)))){j->busy=0;goto done;}
        ConvState *cs=&g->conv[index];
        if(!sixteen) {
            int32_t *params=cs->params;
            int ok=cs->ready && cs->static_q?cs->quant_ok:0;
            if(!(cs->ready && cs->static_q)) {
                params=salloc(g,(size_t)co*12);
                if(!params){j->busy=0;rc=QDG_ERR_MEMORY;goto done;}
                ok=!quant_params(params,w,mult,bias,xt->zp);
                if(!dynamic && g->static_fi[fi]) {
                    cs->params=ok?malloc((size_t)co*12):NULL;
                    if(ok && !cs->params){j->busy=0;rc=QDG_ERR_MEMORY;goto done;}
                    if(ok)memcpy(cs->params,params,(size_t)co*12);
                    cs->quant_ok=ok;cs->static_q=1;cs->ready=1;
                    params=cs->params;
                }
            }
            if(ok) {
                /* transport.conv_quantized: HMX if every channel's epilogue qualifies. */
                int hmx=g->use_hmx && co%32==0;
                for(int32_t k=0;hmx && k<co;k++) {
                    int64_t p=params[3*k+2],half=p>0?((int64_t)1<<(p-1)):0;
                    if(p>15 || (int64_t)params[3*k]-half<-2147483648LL)hmx=0;
                }
                Layout L;
                if(hmx && conv_layout(&L,xx.n[1],xx.n[2],xx.n[3],w->kh,w->kw,co,1,1))hmx=0;
                j->kind=0;j->planes=1;j->pending=1;
                rc=dsp_submit(g,j,0,&xx,w,params,hmx);
                if(rc)j->busy=0;
                goto done;
            }
        }
        if(sixteen) {
            /* One request with the fused u16 epilogue when every channel's exact
             * parameters fit the DSP's 32-bit arithmetic (see QdQuant16). */
            int32_t *q16=salloc(g,(size_t)co*32);
            int ok=q16!=NULL && 255ll*128*w->kh*w->kw*w->ci<(1ll<<29); /* |acc| per plane */
            for(int32_t k=0;ok && k<co;k++) {
                uint32_t h=(uint32_t)mult[k]&65535,shift=34-((h>>10)&31);
                int64_t post=(int64_t)shift-8,M=1024+(h&1023);
                int64_t corr=256*(int64_t)bias[k]-(int64_t)xt->zp*w->wsum[k],ch=corr>>8;
                if(post<1 || post>15 || ch<-(1ll<<29) || ch>(1ll<<29)){ok=0;break;}
                int64_t rd=1ll<<(post-1),top=((int64_t)65535<<post)-rd,pmax=(top+M-1)/M;
                int32_t *e=q16+8*k;
                e[0]=(int32_t)ch;e[1]=(int32_t)(corr-256*ch);e[2]=(int32_t)pmax;e[3]=(int32_t)M;e[4]=(int32_t)post;e[5]=(int32_t)rd;e[6]=e[7]=0;
            }
            if(ok) {
                j->kind=0;j->planes=1;j->pending=1;
                rc=dsp_submit(g,j,0,&xx,w,q16,0);
                if(rc)j->busy=0;
                goto done;
            }
        }
        /* accumulate_direct: u16 as exact high/low byte planes. */
        j->kind=1;j->planes=x.dt==DT_U8?1:2;j->pending=j->planes;
        if((rc=grow((void **)&j->acc[0],&j->acc_cap[0],values*8)) || (rc=grow((void **)&j->acc[1],&j->acc_cap[1],values*8)) ||
           (rc=grow((void **)&j->mult,&j->mult_cap,co*4)) || (rc=grow((void **)&j->bias,&j->bias_cap,co*4)) ||
           (rc=grow((void **)&j->wsum,&j->wsum_cap,co*8))){j->busy=0;goto done;}
        memcpy(j->mult,mult,co*4);memcpy(j->bias,bias,co*4);memcpy(j->wsum,w->wsum,co*8);
        if(x.dt==DT_U8) rc=dsp_submit(g,j,0,&xx,w,NULL,0);
        else {
            size_t n=count4(x.n);
            A hi=xx,lo=xx;hi.dt=lo.dt=DT_U8;
            hi.p=salloc(g,n);lo.p=salloc(g,n);
            if(!hi.p || !lo.p){j->busy=0;rc=QDG_ERR_MEMORY;goto done;}
            for(size_t k=0;k<n;k++){uint16_t v=((uint16_t *)x.p)[k];((uint8_t *)hi.p)[k]=(uint8_t)(v>>8);((uint8_t *)lo.p)[k]=(uint8_t)v;}
            rc=dsp_submit(g,j,0,&hi,w,NULL,0);
            if(!rc)rc=dsp_submit(g,j,1,&lo,w,NULL,0);
        }
        goto done;
    }
    {
        A r;
        int64_t *acc=salloc(g,values*8);
        if(!acc || (rc=anew(g,&r,on,sixteen?DT_U16:DT_U8))){if(!rc)rc=QDG_ERR_MEMORY;goto done;}
        if((rc=cpu_accumulate(g,&x,w,acc,oh,ow)))goto done;
        for(size_t p=0;p<pixels;p++)for(int32_t k=0;k<co;k++)acc[p*co+k]-=(int64_t)xt->zp*w->wsum[k];
        requant(r.p,acc,mult,bias,pixels,co,sixteen);
        rc=t_write_region(g,o,&r,oorigin);
    }
done:
    if(dynamic)free_weights(&tmp);
    g->stats.conv_ns+=now_ns()-start;
    return rc;
}

/* ------------------------------------------------------------- footprints */
static void fp_range(QdgGraph *g, uint64_t *fp, uint32_t pool, uint64_t off, uint64_t bytes) {
    if(!bytes || pool>=QDG_POOLS || !g->pool[pool]) return;
    for(uint64_t b=off/2048;b<=(off+bytes-1)/2048;b++) {
        uint64_t k=g->block_base[pool]+b;
        if(k<g->fp_words*64)fp[k/64]|=1ull<<(k%64);
    }
}
static void fp_tensor(QdgGraph *g, uint64_t *fp, const T *t) {
    if(t->flags&QDR_DIRECT) fp_range(g,fp,t->pool,t->offset,count4(t->m)*t->esize);
    if(t->flags&QDR_BLOCKED) for(uint32_t i=0;i<t->nblocks;i++) fp_range(g,fp,t->blocks[i]&127,t->blocks[i]&~127u,2048);
}
/* Blocks (or the direct byte span) touched by a box access, using exactly the
 * index arithmetic of blocked_impl/direct_xfer. */
static void fp_box(QdgGraph *g, uint64_t *fp, const T *t, const int32_t *dims, const int32_t *origin) {
    for(int i=0;i<4;i++)if(dims[i]<=0)return;
    if(t->flags&QDR_DIRECT) {
        size_t lo=(((size_t)origin[0]*t->m[1]+origin[1])*t->m[2]+origin[2])*t->m[3]+origin[3];
        size_t hi=((((size_t)origin[0]+dims[0]-1)*t->m[1]+origin[1]+dims[1]-1)*t->m[2]+origin[2]+dims[2]-1)*t->m[3]+origin[3]+dims[3]-1;
        fp_range(g,fp,t->pool,t->offset+lo*t->esize,(hi-lo+1)*t->esize);
        return;
    }
    if(!(t->flags&QDR_BLOCKED)) return;
    int64_t cw=t->esize==1?8:t->esize==2?4:2;
    int64_t nbh=(t->m[1]+7)/8,nbw=(t->m[2]+cw-1)/cw,nbd=(t->m[3]+31)/32;
    for(int64_t B=origin[0];B<origin[0]+dims[0];B++)
        for(int64_t bh=origin[1]/8;bh<=(origin[1]+dims[1]-1)/8;bh++)
            for(int64_t bw=origin[2]/cw;bw<=(origin[2]+dims[2]-1)/cw;bw++)
                for(int64_t bd=origin[3]/32;bd<=(origin[3]+dims[3]-1)/32;bd++) {
                    int64_t index=((B*nbh+bh)*nbw+bw)*nbd+bd+t->slice;
                    if(index>=0 && index<t->nblocks){uint32_t e=t->blocks[index];fp_range(g,fp,e&127,e&~127u,2048);}
                }
}
/* Box read by t_read (full: max extent) or written by t_write (lane extension). */
static void fp_read(QdgGraph *g, uint64_t *fp, const T *t, int full) {
    static const int32_t zero[4]={0,0,0,0};
    if(!(t->flags&QDR_HAS_SHAPE)) return;
    fp_box(g,fp,t,full?t->m:t->d,full?zero:t->p);
}
static void fp_write(QdgGraph *g, uint64_t *fp, const T *t) {
    if(!(t->flags&QDR_HAS_SHAPE)) return;
    int32_t dims[4];memcpy(dims,t->d,sizeof dims);
    if(t->crouton){int32_t e=(t->d[3]+t->p[3]+31)/32*32;if(e>t->m[3])e=t->m[3];dims[3]=e-t->p[3];}
    fp_box(g,fp,t,dims,t->p);
}
/* The geometry k_hmx_conv derives (fast_depth.conv), from static shapes only. */
static int conv_boxes(QdgGraph *g, const C *c, int32_t *xdims, int32_t *xorigin, int32_t *odims, int32_t *oorigin) {
    if(c->nin<3 || c->nout!=1) return 1;
    const T *xt=&g->t[c->in[0]],*ft=&g->t[c->in[2]],*o=&g->t[c->out[0]];
    uint32_t wi=c->in[1];
    const T *wt=&g->t[g->static_source[wi]!=QDR_NONE?g->static_source[wi]:wi];
    int s2=c->variant==36,sixteen=xt->cls==('C'|'H'<<8) || xt->cls==('c'|'H'<<8);
    int32_t kh=s2?2:wt->m[0],kw=s2?2:wt->m[1],ci=s2?4*wt->m[2]:wt->m[2];
    int32_t nm=(int32_t)(count4(ft->m)/2),co=(o->d[3]+31)/32*32;
    if(co>wt->m[3])co=wt->m[3];
    if(co>nm)co=nm;
    int32_t xH=xt->m[1]-xt->p[1],xW=xt->m[2]-xt->p[2],cw=sixteen?4:8;
    int32_t oh=(o->d[1]+7)/8*8,ow=(o->d[2]+cw-1)/cw*cw;
    if(oh>o->m[1])oh=o->m[1];if(oh>xH-kh+1)oh=xH-kh+1;
    if(ow>o->m[2])ow=o->m[2];if(ow>xW-kw+1)ow=xW-kw+1;
    if(oh<=0 || ow<=0) return 1;
    int32_t xd[4]={1,oh+kh-1,ow+kw-1,ci},xo[4]={0,xt->p[1],xt->p[2],0},od[4]={1,oh,ow,co},oo[4]={0,o->p[1],o->p[2],0};
    memcpy(xdims,xd,16);memcpy(xorigin,xo,16);memcpy(odims,od,16);memcpy(oorigin,oo,16);
    return 0;
}
static int fp_hit(const QdgGraph *g, const uint64_t *a, const uint64_t *b) {
    for(uint32_t i=0;i<g->fp_words;i++)if(a[i]&b[i])return 1;
    return 0;
}
static int build_footprints(QdgGraph *g) {
    uint64_t blocks=0;
    for(int p=0;p<QDG_POOLS;p++){g->block_base[p]=blocks;if(g->pool[p])blocks+=(g->pool_bytes[p]+2047)/2048;}
    g->fp_words=(uint32_t)((blocks+63)/64);
    g->fp_all=calloc((size_t)g->ncommands*g->fp_words,8);
    g->fp_conv_in=calloc((size_t)g->ncommands,sizeof *g->fp_conv_in);
    g->fp_conv_out=calloc((size_t)g->ncommands,sizeof *g->fp_conv_out);
    g->hoisted=calloc(g->ncommands,1);
    if(!g->fp_all || !g->fp_conv_in || !g->fp_conv_out || !g->hoisted) return QDG_ERR_MEMORY;
    for(uint32_t i=0;i<g->ncommands;i++) {
        const C *c=&g->c[i];
        uint64_t *fp=g->fp_all+(size_t)i*g->fp_words;
        switch(c->op) {
        case QDR_HMX_CONV: {
            int32_t xd[4],xo[4],od[4],oo[4];
            g->fp_conv_in[i]=calloc(g->fp_words,8);g->fp_conv_out[i]=calloc(g->fp_words,8);
            if(!g->fp_conv_in[i] || !g->fp_conv_out[i]) return QDG_ERR_MEMORY;
            if(conv_boxes(g,c,xd,xo,od,oo)) {
                for(uint32_t k=0;k<c->nin;k++)fp_tensor(g,g->fp_conv_in[i],&g->t[c->in[k]]);
                fp_tensor(g,g->fp_conv_out[i],&g->t[c->out[0]]);
            } else {
                fp_box(g,g->fp_conv_in[i],&g->t[c->in[0]],xd,xo);
                for(uint32_t k=1;k<c->nin;k++)fp_tensor(g,g->fp_conv_in[i],&g->t[c->in[k]]);
                fp_box(g,g->fp_conv_out[i],&g->t[c->out[0]],od,oo);
            }
            for(uint32_t k=0;k<g->fp_words;k++)fp[k]=g->fp_conv_in[i][k]|g->fp_conv_out[i][k];
            break;
        }
        case QDR_WEIGHTS_TO_VTCM:
            for(uint32_t k=0;k<c->nin;k++)fp_tensor(g,fp,&g->t[c->in[k]]);
            for(uint32_t k=0;k<c->nout;k++)fp_tensor(g,fp,&g->t[c->out[k]]);
            break;
        case QDR_COPY_ACTIVATION: case QDR_SPACE_TO_DEPTH:
            for(uint32_t k=0;k<c->nin;k++)fp_read(g,fp,&g->t[c->in[k]],1);
            for(uint32_t k=0;k<c->nout;k++)fp_read(g,fp,&g->t[c->out[k]],1);
            break;
        case QDR_SPILL_FILL:
            if(c->nfv>=4) {
                uint64_t offset=c->fv[3];
                for(uint32_t k=4;k+1<c->nfv;k+=2) {
                    fp_range(g,fp,c->fv[k]&0x7ff,c->fv[k]&~0x7ffu,c->fv[k+1]);
                    fp_range(g,fp,QDG_POOL_SPILL,offset,c->fv[k+1]);offset+=c->fv[k+1];
                }
            }
            break;
        case QDR_BLOCK_ZAP:
            for(uint32_t k=2;k+1<c->nfv;k+=2){fp_range(g,fp,1,c->fv[k],2048);fp_range(g,fp,1,c->fv[k+1],2048);}
            break;
        case QDR_CONVERT_WEIGHTS:
            for(uint32_t k=0;k<c->nin;k++)fp_read(g,fp,&g->t[c->in[k]],0);
            if(c->nin>=1 && c->nout==1){const T *x=&g->t[c->in[0]],*o=&g->t[c->out[0]];fp_range(g,fp,o->pool,o->offset,count4(x->d));}
            break;
        default:
            /* Every other kernel reads inputs with t_read (dims at padding) and
             * writes outputs with t_write; direct raw access is listed above. */
            for(uint32_t k=0;k<c->nin;k++)fp_read(g,fp,&g->t[c->in[k]],0);
            for(uint32_t k=0;k<c->nout;k++)fp_write(g,fp,&g->t[c->out[k]]);
            break;
        }
    }
    return 0;
}

/* ----------------------------------------------------------------- dispatch */
static int run_command(QdgGraph *g, uint32_t index) {
#ifdef QDG_DSP
    C local;dsp_command_begin(g,index,&local);
    const C *c=&local;
#else
    const C *c=&g->c[index];
#endif
    g->current=index;
#ifdef QDG_DSP
    /* Convolution owns VTCM until every worker returns. Between convolutions
     * the same verified scratch region belongs to the serial glue kernel.
     * Discard the host residency mirror before any such VTCM write. */
    g->dsp_l2_used=0;g->dsp_l2_active=c->op!=QDR_HMX_CONV;
    if(c->op==QDR_HMX_CONV){g->scratch=g->dsp_scratch;g->scratch_bytes=g->dsp_scratch_bytes;}
    else {g->scratch=(uint8_t *)0x31000000u;g->scratch_bytes=0x250000u;g->resident_key=0;}
#endif
    switch(c->op) {
    case QDR_NOP: return 0;
    case QDR_INPUT_SLICE_PAD: return k_input_slice_pad(g,c);
    case QDR_INPUT_SLICE: return k_input_slice(g,c);
    case QDR_WEIGHTS_TO_VTCM: return k_weights_to_vtcm(g,c);
    case QDR_IM2COL: return k_im2col(g,c);
    case QDR_HMX_CONV: return k_hmx_conv(g,c,index);
    case QDR_SPACE_TO_DEPTH: return k_space_to_depth(g,c);
    case QDR_REQUANT_U8: return k_requant_u8(g,c);
    case QDR_REQUANT_U16: return k_requant_u16(g,c);
    case QDR_ADD_U8: return k_add_u8(g,c);
    case QDR_ADD_U16: return k_add_u16(g,c);
    case QDR_ADDSUB_U16: return k_addsub_u16(g,c);
    case QDR_ADD1111_U16: return k_add1111_u16(g,c);
    case QDR_MUL_U16: return k_mul_u16(g,c);
    case QDR_MUL1111_U16: return k_mul1111_u16(g,c);
    case QDR_COPY_VALUES: return k_copy_values(g,c);
    case QDR_COPY_ACTIVATION: return k_copy_activation(g,c);
    case QDR_TRANSPOSE: return k_transpose(g,c);
    case QDR_CONCAT: return k_concat(g,c);
    case QDR_GATHER: return k_gather(g,c);
    case QDR_SLICE: return k_slice(g,c);
    case QDR_SLICE_PAD: return k_slice_pad(g,c);
    case QDR_DECIMATE: return k_decimate(g,c);
    case QDR_RESIZE: return k_resize(g,c);
    case QDR_DEPTH_TO_SPACE: return k_depth_to_space(g,c);
    case QDR_SPILL_FILL: return k_spill_fill(g,c,c->variant==9);
    case QDR_OUTPUT_SLICE: return k_output_slice(g,c);
    case QDR_QUANTIZE: return k_quantize(g,c);
    case QDR_DEQUANTIZE: return k_dequantize(g,c);
    case QDR_SIGN: return k_sign(g,c);
    case QDR_MEAN: return k_mean(g,c);
    case QDR_WEIGHT_SCALES: return k_weight_scales(g,c);
    case QDR_INVSCALE: return k_invscale(g,c);
    case QDR_COMBINE_SCALES: return k_combine_scales(g,c);
    case QDR_CONVERT_WEIGHTS: return k_convert_weights(g,c);
    case QDR_SCALE_CONVERT: return k_scale_convert(g,c);
    case QDR_DYNAMIC_BIAS: return k_dynamic_bias(g,c);
    case QDR_BIAS_SHUFFLE: return k_bias_shuffle(g,c);
    case QDR_LINEARCLIP: return k_linearclip(g,c);
    case QDR_ABS: return k_abs(g,c);
    case QDR_MUL_INLINE: return k_mul_inline(g,c);
    case QDR_ADDSUB_FLAT: return k_addsub_flat(g,c);
    case QDR_ADDSUB_INLINE: return k_addsub_inline(g,c);
    case QDR_BLOCK_ZAP: return k_block_zap(g,c);
    default: return QDG_ERR_UNSUPPORTED;
    }
}

/* Command i needs the in-flight result. Submit the next convolution j instead
 * when it is independent of the in-flight request and of every pending command
 * in [i, j) (no shared footprint block), so the DSP computes it while ARM runs
 * them. Its predecessor is collected during the submission. */
#define HOIST_WINDOW 64u
static int run_command(QdgGraph *g, uint32_t index);
static int hoist(QdgGraph *g, uint32_t i, uint32_t last, int *done) {
    *done=0;
    const uint64_t *busy=g->fp_conv_out[g->inflight.job->cmd];
    for(uint32_t j=i+1;j<last && j<i+HOIST_WINDOW;j++) {
        if(g->hoisted[j] || g->c[j].op!=QDR_HMX_CONV) continue;
        const uint64_t *in=g->fp_conv_in[j],*outj=g->fp_conv_out[j];
        if(fp_hit(g,in,busy)) {g->stats.variant_count[76]++;return 0;}
        for(uint32_t k=i;k<j;k++) {
            if(g->hoisted[k]) continue;
            const uint64_t *fk=g->fp_all+(size_t)k*g->fp_words;
            if(fp_hit(g,in,fk)) {g->stats.variant_count[77]++;g->stats.variant_ns[77]+=j-i;return 0;}
            if(fp_hit(g,outj,fk)) {g->stats.variant_count[79]++;g->stats.variant_ns[79]+=j-i;return 0;}
        }
        g->scratch_used=0;
        g->hoisted[j]=1;
        g->stats.variant_count[78]++;
        *done=1;
        return run_command(g,j);
    }
    g->stats.variant_count[75]++;
    return 0;
}
int qdg_run(QdgGraph *g, const void *const *inputs, const uint64_t *input_bytes,
            void *const *outputs, const uint64_t *output_bytes, uint32_t first, uint32_t last) {
    if(last>g->ncommands)last=g->ncommands;
    g->inputs=inputs;g->input_bytes=input_bytes;g->outputs=outputs;g->output_bytes=output_bytes;
    g->error_detail=0;
    uint64_t start=now_ns();
    helpers_active(g,1);
    g->resident_key=0; /* another graph may have used the DSP since */
    memset(g->hoisted+first,0,last-first);
    if(first==0) {
        for(int p=1;p<QDG_POOLS;p++)if(p!=QDG_POOL_CONST && g->pool[p])memset(g->pool[p],0,g->pool_bytes[p]);
        for(int i=0;i<2;i++)memset(outputs[i],0,output_bytes[i]);
        g->stats.zero_ns+=now_ns()-start;
    }
    for(uint32_t i=first;i<last;i++) {
        g->scratch_used=0;
        int rc=0;
#ifdef QDG_DSP
        if(resident_cancelled()){rc=12;goto failed;}
#endif
        if(g->hoisted[i]) { if(i+1==last) rc=dsp_complete(g); if(rc) goto failed; continue; }
        if(g->inflight.active) {
            const uint64_t *mine=g->fp_conv_in[i]?g->fp_conv_in[i]:g->fp_all+(size_t)i*g->fp_words;
            if(fp_hit(g,mine,g->fp_conv_out[g->inflight.job->cmd])) {
                uint64_t d0=now_ns();
                int moved=0;
                if(g->backend==QDG_BACKEND_DSP && g->hoist) rc=hoist(g,i,last,&moved);
                if(!rc && !moved) rc=dsp_complete(g);
                else if(!rc && g->inflight.active && fp_hit(g,mine,g->fp_conv_out[g->inflight.job->cmd])) rc=dsp_complete(g);
                g->stats.variant_ns[0]+=now_ns()-d0;g->stats.variant_count[0]++; /* hazard drains */
            }
        }
        uint64_t t0=g->profile?now_ns():0;
        if(!rc) rc=run_command(g,i);
        if(!rc && i+1==last) rc=dsp_complete(g);
        if(rc) {
failed:
            if(g->inflight.active && rc!=QDG_ERR_TIMEOUT && rc!=QDG_ERR_WAKE && !dsp_wait(g,g->inflight.seq)) g->inflight.active=0;
            g->inflight.active=0;
            for(int k=0;k<2;k++)g->jobs[k].busy=0;
            helpers_active(g,0);
            g->stats.error_command=i;g->stats.error_detail=g->error_detail;
            g->stats.sequence=g->sequence;g->stats.persistent=g->persistent;
            return rc;
        }
        if(g->profile) {
            uint32_t v=g->c[i].variant;
            if(v<80){g->stats.variant_ns[v]+=now_ns()-t0;g->stats.variant_count[v]++;}
        }
    }
    helpers_active(g,0);
    g->stats.total_ns+=now_ns()-start;
    g->stats.sequence=g->sequence;g->stats.persistent=g->persistent;
    return 0;
}

/* -------------------------------------------------------------------- setup */
static uint32_t w32(const uint8_t *r, uint32_t i) {return qdr_u32(r+4*i);}
QdgGraph *qdg_create(const void *image, uint32_t bytes, uint8_t *const *pools, const uint64_t *pool_bytes, int *error) {
    QdrStatus status;
    QdgGraph *g=calloc(1,sizeof *g);
    *error=QDG_ERR_MEMORY;
    if(!g) return NULL;
    g->image=malloc(bytes+8);
    if(!g->image) goto fail;
    memcpy(g->image,image,bytes);
    *error=QDG_ERR_IMAGE;
    if(qdr_open(g->image,bytes,&g->img,&status)) goto fail;
    g->ntensors=g->img.count[QDR_TENSORS];g->ncommands=g->img.count[QDR_COMMANDS];
    for(int p=0;p<QDG_POOLS;p++){g->pool[p]=pools[p];g->pool_bytes[p]=pool_bytes[p];}
    *error=QDG_ERR_MEMORY;
    g->t=calloc(g->ntensors,sizeof *g->t);g->c=calloc(g->ncommands,sizeof *g->c);
    g->static_source=malloc(g->ntensors*4);g->static_fi=calloc(g->ntensors,1);
    g->conv=calloc(g->ncommands,sizeof *g->conv);
#if defined(QDG_DSP) && QDS_BLOCKED_GLUE
    g->blocked_lc=calloc(g->ncommands,sizeof *g->blocked_lc);
    g->blocked_lc_u8=calloc(g->ncommands,sizeof *g->blocked_lc_u8);
    g->blocked_sign=calloc(g->ncommands,sizeof *g->blocked_sign);
    g->blocked_add=calloc(g->ncommands,sizeof *g->blocked_add);
    if(!g->blocked_lc || !g->blocked_lc_u8 || !g->blocked_sign || !g->blocked_add)goto fail;
#if QDS_BLOCKED_S2D
    g->blocked_s2d=calloc(g->ncommands,sizeof *g->blocked_s2d);
    if(!g->blocked_s2d)goto fail;
#endif
#endif
    g->scratch_bytes=
#ifdef QDG_DSP
        (16u<<20)-4096;
#else
        64u<<20;
#endif
#ifdef QDG_DSP
    g->scratch=(uint8_t *)0xa4000000u;
#else
    g->scratch=malloc(g->scratch_bytes);
#endif
#ifdef QDG_DSP
    g->dsp_scratch=g->scratch;g->dsp_scratch_bytes=g->scratch_bytes;
#endif
#ifdef QDG_DSP
    g->staging=(uint8_t *)0xa5000000u;
#else
    g->staging=malloc(ARENA_BYTES);
#endif
    if(!g->t || !g->c || !g->static_source || !g->static_fi || !g->conv || !g->scratch || !g->staging) goto fail;
    const uint32_t *blocks=(const uint32_t *)(g->image+g->img.offset[QDR_BLOCKS]);
    const uint32_t *refs=(const uint32_t *)(g->image+g->img.offset[QDR_REFS]);
    const uint32_t *words=(const uint32_t *)(g->image+g->img.offset[QDR_WORDS]);
    *error=QDG_ERR_TENSOR;
    for(uint32_t i=0;i<g->ntensors;i++) {
        const uint8_t *r=qdr_record(&g->img,QDR_TENSORS,i);
        T *t=&g->t[i];
        t->id=w32(r,0);t->dtype=w32(r,1);t->cls=w32(r,2);t->flags=w32(r,3);
        t->esize=dsize(t->dtype);t->crouton=!!(t->flags&QDR_CROUTON);
        for(int j=0;j<4;j++){t->d[j]=(int32_t)w32(r,4+j);t->m[j]=(int32_t)w32(r,8+j);t->p[j]=(int32_t)w32(r,12+j);t->sv[j]=(int32_t)w32(r,26+j);}
        t->pool=w32(r,16);t->offset=w32(r,17);
        t->blocks=blocks+w32(r,18);t->nblocks=w32(r,19);t->slice=w32(r,20);
        t->zp=(int32_t)w32(r,22);
        uint64_t sb=(uint64_t)w32(r,23)|((uint64_t)w32(r,24)<<32);memcpy(&t->scale,&sb,8);
        t->scalar=w32(r,25);
#ifdef QDG_DSP
        if(count4(t->m)>0x20000000u || t->nblocks>0x10000000u || t->slice>0x10000000u)goto fail;
        for(int j=0;j<4;j++)if(t->m[j]>65535)goto fail;
#endif
        g->static_source[i]=QDR_NONE;
        if(t->flags&QDR_DIRECT) {
            if(t->pool>=QDG_POOLS || !g->pool[t->pool] ||
               (uint64_t)t->offset+count4(t->m)*t->esize>g->pool_bytes[t->pool]) {
                /* Direct storage outside the provided pools is rejected at use. */
                if(t->pool<QDG_POOLS && g->pool[t->pool]) goto fail;
            }
        }
        if(t->flags&QDR_BLOCKED) for(uint32_t b=0;b<t->nblocks;b++) {
            uint32_t e=t->blocks[b],p=e&127;
            if(p>=QDG_POOLS || !g->pool[p] || (uint64_t)(e&~127u)+2048>g->pool_bytes[p]) goto fail;
        }
    }
    *error=QDG_ERR_IMAGE;
    for(uint32_t i=0;i<g->ncommands;i++) {
        const uint8_t *r=qdr_record(&g->img,QDR_COMMANDS,i);
        C *c=&g->c[i];
        c->seq=w32(r,0);c->op=w32(r,1);c->variant=w32(r,2);c->io=w32(r,5);
        c->in=refs+w32(r,6);c->nin=w32(r,7);c->out=refs+w32(r,8);c->nout=w32(r,9);
        c->fv=words+w32(r,14);c->nfv=w32(r,15);
        c->calls=qdr_record(&g->img,QDR_CALL_TABLE,w32(r,16));c->ncalls=w32(r,17);
        if(c->ncalls && !c->calls) goto fail;
        if(c->op<1 || c->op>44) goto fail;
        /* Immutable conv inputs: weights_to_vtcm from the constant pool. */
        if(c->op==QDR_WEIGHTS_TO_VTCM && c->nin==1 && c->nout==1) {
            const T *s=&g->t[c->in[0]];
            if((s->flags&QDR_DIRECT) && s->pool==QDG_POOL_CONST) {
                if(is_class(s,"Fi","fi"))g->static_fi[c->out[0]]=1;
                else if(is_class(s,"Xwb","xwb") || is_class(s,"Xpb","xpb"))g->static_source[c->out[0]]=c->in[0];
            }
        }
    }
    *error=QDG_ERR_MEMORY;
#ifndef QDG_DSP
    if(build_footprints(g)) goto fail;
#else
    g->hoisted=calloc(g->ncommands,1);
    if(!g->hoisted)goto fail;
#endif
    g->timeout=10;
    g->hoist=1;
    *error=0;
    return g;
fail:
    qdg_destroy(g);
    return NULL;
}
void qdg_destroy(QdgGraph *g) {
    if(!g) return;
#if defined(QDG_DSP) && QDS_BLOCKED_GLUE
    if(g->blocked_lc)for(uint32_t i=0;i<g->ncommands;i++)if((uintptr_t)g->blocked_lc[i]>1)free(g->blocked_lc[i]);
    free(g->blocked_lc);
    if(g->blocked_lc_u8)for(uint32_t i=0;i<g->ncommands;i++)if((uintptr_t)g->blocked_lc_u8[i]>1)free(g->blocked_lc_u8[i]);
    free(g->blocked_lc_u8);
    if(g->blocked_sign)for(uint32_t i=0;i<g->ncommands;i++)if((uintptr_t)g->blocked_sign[i]>1)free(g->blocked_sign[i]);
    free(g->blocked_sign);
    if(g->blocked_add)for(uint32_t i=0;i<g->ncommands;i++)if((uintptr_t)g->blocked_add[i]>1)free(g->blocked_add[i]);
    free(g->blocked_add);
#if QDS_BLOCKED_S2D
    if(g->blocked_s2d)for(uint32_t i=0;i<g->ncommands;i++)if((uintptr_t)g->blocked_s2d[i]>1)free(g->blocked_s2d[i]);
    free(g->blocked_s2d);
#endif
#endif
    stop_helpers(g);
    for(uint32_t i=0;i<g->nweights;i++)free_weights(&g->weights[i]);
    if(g->conv)for(uint32_t i=0;i<g->ncommands;i++)free(g->conv[i].params);
    for(int k=0;k<2;k++){Job *j=&g->jobs[k];free(j->r);free(j->acc[0]);free(j->acc[1]);free(j->mult);free(j->bias);free(j->wsum);}
    if(g->fp_conv_in)for(uint32_t i=0;i<g->ncommands;i++){free(g->fp_conv_in[i]);free(g->fp_conv_out[i]);}
    free(g->fp_all);free(g->fp_conv_in);free(g->fp_conv_out);free(g->hoisted);
    free(g->weights);free(g->conv);free(g->t);free(g->c);free(g->static_source);free(g->static_fi);
#ifndef QDG_DSP
    free(g->scratch);free(g->staging);
#endif
    free(g->image);free(g);
}
void qdg_set_dsp(QdgGraph *g, uint8_t *arena, const volatile uint32_t *callback_word, uint32_t callback_value,
                 uint32_t sequence, uint32_t persistent, uint32_t raise_clock, uint32_t use_hmx,
                 double timeout, QdgWake wake) {
    g->backend=QDG_BACKEND_DSP;g->arena=arena;g->callback_word=callback_word;g->callback_value=callback_value;
    g->sequence=sequence;g->persistent=persistent;g->raise_clock=raise_clock;g->use_hmx=use_hmx;
    g->timeout=timeout;g->wake=wake;
}
void qdg_set_cpu(QdgGraph *g) {g->backend=QDG_BACKEND_CPU;g->arena=NULL;g->wake=NULL;}
void qdg_set_profile(QdgGraph *g, int enable) {g->profile=enable;}
void qdg_set_native_hmx(QdgGraph *g,int enable){g->native_hmx=enable;}
void qdg_set_hoist(QdgGraph *g, int enable) {g->hoist=enable;}
void qdg_set_block_output(QdgGraph *g, int enable) {g->block_output=!!enable;}
int qdg_commands(const QdgGraph *g) {return (int)g->ncommands;}
const QdgStats *qdg_stats(const QdgGraph *g) {return &g->stats;}
void qdg_reset_stats(QdgGraph *g) {
    uint32_t s=g->sequence,p=g->persistent;
    memset(&g->stats,0,sizeof g->stats);
#ifdef QDG_DSP
    g->dsp_direct_calls=0;g->dsp_read_cycles=g->dsp_write_cycles=0;memset(g->dsp_pack_cycles,0,sizeof g->dsp_pack_cycles);
    g->dsp_weights_cycles=g->dsp_quant_cycles=g->dsp_weights_miss_cycles=g->dsp_weights_misses=0;
    memset(g->dsp_conv_phase,0,sizeof g->dsp_conv_phase);
    g->dsp_dynamic_tcm_calls=0;
#if QDS_NATIVE_1X1
    memset(g->dsp_native_1x1_diag,0,sizeof g->dsp_native_1x1_diag);
#endif
#if QDS_NATIVE_3X3
    memset(g->dsp_native_3x3_diag,0,sizeof g->dsp_native_3x3_diag);
#endif
    memset(g->dsp_fallback_phase,0,sizeof g->dsp_fallback_phase);g->dsp_fallback_tcm_calls=g->dsp_fallback_l2_calls=0;
#if QDS_BLOCKED_GLUE
    g->dsp_blocked_lc_calls=g->dsp_blocked_lc_prepare=g->dsp_blocked_lc_apply=g->dsp_blocked_lc_u8_calls=0;
    g->dsp_blocked_sign_calls=g->dsp_blocked_sign_prepare=g->dsp_blocked_sign_apply=0;
    g->dsp_blocked_add_calls=g->dsp_blocked_add_prepare=g->dsp_blocked_add_apply=0;
    g->dsp_direct_input_pad_calls=0;
    g->dsp_direct_im2col_calls=0;
    g->dsp_direct_space_to_depth_calls=0;
#if QDS_BLOCKED_S2D
    memset(g->dsp_s2d_diag,0,sizeof g->dsp_s2d_diag);
#endif
#endif
#endif
    g->stats.sequence=s;g->stats.persistent=p;
}
