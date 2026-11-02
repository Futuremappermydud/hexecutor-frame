/* Exact dense ports of qd_graph.c's CPU-backend kernels; see prepared_reference.h.
 * Each kernel keeps the reference's statement order, rounding and checks. In a
 * dry pass, xfer only proves bounds, salloc only measures, and the arithmetic
 * blocks guarded by !g->dry are skipped. */
#include "prepared_reference.h"
#include "prepared_vcopy.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__HVX__) && __HVX_LENGTH__==128
#include <hexagon_types.h>
#include <hexagon_protos.h>
#endif

enum { DT_U8=1, DT_I8=2, DT_U16=3, DT_I16=4, DT_I32=5, DT_F32=6, DT_Q32=7 };
enum { F_SHAPE=1, F_BLOCKED=2, F_DIRECT=4, F_SV=8, F_SCALAR=16, F_QUANT=32, F_CROUTON=64 };
enum { E_TENSOR=2, E_SHAPE=3, E_ARGS=4, E_RANGE=5, E_UNSUPPORTED=6, E_MEMORY=7 };
enum { CALL_BYTES=88, BLOCK_BYTES=2048, CACHED_INPUTS=8, SPILL_POOL=6 };

static uint32_t reference_error_line;
#define FAIL(code) do { reference_error_line=__LINE__; return (code); } while(0)
#define CHECK(cond,code) do { if(!(cond)) FAIL(code); } while(0)
#define TRY(expr) do { int e_=(expr); if(e_) return e_; } while(0)

#if defined(__HVX__) && __HVX_LENGTH__==128
#define BLK_HVX 1
/* Bytes v[src..src+n) to d, with masked aligned stores; src+n<=128. */
static inline void store_part(uint8_t *d, HVX_Vector v, uint32_t src, uint32_t n) {
    uint32_t off=(uint32_t)((uintptr_t)d&127u),end=off+n;
    if(!off && !src && n==128){*(HVX_Vector *)d=v;return;}
    HVX_Vector *base=(HVX_Vector *)(d-off),r=Q6_V_vror_VR(v,(int32_t)((src-off)&127u));
    HVX_VectorPred q=Q6_Q_not_Q(Q6_Q_vsetq_R((int32_t)off));
    if(end<128) q=Q6_Q_and_QQ(q,Q6_Q_vsetq_R((int32_t)end));
    Q6_vmem_QRIV(q,base,r);
    if(end>128) Q6_vmem_QRIV(Q6_Q_vsetq_R((int32_t)(end-128)),base+1,r);
}
/* A vector whose bytes [0,n) are s[0..n), 0<n<=128. Only the aligned vectors
 * holding s[0] and s[n-1] are loaded, so no page past the buffer is touched. */
static inline HVX_Vector load_part(const uint8_t *s, uint32_t n) {
    uintptr_t a=(uintptr_t)s;
    const HVX_Vector *lo=(const HVX_Vector *)(a&~(uintptr_t)127);
    if(!(a&127u)) return *lo;
    const HVX_Vector *hi=(const HVX_Vector *)((a+n-1)&~(uintptr_t)127);
    return Q6_V_valign_VVR(*hi,*lo,(int32_t)(a&127u));
}
#else
#define BLK_HVX 0
#endif

typedef struct {
    uint32_t dtype, flags, esize, crouton, cls, scalar;
    int32_t d[4], m[4], p[4], sv[4];
    uint32_t pool, offset, nblocks, slice;
    const uint32_t *blocks;
    int32_t zp;
    double scale;
} T;

typedef struct { int32_t n[4]; uint32_t dt, scalar, unknown; void *p; } A;

/* Conv epilogue fused into the conv's output step (see prepared_fuse.h and
 * EXECUTOR.md): per output byte q (the conv's requantized u8 value),
 *   kind 1 (REQUANT_U8):  e = min((q*mc + rd) >> sh, 255)
 *   kind 2 (ADD_U8):      e = min((q*mc + b*mo + rd) >> sh, 255)
 * in unsigned 32-bit arithmetic that fuse_decode proves cannot overflow;
 * b is the other operand's byte at the same position. The epilogue's output
 * oe is written, the conv's own output only unless dead. eb/bb are the
 * scalar path's chunk buffers (workspace). */
typedef struct {
    int kind, dead;
    uint32_t mc, mo, rd, sh;
    T oe, ob;
    uint8_t *eb, *bb;
} Fuse;

typedef struct {
    const Nprg2ExecCall *call;
    int dry;
    uint8_t *scratch;
    size_t used, cap;
    uint8_t *pool[7];
    uint64_t pool_bytes[7], write_limit[7];
    uint32_t op, variant, io, nin, nout, nfv, ncalls;
    const uint8_t *calls, *fv;
    T in[CACHED_INPUTS], out;
    uint8_t measure_only;
    const Nprg2ReferenceConv *conv;
    size_t wused;
    Fuse fx;
    int fx_ok;
    uint32_t par_slots; /* usable helper near slots (Nprg2ReferenceConv.par_near) */
} G;

static uint32_t rd32(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint64_t rd64(const uint8_t *p){return (uint64_t)rd32(p)|((uint64_t)rd32(p+4)<<32);}
static int32_t rdi16(const uint8_t *p){return (int16_t)(uint16_t)(p[0]|p[1]<<8);}

/* ------------------------------------------------------------------ scratch */
static void *salloc(G *g, size_t bytes) {
    size_t at=(g->used+63)&~(size_t)63;
    if(at<g->used || bytes>SIZE_MAX-at || at+bytes>g->cap) return NULL;
    g->used=at+bytes;
    if(g->measure_only) return (void *)g; /* never dereferenced in a dry pass */
    return g->scratch+at;
}
static size_t count4(const int32_t *n) {return (size_t)n[0]*n[1]*n[2]*n[3];}
static uint32_t dsize(uint32_t dt) {return dt==DT_U8||dt==DT_I8?1:dt==DT_U16||dt==DT_I16?2:4;}
static int anew(G *g, A *a, const int32_t *n, uint32_t dt) {
    for(int i=0;i<4;i++){if(n[i]<0)FAIL(E_SHAPE);a->n[i]=n[i];}
    CHECK(count4(n)<=(SIZE_MAX>>3),E_SHAPE);
    a->dt=dt;a->scalar=0;a->unknown=0;
    a->p=salloc(g,count4(n)*dsize(dt)+64);
    return a->p?0:E_MEMORY;
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
static int to_i32(G *g, const A *a, int32_t **out) {
    size_t n=count4(a->n);
    int32_t *r=salloc(g,n*4+64);
    if(!r) FAIL(E_MEMORY);
    if(a->dt==DT_F32) FAIL(E_UNSUPPORTED);
    if(!g->dry) {
        if(a->dt==DT_U8){const uint8_t *s=a->p;for(size_t i=0;i<n;i++)r[i]=s[i];}
        else if(a->dt==DT_U16){const uint16_t *s=a->p;for(size_t i=0;i<n;i++)r[i]=s[i];}
        else for(size_t i=0;i<n;i++)r[i]=(int32_t)aget(a,i);
    }
    *out=r;return 0;
}
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
static int decode(const Nprg2TensorView *v, T *t) {
    const uint8_t *r=v->geometry_record;
    if(!r) return 0;
    memset(t,0,sizeof *t);
    t->dtype=r[0];t->flags=r[1]&~(uint32_t)(F_BLOCKED|F_DIRECT);
    t->cls=rd32(r+2);t->esize=dsize(t->dtype);t->crouton=!!(t->flags&F_CROUTON);
    /* The record payload is a union. A shape tensor whose QDRX source also
     * had a value/scalar payload loses those flags, so their users reject. */
    if(t->flags&F_SHAPE) {
        t->flags&=~(uint32_t)(F_SV|F_SCALAR);
        for(int i=0;i<4;i++){t->d[i]=rdi16(r+6+2*i);t->m[i]=rdi16(r+14+2*i);t->p[i]=rdi16(r+22+2*i);}
    } else if(t->flags&F_SV) {
        t->flags&=~(uint32_t)F_SCALAR;
        for(int i=0;i<4;i++)t->sv[i]=(int32_t)rd32(r+6+4*i);
    } else if(t->flags&F_SCALAR) t->scalar=rd32(r+6);
    t->zp=(int32_t)rd32(r+30);memcpy(&t->scale,r+34,8);
    if(v->storage_kind==1){t->flags|=F_DIRECT;t->pool=v->pool;t->offset=v->offset;}
    else if(v->storage_kind==2) {
        if(!v->blocks || !v->block_count) return 0;
        t->flags|=F_BLOCKED;t->blocks=v->blocks;t->nblocks=v->block_count;t->slice=v->slice_origin;
    }
    return 1;
}
static int input_at(G *g, uint32_t i, T *t) {
    uint32_t index;Nprg2TensorView v;
    if(i>=g->nin || nprg2_exec_input(g->call,i,&index,&v)!=NPRG2_EXEC_OK || !decode(&v,t)) FAIL(E_TENSOR);
    return 0;
}
static const T *in(G *g, uint32_t i) {return i<g->nin && i<CACHED_INPUTS?&g->in[i]:NULL;}
static const T *out0(G *g) {return &g->out;}
#define NEED(nin_min) CHECK(g->nin>=(nin_min) && (nin_min)<=CACHED_INPUTS && g->nout>=1,E_ARGS)

/* Every block a blocked transfer touches: absent entries read as zero and
 * are skipped on store, exactly as qd_graph's blocked_impl. */
static int blocked_check(G *g, const T *t, const int32_t *dims, const int32_t *origin, int store) {
    for(int i=0;i<4;i++)CHECK(origin[i]>=0 && dims[i]>=0,E_RANGE);
    if(!dims[0] || !dims[1] || !dims[2] || !dims[3]) return 0;
    CHECK(t->esize==1 || t->esize==2 || t->esize==4,E_TENSOR);
    int64_t cw=t->esize==1?8:t->esize==2?4:2;
    int64_t nbh=((int64_t)t->m[1]+7)/8,nbw=((int64_t)t->m[2]+cw-1)/cw,nbd=((int64_t)t->m[3]+31)/32;
    for(int64_t b=origin[0];b<(int64_t)origin[0]+dims[0];b++)
     for(int64_t h=origin[1]/8;h<=((int64_t)origin[1]+dims[1]-1)/8;h++)
      for(int64_t w=origin[2]/cw;w<=((int64_t)origin[2]+dims[2]-1)/cw;w++)
       for(int64_t z=origin[3]/32;z<=((int64_t)origin[3]+dims[3]-1)/32;z++) {
        int64_t index=((b*nbh+h)*nbw+w)*nbd+t->slice+z;
        if(index<0 || index>=(int64_t)t->nblocks) continue;
        uint32_t e=t->blocks[index],pool=e&127u;
        if(!pool) continue;
        CHECK(pool<7 && g->pool[pool],E_RANGE);
        uint64_t limit=store?g->write_limit[pool]:g->pool_bytes[pool];
        CHECK((uint64_t)(e&~127u)+BLOCK_BYTES<=limit,E_RANGE);
       }
    return 0;
}
#if BLK_HVX
/* u8 blocked_impl for the 8 pixels W..W+7 (W%8==0) of one row. In a block
 * they are 256 contiguous bytes per 32-channel group, 32 per pixel; values
 * holds them pixel-major with nd channels each. Whole groups move as vectors. */
static void blocked_run8(uint8_t *const *pools, const uint32_t *blocks, int64_t count, uint8_t *values,
                         int64_t wrow, size_t hin, int64_t nd, int64_t d0, int store) {
    for(int64_t d=0;d<nd;) {
        int64_t D=d+d0,n=32-D%32;
        if(n>nd-d)n=nd-d;
        int64_t index=wrow+D/32;
        uint32_t e=(index>=0 && index<count)?blocks[index]:0,pool=e&127;
        uint8_t *base=pool?pools[pool]:NULL,*v=values+d;
        if(!base) {
            if(!store)for(int k=0;k<8;k++)memset(v+k*nd,0,(size_t)n);
        } else {
            uint8_t *blk=base+(e&~127u)+hin;
            if(n==32 && nd==32) {
                if(store){store_part(blk,load_part(v,128),0,128);store_part(blk+128,load_part(v+128,128),0,128);}
                else {store_part(v,load_part(blk,128),0,128);store_part(v+128,load_part(blk+128,128),0,128);}
            } else if(n==32) {
                if(store)for(int k=0;k<8;k++)store_part(blk+32*k,load_part(v+k*nd,32),0,32);
                else {
                    HVX_Vector b0=load_part(blk,128),b1=load_part(blk+128,128);
                    for(int k=0;k<8;k++)store_part(v+k*nd,k<4?b0:b1,(uint32_t)(k%4)*32,32);
                }
            } else {
                size_t at=(size_t)(D%32);
                for(int k=0;k<8;k++) {
                    if(store)memcpy(blk+32*k+at,v+k*nd,(size_t)n);else memcpy(v+k*nd,blk+32*k+at,(size_t)n);
                }
            }
        }
        d+=n;
    }
}
/* u16 blocked_impl for the 4 pixels W..W+3 (W%4==0) of one row: per
 * 32-channel group a block row is two 128-byte pixel pairs, channel c of pixel
 * 2q+r at u16 index 2c+r of pair q (hin: the row's byte offset). values holds
 * the pixels pixel-major with nd u16 channels each. Whole groups move as
 * vectors (vdeal/vshuff of halfwords); partial groups element by element. */
static void blocked_run4_16(uint8_t *const *pools, const uint32_t *blocks, int64_t count, uint8_t *values,
                            int64_t wrow, size_t hin, int64_t nd, int64_t d0, int store) {
    const size_t ps=(size_t)nd*2;
    for(int64_t d=0;d<nd;) {
        int64_t D=d+d0,n=32-D%32;
        if(n>nd-d)n=nd-d;
        int64_t index=wrow+D/32;
        uint32_t e=(index>=0 && index<count)?blocks[index]:0,pool=e&127;
        uint8_t *base=pool?pools[pool]:NULL,*v=values+(size_t)d*2;
        if(!base) {
            if(!store)for(int k=0;k<4;k++)memset(v+k*ps,0,(size_t)n*2);
        } else {
            uint8_t *blk=base+(e&~127u)+hin;
            if(n==32) {
                for(int q=0;q<2;q++) {
                    uint8_t *pv=v+(size_t)(2*q)*ps,*bq=blk+128*q;
                    if(store) {
                        HVX_Vector a=load_part(pv,64),b=load_part(pv+ps,64);
                        HVX_Vector x=Q6_V_vmux_QVV(Q6_Q_vsetq_R(64),a,Q6_V_vror_VR(b,64));
                        *(HVX_Vector *)bq=Q6_Vh_vshuff_Vh(x);
                    } else {
                        HVX_Vector x=Q6_Vh_vdeal_Vh(*(const HVX_Vector *)bq);
                        store_part(pv,x,0,64);store_part(pv+ps,x,64,64);
                    }
                }
            } else {
                for(int k=0;k<4;k++)for(int64_t c=0;c<n;c++) {
                    uint8_t *ptr=blk+(size_t)(k>>1)*128+((size_t)(D%32+c)*2+(size_t)(k&1))*2,*val=v+k*ps+(size_t)c*2;
                    if(store)memcpy(ptr,val,2);else memcpy(val,ptr,2);
                }
            }
        }
        d+=n;
    }
}
#endif
static inline __attribute__((always_inline)) void blocked_impl(uint8_t *const *pools, const T *t, uint8_t *values,
        const int32_t *dims, const int32_t *origin, const uint32_t size, const int store) {
    const int64_t cw=size==1?8:size==2?4:2;
    int64_t maxh=t->m[1],maxw=t->m[2],maxd=t->m[3],slice=t->slice,count=t->nblocks;
    int64_t nbh=(maxh+7)/8,nbw=(maxw+cw-1)/cw,nbd=(maxd+31)/32;
    const uint32_t *blocks=t->blocks;
    size_t pos=0;
    for(int64_t b=0;b<dims[0];b++)for(int64_t h=0;h<dims[1];h++) {
        int64_t B=b+origin[0],H=h+origin[1];
        int64_t hrow=(B*nbh+H/8)*nbw;
        size_t hin=size==1?(H%8)*256:size==2?(H%8)*128:(H%8)*64;
        for(int64_t w=0;w<dims[2];w++) {
            int64_t W=w+origin[2];
            int64_t wrow=(hrow+W/cw)*nbd+slice;
#if BLK_HVX
            if(size==1 && W%8==0 && dims[2]-w>=8) {
                blocked_run8(pools,blocks,count,values+pos,wrow,hin,dims[3],origin[3],store);
                pos+=(size_t)8*dims[3];w+=7;
                continue;
            }
            if(size==2 && W%4==0 && dims[2]-w>=4) {
                blocked_run4_16(pools,blocks,count,values+pos,wrow,(size_t)(H%8)*256,dims[3],origin[3],store);
                pos+=(size_t)8*dims[3];w+=3;
                continue;
            }
#endif
            size_t win=hin+(size==1?(W%8)*32:size==2?((W%4)/2)*64+W%2:(W%2)*32);
            for(int64_t d=0;d<dims[3];) {
                int64_t D=d+origin[3],n=32-D%32;
                if(n>dims[3]-d)n=dims[3]-d;
                int64_t index=wrow+D/32;
                uint32_t e=(index>=0 && index<count)?blocks[index]:0,pool=e&127;
                uint8_t *base=pool?pools[pool]:NULL;
                if(base) {
                    size_t within=win+(size==2?(D%32)*2:D%32);
                    uint8_t *ptr=base+(e&~127u)+within*size;
                    if(size!=2) {
                        if(store)memcpy(ptr,values+pos,n*size);else memcpy(values+pos,ptr,n*size);
                    } else {
                        uint16_t v16;
                        for(int64_t c=0;c<n;c++) {
                            if(store){memcpy(&v16,values+pos+2*c,2);memcpy(ptr+4*c,&v16,2);}
                            else {memcpy(&v16,ptr+4*c,2);memcpy(values+pos+2*c,&v16,2);}
                        }
                    }
                } else if(!store)memset(values+pos,0,n*size);
                pos+=n*size;d+=n;
            }
        }
    }
}
static int direct_check(G *g, const T *t, const int32_t *dims, const int32_t *origin, int store) {
    for(int i=0;i<4;i++)CHECK(origin[i]>=0 && dims[i]>=0 && (int64_t)origin[i]+dims[i]<=t->m[i],E_RANGE);
    CHECK(t->pool<7 && g->pool[t->pool],E_TENSOR);
    if(!dims[0] || !dims[1] || !dims[2] || !dims[3]) return 0;
    uint64_t es=t->esize;
    uint64_t first=((((uint64_t)origin[0]*t->m[1]+origin[1])*t->m[2]+origin[2])*t->m[3]+origin[3]);
    uint64_t last=((((uint64_t)(origin[0]+dims[0]-1)*t->m[1]+origin[1]+dims[1]-1)*t->m[2]+origin[2]+dims[2]-1)*t->m[3]+origin[3]+dims[3]-1);
    uint64_t limit=store?g->write_limit[t->pool]:g->pool_bytes[t->pool];
    (void)first;
    CHECK((uint64_t)t->offset+(last+1)*es<=limit,E_RANGE);
    return 0;
}
static int xfer_check(G *g, const T *t, const int32_t *dims, const int32_t *origin, int store) {
    if(t->flags&F_DIRECT) return direct_check(g,t,dims,origin,store);
    if(t->flags&F_BLOCKED) return blocked_check(g,t,dims,origin,store);
    FAIL(E_TENSOR);
}
/* The data movement of xfer for a region xfer_check accepted. */
static void xfer_copy(G *g, const T *t, void *values, const int32_t *dims, const int32_t *origin, int store) {
    if(t->flags&F_DIRECT) {
        /* Rows that are adjacent in the tensor (the region spans the whole of
         * the faster dimensions) move as one run; a run goes through
         * qd_vcopy (memmove below 512 bytes and off Hexagon). */
        uint8_t *base=g->pool[t->pool];
        size_t es=t->esize,row=(size_t)dims[3]*es,pos=0;
        int32_t n0=dims[0],n1=dims[1],n2=dims[2];
        if(dims[3]==t->m[3]) {
            row*=(size_t)n2;n2=1;
            if(dims[2]==t->m[2]) {
                row*=(size_t)n1;n1=1;
                if(dims[1]==t->m[1]){row*=(size_t)n0;n0=1;}
            }
        }
        for(int32_t b=0;b<n0;b++)for(int32_t h=0;h<n1;h++)for(int32_t w=0;w<n2;w++) {
            size_t lin=(((size_t)(b+origin[0])*t->m[1]+h+origin[1])*t->m[2]+w+origin[2])*t->m[3]+origin[3];
            uint8_t *ptr=base+t->offset+lin*es;
            if(store)qd_vcopy(ptr,(uint8_t *)values+pos,row);else qd_vcopy((uint8_t *)values+pos,ptr,row);
            pos+=row;
        }
        return;
    }
    switch(t->esize*2+!!store) {
    case 2: blocked_impl(g->pool,t,values,dims,origin,1,0);break;
    case 3: blocked_impl(g->pool,t,values,dims,origin,1,1);break;
    case 4: blocked_impl(g->pool,t,values,dims,origin,2,0);break;
    case 5: blocked_impl(g->pool,t,values,dims,origin,2,1);break;
    case 8: blocked_impl(g->pool,t,values,dims,origin,4,0);break;
    default: blocked_impl(g->pool,t,values,dims,origin,4,1);break;
    }
}
static int xfer(G *g, const T *t, void *values, const int32_t *dims, const int32_t *origin, int store) {
    TRY(xfer_check(g,t,dims,origin,store));
    if(!g->dry) xfer_copy(g,t,values,dims,origin,store);
    return 0;
}
static int t_read(G *g, const T *t, A *a, int full) {
    static const int32_t zero[4]={0,0,0,0};
    CHECK(t->flags&F_SHAPE,E_TENSOR);
    TRY(anew(g,a,full?t->m:t->d,t->dtype));
    return xfer(g,t,a->p,a->n,full?zero:t->p,0);
}
static int t_read_region(G *g, const T *t, A *a, const int32_t *dims, const int32_t *origin) {
    TRY(anew(g,a,dims,t->dtype));
    return xfer(g,t,a->p,a->n,origin,0);
}
/* Memory.write: crouton depth extends to its 32-lane boundary, a narrower 4-D
 * value is zero padded in depth, then broadcast. A dry value whose shape
 * depends on non-constant live data (unknown) takes the broadcast path's
 * scratch bound. */
static int t_write(G *g, const T *t, const A *v) {
    int32_t dims[4];
    CHECK(t->flags&F_SHAPE,E_TENSOR);
    CHECK(v->dt==t->dtype || (dsize(v->dt)==t->esize && v->dt!=DT_F32 && t->dtype!=DT_F32),E_TENSOR);
    memcpy(dims,t->d,sizeof dims);
    if(t->crouton) {
        int32_t e=(t->d[3]+t->p[3]+31)/32*32;
        if(e>t->m[3])e=t->m[3];
        dims[3]=e-t->p[3];
    }
    int32_t vn[4];memcpy(vn,v->n,sizeof vn);
    int pad=!v->scalar && vn[3]<dims[3];
    int same=!pad;
    if(v->unknown) same=0;
    else for(int i=0;i<4;i++) {
        int32_t have=(pad && i==3)?dims[3]:vn[i];
        if(have!=dims[i] && have!=1) FAIL(E_SHAPE);
        if(have!=dims[i]) same=0;
    }
    if(same) return xfer(g,t,v->p,dims,t->p,1);
    A full;TRY(anew(g,&full,dims,t->dtype));
    if(!g->dry) {
        size_t es=t->esize,row=(size_t)dims[3]*es;
        uint8_t *dst=full.p;
        for(int32_t b=0;b<dims[0];b++)for(int32_t h=0;h<dims[1];h++)for(int32_t w=0;w<dims[2];w++,dst+=row) {
            int32_t ib=vn[0]==1?0:b,ih=vn[1]==1?0:h,iw=vn[2]==1?0:w;
            const uint8_t *src=(const uint8_t *)v->p+((((size_t)ib*vn[1]+ih)*vn[2]+iw)*vn[3])*es;
            if(pad){memcpy(dst,src,(size_t)vn[3]*es);memset(dst+(size_t)vn[3]*es,0,row-(size_t)vn[3]*es);}
            else if(vn[3]==dims[3])memcpy(dst,src,row);
            else for(int32_t d=0;d<dims[3];d++)memcpy(dst+d*es,src,es);
        }
    }
    return xfer(g,t,full.p,dims,t->p,1);
}
static int t_write_region(G *g, const T *t, const A *v, const int32_t *origin) {
    CHECK(dsize(v->dt)==t->esize,E_TENSOR);
    return xfer(g,t,v->p,v->n,origin,1);
}
static int read_scalar(G *g, const T *t, int64_t *value) {
    A a;TRY(t_read(g,t,&a,0));
    CHECK(count4(a.n)>0 && a.dt!=DT_F32,E_TENSOR);
    *value=g->dry?0:aget(&a,0);return 0;
}
static int read_i32(G *g, const T *t, int32_t **values, size_t *n) {
    A a;TRY(t_read(g,t,&a,0));
    CHECK(a.dt==DT_I32 || a.dt==DT_Q32,E_TENSOR);
    *values=a.p;*n=count4(a.n);return 0;
}
/* Element i of a direct constant (pool 5) tensor's dense region, read as
 * aget would. The constant pool is immutable, so a dry pass may use it to
 * size data-dependent results exactly. */
static int const_value(const G *g, const T *t, size_t i, int64_t *v) {
    if(!(t->flags&F_DIRECT) || !(t->flags&F_SHAPE) || t->pool!=5 || !g->pool[5] || t->dtype==DT_F32) return 0;
    for(int k=0;k<4;k++)if(t->d[k]<0 || t->p[k]<0 || (int64_t)t->p[k]+t->d[k]>t->m[k]) return 0;
    if(i>=count4(t->d)) return 0;
    size_t c[4];
    for(int k=3;k>=0;k--){c[k]=i%(size_t)t->d[k]+(size_t)t->p[k];i/=(size_t)t->d[k];}
    uint64_t lin=(((uint64_t)c[0]*t->m[1]+c[1])*t->m[2]+c[2])*t->m[3]+c[3];
    if((uint64_t)t->offset+(lin+1)*t->esize>g->pool_bytes[5]) return 0;
    uint32_t raw=0;
    memcpy(&raw,g->pool[5]+t->offset+lin*t->esize,t->esize);
    A a={.dt=t->dtype,.p=&raw};
    *v=aget(&a,0);
    return 1;
}
static int sv_or_scalar(const T *t, int64_t *v) {
    if(t->flags&F_SCALAR){*v=t->dtype==DT_F32?0:(int32_t)t->scalar;return t->dtype==DT_F32;}
    return 1;
}
static int shape_values_nonnegative(const T *t) {
    return (t->flags&F_SV) && t->sv[0]>=0 && t->sv[1]>=0 && t->sv[2]>=0 && t->sv[3]>=0;
}

/* ---------------------------------------------------------------- arguments */
static int call_args(G *g, uint32_t func, int64_t *args, uint32_t used) {
    const uint8_t *found=NULL;
    for(uint32_t i=0;i<g->ncalls;i++) {
        const uint8_t *r=g->calls+(size_t)i*CALL_BYTES;
        if(rd32(r)==func){CHECK(!found,E_ARGS);found=r;}
    }
    CHECK(found,E_ARGS);
    uint32_t mask=rd32(found+4);
    for(uint32_t j=0;j<10;j++) {
        uint64_t v=rd64(found+8+8*j);
        if(used&(1u<<j)) CHECK(!(mask&(1u<<j)),E_ARGS);
        args[j]=(int64_t)v;
    }
    return 0;
}
static uint32_t fv(const G *g, uint32_t i) {return rd32(g->fv+(size_t)i*4u);}
static int shift_ok(int64_t s) {return s>=0 && s<=48;}
static int small(const int64_t *r, uint32_t mask) {
    for(int j=0;j<10;j++)if((mask&(1u<<j)) && (uint64_t)r[j]>=(1ull<<40))return 0;
    return 1;
}

#if defined(DM_SHAPES) || defined(CONV_SHAPES)
#include <stdio.h>
#endif
#ifdef DM_SHAPES
static void shp_t(const char *n,const T *t){if(t->flags&F_SV){fprintf(stderr,"  %s sv%d,%d,%d,%d\n",n,t->sv[0],t->sv[1],t->sv[2],t->sv[3]);return;}fprintf(stderr,"  %s fl%u es%u cr%u d%d,%d,%d,%d m%d,%d,%d,%d p%d,%d,%d,%d sl%u nb%u pool%u off%u\n",n,t->flags,t->esize,t->crouton,t->d[0],t->d[1],t->d[2],t->d[3],t->m[0],t->m[1],t->m[2],t->m[3],t->p[0],t->p[1],t->p[2],t->p[3],t->slice,t->nblocks,t->pool,t->offset);}
#define SHP(name,g,np) do{ if(!(g)->dry){fprintf(stderr,"SHP %s op=%u var=%u\n",name,(g)->op,(g)->variant); for(uint32_t q_=0;q_<(g)->nin&&q_<CACHED_INPUTS;q_++)shp_t("in",&(g)->in[q_]); shp_t("out",&(g)->out);} }while(0)
#else
#define SHP(name,g,np) ((void)0)
#endif
/* Exact direct-copy fast paths for the data-movement kernels below. */
#include "prepared_datamove.inc"

/* ------------------------------------------------------------------ kernels */
static int k_input_slice_pad(G *g) {
    CHECK(g->nin==7 && g->nout==1 && g->io<2 && g->io<g->call->input_count,E_ARGS);
    const T *pb=in(g,0),*st=in(g,1),*os=in(g,2),*fl=in(g,3),*is=in(g,6),*o=out0(g);
    CHECK((pb->flags&st->flags&os->flags&is->flags&F_SV),E_ARGS);
    CHECK(shape_values_nonnegative(os) && shape_values_nonnegative(is),E_ARGS);
    int64_t fill=0;
    if(fl->flags&F_SCALAR)CHECK(!sv_or_scalar(fl,&fill),E_ARGS);
    else CHECK(fl->flags&F_SV && !fl->sv[0] && !fl->sv[1] && !fl->sv[2] && !fl->sv[3],E_ARGS);
    const Nprg2ExecBuffer *input=&g->call->inputs[g->io];
    const uint8_t *img=input->data;
    CHECK(img && o->dtype==DT_U8 && input->bytes==count4(is->sv),E_ARGS);
    A a;TRY(anew(g,&a,os->sv,DT_U8));
    if(!g->dry) {
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
    }
    return t_write(g,o,&a);
}
static int k_input_slice(G *g) {
    CHECK(g->nout==1 && (g->io==2 || g->io==3) && g->io<g->call->input_count,E_ARGS);
    const T *o=out0(g);
    const Nprg2ExecBuffer *input=&g->call->inputs[g->io];
    CHECK(o->flags&F_SHAPE,E_TENSOR);
    for(int i=0;i<4;i++)CHECK(o->d[i]>=0,E_SHAPE);
    CHECK(input->data && o->dtype==DT_F32 && input->bytes==count4(o->d)*4,E_ARGS);
    A a={{o->d[0],o->d[1],o->d[2],o->d[3]},DT_F32,0,0,(void *)input->data};
    return t_write(g,o,&a);
}
static int k_weights_to_vtcm(G *g) {
    CHECK(g->nin==1 && g->nout==1,E_ARGS);
    const T *s=in(g,0),*d=out0(g);
    CHECK((s->flags&d->flags&F_DIRECT) && s->pool<7 && d->pool<7 && g->pool[s->pool] && g->pool[d->pool],E_TENSOR);
    for(int i=0;i<4;i++)CHECK(s->m[i]>=0 && d->m[i]>=0,E_SHAPE);
    size_t ns=count4(s->m)*s->esize,nd=count4(d->m)*d->esize,n=ns<nd?ns:nd;
    CHECK(s->offset+(uint64_t)n<=g->pool_bytes[s->pool] && d->offset+(uint64_t)n<=g->write_limit[d->pool],E_RANGE);
    if(!g->dry)qd_vcopy(g->pool[d->pool]+d->offset,g->pool[s->pool]+s->offset,n);
    return 0;
}
static int k_im2col(G *g) {
    SHP("im2col",g,0);
    CHECK(g->nin==3 && g->nout==1,E_ARGS);
    const T *i=in(g,0),*k=in(g,1),*s=in(g,2),*o=out0(g);
    CHECK(k->flags&s->flags&F_SV,E_ARGS);
    {
        int fr=im2col_u8_fast(g,i,o,k->sv[0],k->sv[1],k->sv[2],s->sv[1],s->sv[2]);
        if(fr<0)fr=im2col_u16_fast(g,i,o,k->sv[0],k->sv[1],k->sv[2],s->sv[1],s->sv[2]);
        if(fr>=0) return fr;
    }
    A x;TRY(t_read(g,i,&x,0));
    CHECK(x.n[0]==1,E_SHAPE);
    int32_t kh=k->sv[0],kw=k->sv[1],cin=k->sv[2],sh=s->sv[1],sw=s->sv[2],oh=o->d[1],ow=o->d[2];
    CHECK(sh>0 && sw>0 && cin<=x.n[3] && kh>0 && kw>0 && cin>=0,E_ARGS);
    CHECK(oh>=0 && ow>=0,E_SHAPE);
    CHECK((kh-1)+sh*(oh-1)<x.n[1] && (kw-1)+sw*(ow-1)<x.n[2],E_SHAPE);
    int32_t depth=kh*kw*cin,keep=o->d[3]<depth?o->d[3]:depth;
    int32_t n[4]={1,oh,ow,keep};
    A r;TRY(anew(g,&r,n,x.dt));
    if(!g->dry) {
        size_t es=dsize(x.dt);
        for(int32_t h=0;h<oh;h++)for(int32_t w=0;w<ow;w++) {
            uint8_t *dst=(uint8_t *)r.p+((size_t)h*ow+w)*keep*es;
            int32_t col=0;
            for(int32_t y=0;y<kh && col<keep;y++)for(int32_t xx=0;xx<kw && col<keep;xx++) {
                int32_t nn=cin<keep-col?cin:keep-col;
                size_t si=((size_t)(y+sh*h)*x.n[2]+(xx+sw*w))*x.n[3];
                if(es==1){const uint8_t *src=(const uint8_t *)x.p+si;for(int32_t ch=0;ch<nn;ch++)dst[col+ch]=src[ch];}
                else memcpy(dst+col*es,(const uint8_t *)x.p+si*es,(size_t)nn*es);
                col+=nn;
            }
        }
    }
    return t_write(g,o,&r);
}
static int k_space_to_depth(G *g) {
    CHECK(g->nin>=1 && g->nout==1,E_ARGS);
    const T *i=in(g,0),*o=out0(g);
    int32_t ph=i->p[1],pw=i->p[2],oh=o->d[1],oc=o->d[3],ci=i->m[3];
    int32_t ow=(i->m[2]-pw)/2;
    if(o->m[2]<ow)ow=o->m[2];
    CHECK(i->dtype==DT_U8 && i->m[0]>=1 && oh>0 && ow>0 && ph+2*(oh-1)+1<i->m[1] && pw+2*(ow-1)<i->m[2] && pw>=1,E_SHAPE);
    {int fr=s2d_fast(g,i,o,ph,pw,oh,ow,ci,oc);if(fr>=0)return fr;}
    int32_t rd[4]={1,2*oh,2*ow,ci},ro[4]={0,ph,pw-1,0};
    A xf;TRY(t_read_region(g,i,&xf,rd,ro));
    ph=0;pw=1;
    int32_t n[4]={1,oh,ow,oc};
    A r;TRY(anew(g,&r,n,DT_U8));
    for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++)for(int32_t cb=0;cb<(ci+31)/32;cb++) {
        int32_t j=cb*4+2*dy+dx;
        if(32*j>=oc)continue;
        int32_t width=32;
        if(32*j+width>oc)width=oc-32*j;
        int32_t src_width=ci-32*cb<32?ci-32*cb:32;
        CHECK(width==src_width,E_SHAPE);
    }
    if(!g->dry) {
        memset(r.p,0,count4(n));
        const uint8_t *s=xf.p;uint8_t *d=r.p;
        for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++)for(int32_t cb=0;cb<(ci+31)/32;cb++) {
            int32_t j=cb*4+2*dy+dx;
            if(32*j>=oc)continue;
            int32_t width=32;
            if(32*j+width>oc)width=oc-32*j;
            for(int32_t h=0;h<oh;h++)for(int32_t w=0;w<ow;w++)
                memcpy(d+((size_t)h*ow+w)*oc+32*j,s+((size_t)(ph+2*h+dy)*xf.n[2]+(pw+2*w-dx))*ci+32*cb,width);
        }
    }
    static const int32_t zero[4]={0,0,0,0};
    return t_write_region(g,o,&r,zero);
}

/* Elementwise helpers: binary broadcasting as NumPy for two 4-D arrays. */
static int bshape(const A *a, const A *b, int32_t *n) {
    for(int i=0;i<4;i++) {
        if(a->n[i]==b->n[i] || b->n[i]==1)n[i]=a->n[i];
        else if(a->n[i]==1)n[i]=b->n[i];
        else return E_SHAPE;
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
#define BINARY_BODY(F) do { \
    int32_t n[4];TRY(bshape(a,b,n)); \
    int32_t *ia,*ib;TRY(to_i32(g,a,&ia));TRY(to_i32(g,b,&ib)); \
    A r;TRY(anew(g,&r,n,odt)); \
    if(!g->dry) { \
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
    } \
    return t_write(g,o,&r); } while(0)
#define DEFINE_BINARY(NAME,F) static int NAME(G *g, const A *a, const A *b, const int64_t *k, uint32_t odt, const T *o) { BINARY_BODY(F); }
DEFINE_BINARY(b_add_u8,f_add_u8)
DEFINE_BINARY(b_add_u16,f_add_u16)
DEFINE_BINARY(b_addsub_u16,f_addsub_u16)
DEFINE_BINARY(b_add1111,f_add1111)
DEFINE_BINARY(b_mul_u16,f_mul_u16)
DEFINE_BINARY(b_addsub_flat,f_addsub_flat)

static int read2(G *g, A *a, A *b) {
    NEED(2);
    TRY(t_read(g,in(g,0),a,0));
    return t_read(g,in(g,1),b,0);
}
static int k_add_u8(G *g) {
    int64_t r[10],k[4];
    TRY(call_args(g,QDR_CALL_ELEMENTWISE_ADD_CROUTON_U8,r,0x1f0));
    CHECK(r[7]==0 && small(r,0x70),E_UNSUPPORTED);
    k[0]=r[4];k[1]=r[5];k[2]=r[6]+8;
    CHECK(shift_ok(k[2]) && k[2]>=1,E_ARGS);
    A a,b;TRY(read2(g,&a,&b));
    if(!memcmp(a.n,b.n,16) && a.dt==DT_U8 && b.dt==DT_U8 && k[0]>=0 && k[1]>=0 && k[2]<=31 &&
       ((uint64_t)k[0]+(uint64_t)k[1])*255+(1ull<<(k[2]-1))<=0xffffffffull) {
        A o;TRY(anew(g,&o,a.n,DT_U8));
        if(!g->dry) {
            uint32_t ma=(uint32_t)k[0],mb=(uint32_t)k[1],rd=1u<<(k[2]-1),s=(uint32_t)k[2];
            const uint8_t *pa=a.p,*pb=b.p;uint8_t *po=o.p;
            for(size_t i=0,n=count4(a.n);i<n;i++){uint32_t v=(pa[i]*ma+pb[i]*mb+rd)>>s;po[i]=(uint8_t)(v>255?255:v);}
        }
        return t_write(g,out0(g),&o);
    }
    return b_add_u8(g,&a,&b,k,DT_U8,out0(g));
}
static int k_add_u16(G *g) {
    int64_t r[10],k[3];
    TRY(call_args(g,QDR_CALL_ELEMENTWISE_ADD_CROUTON_U16,r,0x1f0));
    CHECK(r[7]==0 && small(r,0x70) && shift_ok(r[6]),E_ARGS);
    k[0]=r[4];k[1]=r[5];k[2]=r[6];
    A a,b;TRY(read2(g,&a,&b));
    return b_add_u16(g,&a,&b,k,DT_U16,out0(g));
}
static int k_addsub_u16(G *g) {
    int64_t r[10],k[4];
    TRY(call_args(g,QDR_CALL_ELEMENTWISE_ADDSUB_CROUTON_U16,r,0x70));
    const T *o=out0(g);
    CHECK(small(r,0x70) && shift_ok(r[6]+1),E_ARGS);
    k[0]=r[4];k[1]=r[5];k[2]=r[6]+1;k[3]=((int64_t)o->zp<<k[2])+((int64_t)1<<r[6])+1;
    A a,b;TRY(read2(g,&a,&b));
    return b_addsub_u16(g,&a,&b,k,DT_U16,o);
}
static int k_add1111_u16(G *g) {
    int64_t r[10],k[4],s;
    NEED(2);
    TRY(call_args(g,QDR_CALL_ELEMENTWISE_ADD1111_CROUTON_U16,r,0xf0));
    CHECK(small(r,0x70) && shift_ok(r[6]+1),E_ARGS);
    k[0]=r[4];k[1]=r[5];k[2]=r[6]+1;k[3]=2*s32((uint64_t)r[7])+((int64_t)1<<r[6]);
    A x;TRY(t_read(g,in(g,0),&x,0));
    TRY(read_scalar(g,in(g,1),&s));
    int32_t sv=(int32_t)s;
    A b={{1,1,1,1},DT_I32,1,0,&sv};
    return b_add1111(g,&x,&b,k,DT_U16,out0(g));
}
static int k_mul_u16(G *g) {
    int64_t r[10],k[5];
    TRY(call_args(g,QDR_CALL_ELEMENTWISE_MUL_CROUTON_U16,r,0x1f0));
    k[0]=r[4];k[1]=r[5];k[2]=r[6];k[3]=r[7];k[4]=r[8];
    A a,b;TRY(read2(g,&a,&b));
    return b_mul_u16(g,&a,&b,k,DT_U16,out0(g));
}
static int k_mul1111_u16(G *g) {
    int64_t r[10],k[5],s;
    NEED(2);
    TRY(call_args(g,QDR_CALL_ELEMENTWISE_MUL1111_CROUTON_U16,r,0x1f0));
    k[0]=r[4];k[1]=r[5];k[2]=r[6];k[3]=r[7];k[4]=r[8];
    A x;TRY(t_read(g,in(g,0),&x,0));
    TRY(read_scalar(g,in(g,1),&s));
    if(!g->dry)CHECK(s>=INT32_MIN && s<=INT32_MAX,E_ARGS);
    int32_t sv=(int32_t)s;
    A b={{1,1,1,1},DT_I32,1,0,&sv};
    return b_mul_u16(g,&x,&b,k,DT_U16,out0(g));
}
static int k_mul_inline(G *g) {
    CHECK(g->nin==3,E_ARGS);
    NEED(3);
    int32_t *p;size_t np;int64_t s,k[5]={0,0,0,0,0};
    TRY(read_i32(g,in(g,2),&p,&np));
    CHECK(np>=6,E_ARGS);
    if(!g->dry) {
        uint64_t z=(uint32_t)p[3];
        k[0]=p[2];k[1]=(int64_t)(z&65535);k[2]=(int64_t)(z>>16);k[3]=p[4];k[4]=p[5];
    }
    A x;TRY(t_read(g,in(g,0),&x,0));
    TRY(read_scalar(g,in(g,1),&s));
    if(!g->dry)CHECK(s>=INT32_MIN && s<=INT32_MAX,E_ARGS);
    int32_t sv=(int32_t)s;
    A b={{1,1,1,1},DT_I32,1,0,&sv};
    return b_mul_u16(g,&x,&b,k,DT_U16,out0(g));
}
static int k_addsub_flat(G *g) {
    CHECK(g->ncalls>=1,E_ARGS);
    uint32_t fn=rd32(g->calls);
    CHECK(fn==QDR_CALL_ELEMENTWISE_SUB_FLAT_U16 || fn==QDR_CALL_ELEMENTWISE_ADD_FLAT_U16,E_ARGS);
    int64_t r[10],k[5];
    uint32_t mask=rd32(g->calls+4);
    for(int j=0;j<10;j++)r[j]=(int64_t)rd64(g->calls+8+8*j);
    CHECK(!(mask&0xf0) && small(r,0x70) && shift_ok(r[6]+1),E_ARGS);
    k[0]=r[4];k[1]=r[5];k[2]=r[6];k[3]=s32((uint64_t)r[7]);k[4]=fn==QDR_CALL_ELEMENTWISE_SUB_FLAT_U16?-1:1;
    A a,b;TRY(read2(g,&a,&b));
    return b_addsub_flat(g,&a,&b,k,DT_U16,out0(g));
}
static int k_addsub_inline(G *g) {
    CHECK(g->nin==3,E_ARGS);
    NEED(3);
    int32_t *p;size_t np;int64_t k[4]={0,0,1,0};
    TRY(read_i32(g,in(g,2),&p,&np));
    CHECK(np>=5,E_ARGS);
    const T *o=out0(g);
    if(!g->dry) {
        int64_t sh=(int64_t)p[4]>>16;
        CHECK(shift_ok(sh+1),E_ARGS);
        k[0]=(int64_t)p[1]>>16;k[1]=p[2];k[2]=sh+1;k[3]=((int64_t)o->zp<<(sh+1))+((int64_t)1<<sh)+1;
    }
    A a,b;TRY(read2(g,&a,&b));
    return b_addsub_u16(g,&a,&b,k,DT_U16,o);
}
static int k_requant_u8(G *g) {
    int64_t r[10];
    NEED(1);
    TRY(call_args(g,QDR_CALL_REQUANTIZE_CROUTON_U8,r,0x38));
    CHECK(r[4]==0 && r[5]==0 && small(r,0x8),E_UNSUPPORTED);
    int64_t m=r[3]&0xffff,s=15-(r[3]>>24);
    CHECK(s>=1 && s<=31,E_ARGS);
    A x;TRY(t_read(g,in(g,0),&x,0));
    CHECK(x.dt==DT_U8,E_TENSOR);
    A o;TRY(anew(g,&o,x.n,DT_U8));
    if(!g->dry) {
        uint32_t mm=(uint32_t)m,rd=1u<<(s-1),ss=(uint32_t)s;
        const uint8_t *px=x.p;uint8_t *po=o.p;
        for(size_t i=0,n=count4(x.n);i<n;i++){uint32_t v=(px[i]*mm+rd)>>ss;po[i]=(uint8_t)(v>255?255:v);}
    }
    return t_write(g,out0(g),&o);
}
static int k_requant_u16(G *g) {
    int64_t r[10];
    NEED(1);
    TRY(call_args(g,QDR_CALL_REQUANTIZE_CROUTON_U16,r,0x18));
    CHECK(small(r,0x18),E_ARGS);
    int64_t m=r[3]&0xffff,s=15-(r[3]>>24),zin=r[4]&0xffff,zout=r[4]>>16;
    CHECK(s>=1 && s<=48,E_ARGS);
    A x;TRY(t_read(g,in(g,0),&x,0));
    int32_t *v;TRY(to_i32(g,&x,&v));
    A o;TRY(anew(g,&o,x.n,DT_U16));
    if(!g->dry) {
        uint16_t *po=o.p;
        for(size_t i=0,n=count4(x.n);i<n;i++)po[i]=(uint16_t)clip64((((v[i]-zin)*m+((int64_t)1<<(s-1)))>>s)+zout,0,65535);
    }
    return t_write(g,out0(g),&o);
}
static int k_linearclip(G *g) {
    CHECK(g->nin==2,E_ARGS);
    NEED(2);
    const T *i=in(g,0),*o=out0(g);
    int32_t *p;size_t np;
    TRY(read_i32(g,in(g,1),&p,&np));
    A x;TRY(t_read(g,i,&x,0));
    int32_t *v;TRY(to_i32(g,&x,&v));
    A r;TRY(anew(g,&r,x.n,o->dtype));
    if(i->dtype==DT_U8 && o->dtype==DT_U16) CHECK(np>=2,E_ARGS);
    else CHECK(np>=12,E_ARGS);
    if(!g->dry) {
        int64_t hi=dtmax(o->dtype);
        size_t n=count4(x.n);
        if(i->dtype==DT_U8 && o->dtype==DT_U16) {
            int64_t m=p[1]&0xffff,sh=7-((int64_t)p[1]>>16);
            CHECK(sh>=1 && sh<=48,E_ARGS);
            for(size_t j=0;j<n;j++)from_i64(&r,j,clip64((v[j]*m+((int64_t)1<<(sh-1)))>>sh,0,hi));
        } else {
            int64_t P=(uint32_t)p[10],zp=(uint32_t)p[11],m=P&0xffff,sh=15-(P>>24);
            if(p[6]<0)m=-m;
            if(i->dtype==DT_U8) {
                CHECK(sh==14 && zp==0 && m>=0,E_UNSUPPORTED);
                for(size_t j=0;j<n;j++)from_i64(&r,j,clip64((((v[j]*m+128)>>8)+32)>>6,0,hi));
            } else {
                CHECK(sh>=1 && sh<=48,E_ARGS);
                int64_t zin=zp&0xffff,zout=zp>>16;
                for(size_t j=0;j<n;j++)from_i64(&r,j,clip64((((v[j]-zin)*m+((int64_t)1<<(sh-1)))>>sh)+zout,0,hi));
            }
        }
    }
    return t_write(g,o,&r);
}
static int k_copy_values(G *g) {
    SHP("copy_values",g,0);
    CHECK(g->nin>=1 && g->nout==1,E_ARGS);
    const T *o=out0(g);
    {
        int fr=copy_c2_fast(g,in(g,0),o,1);   /* flat flow -> padded blocked */
        if(fr<0)fr=copy_c2_fast(g,o,in(g,0),0);   /* padded blocked -> flat flow */
        if(fr>=0) return fr;
    }
    A x;TRY(t_read(g,in(g,0),&x,0));
    CHECK(o->flags&F_SHAPE,E_TENSOR);
    CHECK(count4(x.n)==count4(o->d),E_SHAPE);
    memcpy(x.n,o->d,16);
    CHECK(x.dt==o->dtype,E_TENSOR);
    return t_write(g,o,&x);
}
static int k_copy_activation(G *g) {
    static const int32_t zero[4]={0,0,0,0};
    NEED(1);
    const T *i=in(g,0),*o=out0(g);
    CHECK(i->dtype==o->dtype,E_TENSOR);
    A x;TRY(t_read(g,i,&x,1));
    return t_write_region(g,o,&x,zero);
}
static int k_transpose(G *g) {
    SHP("transpose",g,0);
    CHECK(g->nin>=2,E_ARGS);
    NEED(1);
    T pt;TRY(input_at(g,g->nin-1,&pt));
    int32_t *perm;size_t np;
    TRY(read_i32(g,&pt,&perm,&np));
    CHECK(np==4,E_ARGS);
#ifdef DM_SHAPES
    if(!g->dry){int64_t q[4];if(const_value(g,&pt,0,&q[0])&&const_value(g,&pt,1,&q[1])&&const_value(g,&pt,2,&q[2])&&const_value(g,&pt,3,&q[3]))fprintf(stderr,"  perm %d%d%d%d\n",(int)q[0],(int)q[1],(int)q[2],(int)q[3]);else fprintf(stderr,"  perm dynamic\n");}
#endif
    {
        int64_t pv[4];
        if(const_value(g,&pt,0,&pv[0]) && const_value(g,&pt,1,&pv[1]) && const_value(g,&pt,2,&pv[2]) && const_value(g,&pt,3,&pv[3])) {
            int32_t pm[4]={(int32_t)pv[0],(int32_t)pv[1],(int32_t)pv[2],(int32_t)pv[3]};
            int fr=tr_fast_dispatch(g,in(g,0),out0(g),pm);
            if(fr>=0) return fr;
        }
    }
    A x;TRY(t_read(g,in(g,0),&x,0));
    int32_t known[4];
    if(g->dry) {
        int64_t v;
        for(int i=0;i<4;i++) {
            if(!const_value(g,&pt,(size_t)i,&v)) {
                A r;TRY(anew(g,&r,x.n,x.dt));
                r.unknown=1;
                return t_write(g,out0(g),&r);
            }
            known[i]=(int32_t)v;
        }
        perm=known;
    }
    int used=0;
    for(int i=0;i<4;i++){CHECK(perm[i]>=0 && perm[i]<4 && !(used&(1<<perm[i])),E_ARGS);used|=1<<perm[i];}
    int32_t n[4];for(int i=0;i<4;i++)n[i]=x.n[perm[i]];
    A r;TRY(anew(g,&r,n,x.dt));
    if(g->dry) return t_write(g,out0(g),&r);
    size_t es=dsize(x.dt),st[4]={(size_t)x.n[1]*x.n[2]*x.n[3],(size_t)x.n[2]*x.n[3],(size_t)x.n[3],1},k=0;
    size_t ps[4]={st[perm[0]],st[perm[1]],st[perm[2]],st[perm[3]]};
    for(int32_t a=0;a<n[0];a++)for(int32_t b=0;b<n[1];b++)for(int32_t d2=0;d2<n[2];d2++) {
        size_t base=a*ps[0]+b*ps[1]+d2*ps[2];
        if(es==1){const uint8_t *s=x.p;uint8_t *d=(uint8_t *)r.p+k;for(int32_t e=0;e<n[3];e++)d[e]=s[base+e*ps[3]];}
        else if(es==2){const uint16_t *s=x.p;uint16_t *d=(uint16_t *)r.p+k;for(int32_t e=0;e<n[3];e++)d[e]=s[base+e*ps[3]];}
        else {const uint32_t *s=x.p;uint32_t *d=(uint32_t *)r.p+k;for(int32_t e=0;e<n[3];e++)d[e]=s[base+e*ps[3]];}
        k+=n[3];
    }
    return t_write(g,out0(g),&r);
}
static int k_concat(G *g) {
    CHECK(g->nin>=2 && g->nout>=1,E_ARGS);
    int64_t axis;
    CHECK(!sv_or_scalar(in(g,0),&axis),E_ARGS);
    if(axis<0)axis+=4;
    CHECK(axis>=0 && axis<4,E_ARGS);
    int32_t n[4];uint32_t dt=0;
    for(uint32_t i=1;i<g->nin;i++) {
        T t;TRY(input_at(g,i,&t));
        CHECK(t.flags&F_SHAPE,E_TENSOR);
        if(i==1){memcpy(n,t.d,16);dt=t.dtype;continue;}
        for(int d=0;d<4;d++)CHECK(d==axis || t.d[d]==n[d],E_SHAPE);
        CHECK(t.dtype==dt,E_TENSOR);
        n[axis]+=t.d[axis];
    }
    A r;TRY(anew(g,&r,n,dt));
    size_t es=dsize(r.dt),outer=1,inner=es;
    for(int d=0;d<axis;d++)outer*=n[d];
    for(int d=axis+1;d<4;d++)inner*=n[d];
    size_t at=0;
    for(uint32_t i=1;i<g->nin;i++) {
        T t;TRY(input_at(g,i,&t));
        A part;TRY(t_read(g,&t,&part,0));
        size_t chunk=inner*part.n[axis];
        if(!g->dry)for(size_t o=0;o<outer;o++)qd_vcopy((uint8_t *)r.p+o*inner*n[axis]+at,(uint8_t *)part.p+o*chunk,chunk);
        at+=chunk;
    }
    return t_write(g,out0(g),&r);
}
static int k_gather(G *g) {
    CHECK(g->nin==4,E_ARGS);
    NEED(3);
    int64_t axis;TRY(read_scalar(g,in(g,1),&axis));
    int32_t *idx;size_t ni;TRY(read_i32(g,in(g,2),&idx,&ni));
    {int fr=gather_fast(g,in(g,0),in(g,1),out0(g),idx,ni);if(fr>=0)return fr;}
    A x;TRY(t_read(g,in(g,0),&x,0));
    if(g->dry && !const_value(g,in(g,1),0,&axis)) {
        size_t worst=0;
        for(int a=0;a<4;a++)if(x.n[a]){size_t c=count4(x.n)/(size_t)x.n[a]*ni;if(c>worst)worst=c;}
        CHECK(worst<=INT32_MAX,E_SHAPE);
        int32_t n[4]={1,1,1,(int32_t)worst};
        A r;TRY(anew(g,&r,n,x.dt));
        r.unknown=1;
        return t_write(g,out0(g),&r);
    }
    if(axis<0)axis+=4;
    CHECK(axis>=0 && axis<4,E_ARGS);
    CHECK(ni<=INT32_MAX,E_SHAPE);
    int32_t n[4];memcpy(n,x.n,16);n[axis]=(int32_t)ni;
    A r;TRY(anew(g,&r,n,x.dt));
    size_t es=dsize(x.dt),outer=1,inner=es;
    for(int d=0;d<axis;d++)outer*=n[d];
    for(int d=axis+1;d<4;d++)inner*=n[d];
    if(!g->dry)for(size_t o=0;o<outer;o++)for(size_t j=0;j<ni;j++) {
        int64_t k=idx[j];
        if(k<0)k+=x.n[axis];
        CHECK(k>=0 && k<x.n[axis],E_RANGE);
        memcpy((uint8_t *)r.p+(o*ni+j)*inner,(uint8_t *)x.p+(o*x.n[axis]+k)*inner,inner);
    }
    return t_write(g,out0(g),&r);
}
static int k_slice(G *g) {
    CHECK(g->nin==3,E_ARGS);
    NEED(3);
    const T *s=in(g,1),*n=in(g,2);
    CHECK(s->flags&n->flags&F_SV,E_ARGS);
    const T *i=in(g,0);
    int32_t origin[4];
    for(int k=0;k<4;k++){CHECK(s->sv[k]>=0 && n->sv[k]>=0 && (int64_t)s->sv[k]+n->sv[k]<=i->d[k],E_RANGE);origin[k]=i->p[k]+s->sv[k];}
    {int fr=slice_fast(g,i,out0(g),n->sv,origin);if(fr>=0)return fr;}
    A r;TRY(t_read_region(g,i,&r,n->sv,origin));
    return t_write(g,out0(g),&r);
}
static int k_slice_pad(G *g) {
    CHECK(g->nin==5,E_ARGS);
    NEED(5);
    const T *pad=in(g,1),*st=in(g,2),*sz=in(g,3);
    CHECK(pad->flags&st->flags&sz->flags&F_SV,E_ARGS);
    int64_t fill;TRY(read_scalar(g,in(g,4),&fill));
    {int fr=slice_pad_fast(g,in(g,0),out0(g),pad,st,sz,fill);if(fr>=0)return fr;}
    A x;TRY(t_read(g,in(g,0),&x,0));
    CHECK(x.dt!=DT_F32,E_ARGS);
    A r;TRY(anew(g,&r,sz->sv,x.dt));
    if(!g->dry) {
        CHECK(fill>=0 && fill<=dtmax(x.dt),E_ARGS);
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
    }
    return t_write(g,out0(g),&r);
}
static int k_decimate(G *g) {
    NEED(2);
    const T *s=in(g,1);
    CHECK(g->nin==2 && (s->flags&F_SV),E_ARGS);
    A x;TRY(t_read(g,in(g,0),&x,0));
    int32_t n[4];
    for(int i=0;i<4;i++){CHECK(s->sv[i]>0,E_ARGS);n[i]=(x.n[i]+s->sv[i]-1)/s->sv[i];}
    A r;TRY(anew(g,&r,n,x.dt));
    if(!g->dry) {
        size_t es=dsize(x.dt),k=0;
        for(int32_t b=0;b<n[0];b++)for(int32_t h=0;h<n[1];h++)for(int32_t w=0;w<n[2];w++)for(int32_t d=0;d<n[3];d++,k++) {
            size_t si=((((size_t)b*s->sv[0])*x.n[1]+(size_t)h*s->sv[1])*x.n[2]+(size_t)w*s->sv[2])*x.n[3]+(size_t)d*s->sv[3];
            if(es==1)((uint8_t *)r.p)[k]=((const uint8_t *)x.p)[si];
            else if(es==2)((uint16_t *)r.p)[k]=((const uint16_t *)x.p)[si];
            else memcpy((uint8_t *)r.p+k*4,(uint8_t *)x.p+si*4,4);
        }
    }
    return t_write(g,out0(g),&r);
}
static int k_resize(G *g) {
    CHECK(g->nin==3,E_ARGS);
    NEED(3);
    int32_t *hp,*wp;size_t nh,nw;
    TRY(read_i32(g,in(g,1),&hp,&nh));
    TRY(read_i32(g,in(g,2),&wp,&nw));
    CHECK(nh>=6 && nw>=6,E_ARGS);
    {int fr=resize_fast(g,in(g,0),in(g,2),out0(g),hp);if(fr>=0)return fr;}
    A x;TRY(t_read(g,in(g,0),&x,0));
    const T *o=out0(g);
    CHECK((o->flags&F_SHAPE) && o->d[1]>=0 && o->d[2]>=0,E_SHAPE);
    int32_t n[4]={x.n[0],o->d[1],o->d[2],x.n[3]};
    int32_t *hi=salloc(g,4*(size_t)n[1]+4),*wi=salloc(g,4*(size_t)n[2]+4);
    CHECK(hi && wi,E_MEMORY);
    if(!g->dry) {
        for(int32_t i=0;i<n[1];i++) {
            int64_t v=(int64_t)((((uint64_t)i*(uint32_t)hp[2])+(uint32_t)hp[4])>>32)+hp[5];
            if(v<0)v+=x.n[1];
            CHECK(v>=0 && v<x.n[1],E_RANGE);hi[i]=(int32_t)v;
        }
        for(int32_t i=0;i<n[2];i++) {
            int64_t v=(int64_t)((((uint64_t)i*(uint32_t)wp[2])+(uint32_t)wp[4])>>32)+wp[5];
            if(v<0)v+=x.n[2];
            CHECK(v>=0 && v<x.n[2],E_RANGE);wi[i]=(int32_t)v;
        }
    }
    A r;TRY(anew(g,&r,n,x.dt));
    if(!g->dry) {
        size_t es=dsize(x.dt),row=(size_t)n[3]*es;
        for(int32_t b=0;b<n[0];b++)for(int32_t h=0;h<n[1];h++)for(int32_t w=0;w<n[2];w++)
            memcpy((uint8_t *)r.p+(((size_t)b*n[1]+h)*n[2]+w)*row,(uint8_t *)x.p+(((size_t)b*x.n[1]+hi[h])*x.n[2]+wi[w])*row,row);
    }
    return t_write(g,o,&r);
}
static int k_depth_to_space(G *g) {
    CHECK(g->nin==2,E_ARGS);
    NEED(2);
    int32_t *f;size_t nf;
    TRY(read_i32(g,in(g,1),&f,&nf));
    CHECK(nf==2,E_ARGS);
    A x;TRY(t_read(g,in(g,0),&x,0));
    int32_t known[2];
    if(g->dry) {
        int64_t v0,v1;
        if(!const_value(g,in(g,1),0,&v0) || !const_value(g,in(g,1),1,&v1)) {
            A r;TRY(anew(g,&r,x.n,x.dt));
            r.unknown=1;
            return t_write(g,out0(g),&r);
        }
        known[0]=(int32_t)v0;known[1]=(int32_t)v1;
        f=known;
    }
    CHECK(f[0]>0 && f[1]>0,E_ARGS);
    int32_t rh=f[0],rw=f[1];
    CHECK(x.n[3]%(rh*rw)==0,E_SHAPE);
    int32_t cc=x.n[3]/(rh*rw),n[4]={x.n[0],x.n[1]*rh,x.n[2]*rw,cc};
    A r;TRY(anew(g,&r,n,x.dt));
    if(g->dry) return t_write(g,out0(g),&r);
    size_t es=dsize(x.dt);
    for(int32_t b=0;b<x.n[0];b++)for(int32_t h=0;h<x.n[1];h++)for(int32_t i=0;i<rh;i++)
        for(int32_t w=0;w<x.n[2];w++)for(int32_t j=0;j<rw;j++) {
            size_t si=((((size_t)b*x.n[1]+h)*x.n[2]+w)*x.n[3]+(size_t)(i*rw+j)*cc);
            size_t di=((((size_t)b*n[1]+h*rh+i)*n[2]+w*rw+j)*cc);
            memcpy((uint8_t *)r.p+di*es,(uint8_t *)x.p+si*es,cc*es);
        }
    return t_write(g,out0(g),&r);
}
static int k_spill_fill(G *g, int spill) {
    CHECK(g->nfv>=4,E_ARGS);
    uint64_t offset=fv(g,3);
    for(uint32_t i=4;i+1<g->nfv;i+=2) {
        uint32_t p=fv(g,i),n=fv(g,i+1),pool=p&0x7ff,addr=p&~0x7ffu;
        CHECK(pool<7 && pool!=SPILL_POOL && g->pool[pool] && g->pool[SPILL_POOL],E_RANGE);
        CHECK((uint64_t)addr+n<=g->pool_bytes[pool] && offset+n<=g->pool_bytes[SPILL_POOL],E_RANGE);
        if(spill)CHECK(offset+n<=g->write_limit[SPILL_POOL],E_RANGE);
        else CHECK((uint64_t)addr+n<=g->write_limit[pool],E_RANGE);
        if(!g->dry) {
            uint8_t *dst=spill?g->pool[SPILL_POOL]+offset:g->pool[pool]+addr;
            const uint8_t *src=spill?g->pool[pool]+addr:g->pool[SPILL_POOL]+offset;
            qd_vcopy(dst,src,n);
        }
        offset+=n;
    }
    return 0;
}
static int k_output_slice(G *g) {
    CHECK(g->nin==4 && g->io<2 && g->io<g->call->output_count,E_ARGS);
    const T *sh=in(g,1),*st=in(g,2);
    CHECK(sh->flags&st->flags&F_SV,E_ARGS);
    CHECK(shape_values_nonnegative(sh),E_ARGS);
    A x;TRY(t_read(g,in(g,0),&x,0));
    size_t es=dsize(x.dt);
    const Nprg2ExecBuffer *output=&g->call->outputs[g->io];
    CHECK(output->data && output->bytes==count4(sh->sv)*es,E_ARGS);
    for(int i=0;i<4;i++)CHECK(st->sv[i]>=0 && (int64_t)st->sv[i]+x.n[i]<=sh->sv[i],E_RANGE);
    if(g->dry) return 0;
    uint8_t *o=output->data;
    /* x is dense; output rows that are adjacent in the destination (the slice
     * spans the whole of the faster dimensions) move as one qd_vcopy run. */
    size_t row=(size_t)x.n[3]*es;
    int32_t n0=x.n[0],n1=x.n[1],n2=x.n[2];
    if(x.n[3]==sh->sv[3]) {
        row*=(size_t)n2;n2=1;
        if(x.n[2]==sh->sv[2]) {
            row*=(size_t)n1;n1=1;
            if(x.n[1]==sh->sv[1]){row*=(size_t)n0;n0=1;}
        }
    }
    for(int32_t b=0;b<n0;b++)for(int32_t h=0;h<n1;h++)for(int32_t w=0;w<n2;w++) {
        size_t di=((((size_t)b+st->sv[0])*sh->sv[1]+h+st->sv[1])*sh->sv[2]+w+st->sv[2])*sh->sv[3]+st->sv[3];
        qd_vcopy(o+di*es,(uint8_t *)x.p+(((size_t)b*x.n[1]+h)*x.n[2]+w)*x.n[3]*es,row);
    }
    return 0;
}
static int k_quantize(G *g) {
    NEED(1);
    const T *o=out0(g);
    A x;TRY(t_read(g,in(g,0),&x,0));
    CHECK(x.dt==DT_F32 && o->dtype==DT_U16 && (o->flags&F_QUANT),E_TENSOR);
    A r;TRY(anew(g,&r,x.n,DT_U16));
    if(!g->dry) {
        double inv=(double)(float)(1.0/o->scale);
        const float *s=x.p;uint16_t *d=r.p;
        for(size_t i=0,n=count4(x.n);i<n;i++) {
            double v=floor((double)s[i]*inv+0.5)+o->zp;
            d[i]=(uint16_t)(v<0?0:v>65535?65535:v);
        }
    }
    return t_write(g,o,&r);
}
static int k_dequantize(G *g) {
    NEED(1);
    const T *i=in(g,0);
    A x;TRY(t_read(g,i,&x,0));
    int32_t *v;TRY(to_i32(g,&x,&v));
    A r;TRY(anew(g,&r,x.n,DT_F32));
    if(!g->dry) {
        double sc=(double)(float)i->scale;
        float *d=r.p;
        for(size_t j=0,n=count4(x.n);j<n;j++) {
            double exact=((double)v[j]-i->zp)*sc;
            float rounded=(float)exact;
            float away=nextafterf(rounded,exact<0?-INFINITY:INFINITY);
            d[j]=exact==((double)rounded+(double)away)*0.5?away:rounded;
        }
    }
    return t_write(g,out0(g),&r);
}
static int k_sign(G *g) {
    NEED(1);
    const T *i=in(g,0),*o=out0(g);
    A x;TRY(t_read(g,i,&x,0));
    int32_t *v;TRY(to_i32(g,&x,&v));
    A r;TRY(anew(g,&r,x.n,o->dtype));
    if(!g->dry)for(size_t j=0,n=count4(x.n);j<n;j++) {
        int64_t d=(int64_t)v[j]-i->zp,s=d>0?1:d<0?-1:0;
        double y=floor((double)s/o->scale+o->zp+0.5);
        from_i64(&r,j,y<0?0:y>65535?65535:(int64_t)y);
    }
    return t_write(g,o,&r);
}
static int k_mean(G *g) {
    NEED(1);
    const T *i=in(g,0),*o=out0(g);
    A x;TRY(t_read(g,i,&x,0));
    CHECK(count4(x.n)==32 && i->zp==o->zp && i->scale==o->scale,E_UNSUPPORTED);
    int32_t one[4]={1,1,1,1};
    A r;TRY(anew(g,&r,one,o->dtype));r.scalar=1;
    if(!g->dry){int64_t s=0;for(size_t j=0;j<32;j++)s+=aget(&x,j);from_i64(&r,0,clip64((s+33)>>5,0,65535));}
    return t_write(g,o,&r);
}
static float f32_from_bits(uint32_t b){float f;memcpy(&f,&b,4);return f;}
static uint32_t f32_bits(float f){uint32_t b;memcpy(&b,&f,4);return b;}
static int k_weight_scales(G *g) {
    NEED(1);
    const T *i=in(g,0),*o=out0(g);
    A x;TRY(t_read(g,i,&x,0));
    CHECK(x.dt==DT_U8 && o->dtype==DT_F32,E_TENSOR);
    int32_t co=x.n[3];
    CHECK(co>0 && (size_t)co==count4(o->d),E_SHAPE);
    A r;TRY(anew(g,&r,o->d,DT_F32));
    if(!g->dry) {
        size_t rows=count4(x.n)/co;
        for(int32_t k=0;k<co;k++) {
            int32_t mn=INT32_MAX,mx=INT32_MIN;
            for(size_t j=0;j<rows;j++){int32_t v=((uint8_t *)x.p)[j*co+k]-i->zp;if(v<mn)mn=v;if(v>mx)mx=v;}
            int32_t a=-127*mn>128*mx?-127*mn:128*mx;
            float s=(float)a/(float)16256;
            if(!(s>=1.0f))s=s!=s?s:1.0f;
            ((float *)r.p)[k]=f32_from_bits((f32_bits(s)+0x800u)&0xfffff000u);
        }
    }
    return t_write(g,o,&r);
}
static int k_invscale(G *g) {
    NEED(1);
    const T *o=out0(g);
    A x;TRY(t_read(g,in(g,0),&x,0));
    CHECK(x.dt==DT_F32 && (o->dtype==DT_Q32 || o->dtype==DT_I32),E_TENSOR);
    A r;TRY(anew(g,&r,x.n,o->dtype));
    if(!g->dry)for(size_t j=0,n=count4(x.n);j<n;j++) {
        volatile float q=(float)65536/((float *)x.p)[j];
        volatile float s=q+0.5f;
        float t=truncf(s);
        CHECK(t>-2147483648.0f && t<2147483648.0f,E_RANGE);
        ((int32_t *)r.p)[j]=(int32_t)t;
    }
    return t_write(g,o,&r);
}
static int k_combine_scales(G *g) {
    CHECK(g->nin==3,E_ARGS);
    NEED(3);
    const T *s=in(g,2);
    CHECK((s->flags&F_SCALAR) && s->dtype==DT_F32,E_ARGS);
    A a,b;TRY(read2(g,&a,&b));
    CHECK(a.dt==DT_F32 && b.dt==DT_F32 && !memcmp(a.n,b.n,16),E_SHAPE);
    A r;TRY(anew(g,&r,a.n,DT_F32));
    if(!g->dry) {
        float k=f32_from_bits(s->scalar);
        for(size_t j=0,n=count4(a.n);j<n;j++){volatile float p=((float *)a.p)[j]*((float *)b.p)[j];((float *)r.p)[j]=p*k;}
    }
    return t_write(g,out0(g),&r);
}
static int k_convert_weights(G *g) {
    CHECK(g->nin==4,E_ARGS);
    NEED(3);
    const T *i=in(g,0),*o=out0(g);
    A x,s;TRY(t_read(g,i,&x,0));TRY(t_read(g,in(g,2),&s,0));
    CHECK(x.dt==DT_U8 && (s.dt==DT_Q32 || s.dt==DT_I32) && (o->flags&F_DIRECT),E_TENSOR);
    int32_t H=x.n[0],W=x.n[1],I=x.n[2],O=x.n[3];
    CHECK(I%4==0 && O%32==0,E_SHAPE);
    int32_t bn[4];CHECK(!bshape(&x,&s,bn) && !memcmp(bn,x.n,16),E_SHAPE);
    size_t total=count4(x.n);
    CHECK(o->pool<7 && g->pool[o->pool] && o->offset+(uint64_t)total<=g->write_limit[o->pool],E_RANGE);
    if(g->dry) return 0;
    int8_t *raw=(int8_t *)g->pool[o->pool]+o->offset;
    for(int32_t h=0;h<H;h++)for(int32_t w=0;w<W;w++)for(int32_t ii=0;ii<I;ii++)for(int32_t oo=0;oo<O;oo++) {
        int64_t v=((uint8_t *)x.p)[(((size_t)h*W+w)*I+ii)*O+oo]-(int64_t)i->zp;
        int64_t sc=((int32_t *)s.p)[bindex(s.n,h,w,ii,oo)];
        int64_t q=clip64((v*sc+32768)>>16,-128,127);
        size_t at=(((((size_t)(oo/32)*H+h)*W+w)*(I/4)+ii/4)*32+oo%32)*4+ii%4;
        raw[at]=(int8_t)q;
    }
    return 0;
}
static int k_scale_convert(G *g) {
    NEED(1);
    const T *o=out0(g);
    A x;TRY(t_read(g,in(g,0),&x,0));
    CHECK(x.dt==DT_F32 && o->esize==4 && o->dtype!=DT_F32,E_TENSOR);
    A r;TRY(anew(g,&r,x.n,o->dtype));
    if(!g->dry)for(size_t j=0,n=count4(x.n);j<n;j++) {
        int64_t h=((int64_t)((uint32_t *)x.p)[j]+0x1000)>>13;
        ((int32_t *)r.p)[j]=(int32_t)((h&0x3ff)+(((h&0x7c00)+0x6400)&0x7c00)+0x400000);
    }
    return t_write(g,o,&r);
}
static int k_dynamic_bias(G *g) {
    CHECK(g->nout>=1,E_ARGS);
    for(uint32_t i=3;i<g->nin;i++) {
        T t;TRY(input_at(g,i,&t));int64_t v;
        CHECK(!sv_or_scalar(&t,&v) && v==0,E_UNSUPPORTED);
    }
    const T *o=out0(g);
    int32_t zero=0;
    A r={{1,1,1,1},o->dtype,1,0,&zero};
    CHECK(o->esize==4 && o->dtype!=DT_F32,E_TENSOR);
    return t_write(g,o,&r);
}
static int k_bias_shuffle(G *g) {
    CHECK(g->nin==3,E_ARGS);
    NEED(3);
    int64_t mode;
    CHECK(!sv_or_scalar(in(g,2),&mode) && mode==0,E_UNSUPPORTED);
    int32_t *a,*b;size_t na,nb;
    TRY(read_i32(g,in(g,0),&a,&na));TRY(read_i32(g,in(g,1),&b,&nb));
    const T *o=out0(g);
    CHECK(o->flags&F_SHAPE,E_TENSOR);
    CHECK(na==nb && na%32==0 && 2*na==count4(o->d) && o->esize==4 && o->dtype!=DT_F32,E_SHAPE);
    A r;TRY(anew(g,&r,o->d,o->dtype));
    if(!g->dry){int32_t *d=r.p;for(size_t row=0;row<na/32;row++){memcpy(d+row*64,a+row*32,128);memcpy(d+row*64+32,b+row*32,128);}}
    return t_write(g,o,&r);
}
static int k_abs(G *g) {
    NEED(1);
    const T *i=in(g,0),*o=out0(g);
    A x;TRY(t_read(g,i,&x,0));
    int32_t *v;TRY(to_i32(g,&x,&v));
    CHECK(i->esize==2,E_TENSOR);
    A r;TRY(anew(g,&r,x.n,o->dtype));
    if(!g->dry) {
        float ratio=(float)i->scale*(float)(1.0/o->scale);
        int64_t mult=(int64_t)rintf(ratio*262144.0f);
        volatile float a=ratio*32768.0f;
        volatile float b=a+(float)o->zp;
        volatile float e=b*8.0f;
        int64_t offset=(int64_t)rintf(e);
        size_t k=0;
        for(int32_t bb=0;bb<x.n[0];bb++)for(int32_t h=0;h<x.n[1];h++)for(int32_t w=0;w<x.n[2];w++)for(int32_t d=0;d<x.n[3];d++,k++) {
            int64_t lane;
            int64_t H=h+i->p[1],W=w+i->p[2],B=bb+i->p[0],D=d+i->p[3];
            if(i->flags&F_DIRECT) {
                int64_t lin=((B*i->m[1]+H)*i->m[2]+W)*i->m[3]+D;
                lane=((i->offset+lin*2)/2)%2==0;
            } else lane=W%2==0;
            int64_t centered=llabs((int64_t)v[k]-i->zp)-32768;
            int64_t y=((centered*mult+lane*16384)>>15)+offset;
            from_i64(&r,k,clip64((y+4)>>3,0,65535));
        }
    }
    return t_write(g,o,&r);
}
static int k_block_zap(G *g) {
    CHECK(g->nfv>=2 && g->pool[1],E_ARGS);
    uint32_t fill=fv(g,0),header=fv(g,1),count=header&0x7fff;
    CHECK((header&0x8000) && (header&0xc00000)==0xc00000,E_UNSUPPORTED);
    uint32_t nzero=((header>>16)&7)*256;
    CHECK(nzero<=BLOCK_BYTES,E_UNSUPPORTED);
    uint8_t *tile=salloc(g,BLOCK_BYTES);
    CHECK(tile,E_MEMORY);
    for(uint32_t j=0;j<count;j++) {
        CHECK(2+2*j+1<g->nfv,E_ARGS);
        uint32_t src=fv(g,2+2*j),dst=fv(g,3+2*j);
        CHECK((uint64_t)src+BLOCK_BYTES<=g->pool_bytes[1] && (uint64_t)dst+BLOCK_BYTES<=g->write_limit[1],E_RANGE);
        if(g->dry) continue;
        memcpy(tile,g->pool[1]+src,BLOCK_BYTES);
        memset(tile+BLOCK_BYTES-nzero,fill&255,nzero);
        memcpy(g->pool[1]+dst,tile,BLOCK_BYTES);
    }
    return 0;
}

/* -------------------------------------------------------------- convolution */
typedef struct {
    int32_t kh, kw, ci, co;
    int8_t *w;
    int64_t *wsum;
} Weights;
static const int S2_TAPS[9][2]={{0,2},{0,0},{2,2},{2,0},{0,1},{2,1},{1,2},{1,0},{1,1}};
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
/* qd_graph conv_weights without its static-source cache: dense HWIO int8
 * (stride-2 regrouped to 2x2 taps, Xwb width-reversed) and channel sums. */
static int conv_weights(G *g, const T *t, int s2, int32_t co, Weights *e) {
    int reverse=!s2 && t->cls==('X'|'w'<<8|'b'<<16);
    CHECK((t->flags&F_DIRECT) && t->pool<7 && g->pool[t->pool],E_TENSOR);
    int32_t H=t->m[0],W=t->m[1],I=t->m[2],O=t->m[3];
    CHECK(H>0 && W>0 && I>0 && O>0,E_SHAPE);
    int packed=is_class(t,"Xpb","xpb");
    size_t n=(size_t)H*W*I*O;
    CHECK(I%4==0 && O%32==0 && (packed || I%32==0),E_SHAPE);
    CHECK(t->offset+(uint64_t)n<=g->pool_bytes[t->pool],E_RANGE);
    int8_t *w=salloc(g,n),*final=NULL;
    CHECK(w,E_MEMORY);
    if(!g->dry)unpack_weights(w,(const int8_t *)g->pool[t->pool]+t->offset,H,W,I,O,packed);
    int32_t kh=H,kw=W,ci=I;
    if(s2) {
        CHECK(H==3 && W==3 && !(I%32),E_SHAPE);
        int8_t *tr=salloc(g,n);
        CHECK(tr,E_MEMORY);
        kh=kw=2;ci=4*I;
        final=salloc(g,(size_t)4*ci*O);
        CHECK(final,E_MEMORY);
        if(!g->dry) {
            memset(tr,0,n);memset(final,0,(size_t)4*ci*O);
            for(int k=0;k<9;k++)memcpy(tr+((size_t)S2_TAPS[k][0]*3+S2_TAPS[k][1])*I*O,w+(size_t)k*I*O,(size_t)I*O);
            for(int a=0;a<2;a++)for(int b=0;b<2;b++)for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++) {
                int ky=2*a+dy,kx=2*b-dx;
                if(ky<3 && kx>=0 && kx<3)for(int32_t cb=0;cb<I/32;cb++) {
                    int32_t j=cb*4+2*dy+dx;
                    for(int32_t ch=0;ch<32;ch++)
                        memcpy(final+(((size_t)a*2+b)*ci+32*j+ch)*O,tr+(((size_t)ky*3+kx)*I+32*cb+ch)*O,O);
                }
            }
        }
    } else if(reverse) {
        final=salloc(g,n);
        CHECK(final,E_MEMORY);
        if(!g->dry)for(int32_t h=0;h<H;h++)for(int32_t x=0;x<W;x++)
            memcpy(final+((size_t)h*W+x)*I*O,w+((size_t)h*W+(W-1-x))*I*O,(size_t)I*O);
    } else final=w;
    CHECK(co>0 && co<=O,E_SHAPE);
    size_t terms=(size_t)kh*kw*ci;
    e->kh=kh;e->kw=kw;e->ci=ci;e->co=co;
    e->w=salloc(g,terms*co);e->wsum=salloc(g,(size_t)co*sizeof(int64_t));
    CHECK(e->w && e->wsum,E_MEMORY);
    if(!g->dry) {
        memset(e->wsum,0,(size_t)co*sizeof(int64_t));
        for(size_t k=0;k<terms;k++)for(int32_t o=0;o<co;o++){int8_t v=final[k*O+o];e->w[k*co+o]=v;e->wsum[o]+=v;}
    }
    return 0;
}
static void cpu_accumulate(const A *x, const Weights *w, int64_t *acc, int32_t *row, int32_t oh, int32_t ow) {
    int32_t co=w->co,ci=w->ci;
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
}
static void requant_params(const int32_t *mult, int32_t co, int sixteen, int32_t *M, int32_t *pre, int32_t *post) {
    for(int32_t c=0;c<co;c++) {
        uint32_t h=(uint32_t)mult[c]&65535,shift=34-((h>>10)&31);
        M[c]=1024+(h&1023);pre[c]=sixteen?8:(int32_t)shift-14;post[c]=(int32_t)shift-pre[c];
    }
}
static inline int64_t requant_value(int64_t acc, int32_t bias, int sixteen, int32_t M, int32_t pre, int32_t post) {
    int64_t a=acc+(int64_t)bias*(sixteen?256:1);
    int64_t q=(((a>>pre)*M)+((int64_t)1<<(post-1)))>>post;
    return clip64(q,0,sixteen?65535:255);
}
static void requant(uint8_t *o, const int64_t *acc, const int32_t *mult, const int32_t *bias, size_t pixels,
                    int32_t co, int sixteen, int32_t *M, int32_t *pre, int32_t *post) {
    requant_params(mult,co,sixteen,M,pre,post);
    for(size_t p=0;p<pixels;p++)for(int32_t c=0;c<co;c++) {
        size_t j=p*co+c;
        int64_t q=requant_value(acc[j],bias[c],sixteen,M[c],pre[c],post[c]);
        if(sixteen)((uint16_t *)o)[j]=(uint16_t)q;else o[j]=(uint8_t)q;
    }
}

/* Fast exact path. The workspace is a second bump arena; a dry pass only
 * measures it. */
static void *walloc(G *g, size_t bytes) {
    size_t at=(g->wused+127)&~(size_t)127;
    if(at<g->wused || bytes>SIZE_MAX-at) return NULL;
    g->wused=at+bytes;
    if(g->dry) return (void *)g;
    if(g->wused>g->conv->workspace_bytes) return NULL;
    return g->conv->workspace+at;
}
#define NO_TILE 0xffffffffu
enum { PIXEL_BLOCK=8 };
/* Smallest conv (multiply-accumulates) worth waking helper threads for. */
#ifndef PAR_CONV_MIN_MACS
#define PAR_CONV_MIN_MACS 2000000ull
#endif
/* Byte offset in the raw weight tensor of the [O%32][I%4] tile that feeds
 * final tap (ky,kx), final input channels 4*group..+3 and output block j, or
 * NO_TILE where the stride-2 regrouping leaves zeros. Mirrors unpack_weights,
 * the S2 regrouping and the Xwb width reversal in conv_weights. */
static uint32_t weight_tile(const T *t, int s2, int reverse, int packed, int32_t ky, int32_t kx, int32_t group, int32_t j) {
    int32_t H=t->m[0],W=t->m[1],I=t->m[2],h=ky,x=kx,i=4*group;
    if(s2) {
        int32_t jj=i/32,cb=jj/4,dy=(jj%4)/2,dx=jj%2,y3=2*ky+dy,x3=2*kx-dx,k=0;
        if(y3>=3 || x3<0 || x3>=3) return NO_TILE;
        while(S2_TAPS[k][0]!=y3 || S2_TAPS[k][1]!=x3) k++;
        h=k/3;x=k%3;i=32*cb+i%32;
    } else if(reverse) x=W-1-kx;
    size_t off=packed?(((size_t)j*H+h)*W+x)*(size_t)(I/4)*128+(size_t)(i/4)*128:
        ((((size_t)j*(I/32)+i/32)*H+h)*W+x)*1024+(size_t)((i/4)%8)*128;
    return (uint32_t)off;
}
/* Vector requantization. Per 32-channel block, RQ_VECS lane vectors hold
 * (lane l = channel 32*j+l, all precomputed by rq_prepare):
 *   u8:  {bias-zp*wsum, pre, M, post, r, c, 0, 0}
 *   u16: {(256*bias-zp*wsum)>>8, &255, M, post, r, c, sl, 0}
 * with r=2^(post-1). Per lane x=floor(a/2^pre) is clamped to [0,c]; q is
 * monotone and already saturated at c, so the packs clip it exactly.
 * u8: c=2^max(post-2,0) (c*M>=2^(post+8), so q(c)>=256); x*M+r<=
 * 2^(post-2)*2047+2^(post-1)<2^32 for post<=22, so one vmpyieacc (mod 2^32)
 * and a logical shift by post give q exactly.
 * u16: c=min(2^(post+6),2^22) and q=(x*M+r)>>post is formed from 16-bit
 * halves of x, sl=16-post, so every intermediate fits int32.
 * Equal to requant_value for M 1024..2047 with post 1..22 (u8) or 1..16
 * (u16), the ranges rq_prepare admits. */
enum { RQ_VECS=8 };
#if defined(__HVX__) && __HVX_LENGTH__==128 && !defined(NPRG2_SCALAR_REQUANT)
#define RQ_HVX 1
#else
#define RQ_HVX 0
#endif
/* Fills par (co/32 blocks of RQ_VECS*32 int32) or returns 0 when a channel's
 * parameters or its int32 accumulator range fall outside the vector path. */
static int rq_prepare(int32_t *par, int32_t co, int sixteen, int64_t bound, int32_t zp, const int64_t *wsum,
                      const int32_t *bias, const int32_t *M, const int32_t *pre, const int32_t *post) {
    for(int32_t c=0;c<co;c++) {
        int32_t *v=par+(size_t)(c/32)*RQ_VECS*32+c%32;
        int64_t off=(int64_t)bias[c]*(sixteen?256:1)-(int64_t)zp*wsum[c];
        if(M[c]<1024 || M[c]>2047) return 0;
        if(sixteen) {
            int64_t offh=off>>8;
            if(pre[c]!=8 || post[c]<1 || post[c]>16) return 0;
            if(bound+255>INT32_MAX || (offh<0?-offh:offh)+bound+bound/256+1>INT32_MAX) return 0;
            v[0]=(int32_t)offh;v[32]=(int32_t)(off&255);
            v[160]=post[c]+6<22?1<<(post[c]+6):1<<22;v[192]=16-post[c];
        } else {
            if(pre[c]<0 || pre[c]>31 || post[c]<1 || post[c]>22) return 0;
            if((off<0?-off:off)+bound>INT32_MAX) return 0;
            v[0]=(int32_t)off;v[32]=pre[c];
            v[160]=post[c]>=2?1<<(post[c]-2):1;v[192]=0;
        }
        v[64]=M[c];v[96]=post[c];v[128]=1<<(post[c]-1);v[224]=0;
    }
    return 1;
}
#if defined(__HVX__) && __HVX_LENGTH__==128
/* Where tile_block requantizes: block j's parameters, the output byte for
 * pixel 0 channel 32*j, and the byte stride between pixels.
 * With blocked set, block b's pixels go to the final blocked layout instead:
 * blk[b][i] is the start (row H%8 included) of the i-th block that the 8
 * pixels touch, a is the first pixel's slot in its block (W%8 for u8, W%4 for
 * u16, even for u16) and a NULL entry is an absent block, which stores skip. */
/* Fused epilogue (u8, aligned full 8-pixel blocks only; see Fuse): block b's
 * row start in the epilogue's output (eblk) and, for kind 2, in the other
 * operand (bblk). */
typedef struct { uint32_t mc, mo, rd, sh; int kind, dead; uint8_t *eblk[2]; const uint8_t *bblk[2]; } FxOut;
typedef struct { const HVX_Vector *par; uint8_t *out; uint32_t pstride; int sixteen, blocked, a; uint8_t *blk[2][3]; const FxOut *fx; } RqOut;
#define RQ_INLINE static inline __attribute__((always_inline))
/* Constant trip counts (np only guards), so every copy unrolls fully and the
 * accumulator arrays below stay in registers. */
#define RQ_UNROLL _Pragma("clang loop unroll(full)")
RQ_INLINE HVX_Vector rq_lane(HVX_Vector a, const HVX_Vector *lo, const HVX_Vector *k, int sixteen) {
    if(!sixteen) {
        HVX_Vector x=Q6_Vw_vasr_VwVw(Q6_Vw_vadd_VwVw(a,k[0]),k[1]);
        x=Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(x,Q6_V_vzero()),k[5]);
        return Q6_Vw_vlsr_VwVw(Q6_Vw_vmpyieacc_VwVwVuh(k[4],x,k[2]),k[3]);
    }
    HVX_Vector m16=Q6_V_vsplat_R(0xffff);
    HVX_Vector x=Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a,k[0]),Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(*lo,k[1]),8));
    x=Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(x,Q6_V_vzero()),k[5]);
    HVX_Vector l=Q6_Vw_vmpyieacc_VwVwVuh(k[4],Q6_V_vand_VV(x,m16),k[2]);
    HVX_Vector h=Q6_Vw_vmpyieacc_VwVwVuh(Q6_Vuw_vlsr_VuwR(l,16),Q6_Vw_vasr_VwR(x,16),k[2]);
    return Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwVw(h,k[6]),Q6_Vw_vlsr_VwVw(Q6_V_vand_VV(l,m16),k[3]));
}
RQ_INLINE HVX_Vector rq_pack8(HVX_Vector q0, HVX_Vector q1, HVX_Vector q2, HVX_Vector q3) {
    return Q6_Vub_vpack_VhVh_sat(Q6_Vh_vpack_VwVw_sat(q3,q2),Q6_Vh_vpack_VwVw_sat(q1,q0));
}
/* cnt (1..4) u8 pixels of 32 channels, v bytes 0..32*cnt, to slots a+p0.. of
 * output block b. aligned: the caller guarantees a==0. */
RQ_INLINE void rq_put8(const RqOut *o, int32_t b, HVX_Vector v, int32_t p0, int32_t cnt, int aligned) {
    if(aligned) {
        uint8_t *d=o->blk[b][0];
        if(d) store_part(d+32*p0,v,0,(uint32_t)cnt*32);
        return;
    }
    int32_t s=o->a+p0,idx=s>>3,slot=s&7,n1=8-slot<cnt?8-slot:cnt;
    uint8_t *d=o->blk[b][idx];
    if(d) store_part(d+32*slot,v,0,(uint32_t)n1*32);
    if(cnt>n1 && (d=o->blk[b][idx+1])!=NULL) store_part(d,v,(uint32_t)n1*32,(uint32_t)(cnt-n1)*32);
}
/* The pixel pair p0,p0+1 (v halfword 2c+i is channel c of pixel p0+i), a
 * whole 128-byte vector of u16 output block b. */
RQ_INLINE void rq_put16(const RqOut *o, int32_t b, HVX_Vector v, int32_t p0, int aligned) {
    int32_t s=aligned?p0:o->a+p0;
    uint8_t *d=o->blk[b][s>>2];
    if(d) store_part(d+128*((s>>1)&1),v,0,128);
}
/* Requantizes acc[b*PIXEL_BLOCK+p] (u16: with lo[b*PIXEL_BLOCK+p]), b<nb,
 * p<np, and stores them; saturating packs clip. Each output vector's lanes
 * are formed as it is packed, so no array of results is live. nb, sixteen
 * and aligned are compile-time constants at every use. u16 blocked output
 * needs np even. */
RQ_INLINE void rq_emit_k(const RqOut *o, const HVX_Vector *acc, const HVX_Vector *lo, int32_t nb, int32_t np,
                         int sixteen, int aligned) {
    HVX_Vector z=Q6_V_vzero();
#define RQ(b,p) ((p)<np?rq_lane(acc[(b)*PIXEL_BLOCK+(p)],sixteen?lo+(b)*PIXEL_BLOCK+(p):NULL, \
                                o->par+(size_t)(b)*RQ_VECS,sixteen):z)
    if(o->blocked) {
        if(!sixteen) {
            RQ_UNROLL for(int32_t b=0;b<nb;b++) RQ_UNROLL for(int32_t p=0;p<PIXEL_BLOCK;p+=4) if(p<np)
                rq_put8(o,b,rq_pack8(RQ(b,p),RQ(b,p+1),RQ(b,p+2),RQ(b,p+3)),p,np-p<4?np-p:4,aligned);
        } else {
            RQ_UNROLL for(int32_t b=0;b<nb;b++) RQ_UNROLL for(int32_t p=0;p<PIXEL_BLOCK;p+=2) if(p<np)
                rq_put16(o,b,Q6_Vuh_vsat_VuwVuw(RQ(b,p+1),RQ(b,p)),p,aligned);
        }
        return;
    }
    /* Output order is pixel-major, 32 channels (one block) per segment; a
     * packed vector holds g segments. */
    uint32_t seg=sixteen?64:32,g=sixteen?2:4,run=(uint32_t)nb*seg,n=(uint32_t)(nb*np);
    int contig=o->pstride==run;
#define SQ(i) RQ((int32_t)((i)%(uint32_t)nb),(int32_t)((i)/(uint32_t)nb))
    RQ_UNROLL for(uint32_t s0=0;s0<(uint32_t)nb*PIXEL_BLOCK;s0+=g) if(s0<n) {
        HVX_Vector v=sixteen?Q6_Vuh_vpack_VwVw_sat(SQ(s0+1),SQ(s0)):rq_pack8(SQ(s0),SQ(s0+1),SQ(s0+2),SQ(s0+3));
        if(contig) store_part(o->out+(size_t)s0*seg,v,0,(n-s0<g?n-s0:g)*seg);
        else RQ_UNROLL for(uint32_t s=s0;s<s0+g;s+=(uint32_t)nb) if(s<n)
            store_part(o->out+(size_t)(s/(uint32_t)nb)*o->pstride,v,(s-s0)*seg,run);
    }
#undef SQ
#undef RQ
}
RQ_INLINE void rq_emit(const RqOut *o, const HVX_Vector *acc, const HVX_Vector *lo, int32_t nb, int32_t np, int aligned) {
    if(o->sixteen) {
        if(nb==2) rq_emit_k(o,acc,lo,2,np,1,aligned);
        else rq_emit_k(o,acc,lo,1,np,1,aligned);
    } else if(nb==2) rq_emit_k(o,acc,lo,2,np,0,aligned);
    else rq_emit_k(o,acc,lo,1,np,0,aligned);
}
/* Partial blocks (np<PIXEL_BLOCK) and misaligned blocked output are rare:
 * keep their copy out of line. */
static __attribute__((noinline)) void rq_emit_part(const RqOut *o, const HVX_Vector *acc, const HVX_Vector *lo,
                                                   int32_t nb, int32_t np) {
    rq_emit(o,acc,lo,nb,np,0);
}
/* Fused conv output step for u8 blocked output with a == 0 and np == 8:
 * the conv's requantized value q (clipped to the u8 range exactly as the
 * saturating pack would) feeds the epilogue's integer formula, e =
 * min((q*mc + b*mo + rd) >> sh, 255), whose u32 sum Fuse proved cannot wrap;
 * vmpyieacc multiplies by the low unsigned halfword (all multipliers are
 * <= 65535) and wraps mod 2^32, vlsr is logical. The raw q is stored too
 * unless dead. Out of line so the unfused emit keeps its register budget. */
static __attribute__((noinline)) void rq_emit_fx(const RqOut *o, const HVX_Vector *acc, int32_t nb) {
    const FxOut *f=o->fx;
    HVX_Vector rdv=Q6_V_vsplat_R((int32_t)f->rd),mcv=Q6_V_vsplat_R((int32_t)f->mc),mov=Q6_V_vsplat_R((int32_t)f->mo),
               shv=Q6_V_vsplat_R((int32_t)f->sh),c255=Q6_V_vsplat_R(255);
    for(int32_t b=0;b<nb;b++) {
        const HVX_Vector *k=o->par+(size_t)b*RQ_VECS;
        for(int32_t p0=0;p0<PIXEL_BLOCK;p0+=4) {
            HVX_Vector q[4],r[4];
            for(int i=0;i<4;i++) q[i]=rq_lane(acc[b*PIXEL_BLOCK+p0+i],NULL,k,0);
            if(!f->dead) rq_put8(o,b,rq_pack8(q[0],q[1],q[2],q[3]),p0,4,1);
            HVX_Vector u[4];
            if(f->kind==2) {
                HVX_Vector bv=*(const HVX_Vector *)(f->bblk[b]+32*p0);
                HVX_VectorPair w=Q6_Wuh_vunpack_Vub(bv);
                HVX_VectorPair w0=Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(w)),w1=Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(w));
                u[0]=Q6_V_lo_W(w0);u[1]=Q6_V_hi_W(w0);u[2]=Q6_V_lo_W(w1);u[3]=Q6_V_hi_W(w1);
            }
            for(int i=0;i<4;i++) {
                HVX_Vector t=Q6_Vw_vmpyieacc_VwVwVuh(rdv,Q6_Vw_vmin_VwVw(q[i],c255),mcv);
                if(f->kind==2) t=Q6_Vw_vmpyieacc_VwVwVuh(t,u[i],mov);
                r[i]=Q6_Vw_vmin_VwVw(Q6_Vw_vlsr_VwVw(t,shv),c255);
            }
            *(HVX_Vector *)(f->eblk[b]+32*p0)=rq_pack8(r[0],r[1],r[2],r[3]);
        }
    }
}
#else
typedef struct RqOut RqOut;
typedef int32_t HVX_Vector;
#endif
/* Tile tables hold absolute weight addresses on 32-bit targets (one add less
 * per vrmpy operand), byte offsets otherwise. */
#if UINTPTR_MAX==0xffffffffu
#define WREF(wb,off) ((uint32_t)(uintptr_t)((wb)+(off)))
#define WPTR(wb,r) ((const int8_t *)(uintptr_t)(r))
#else
#define WREF(wb,off) ((uint32_t)(off))
#define WPTR(wb,r) ((wb)+(r))
#endif
/* For output block b (b=0,1: wp[2*i+b] is tap i's weight tile) and pixel
 * p < np: acc[b][p][o] = sum over taps i of the four u8 channels at
 * xb[segment of i]+xoff[i]+p*stride times the tile's [o][0..3] weights. The
 * taps are the compact list from build_taps (no NO_TILE entries); segend[s]
 * is the end of segment s, which reads from xb[s] (one per kernel row in
 * near memory, so a row ring needs no per-phase tables). two: both blocks.
 * With rq the sums are requantized and stored by rq_emit instead (lo: the
 * u16 low-byte sums from an earlier call's acc). Full blocks then keep acc
 * untouched; partial ones (np<PIXEL_BLOCK) pass their sums through acc,
 * which must be 128-byte aligned, as lo already is. */
#if defined(__HVX__) && __HVX_LENGTH__==128
/* Activation words are 4-byte aligned: ci%4==0 and the staged input is
 * 128-byte aligned. */
#define SPLAT(p) HVX_Vector s##p=Q6_V_vsplat_R(*(const uint32_t *)(x+(size_t)(p)*stride))
#define MAC(a,p,w) a##p=Q6_Vw_vrmpyacc_VwVubVb(a##p,s##p,w)
#define MAC8(a,w) MAC(a,0,w);MAC(a,1,w);MAC(a,2,w);MAC(a,3,w);MAC(a,4,w);MAC(a,5,w);MAC(a,6,w);MAC(a,7,w)
#define WLOAD(i,b) (*(const HVX_UVector *)WPTR(wbase,wp[2*(i)+(b)]))
static void tile_block(int32_t *acc, int32_t np, const uint8_t *const *xb, uint32_t nseg, const uint32_t *segend,
                       uint32_t stride, const uint32_t *wp, int two, const uint32_t *xoff, const int8_t *wbase,
                       const RqOut *rq, const HVX_Vector *lo) {
    HVX_UVector *o=(HVX_UVector *)acc;
    (void)wbase;
    if(np==PIXEL_BLOCK && two) {
        HVX_Vector a0=Q6_V_vzero(),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
        HVX_Vector b0=a0,b1=a0,b2=a0,b3=a0,b4=a0,b5=a0,b6=a0,b7=a0;
        uint32_t i=0;
        for(uint32_t s=0;s<nseg;s++) {
            const uint8_t *xs=xb[s];
            for(uint32_t e=segend[s];i<e;i++) {
                HVX_Vector w0=WLOAD(i,0),w1=WLOAD(i,1);
                const uint8_t *x=xs+xoff[i];
                SPLAT(0);SPLAT(1);SPLAT(2);SPLAT(3);SPLAT(4);SPLAT(5);SPLAT(6);SPLAT(7);
                MAC8(a,w0);MAC8(b,w1);
            }
        }
        if(rq && rq->fx) {
            o[0]=a0;o[1]=a1;o[2]=a2;o[3]=a3;o[4]=a4;o[5]=a5;o[6]=a6;o[7]=a7;
            o[8]=b0;o[9]=b1;o[10]=b2;o[11]=b3;o[12]=b4;o[13]=b5;o[14]=b6;o[15]=b7;
            rq_emit_fx(rq,(const HVX_Vector *)acc,2);
            return;
        }
        if(rq && !rq->a) {
            HVX_Vector v[2*PIXEL_BLOCK]={a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7};
            rq_emit(rq,v,lo,2,PIXEL_BLOCK,1);
            return;
        }
        o[0]=a0;o[1]=a1;o[2]=a2;o[3]=a3;o[4]=a4;o[5]=a5;o[6]=a6;o[7]=a7;
        o[8]=b0;o[9]=b1;o[10]=b2;o[11]=b3;o[12]=b4;o[13]=b5;o[14]=b6;o[15]=b7;
        if(rq) rq_emit_part(rq,(const HVX_Vector *)acc,lo,2,PIXEL_BLOCK);
        return;
    }
    if(np==PIXEL_BLOCK) {
        HVX_Vector a0=Q6_V_vzero(),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
        uint32_t i=0;
        for(uint32_t s=0;s<nseg;s++) {
            const uint8_t *xs=xb[s];
            for(uint32_t e=segend[s];i<e;i++) {
                HVX_Vector w0=WLOAD(i,0);
                const uint8_t *x=xs+xoff[i];
                SPLAT(0);SPLAT(1);SPLAT(2);SPLAT(3);SPLAT(4);SPLAT(5);SPLAT(6);SPLAT(7);
                MAC8(a,w0);
            }
        }
        if(rq && rq->fx) {
            o[0]=a0;o[1]=a1;o[2]=a2;o[3]=a3;o[4]=a4;o[5]=a5;o[6]=a6;o[7]=a7;
            rq_emit_fx(rq,(const HVX_Vector *)acc,1);
            return;
        }
        if(rq && !rq->a) {
            HVX_Vector v[PIXEL_BLOCK]={a0,a1,a2,a3,a4,a5,a6,a7};
            rq_emit(rq,v,lo,1,PIXEL_BLOCK,1);
            return;
        }
        o[0]=a0;o[1]=a1;o[2]=a2;o[3]=a3;o[4]=a4;o[5]=a5;o[6]=a6;o[7]=a7;
        if(rq) rq_emit_part(rq,(const HVX_Vector *)acc,lo,1,PIXEL_BLOCK);
        return;
    }
    for(int32_t b=0;b<(two?2:1);b++)for(int32_t p=0;p<np;p++) {
        HVX_Vector a=Q6_V_vzero();
        uint32_t i=0;
        for(uint32_t s=0;s<nseg;s++) {
            const uint8_t *xs=xb[s]+(size_t)p*stride;
            for(uint32_t e=segend[s];i<e;i++)
                a=Q6_Vw_vrmpyacc_VwVubVb(a,Q6_V_vsplat_R(*(const uint32_t *)(xs+xoff[i])),WLOAD(i,b));
        }
        o[b*PIXEL_BLOCK+p]=a;
    }
    if(rq) rq_emit_part(rq,(const HVX_Vector *)acc,lo,two?2:1,np);
}
#undef WLOAD
#undef MAC8
#undef MAC
#undef SPLAT
#else
static void tile_block(int32_t *acc, int32_t np, const uint8_t *const *xb, uint32_t nseg, const uint32_t *segend,
                       uint32_t stride, const uint32_t *wp, int two, const uint32_t *xoff, const int8_t *wbase,
                       const RqOut *rq, const HVX_Vector *lo) {
    (void)rq;(void)lo;(void)wbase;
    for(int32_t b=0;b<(two?2:1);b++) {
        int32_t *ab=acc+(size_t)b*PIXEL_BLOCK*32;
        memset(ab,0,(size_t)np*32*4);
        uint32_t i=0;
        for(uint32_t s=0;s<nseg;s++)for(uint32_t e=segend[s];i<e;i++) {
            const int8_t *w=WPTR(wbase,wp[2*i+(uint32_t)b]);
            const uint8_t *x=xb[s]+xoff[i];
            for(int32_t p=0;p<np;p++) {
                const uint8_t *v=x+(size_t)p*stride;
                int32_t v0=v[0],v1=v[1],v2=v[2],v3=v[3],*r=ab+32*p;
                for(int32_t q=0;q<32;q++)r[q]+=v0*w[4*q]+v1*w[4*q+1]+v2*w[4*q+2]+v3*w[4*q+3];
            }
        }
    }
}
#endif
/* The compact tap list: taps (ky,kx,group) whose tile exists (the stride-2
 * regrouping leaves some zero; that depends on the tap only, not on the output
 * block). */
static uint32_t count_taps(const T *wt, int s2, int reverse, int packed, int32_t kh, int32_t kw, int32_t groups) {
    uint32_t n=0;
    for(int32_t ky=0;ky<kh;ky++)for(int32_t kx=0;kx<kw;kx++)for(int32_t q=0;q<groups;q++)
        if(weight_tile(wt,s2,reverse,packed,ky,kx,q,0)!=NO_TILE) n++;
    return n;
}
/* wp: (blocks+1)/2 pairs x nt taps x 2 weight addresses (an odd last block
 * repeats its own). xoff[i]=(lead+kx)*ci+4*group, plus ky*kystride when merge
 * (one segment), else segend has one end per kernel row. */
static void build_taps(const T *wt, int s2, int reverse, int packed, int32_t kh, int32_t kw, int32_t groups,
                       int32_t blocks, const int8_t *wbase, int32_t ci, int32_t lead, int merge, size_t kystride,
                       uint32_t nt, uint32_t *wp, uint32_t *xoff, uint32_t *segend) {
    uint32_t n=0;
    (void)wbase;
    for(int32_t ky=0;ky<kh;ky++) {
        for(int32_t kx=0;kx<kw;kx++)for(int32_t q=0;q<groups;q++) {
            if(weight_tile(wt,s2,reverse,packed,ky,kx,q,0)==NO_TILE) continue;
            xoff[n]=(uint32_t)((merge?(size_t)ky*kystride:0)+(size_t)(lead+kx)*ci+4*(size_t)q);
            for(int32_t pj=0;pj*2<blocks;pj++)for(int32_t b=0;b<2;b++) {
                int32_t j=2*pj+b<blocks?2*pj+b:2*pj;
                wp[((size_t)pj*nt+n)*2+(size_t)b]=WREF(wbase,weight_tile(wt,s2,reverse,packed,ky,kx,q,j));
            }
            n++;
        }
        if(!merge) segend[ky]=n;
    }
    if(merge) segend[0]=n;
}
/* wsum[c] = sum over taps of the tile's weights for output channel c. */
static void tap_wsum(int64_t *wsum, int32_t blocks, uint32_t nt, const uint32_t *wp, const int8_t *wbase) {
    (void)wbase;
#if defined(__HVX__) && __HVX_LENGTH__==128
    HVX_Vector ones=Q6_V_vsplat_R(0x01010101);
    int32_t tmp[32] __attribute__((aligned(128)));
    for(int32_t j=0;j<blocks;j++) {
        HVX_Vector a=Q6_V_vzero();
        for(uint32_t i=0;i<nt;i++)
            a=Q6_Vw_vrmpyacc_VwVbVb(a,*(const HVX_UVector *)WPTR(wbase,wp[(((size_t)j/2)*nt+i)*2+(size_t)(j&1)]),ones);
        *(HVX_Vector *)tmp=a;
        for(int32_t c=0;c<32;c++)wsum[j*32+c]+=tmp[c];
    }
#else
    for(int32_t j=0;j<blocks;j++)for(uint32_t i=0;i<nt;i++) {
        const int8_t *w=WPTR(wbase,wp[(((size_t)j/2)*nt+i)*2+(size_t)(j&1)]);
        for(int32_t c=0;c<128;c++)wsum[j*32+c/4]+=w[c];
    }
#endif
}
/* Does any block of the (blocked, batch 0) output region share bytes with a
 * block of the input region? Conservative (1) unless both live in 2048-aligned
 * blocks of pool 1. */
static int blocked_overlap(const T *xt, const int32_t *xd, const int32_t *xo, const T *o, const int32_t *od,
                           const int32_t *oo) {
    uint32_t bits[128];
    memset(bits,0,sizeof bits);
    if(!(o->flags&F_BLOCKED) || !(xt->flags&F_BLOCKED)) return 1;
    for(int pass=0;pass<2;pass++) {
        const T *t=pass?xt:o;
        const int32_t *d=pass?xd:od,*org=pass?xo:oo;
        const int sh=t->esize==1?3:t->esize==2?2:1;
        const int32_t nbw=(t->m[2]+(1<<sh)-1)>>sh,nbd=(t->m[3]+31)>>5,nbh=(t->m[1]+7)>>3;
        for(int32_t h=org[1]>>3;h<=(org[1]+d[1]-1)>>3;h++)
         for(int32_t w=org[2]>>sh;w<=(org[2]+d[2]-1)>>sh;w++)
          for(int32_t z=org[3]>>5;z<=(org[3]+d[3]-1)>>5;z++) {
            int64_t index=(int64_t)(((org[0]*nbh+h)*nbw+w)*nbd)+t->slice+z;
            if(index<0 || index>=(int64_t)t->nblocks) continue;
            uint32_t e=t->blocks[index],pool=e&127u,off=e&~127u;
            if(!pool) continue;
            if(pool!=1) { if(!pass) return 1; continue; }
            if((off&2047u) || (off>>11)>=4096u) return 1;
            uint32_t bit=off>>11;
            if(pass) { if(bits[bit>>5]&(1u<<(bit&31))) return 1; }
            else bits[bit>>5]|=1u<<(bit&31);
        }
    }
    return 0;
}
#if RQ_HVX
/* Where pixels of channel group z in block column wblk (of nbw, nbd channel
 * groups) of row h start (row h%8 included) in a u8 or u16 blocked tensor (batch 0), or NULL
 * for an absent block, as blocked_impl finds them. */
static inline uint8_t *blocked_rowptr(uint8_t *const *pools, const T *t, int32_t h, int32_t wblk, int32_t nbw,
                                      int32_t nbd, int32_t z) {
    int64_t index=(int64_t)(((h>>3)*nbw+wblk)*nbd)+t->slice+z;
    uint32_t e=(index>=0 && index<(int64_t)t->nblocks)?t->blocks[index]:0,pool=e&127;
    return pool && pools[pool]?pools[pool]+(e&~127u)+(size_t)(h&7)*256:NULL;
}
#endif
/* Input pixels W0..W0+npix-1 of row h of the u8 blocked tensor t, channels
 * [0,ci), staged by 8-pixel runs (256 contiguous bytes per 32-channel group)
 * to dst, pixel W at dst+(W-Wbase)*ci. Absent blocks stage zeros, as
 * blocked_impl. ci%128==0 stores exactly the pixels asked for; otherwise Wbase
 * is 8-aligned and whole runs are written (the pixels before W0 and after the
 * end are other columns of the same blocks). dst is 128-byte aligned. */
static void stage_row8(uint8_t *const *pools, const T *t, int64_t h, int64_t W0, int32_t npix, int64_t Wbase,
                       int32_t ci, uint8_t *dst) {
    const int64_t nbw=((int64_t)t->m[2]+7)/8,nbd=((int64_t)t->m[3]+31)/32,count=t->nblocks;
    const int64_t hrow=(h/8)*nbw;
    const size_t hin=(size_t)(h%8)*256;
    const int ng=(ci+31)/32,compact=ci%128==0;
    for(int64_t Wa=W0&~(int64_t)7;Wa<W0+npix;Wa+=8) {
        int lo=0,hi=8;
        if(compact) {
            if(Wa<W0)lo=(int)(W0-Wa);
            if(Wa+8>W0+npix)hi=(int)(W0+npix-Wa);
        }
        const int64_t wrow=(hrow+Wa/8)*nbd+t->slice,pix0=Wa-Wbase;
#if BLK_HVX
        HVX_Vector G[4][2];
#define LDG(gi,B) do { int64_t ix_=wrow+(gi); uint32_t e_=(ix_>=0 && ix_<count)?t->blocks[ix_]:0,pl_=e_&127; \
            if(pl_ && pools[pl_]) { const uint8_t *bk_=pools[pl_]+(e_&~127u)+hin; (B)[0]=load_part(bk_,128); \
                (B)[1]=load_part(bk_+128,128); } else (B)[0]=(B)[1]=Q6_V_vzero(); } while(0)
        if(ci==32) {
            LDG(0,G[0]);
            uint8_t *d=dst+(size_t)(pix0*32);
            *(HVX_Vector *)d=G[0][0];*(HVX_Vector *)(d+128)=G[0][1];
        } else if(ci==64) {
            LDG(0,G[0]);LDG(1,G[1]);
            uint8_t *d=dst+(size_t)(pix0*64);
            for(int h2=0;h2<2;h2++) {
                HVX_VectorPair a=Q6_W_vshuff_VVR(G[1][h2],G[0][h2],-32);
                *(HVX_Vector *)(d+256*h2)=Q6_V_lo_W(a);*(HVX_Vector *)(d+256*h2+128)=Q6_V_hi_W(a);
            }
        } else if(compact) {
            for(int sg=0;sg<ci/128;sg++) {
                HVX_Vector H[4][2];
                for(int gg=0;gg<4;gg++)LDG(4*sg+gg,H[gg]);
                for(int h2=0;h2<2;h2++) {
                    HVX_VectorPair a=Q6_W_vshuff_VVR(H[1][h2],H[0][h2],-32),b=Q6_W_vshuff_VVR(H[3][h2],H[2][h2],-32);
                    HVX_VectorPair c=Q6_W_vshuff_VVR(Q6_V_lo_W(b),Q6_V_lo_W(a),-64),
                                   d2=Q6_W_vshuff_VVR(Q6_V_hi_W(b),Q6_V_hi_W(a),-64);
                    HVX_Vector P[4]={Q6_V_lo_W(c),Q6_V_hi_W(c),Q6_V_lo_W(d2),Q6_V_hi_W(d2)};
                    for(int k=0;k<4;k++) {
                        int px=4*h2+k;
                        if(px>=lo && px<hi) *(HVX_Vector *)(dst+(size_t)((pix0+px)*ci)+128*(size_t)sg)=P[k];
                    }
                }
            }
        } else {
            for(int gi=0;gi<ng;gi++) {
                uint32_t n=(uint32_t)(ci-32*gi<32?ci-32*gi:32);
                LDG(gi,G[0]);
                for(int k=lo;k<hi;k++)
                    store_part(dst+(size_t)((pix0+k)*ci)+32*(size_t)gi,G[0][k>>2],(uint32_t)(k&3)*32,n);
            }
        }
#undef LDG
#else
        for(int k=lo;k<hi;k++)for(int gi=0;gi<ng;gi++) {
            int64_t ix=wrow+gi;
            uint32_t e=(ix>=0 && ix<count)?t->blocks[ix]:0,pl=e&127;
            size_t n=(size_t)(ci-32*gi<32?ci-32*gi:32);
            uint8_t *d=dst+(size_t)((pix0+k)*ci)+32*(size_t)gi;
            if(pl && pools[pl]) memcpy(d,pools[pl]+(e&~127u)+hin+32*(size_t)k,n);
            else memset(d,0,n);
        }
#endif
    }
}
/* u16 counterpart of stage_row8: input pixels W0..W0+npix-1 of row h of the
 * u16 blocked tensor t, channels [0,ci), split into low bytes (lo) and high
 * bytes (hi), pixel W at (W-Wbase)*ci in both. A u16 block row is 256 bytes:
 * two 128-byte pixel pairs, each holding channel c of pixel 2q+r at u16 index
 * 2c+r (blocked_impl). Runs are 4 pixels; the rules for which pixels are
 * written are stage_row8's (ci%128==0 stores exactly the pixels asked for,
 * otherwise Wbase is 4-aligned and whole runs are written). lo and hi are
 * 128-byte aligned. Absent blocks stage zeros. */
static void stage_row16(uint8_t *const *pools, const T *t, int64_t h, int64_t W0, int32_t npix, int64_t Wbase,
                        int32_t ci, uint8_t *lo, uint8_t *hi) {
    const int64_t nbw=((int64_t)t->m[2]+3)/4,nbd=((int64_t)t->m[3]+31)/32,count=t->nblocks;
    const int64_t hrow=(h/8)*nbw;
    const size_t hin=(size_t)(h%8)*256;
    const int ng=(ci+31)/32,compact=ci%128==0;
    for(int64_t Wa=W0&~(int64_t)3;Wa<W0+npix;Wa+=4) {
        int a=0,b=4;
        if(compact) {
            if(Wa<W0)a=(int)(W0-Wa);
            if(Wa+4>W0+npix)b=(int)(W0+npix-Wa);
        }
        const int64_t wrow=(hrow+Wa/4)*nbd+t->slice,pix0=Wa-Wbase;
#if BLK_HVX
        /* L/H: low/high bytes of the run's 4 pixels x 32 channels, pixel-major. */
#define LDG16(gi,L,H) do { int64_t ix_=wrow+(gi); uint32_t e_=(ix_>=0 && ix_<count)?t->blocks[ix_]:0,pl_=e_&127; \
            if(pl_ && pools[pl_]) { const uint8_t *bk_=pools[pl_]+(e_&~127u)+hin; \
                HVX_Vector p01_=Q6_Vh_vdeal_Vh(load_part(bk_,128)),p23_=Q6_Vh_vdeal_Vh(load_part(bk_+128,128)); \
                (L)=Q6_Vb_vpacke_VhVh(p23_,p01_);(H)=Q6_Vb_vpacko_VhVh(p23_,p01_); } \
            else (L)=(H)=Q6_V_vzero(); } while(0)
        if(ci==32) {
            HVX_Vector L,H;
            LDG16(0,L,H);
            *(HVX_Vector *)(lo+(size_t)(pix0*32))=L;*(HVX_Vector *)(hi+(size_t)(pix0*32))=H;
        } else if(ci==64) {
            HVX_Vector L0,H0,L1,H1;
            LDG16(0,L0,H0);LDG16(1,L1,H1);
            HVX_VectorPair x=Q6_W_vshuff_VVR(L1,L0,-32),y=Q6_W_vshuff_VVR(H1,H0,-32);
            uint8_t *dl=lo+(size_t)(pix0*64),*dh=hi+(size_t)(pix0*64);
            *(HVX_Vector *)dl=Q6_V_lo_W(x);*(HVX_Vector *)(dl+128)=Q6_V_hi_W(x);
            *(HVX_Vector *)dh=Q6_V_lo_W(y);*(HVX_Vector *)(dh+128)=Q6_V_hi_W(y);
        } else if(compact) {
            for(int sg=0;sg<ci/128;sg++) {
                HVX_Vector P[2][4];
                {
                    HVX_Vector L[4],H[4];
                    for(int gg=0;gg<4;gg++)LDG16(4*sg+gg,L[gg],H[gg]);
                    for(int pl=0;pl<2;pl++) {
                        const HVX_Vector *V=pl?H:L;
                        HVX_VectorPair x=Q6_W_vshuff_VVR(V[1],V[0],-32),y=Q6_W_vshuff_VVR(V[3],V[2],-32);
                        HVX_VectorPair c=Q6_W_vshuff_VVR(Q6_V_lo_W(y),Q6_V_lo_W(x),-64),
                                       d2=Q6_W_vshuff_VVR(Q6_V_hi_W(y),Q6_V_hi_W(x),-64);
                        P[pl][0]=Q6_V_lo_W(c);P[pl][1]=Q6_V_hi_W(c);P[pl][2]=Q6_V_lo_W(d2);P[pl][3]=Q6_V_hi_W(d2);
                    }
                }
                for(int k=a;k<b;k++) {
                    size_t at=(size_t)((pix0+k)*ci)+128*(size_t)sg;
                    *(HVX_Vector *)(lo+at)=P[0][k];*(HVX_Vector *)(hi+at)=P[1][k];
                }
            }
        } else {
            for(int gi=0;gi<ng;gi++) {
                uint32_t n=(uint32_t)(ci-32*gi<32?ci-32*gi:32);
                HVX_Vector L,H;
                LDG16(gi,L,H);
                for(int k=a;k<b;k++) {
                    size_t at=(size_t)((pix0+k)*ci)+32*(size_t)gi;
                    store_part(lo+at,L,(uint32_t)k*32,n);store_part(hi+at,H,(uint32_t)k*32,n);
                }
            }
        }
#undef LDG16
#else
        for(int k=a;k<b;k++)for(int gi=0;gi<ng;gi++) {
            int64_t ix=wrow+gi;
            uint32_t e=(ix>=0 && ix<count)?t->blocks[ix]:0,pl=e&127;
            int n=ci-32*gi<32?ci-32*gi:32;
            size_t at=(size_t)((pix0+k)*ci)+32*(size_t)gi;
            const uint8_t *s=pl && pools[pl]?pools[pl]+(e&~127u)+hin+(size_t)(k>>1)*128+(size_t)(k&1)*2:NULL;
            for(int c=0;c<n;c++) {
                lo[at+(size_t)c]=s?s[4*c]:0;hi[at+(size_t)c]=s?s[4*c+1]:0;
            }
        }
#endif
    }
}
/* Near-memory layout for one command: output columns per chunk, the staged
 * row ring (kh rows of rowbytes), and 128-aligned offsets into near. direct:
 * rows are staged straight from the blocked input (lead pixels before the
 * chunk's first column keep the 8-pixel runs aligned). */
typedef struct { int32_t cols, lead, direct; uint32_t nt; size_t rowbytes, acc, wp, xoff, segend, mult, bias, M, pre, post, wsum, rq, out, lo, hi; } NearPlan;
static size_t nbump(size_t *at, size_t bytes) {size_t a=(*at+127)&~(size_t)127;*at=a+bytes;return a;}
static size_t near_layout(NearPlan *p, int32_t cols, int32_t kh, int32_t kw, int32_t ci, int32_t co, uint32_t nt,
                          int sixteen, int direct, int32_t head0) {
    size_t at=0;
    p->cols=cols;p->direct=direct;p->nt=nt;
    p->lead=direct && ci%128?head0:0;
    size_t npix=(size_t)p->lead+(size_t)cols+(size_t)kw-1,run=sixteen?4:8;
    if(direct && ci%128) npix=(npix+run-1)&~(run-1);
    p->rowbytes=(npix*(size_t)ci+127)&~(size_t)127;
    p->acc=nbump(&at,(size_t)2*PIXEL_BLOCK*32*4*(sixteen?2:1));
    p->wp=nbump(&at,(size_t)((co/32+1)/2)*nt*8);p->xoff=nbump(&at,(size_t)nt*4);p->segend=nbump(&at,(size_t)kh*4);
    p->mult=nbump(&at,(size_t)co*4);p->bias=nbump(&at,(size_t)co*4);p->M=nbump(&at,(size_t)co*4);
    p->pre=nbump(&at,(size_t)co*4);p->post=nbump(&at,(size_t)co*4);p->wsum=nbump(&at,(size_t)co*8);
    p->rq=nbump(&at,(size_t)co*RQ_VECS*4);
    p->out=nbump(&at,(size_t)cols*co*(sixteen?2:1));
    p->lo=nbump(&at,(size_t)kh*p->rowbytes);p->hi=sixteen?nbump(&at,(size_t)kh*p->rowbytes):0;
    return at;
}
/* The widest whole-block chunk that fits, or 0. Depends on shapes only. */
static int near_plan(const G *g, NearPlan *p, int32_t ow, int32_t kh, int32_t kw, int32_t ci, int32_t co,
                     uint32_t nt, int sixteen, int direct, int32_t head0) {
    const Nprg2ReferenceConv *c=g->conv;
    if(!c->near || ((uintptr_t)c->near&127u)) return 0;
    for(int32_t cols=(ow+PIXEL_BLOCK-1)/PIXEL_BLOCK*PIXEL_BLOCK;cols>0;cols-=PIXEL_BLOCK)
        if(near_layout(p,cols,kh,kw,ci,co,nt,sixteen,direct,head0)<=c->near_bytes) return 1;
    return 0;
}
/* Exact copies: they read up to 127 bytes past src+n (255 past 2n for the
 * u16 split) but write exactly n bytes at each destination. */
#if BLK_HVX
static void near_copy(uint8_t *d, const uint8_t *s, size_t n) {
    size_t i=0;
    for(;i+128<=n;i+=128) *(HVX_UVector *)(d+i)=*(const HVX_UVector *)(s+i);
    if(i<n) store_part(d+i,load_part(s+i,(uint32_t)(n-i)),0,(uint32_t)(n-i));
}
static void near_split16(uint8_t *lo, uint8_t *hi, const uint8_t *s, size_t n) {
    size_t i=0;
    for(;i+128<=n;i+=128) {
        HVX_Vector a=*(const HVX_UVector *)(s+2*i),b=*(const HVX_UVector *)(s+2*i+128);
        *(HVX_UVector *)(lo+i)=Q6_Vb_vpacke_VhVh(b,a);
        *(HVX_UVector *)(hi+i)=Q6_Vb_vpacko_VhVh(b,a);
    }
    if(i<n) {
        uint32_t r=(uint32_t)(n-i);
        HVX_Vector a=load_part(s+2*i,2*r<128?2*r:128),b=2*r>128?load_part(s+2*i+128,2*r-128):Q6_V_vzero();
        store_part(lo+i,Q6_Vb_vpacke_VhVh(b,a),0,r);
        store_part(hi+i,Q6_Vb_vpacko_VhVh(b,a),0,r);
    }
}
#else
static void near_copy(uint8_t *d, const uint8_t *s, size_t n) {memcpy(d,s,n);}
static void near_split16(uint8_t *lo, uint8_t *hi, const uint8_t *s, size_t n) {
    for(size_t i=0;i<n;i++){uint16_t v;memcpy(&v,s+2*i,2);lo[i]=(uint8_t)v;hi[i]=(uint8_t)(v>>8);}
}
#endif
static void fast_conv_compute(G *g, const T *xt, const T *wt, const T *ft, int s2, int reverse, int packed, int sixteen,
                              int32_t kh, int32_t kw, int32_t ci, int32_t co, int32_t oh, int32_t ow, uint8_t *x,
                              uint8_t *xlo, uint8_t *xhi, uint8_t *r, uint32_t nt, uint32_t *wp, uint32_t *xoff,
                              uint32_t *segend, int32_t *acc, int32_t *mult, int32_t *bias, int32_t *M, int32_t *pre,
                              int32_t *post, int64_t *wsum, int32_t *rqp);
static void near_compute(G *g, const T *xt, const T *wt, const T *ft, const T *o, int s2, int reverse, int packed,
                         int sixteen, int32_t kh, int32_t kw, int32_t ci, int32_t co, int32_t oh, int32_t ow,
                         const uint8_t *x, const NearPlan *p);
/* Scalar fused epilogue for one output row chunk: q holds nc pixels x co
 * requantized bytes; the epilogue formula of Fuse is applied and written to
 * the epilogue's output (the other operand read from its blocks first). */
static void fuse_scalar(G *g, const Fuse *fx, const uint8_t *q, int32_t nc, int32_t co, int32_t y, int32_t x0) {
    size_t n=(size_t)nc*(size_t)co;
    int32_t dims[4]={1,1,nc,co};
    if(fx->kind==2) {
        int32_t bo[4]={0,fx->ob.p[1]+y,fx->ob.p[2]+x0,0};
        xfer_copy(g,&fx->ob,fx->bb,dims,bo,0);
    }
    for(size_t i=0;i<n;i++) {
        uint32_t t=(uint32_t)q[i]*fx->mc+fx->rd;
        if(fx->kind==2) t+=(uint32_t)fx->bb[i]*fx->mo;
        t>>=fx->sh;
        fx->eb[i]=(uint8_t)(t>255?255:t);
    }
    int32_t eo[4]={0,fx->oe.p[1]+y,fx->oe.p[2]+x0,0};
    xfer_copy(g,&fx->oe,fx->eb,dims,eo,1);
}
static uint32_t conv_path;
#include "prepared_hmx_conv.inc"
/* Returns 1 when the command took the fast path (or would, in a dry pass),
 * 0 to use the reference convolution, or a negative error. */
static int fast_conv(G *g, const T *xt, const T *wt, const T *ft, const T *o, int s2, int sixteen, int32_t co,
                     int32_t oh, int32_t ow, int32_t *err) {
    *err=0;conv_path=0;g->fx_ok=0;
    if(!g->conv) return 0;
    int32_t H=wt->m[0],W=wt->m[1],I=wt->m[2],O=wt->m[3];
    int packed=is_class(wt,"Xpb","xpb"),reverse=!s2 && wt->cls==('X'|'w'<<8|'b'<<16);
    if(!(wt->flags&F_DIRECT) || wt->pool>=7 || !g->pool[wt->pool]) return 0;
    if(H<=0 || W<=0 || I<=0 || O<=0 || I%4 || O%32 || (!packed && I%32) || co%32 || co>O) return 0;
    if(s2 && (H!=3 || W!=3 || I%32)) return 0;
    int32_t kh=s2?2:H,kw=s2?2:W,ci=s2?4*I:I;
    if(ci>xt->m[3] || kh>4) return 0;
    if((int64_t)kh*kw*ci*255*128>INT32_MAX) return 0;
    int32_t xH=xt->m[1]-xt->p[1],xW=xt->m[2]-xt->p[2],cw=sixteen?4:8;
    oh=(oh+7)/8*8;if(oh>o->m[1])oh=o->m[1];if(oh>xH-kh+1)oh=xH-kh+1;
    ow=(ow+cw-1)/cw*cw;if(ow>o->m[2])ow=o->m[2];if(ow>xW-kw+1)ow=xW-kw+1;
    if(oh<=0 || ow<=0) return 0;
    if(wt->offset+(uint64_t)H*W*I*O>g->pool_bytes[wt->pool]){*err=E_RANGE;return -1;}
    if(!s2 && !sixteen) {
        int r=hmx_conv(g,xt,wt,ft,o,packed,reverse,co,oh,ow,err);
        if(r) return r;
    }

    int32_t xh=oh+kh-1,xw=ow+kw-1,groups=ci/4,blocks=co/32;
    uint32_t nt=count_taps(wt,s2,reverse,packed,kh,kw,groups);
    size_t es=sixteen?2:1,pixels=(size_t)oh*ow;
    NearPlan plan;
    int32_t dims[4]={1,xh,xw,ci},origin[4]={0,xt->p[1],xt->p[2],0};
    int32_t on[4]={1,oh,ow,co},oorigin[4]={0,o->p[1],o->p[2],0};
    /* Blocked input (u8, or u16 split into byte planes) is staged straight into near unless the output blocks
     * could overwrite input that later rows still need. */
    int direct=0,use_near=0,checked=0;
    if(g->conv->near) {
        if((xt->flags&F_BLOCKED) && xt->esize==es && o->esize==es &&
           !xfer_check(g,xt,dims,origin,0) && !xfer_check(g,o,on,oorigin,1)) {
            checked=1;
            direct=!blocked_overlap(xt,dims,origin,o,on,oorigin);
        }
        int32_t head0=(int32_t)(xt->p[2]&(sixteen?3:7));
        use_near=direct && near_plan(g,&plan,ow,kh,kw,ci,co,nt,sixteen,1,head0);
        if(!use_near){direct=0;use_near=near_plan(g,&plan,ow,kh,kw,ci,co,nt,sixteen,0,head0);}
    }
#ifdef CONV_SHAPES
    if(!g->dry) fprintf(stderr,"CONV wpool=%u packed=%d rev=%d fx=%d op=%u var=%u s16=%d s2=%d k=%dx%d ci=%d co=%d oh=%d ow=%d xes=%u xblk=%d xp2=%d oes=%u oblk=%d op2=%d direct=%d near=%d cols=%d\n",
        wt->pool,is_class(wt,"Xpb","xpb"),!s2 && wt->cls==('X'|'w'<<8|'b'<<16),g->fx.kind,g->op,g->variant,sixteen,s2,kh,kw,ci,co,oh,ow,xt->esize,!!(xt->flags&F_BLOCKED),xt->p[2],o->esize,!!(o->flags&F_BLOCKED),
        o->p[2],direct,use_near,use_near?plan.cols:0);
#endif
    if(use_near) {
        g->wused=0;
        /* Direct staging needs no copy of the input; a token allocation keeps
         * the measured workspace nonzero for fast commands. */
        uint8_t *x=walloc(g,direct?128:(size_t)xh*xw*ci*es+256);
        if(g->wused>g->conv->workspace_bytes) return 0;
        if(!x){*err=E_MEMORY;return -1;}
        if(o->esize!=es){*err=E_TENSOR;return -1;}
        if(g->fx.kind) {
            /* The fused epilogue must cover exactly the region the conv writes. */
            if(sixteen || oh!=o->d[1] || ow!=o->d[2] || co!=o->d[3] || ow%8) return 0;
            g->fx.eb=walloc(g,(size_t)plan.cols*co);
            g->fx.bb=g->fx.kind==2?walloc(g,(size_t)plan.cols*co):NULL;
            if(g->wused>g->conv->workspace_bytes) return 0;
            if(!g->fx.eb || (g->fx.kind==2 && !g->fx.bb)){*err=E_MEMORY;return -1;}
            g->fx_ok=1;
        }
        if(!checked && ((*err=xfer_check(g,xt,dims,origin,0)) || (*err=xfer_check(g,o,on,oorigin,1)))) return -1;
        conv_path=2;
        if(g->dry) return 1;
        const Nprg2ReferenceConv *c=g->conv;
        if(!direct) {
            uint64_t t=c->clock && c->phase?c->clock():0;
            xfer_copy(g,xt,x,dims,origin,0);
            if(c->clock && c->phase) c->phase[sixteen?4:0]+=c->clock()-t;
        }
        near_compute(g,xt,wt,ft,o,s2,reverse,packed,sixteen,kh,kw,ci,co,oh,ow,direct?NULL:x,&plan);
        return 1;
    }
    g->wused=0;
    uint8_t *x=walloc(g,(size_t)xh*xw*ci*es+128),*xlo=sixteen?walloc(g,(size_t)xh*xw*ci+128):NULL,
            *xhi=sixteen?walloc(g,(size_t)xh*xw*ci+128):NULL,*r=walloc(g,pixels*co*es);
    uint32_t *wp=walloc(g,(size_t)((blocks+1)/2)*nt*8),*xoff=walloc(g,(size_t)nt*4),*segend=walloc(g,16);
    int32_t *acc=walloc(g,(size_t)2*PIXEL_BLOCK*32*4*(sixteen?2:1)),*mult=walloc(g,(size_t)co*4),
            *bias=walloc(g,(size_t)co*4),*M=walloc(g,(size_t)co*4),*pre=walloc(g,(size_t)co*4),*post=walloc(g,(size_t)co*4);
    int64_t *wsum=walloc(g,(size_t)co*8);
    int32_t *rqp=walloc(g,(size_t)co*RQ_VECS*4);
    if(g->wused>g->conv->workspace_bytes) return 0;
    if(!x || (sixteen && (!xlo || !xhi)) || !r || !wp || !xoff || !segend || !acc || !mult || !bias || !M || !pre || !post ||
       !wsum || !rqp){
        *err=E_MEMORY;return -1;
    }
    if((*err=xfer(g,xt,x,dims,origin,0))) return -1;
    if(!g->dry) fast_conv_compute(g,xt,wt,ft,s2,reverse,packed,sixteen,kh,kw,ci,co,oh,ow,x,xlo,xhi,r,nt,wp,xoff,segend,acc,
                                  mult,bias,M,pre,post,wsum,rqp);
    A out={.n={1,oh,ow,co},.dt=sixteen?DT_U16:DT_U8,.p=r};
    if((*err=t_write_region(g,o,&out,oorigin))) return -1;
    conv_path=1;
    return 1;
}
static void fast_conv_compute(G *g, const T *xt, const T *wt, const T *ft, int s2, int reverse, int packed, int sixteen,
                              int32_t kh, int32_t kw, int32_t ci, int32_t co, int32_t oh, int32_t ow, uint8_t *x,
                              uint8_t *xlo, uint8_t *xhi, uint8_t *r, uint32_t nt, uint32_t *wp, uint32_t *xoff,
                              uint32_t *segend, int32_t *acc, int32_t *mult, int32_t *bias, int32_t *M, int32_t *pre,
                              int32_t *post, int64_t *wsum, int32_t *rqp) {
    int32_t xw=ow+kw-1,groups=ci/4,blocks=co/32;
    const int8_t *wbase=(const int8_t *)g->pool[wt->pool]+wt->offset;
    const uint8_t *rows=g->pool[ft->pool]+ft->offset;
    for(int32_t c=0;c<co;c++){memcpy(mult+c,rows+(size_t)(c/32)*256+(c%32)*4,4);memcpy(bias+c,rows+(size_t)(c/32)*256+128+(c%32)*4,4);}
    requant_params(mult,co,sixteen,M,pre,post);
    build_taps(wt,s2,reverse,packed,kh,kw,groups,blocks,wbase,ci,0,1,(size_t)xw*ci,nt,wp,xoff,segend);
    memset(wsum,0,(size_t)co*8);
    if(xt->zp) tap_wsum(wsum,blocks,nt,wp,wbase);
    if(sixteen) {
        const uint16_t *s=(const uint16_t *)x;
        for(size_t i=0;i<(size_t)(oh+kh-1)*xw*ci;i++){xlo[i]=(uint8_t)s[i];xhi[i]=(uint8_t)(s[i]>>8);}
    }
    uint32_t stride=(uint32_t)ci;
    int vec=RQ_HVX && rq_prepare(rqp,co,sixteen,(int64_t)kh*kw*ci*255*128,xt->zp,wsum,bias,M,pre,post);
    (void)vec;
    for(int32_t y=0;y<oh;y++)for(int32_t x0=0;x0<ow;x0+=PIXEL_BLOCK) {
        int32_t np=ow-x0<PIXEL_BLOCK?ow-x0:PIXEL_BLOCK;
        size_t at=((size_t)y*xw+x0)*ci;
        for(int32_t j=0;j<blocks;j+=2) {
            int32_t nb=blocks-j<2?blocks-j:2;
            const uint32_t *wpp=wp+(size_t)(j/2)*nt*2;
            int32_t *hi=acc+2*PIXEL_BLOCK*32;
            const uint8_t *xb=x+at,*xbl=xlo+at,*xbh=xhi+at;
#if RQ_HVX
            if(vec) {
                size_t es=sixteen?2:1;
                RqOut ro={(const HVX_Vector *)rqp+(size_t)j*RQ_VECS,r+(((size_t)y*ow+x0)*co+(size_t)j*32)*es,
                          (uint32_t)(co*es),sixteen,0,0,{{NULL}},NULL};
                if(sixteen) {
                    tile_block(acc,np,&xbl,1,segend,stride,wpp,nb==2,xoff,wbase,NULL,NULL);
                    tile_block(hi,np,&xbh,1,segend,stride,wpp,nb==2,xoff,wbase,&ro,(const HVX_Vector *)acc);
                } else tile_block(acc,np,&xb,1,segend,stride,wpp,nb==2,xoff,wbase,&ro,NULL);
                continue;
            }
#endif
            if(sixteen) {
                tile_block(acc,np,&xbl,1,segend,stride,wpp,nb==2,xoff,wbase,NULL,NULL);
                tile_block(hi,np,&xbh,1,segend,stride,wpp,nb==2,xoff,wbase,NULL,NULL);
            } else tile_block(acc,np,&xb,1,segend,stride,wpp,nb==2,xoff,wbase,NULL,NULL);
            for(int32_t b=0;b<nb;b++)for(int32_t p=0;p<np;p++)for(int32_t c=0;c<32;c++) {
                int32_t k=(j+b)*32+c;
                size_t a_at=((size_t)b*PIXEL_BLOCK+p)*32+c;
                int64_t a=sixteen?(int64_t)hi[a_at]*256+acc[a_at]:(int64_t)acc[a_at];
                a-=(int64_t)xt->zp*wsum[k];
                int64_t v=requant_value(a,bias[k],sixteen,M[k],pre[k],post[k]);
                size_t idx=((size_t)y*ow+x0+p)*co+k;
                if(sixteen)((uint16_t *)r)[idx]=(uint16_t)v;else r[idx]=(uint8_t)v;
            }
        }
    }
}
/* Per-thread part of the near layout: accumulators, an output chunk and the
 * staged row ring. Everything else near_layout places (tap tables, requant
 * parameters) is read-only while rows run, so all parts share it. */
typedef struct { int32_t cols; size_t rowbytes, acc, out, lo, hi, bytes; } PrivPlan;
static size_t priv_layout(PrivPlan *q, int32_t cols, int32_t kh, int32_t kw, int32_t ci, int32_t co, int sixteen,
                          int direct, int32_t lead, int with_out) {
    size_t at=0;
    q->cols=cols;
    size_t npix=(size_t)lead+(size_t)cols+(size_t)kw-1,run=sixteen?4:8;
    if(direct && ci%128) npix=(npix+run-1)&~(run-1);
    q->rowbytes=(npix*(size_t)ci+127)&~(size_t)127;
    q->acc=nbump(&at,(size_t)2*PIXEL_BLOCK*32*4*(sixteen?2:1));
    q->out=with_out?nbump(&at,(size_t)cols*co*(sixteen?2:1)):0;
    q->lo=nbump(&at,(size_t)kh*q->rowbytes);q->hi=sixteen?nbump(&at,(size_t)kh*q->rowbytes):0;
    q->bytes=at;
    return at;
}
/* Helper-thread slots (Nprg2ReferenceConv.par_near): the widest chunk that
 * fits a slot, or 0. with_out: the part also needs an output chunk (output
 * not stored straight into the blocks; host only). */
static int priv_plan_slot(const Nprg2ReferenceConv *c, PrivPlan *q, int32_t ow, int32_t kh, int32_t kw, int32_t ci,
                          int32_t co, int sixteen, int direct, int32_t lead, int with_out) {
    for(int32_t cols=(ow+PIXEL_BLOCK-1)/PIXEL_BLOCK*PIXEL_BLOCK;cols>0;cols-=PIXEL_BLOCK)
        if(priv_layout(q,cols,kh,kw,ci,co,sixteen,direct,lead,with_out)<=c->par_near_bytes) return 1;
    return 0;
}
/* Everything the row loop reads, shared by all parts of one command. */
typedef struct {
    G *g; const T *xt, *o;
    int sixteen, vec, direct_out;
    int32_t kh, kw, ci, co, oh, ow, oa, rwsh, onbw, onbd, L;
    const uint8_t *x;
    const NearPlan *p;
    uint32_t parts;
    PrivPlan q; /* the plan helper parts use */
    const uint32_t *wp, *xoff, *segend;
    const int32_t *rqp;
    const int8_t *wbase;
    uint64_t hph[2][4]; /* helper parts' phase cycles, added to phase after the join */
} NearJob;
/* Output rows [oh*part/parts, oh*(part+1)/parts) over every column chunk.
 * Part 0 runs in the command's own near region (its plan p); parts 1.. in the
 * helper slot par_near[part-1] (plan job->q). Parts write only output rows of
 * their own range (distinct 256-byte block rows), and read the input and the
 * shared tables, so they are independent. */
static void near_rows(const NearJob *J, uint32_t part) {
    G *g=J->g;const Nprg2ReferenceConv *c=g->conv;
    const NearPlan *p=J->p;
    const int sixteen=J->sixteen;
    const int32_t kh=J->kh,kw=J->kw,ci=J->ci,co=J->co,ow=J->ow,xw=ow+kw-1,blocks=co/32;
    const uint32_t nt=p->nt;
    uint8_t *n,*out,*lo,*hi;int32_t *acc;int32_t cols;size_t rowb;
    if(part==0) {
        n=c->near;out=n+p->out;lo=n+p->lo;hi=sixteen?n+p->hi:NULL;acc=(int32_t *)(n+p->acc);cols=p->cols;rowb=p->rowbytes;
    } else {
        n=c->par_near[part-1u];out=n+J->q.out;lo=n+J->q.lo;hi=sixteen?n+J->q.hi:NULL;acc=(int32_t *)(n+J->q.acc);
        cols=J->q.cols;rowb=J->q.rowbytes;
    }
    const uint32_t *wp=J->wp,*xoff=J->xoff,*segend=J->segend;
    const int32_t *rqp=J->rqp;const int8_t *wbase=J->wbase;(void)rqp;
    const T *xt=J->xt,*o=J->o;const uint8_t *x=J->x;
    const int vec=J->vec;(void)vec;(void)nt;(void)xoff;(void)blocks;
    const int32_t ys=(int32_t)((int64_t)J->oh*part/J->parts),ye=(int32_t)((int64_t)J->oh*(part+1u)/J->parts);
    /* Phases: u8 in phase[0..3], u16 in phase[4..7]. Helper parts time into
     * their own job slots; near_compute adds them once every part is done. */
    uint64_t (*clock)(void)=c->phase?c->clock:NULL,
             *ph=!clock?NULL:part==0?c->phase+(sixteen?4:0):(uint64_t *)J->hph[part-1u],t=clock?clock():0;
    const Fuse *fx=g->fx.kind?&g->fx:NULL;
#define LAP(i) do { if(clock){uint64_t u_=clock();ph[i]+=u_-t;t=u_;} } while(0)
    size_t es=sixteen?2:1;
    int32_t *ahi=acc+2*PIXEL_BLOCK*32;
    int direct=J->direct_out;
    const int32_t oa=J->oa,RW=sixteen?4:8,L=J->L,rwsh=J->rwsh,onbw=J->onbw,onbd=J->onbd;
    (void)RW;(void)L;(void)rwsh;(void)onbw;(void)onbd;(void)oa;
    for(int32_t x0=0;x0<ow;x0+=cols) {
        int32_t nc=ow-x0<cols?ow-x0:cols;
        int dchunk=direct && !(sixteen && (nc&1));
        int64_t W0=(int64_t)xt->p[2]+x0;
        for(int32_t y=ys;y<ye;y++) {
            for(int32_t r=y>ys?y+kh-1:ys;r<=y+kh-1;r++) {
                size_t rs=(size_t)(r%kh)*rowb;
                uint8_t *dl=lo+rs;
                if(!x && sixteen) stage_row16(g->pool,xt,(int64_t)xt->p[1]+r,W0,nc+kw-1,W0-p->lead,ci,dl,hi+rs);
                else if(!x) stage_row8(g->pool,xt,(int64_t)xt->p[1]+r,W0,nc+kw-1,W0-p->lead,ci,dl);
                else {
                    const uint8_t *src=x+(((size_t)r*xw+x0)*ci)*es;
                    size_t span=(size_t)(nc+kw-1)*ci;
                    if(sixteen) near_split16(dl,hi+rs,src,span);
                    else near_copy(dl,src,span);
                }
            }
            LAP(0);
            size_t slot[4];
            for(int32_t s=0,r=y%kh;s<kh;s++,r=r+1==kh?0:r+1) slot[s]=(size_t)r*rowb;
            for(int32_t px=0;px<nc;px+=PIXEL_BLOCK) {
                int32_t np=nc-px<PIXEL_BLOCK?nc-px:PIXEL_BLOCK;
                const uint8_t *xl[4],*xh[4];
                for(int32_t s=0;s<kh;s++) {
                    size_t at=slot[s]+(size_t)px*ci;
                    xl[s]=lo+at;xh[s]=hi?hi+at:NULL;
                }
                for(int32_t j=0;j<blocks;j+=2) {
                    int32_t nb=blocks-j<2?blocks-j:2;
                    const uint32_t *wpp=wp+(size_t)(j/2)*nt*2;
#if RQ_HVX
                    /* Fused: requant time lands in the mac phase. */
                    if(vec) {
                        RqOut ro={(const HVX_Vector *)rqp+(size_t)j*RQ_VECS,out+((size_t)px*co+(size_t)j*32)*es,
                                  (uint32_t)(co*es),sixteen,0,0,{{NULL}},NULL};
                        FxOut fo;
                        if(dchunk) {
                            int32_t wb=(o->p[2]+x0+px-oa)>>rwsh;
                            ro.blocked=1;ro.a=oa;
                            for(int32_t b=0;b<nb;b++)for(int32_t i=0;i<L;i++)
                                ro.blk[b][i]=blocked_rowptr(g->pool,o,o->p[1]+y,wb+i,onbw,onbd,j+b);
                            if(fx) {
                                /* Same geometry as o (proved by fuse_decode), so the same slots. */
                                fo=(FxOut){fx->mc,fx->mo,fx->rd,fx->sh,fx->kind,fx->dead,{NULL,NULL},{NULL,NULL}};
                                for(int32_t b=0;b<nb;b++) {
                                    fo.eblk[b]=blocked_rowptr(g->pool,&fx->oe,fx->oe.p[1]+y,wb,onbw,onbd,j+b);
                                    if(fx->kind==2) fo.bblk[b]=blocked_rowptr(g->pool,&fx->ob,fx->ob.p[1]+y,wb,onbw,onbd,j+b);
                                }
                                ro.fx=&fo;
                            }
                        }
                        if(sixteen) {
                            tile_block(acc,np,xl,(uint32_t)kh,segend,(uint32_t)ci,wpp,nb==2,xoff,wbase,NULL,NULL);
                            tile_block(ahi,np,xh,(uint32_t)kh,segend,(uint32_t)ci,wpp,nb==2,xoff,wbase,&ro,(const HVX_Vector *)acc);
                        } else tile_block(acc,np,xl,(uint32_t)kh,segend,(uint32_t)ci,wpp,nb==2,xoff,wbase,&ro,NULL);
                        LAP(1);
                        continue;
                    }
#endif
                    {
                    const int64_t *wsum=(const int64_t *)((uint8_t *)c->near+p->wsum);
                    const int32_t *bias=(const int32_t *)((uint8_t *)c->near+p->bias),*M=(const int32_t *)((uint8_t *)c->near+p->M),
                                  *pre=(const int32_t *)((uint8_t *)c->near+p->pre),*post=(const int32_t *)((uint8_t *)c->near+p->post);
                    tile_block(acc,np,xl,(uint32_t)kh,segend,(uint32_t)ci,wpp,nb==2,xoff,wbase,NULL,NULL);
                    if(sixteen) tile_block(ahi,np,xh,(uint32_t)kh,segend,(uint32_t)ci,wpp,nb==2,xoff,wbase,NULL,NULL);
                    LAP(1);
                    for(int32_t b=0;b<nb;b++)for(int32_t q=0;q<np;q++)for(int32_t l=0;l<32;l++) {
                        int32_t k=(j+b)*32+l;
                        size_t a_at=((size_t)b*PIXEL_BLOCK+q)*32+l;
                        int64_t a=sixteen?(int64_t)ahi[a_at]*256+acc[a_at]:(int64_t)acc[a_at];
                        a-=(int64_t)xt->zp*wsum[k];
                        int64_t v=requant_value(a,bias[k],sixteen,M[k],pre[k],post[k]);
                        size_t idx=(size_t)(px+q)*co+k;
                        if(sixteen)((uint16_t *)out)[idx]=(uint16_t)v;else out[idx]=(uint8_t)v;
                    }
                    }
                    LAP(2);
                }
            }
            if(!dchunk) {
                int32_t dims[4]={1,1,nc,co},origin[4]={0,o->p[1]+y,o->p[2]+x0,0};
                if(!(fx && fx->dead)) xfer_copy(g,o,out,dims,origin,1);
                if(fx) fuse_scalar(g,fx,out,nc,co,y,x0);
            }
            LAP(3);
        }
    }
#undef LAP
}
static void near_part(void *ctx, uint32_t part, uint32_t parts) {
    (void)parts;
    near_rows((const NearJob *)ctx,part);
}
/* fast_conv_compute over near memory, one column chunk at a time: the input
 * rows live in a ring of kh staged rows (each input row is staged once per
 * chunk, not kh times); a row is staged from the blocked input (x NULL) or
 * from the workspace copy x. Each output row accumulates, requantizes and,
 * for blocked outputs on the vector path, stores straight into o's blocks
 * (u8 or u16, any column alignment; u16 needs even pixel pairs), else into a
 * near output chunk written by xfer_copy.
 *
 * Threads (call->parallel): setup below fills the shared tables in near;
 * then, if every chunk stores straight into o's blocks and the helper slots
 * fit, the output rows are split over up to 3 parts (near_rows). */
static void near_compute(G *g, const T *xt, const T *wt, const T *ft, const T *o, int s2, int reverse, int packed,
                         int sixteen, int32_t kh, int32_t kw, int32_t ci, int32_t co, int32_t oh, int32_t ow,
                         const uint8_t *x, const NearPlan *p) {
    const Nprg2ReferenceConv *c=g->conv;
    uint8_t *n=c->near;
    uint64_t t_setup=c->clock && c->phase?c->clock():0;
#ifdef NPRG2_FUSE_POISON
    /* Host test: a dead conv output must not be observable, so scribble over it. */
    if(g->fx.kind && g->fx.dead) {
        uint64_t nblk=(uint64_t)(o->m[1]/8)*(uint64_t)(o->m[2]/8)*(uint64_t)(o->m[3]/32);
        for(uint64_t i=0;i<nblk;i++) {
            uint32_t e=o->blocks[o->slice+i];
            if((e&127u)==1u) memset(g->pool[1]+(e&~127u),0xA5,2048);
        }
    }
#endif
    int32_t *mult=(int32_t *)(n+p->mult),*bias=(int32_t *)(n+p->bias);
    int32_t *M=(int32_t *)(n+p->M),*pre=(int32_t *)(n+p->pre),*post=(int32_t *)(n+p->post);
    uint32_t *wp=(uint32_t *)(n+p->wp),*xoff=(uint32_t *)(n+p->xoff),*segend=(uint32_t *)(n+p->segend);
    int64_t *wsum=(int64_t *)(n+p->wsum);
    int32_t *rqp=(int32_t *)(n+p->rq);
    int32_t groups=ci/4,blocks=co/32;
    uint32_t nt=p->nt;
    const int8_t *wbase=(const int8_t *)g->pool[wt->pool]+wt->offset;
    const uint8_t *rows=g->pool[ft->pool]+ft->offset;
    /* The filter rows are in VTCM: one vector copy into the (not yet used)
     * row ring, then scalar reads from near, not ~45-cycle scalar VTCM loads. */
    if((size_t)kh*p->rowbytes>=(size_t)blocks*256) {
        near_copy(n+p->lo,rows,(size_t)blocks*256);
        rows=n+p->lo;
    }
    for(int32_t k=0;k<co;k++){memcpy(mult+k,rows+(size_t)(k/32)*256+(k%32)*4,4);memcpy(bias+k,rows+(size_t)(k/32)*256+128+(k%32)*4,4);}
    requant_params(mult,co,sixteen,M,pre,post);
    build_taps(wt,s2,reverse,packed,kh,kw,groups,blocks,wbase,ci,p->lead,0,0,nt,wp,xoff,segend);
    memset(wsum,0,(size_t)co*8);
    if(xt->zp) tap_wsum(wsum,blocks,nt,wp,wbase);
    int vec=RQ_HVX && rq_prepare(rqp,co,sixteen,(int64_t)kh*kw*ci*255*128,xt->zp,wsum,bias,M,pre,post);
    (void)vec;
    /* Blocked outputs are requantized straight into o's blocks, skipping the
     * near output chunk and its xfer_copy. oa: the first column's slot. */
    size_t es=sixteen?2:1;
    int direct=vec && (o->flags&F_BLOCKED) && o->esize==es;
    int32_t RW=sixteen?4:8,oa=(int32_t)(o->p[2]%RW),L=(oa+PIXEL_BLOCK+RW-1)/RW;
    if(sixteen && (oa&1)) direct=0;
    NearJob J={.g=g,.xt=xt,.o=o,.sixteen=sixteen,.vec=vec,.direct_out=direct,.kh=kh,.kw=kw,.ci=ci,.co=co,.oh=oh,.ow=ow,
               .oa=oa,.rwsh=sixteen?2:3,.onbw=(o->m[2]+RW-1)>>(sixteen?2:3),.onbd=(o->m[3]+31)>>5,.L=L,.x=x,.p=p,.parts=1,
               .wp=wp,.xoff=xoff,.segend=segend,.rqp=rqp,.wbase=wbase};
    /* Parts: every chunk must store straight into o's blocks (u16 needs an
     * even width for the last chunk too), rows must be enough, the helper
     * slots must hold a plan, and the work must pay for the wake-up. */
    uint32_t parts=1;
    if(g->par_slots && (direct || !BLK_HVX) && !(direct && sixteen && (ow&1)) && oh>=4 &&
       (uint64_t)oh*(uint64_t)ow*(uint64_t)co*(uint64_t)kh*(uint64_t)kw*(uint64_t)ci>=PAR_CONV_MIN_MACS &&
       priv_plan_slot(c,&J.q,ow,kh,kw,ci,co,sixteen,p->direct,p->lead,!direct)) {
        /* At least two output rows per part: a part restages kh-1 halo rows,
         * cheap next to the MACs (an 8-row tile on 3 parts stages 14 rows, not 10). */
        uint32_t want=1u+g->par_slots;
        if((uint32_t)oh/2u<want) want=(uint32_t)oh/2u;
        parts=nprg2_parallel_parts(g->call->parallel,want);
    }
    J.parts=parts;
    /* Setup (tables, requant parameters) runs once, before the parts: phase slot 2. */
    if(c->clock && c->phase) c->phase[(sixteen?4:0)+2]+=c->clock()-t_setup;
#ifdef CONV_SHAPES
    fprintf(stderr,"PARTS s16=%d k=%dx%d ci=%d co=%d oh=%d ow=%d parts=%u slots=%u qcols=%d\n",sixteen,kh,kw,ci,co,oh,ow,parts,
            g->par_slots,parts>1?J.q.cols:0);
#endif
    if(parts>1) {
        nprg2_parallel_for(g->call->parallel,NPRG2_PAR_CONV,parts,near_part,&J);
        if(c->phase) for(int k=0;k<4;k++) c->phase[(sixteen?4:0)+k]+=J.hph[0][k]+J.hph[1][k];
    } else near_rows(&J,0);
}

/* ------------------------------------------------------- conv epilogue fusion */
static int setup(G *g, const Nprg2ExecCall *c, int dry, int measure, const Nprg2ReferenceConv *conv);
enum { FZ_BLOCKS=4096, FZ_WORDS=FZ_BLOCKS/32 };
static int fz_tensor_ok(const T *t) {
    if(!(t->flags&F_SHAPE) || !(t->flags&F_BLOCKED) || t->dtype!=DT_U8 || t->esize!=1) return 0;
    for(int i=0;i<4;i++) if(t->p[i]!=0 || t->d[i]!=t->m[i]) return 0;
    if(!(t->m[0]==1 && t->m[1]>0 && t->m[1]%8==0 && t->m[2]>0 && t->m[2]%8==0 && t->m[3]>0 && t->m[3]%32==0)) return 0;
    return (uint64_t)t->slice+(uint64_t)(t->m[1]/8)*(uint64_t)(t->m[2]/8)*(uint64_t)(t->m[3]/32)<=t->nblocks;
}
static uint64_t fz_count(const T *t) {return (uint64_t)(t->m[1]/8)*(uint64_t)(t->m[2]/8)*(uint64_t)(t->m[3]/32);}
/* Strict: every entry a present, aligned, distinct pool-1 block inside limit. */
static int fz_mark(const T *t, uint32_t *bm, uint64_t limit) {
    uint64_t n=fz_count(t);
    if((uint64_t)t->slice+n>t->nblocks) return 0;
    for(uint64_t i=0;i<n;i++) {
        uint32_t e=t->blocks[t->slice+i],off=e&~127u;
        if((e&127u)!=1 || (off&2047u) || off/2048u>=FZ_BLOCKS || (uint64_t)off+2048>limit) return 0;
        uint32_t b=off/2048u;
        if(bm[b>>5]&(1u<<(b&31))) return 0;
        bm[b>>5]|=1u<<(b&31);
    }
    return 1;
}
/* Lenient: marks whatever pool-1 blocks the tensor's extent may touch. */
static int fz_mark_any(const T *t, uint32_t *bm) {
    if(t->flags&F_DIRECT) {
        if(t->pool!=1) return 1;
        uint64_t n=t->esize;
        for(int i=0;i<4;i++){if(t->m[i]<=0) return 0;n*=(uint64_t)t->m[i];}
        for(uint64_t b=t->offset/2048u;b<=((uint64_t)t->offset+n-1)/2048u;b++){
            if(b>=FZ_BLOCKS) return 0;
            bm[b>>5]|=1u<<(b&31);
        }
        return 1;
    }
    uint64_t lo=t->slice,hi=t->nblocks;
    if((t->flags&F_SHAPE) && t->esize) {
        int64_t cw=t->esize==1?8:t->esize==2?4:2;
        uint64_t n=(uint64_t)(t->m[0]>0?t->m[0]:0)*(uint64_t)((t->m[1]+7)/8)*(uint64_t)((t->m[2]+cw-1)/cw)*(uint64_t)((t->m[3]+31)/32);
        if(lo+n<hi) hi=lo+n;
    }
    for(uint64_t i=lo;i<hi;i++) {
        uint32_t e=t->blocks[i],off=e&~127u;
        if((e&127u)!=1) continue;
        if((off&2047u) || off/2048u>=FZ_BLOCKS) return 0;
        uint32_t b=off/2048u;
        bm[b>>5]|=1u<<(b&31);
    }
    return 1;
}
static int fz_same_table(const T *a, const T *b) {
    uint64_t n=fz_count(a);
    for(uint64_t i=0;i<n;i++) if(a->blocks[a->slice+i]!=b->blocks[b->slice+i]) return 0;
    return 1;
}
static int fz_meet(const uint32_t *a, const uint32_t *b) {
    for(int i=0;i<FZ_WORDS;i++) if(a[i]&b[i]) return 1;
    return 0;
}
/* Decodes and proves the epilogue command ec against the conv (output o,
 * input xt, weights wt, Fi ft). Fills g->fx or rejects. Everything about the
 * epilogue is checked here, not trusted from the plan: command key,
 * parameters (the u32 lane arithmetic below cannot overflow), tensor shape
 * and block tables, and block-level aliasing between its views and the
 * conv's. */
static int fuse_decode(G *g, const T *xt, const T *wt, const T *ft, const T *o) {
    const Nprg2ExecCall *ec=g->call->epilogue;
    Fuse *fx=&g->fx;
    memset(fx,0,sizeof *fx);
    CHECK(ec && !ec->epilogue && g->pool[1] && !((uintptr_t)g->pool[1]&127u),E_ARGS);
    G h;
    TRY(setup(&h,ec,1,0,NULL));
    int add;
    if(h.op==QDR_REQUANT_U8 && h.variant==70) add=0;
    else if(h.op==QDR_ADD_U8 && h.variant==13) add=1;
    else FAIL(E_UNSUPPORTED);
    CHECK(h.nout==1 && h.nin>=(add?2u:1u) && h.nin<=CACHED_INPUTS,E_ARGS);
    CHECK(!add || h.nin==2,E_ARGS);
    int64_t r[10];
    const T *a=in(&h,0),*b=add?in(&h,1):NULL,*e=out0(&h);
    uint32_t mc,mo=0,rd,sh;
    int conv_op=0;
    CHECK(fz_tensor_ok(o) && fz_tensor_ok(a) && fz_tensor_ok(e) && (!add || fz_tensor_ok(b)),E_UNSUPPORTED);
    CHECK(!memcmp(o->m,a->m,16) && !memcmp(o->m,e->m,16) && (!add || !memcmp(o->m,b->m,16)),E_UNSUPPORTED);
    if(!add) {
        TRY(call_args(&h,QDR_CALL_REQUANTIZE_CROUTON_U8,r,0x38));
        CHECK(r[4]==0 && r[5]==0 && small(r,0x8),E_UNSUPPORTED);
        int64_t m=r[3]&0xffff,s=15-(r[3]>>24);
        CHECK(s>=1 && s<=31,E_ARGS);
        mc=(uint32_t)m;rd=1u<<(s-1);sh=(uint32_t)s;
        CHECK(fz_same_table(o,a),E_UNSUPPORTED);
    } else {
        TRY(call_args(&h,QDR_CALL_ELEMENTWISE_ADD_CROUTON_U8,r,0x1f0));
        CHECK(r[7]==0 && small(r,0x70),E_UNSUPPORTED);
        int64_t k0=r[4],k1=r[5],k2=r[6]+8;
        CHECK(shift_ok(k2) && k2>=1,E_ARGS);
        CHECK(k0>=0 && k1>=0 && k0<=65535 && k1<=65535 && k2<=31 &&
              ((uint64_t)k0+(uint64_t)k1)*255+(1ull<<(k2-1))<=0xffffffffull,E_UNSUPPORTED);
        int ma=fz_same_table(o,a),mb=fz_same_table(o,b);
        CHECK(ma!=mb,E_UNSUPPORTED);
        conv_op=mb;
        mc=(uint32_t)(conv_op?k1:k0);mo=(uint32_t)(conv_op?k0:k1);rd=1u<<(k2-1);sh=(uint32_t)k2;
        fx->ob=*in(&h,conv_op?0:1);
    }
    fx->kind=add?2:1;fx->dead=g->call->epilogue_dead;fx->mc=mc;fx->mo=mo;fx->rd=rd;fx->sh=sh;fx->oe=*e;
    /* Strict tables, then aliasing. */
    uint32_t O[FZ_WORDS]={0},Z[FZ_WORDS]={0},B[FZ_WORDS]={0},X[FZ_WORDS]={0},W[FZ_WORDS]={0};
    CHECK(fz_mark(o,O,g->write_limit[1]) && fz_mark(e,Z,g->write_limit[1]),E_UNSUPPORTED);
    if(add) CHECK(fz_mark(&fx->ob,B,g->pool_bytes[1]),E_UNSUPPORTED);
    CHECK(fz_mark_any(xt,X) && fz_mark_any(wt,W) && fz_mark_any(ft,W),E_UNSUPPORTED);
    int inplace=fz_meet(O,Z);
    if(inplace) CHECK(fz_same_table(o,e) && fx->dead,E_UNSUPPORTED);
    CHECK(!fz_meet(Z,X) && !fz_meet(Z,W) && !fz_meet(Z,B) && !fz_meet(B,O),E_UNSUPPORTED);
    return 0;
}
static int k_hmx_conv(G *g) {
    CHECK(g->nin>=3 && g->nout==1,E_ARGS);
    const T *xt=in(g,0),*wt=in(g,1),*ft=in(g,2),*o=out0(g);
    int s2=g->variant==36;
    CHECK((ft->flags&F_DIRECT) && ft->dtype==DT_I32 && ft->pool<7 && g->pool[ft->pool],E_TENSOR);
    for(int i=0;i<4;i++)CHECK(ft->m[i]>=0,E_SHAPE);
    size_t nfi=count4(ft->m);
    CHECK(nfi%64==0 && ft->offset+(uint64_t)nfi*4<=g->pool_bytes[ft->pool],E_RANGE);
    size_t nm=nfi/2;
    int sixteen=xt->cls==('C'|'H'<<8) || xt->cls==('c'|'H'<<8);
    CHECK((o->flags&F_SHAPE) && (xt->flags&F_SHAPE),E_TENSOR);
    int32_t oh=o->d[1],ow=o->d[2],co=(o->d[3]+31)/32*32;
    int32_t wco=wt->m[3];
    if(co>wco)co=wco;
    if((size_t)co>nm)co=(int32_t)nm;
    int32_t ferr;
    int fused=g->call->epilogue!=NULL;
    if(fused) {
        CHECK(!sixteen,E_UNSUPPORTED);
        TRY(fuse_decode(g,xt,wt,ft,o));
    }
    int fast=fast_conv(g,xt,wt,ft,o,s2,sixteen,co,oh,ow,&ferr);
    if(fast<0) FAIL(ferr);
    /* A fused conv must take the fused path: its epilogue command is skipped. */
    if(fused) CHECK(fast==1 && g->fx_ok,E_UNSUPPORTED);
    if(fast) return 0;
    int32_t *mult=salloc(g,nm*4+4),*bias=salloc(g,nm*4+4);
    CHECK(mult && bias,E_MEMORY);
    if(!g->dry) {
        const uint8_t *rows=g->pool[ft->pool]+ft->offset;
        for(size_t r=0;r<nfi/64;r++){memcpy(mult+r*32,rows+r*256,128);memcpy(bias+r*32,rows+r*256+128,128);}
    }
    Weights w;
    TRY(conv_weights(g,wt,s2,co,&w));
    int32_t xH=xt->m[1]-xt->p[1],xW=xt->m[2]-xt->p[2];
    CHECK(w.ci<=xt->m[3],E_SHAPE);
    int32_t cw=sixteen?4:8;
    oh=(oh+7)/8*8;if(oh>o->m[1])oh=o->m[1];if(oh>xH-w.kh+1)oh=xH-w.kh+1;
    ow=(ow+cw-1)/cw*cw;if(ow>o->m[2])ow=o->m[2];if(ow>xW-w.kw+1)ow=xW-w.kw+1;
    CHECK(oh>0 && ow>0,E_SHAPE);
    int32_t dims[4]={1,oh+w.kh-1,ow+w.kw-1,w.ci},origin[4]={0,xt->p[1],xt->p[2],0};
    int32_t on[4]={1,oh,ow,co},oorigin[4]={0,o->p[1],o->p[2],0};
    size_t pixels=(size_t)oh*ow,values=pixels*co;
    A x;TRY(t_read_region(g,xt,&x,dims,origin));
    A r;
    int64_t *acc=salloc(g,values*8);
    CHECK(acc,E_MEMORY);
    TRY(anew(g,&r,on,sixteen?DT_U16:DT_U8));
    int32_t *row=salloc(g,(size_t)co*4+64),*M=salloc(g,(size_t)co*4),*pre=salloc(g,(size_t)co*4),*post=salloc(g,(size_t)co*4);
    CHECK(row && M && pre && post,E_MEMORY);
    if(!g->dry) {
        cpu_accumulate(&x,&w,acc,row,oh,ow);
        for(size_t p=0;p<pixels;p++)for(int32_t k=0;k<co;k++)acc[p*co+k]-=(int64_t)xt->zp*w.wsum[k];
        requant(r.p,acc,mult,bias,pixels,co,sixteen,M,pre,post);
    }
    return t_write_region(g,o,&r,oorigin);
}

/* ------------------------------------------------------------------ dispatch */
static int run_kernel(G *g) {
#ifdef ALL_SHAPES
    if(g->op!=QDR_HMX_CONV) SHP("op",g,0);
#endif
    switch(g->op) {
    case QDR_NOP: return 0;
    case QDR_INPUT_SLICE_PAD: return k_input_slice_pad(g);
    case QDR_INPUT_SLICE: return k_input_slice(g);
    case QDR_WEIGHTS_TO_VTCM: return k_weights_to_vtcm(g);
    case QDR_IM2COL: return k_im2col(g);
    case QDR_HMX_CONV: return k_hmx_conv(g);
    case QDR_SPACE_TO_DEPTH: return k_space_to_depth(g);
    case QDR_REQUANT_U8: return k_requant_u8(g);
    case QDR_REQUANT_U16: return k_requant_u16(g);
    case QDR_ADD_U8: return k_add_u8(g);
    case QDR_ADD_U16: return k_add_u16(g);
    case QDR_ADDSUB_U16: return k_addsub_u16(g);
    case QDR_ADD1111_U16: return k_add1111_u16(g);
    case QDR_MUL_U16: return k_mul_u16(g);
    case QDR_MUL1111_U16: return k_mul1111_u16(g);
    case QDR_COPY_VALUES: return k_copy_values(g);
    case QDR_COPY_ACTIVATION: return k_copy_activation(g);
    case QDR_TRANSPOSE: return k_transpose(g);
    case QDR_CONCAT: return k_concat(g);
    case QDR_GATHER: return k_gather(g);
    case QDR_SLICE: return k_slice(g);
    case QDR_SLICE_PAD: return k_slice_pad(g);
    case QDR_DECIMATE: return k_decimate(g);
    case QDR_RESIZE: return k_resize(g);
    case QDR_DEPTH_TO_SPACE: return k_depth_to_space(g);
    case QDR_SPILL_FILL: return k_spill_fill(g,g->variant==9);
    case QDR_OUTPUT_SLICE: return k_output_slice(g);
    case QDR_QUANTIZE: return k_quantize(g);
    case QDR_DEQUANTIZE: return k_dequantize(g);
    case QDR_SIGN: return k_sign(g);
    case QDR_MEAN: return k_mean(g);
    case QDR_WEIGHT_SCALES: return k_weight_scales(g);
    case QDR_INVSCALE: return k_invscale(g);
    case QDR_COMBINE_SCALES: return k_combine_scales(g);
    case QDR_CONVERT_WEIGHTS: return k_convert_weights(g);
    case QDR_SCALE_CONVERT: return k_scale_convert(g);
    case QDR_DYNAMIC_BIAS: return k_dynamic_bias(g);
    case QDR_BIAS_SHUFFLE: return k_bias_shuffle(g);
    case QDR_LINEARCLIP: return k_linearclip(g);
    case QDR_ABS: return k_abs(g);
    case QDR_MUL_INLINE: return k_mul_inline(g);
    case QDR_ADDSUB_FLAT: return k_addsub_flat(g);
    case QDR_ADDSUB_INLINE: return k_addsub_inline(g);
    case QDR_BLOCK_ZAP: return k_block_zap(g);
    default: FAIL(E_UNSUPPORTED);
    }
}

/* Duplicate-pad NOP records keep their source InputSlicePad variant. */
static int variant_matches(uint32_t op, uint32_t variant) {
    if(!op || op>QDR_OPCODES || !variant || variant>QDR_VARIANTS) return 0;
    return qdr_variant_ops[variant]==op || (op==QDR_NOP && qdr_variant_ops[variant]==QDR_INPUT_SLICE_PAD);
}
static int setup(G *g, const Nprg2ExecCall *c, int dry, int measure, const Nprg2ReferenceConv *conv) {
    memset(g,0,sizeof *g);
    if(!c || !c->program || !c->bound || !c->pools) FAIL(E_ARGS);
    const Nprg2Command *k=&c->command;
    if(!variant_matches(k->opcode,k->variant)) FAIL(E_ARGS);
    g->call=c;g->dry=dry;g->measure_only=(uint8_t)measure;
    g->scratch=measure?NULL:(uint8_t *)c->scratch;
    g->cap=measure?SIZE_MAX:(c->scratch?c->scratch_bytes:0);
    if(conv && k->opcode==QDR_HMX_CONV && conv->workspace && conv->workspace_bytes) g->conv=conv;
    if(g->conv && g->conv->near) {
        uintptr_t n=(uintptr_t)conv->near,ne=n+conv->near_bytes,w=(uintptr_t)conv->workspace;
        if(n<w+conv->workspace_bytes && w<ne) FAIL(E_MEMORY);
        if(c->scratch && c->scratch_bytes && n<(uintptr_t)c->scratch+c->scratch_bytes && (uintptr_t)c->scratch<ne)
            FAIL(E_MEMORY);
        for(unsigned i=0;i<7;i++) {
            uintptr_t b=(uintptr_t)c->pools[i].base;
            if(!c->pools[i].bytes || n>=b+c->pools[i].bytes || b>=ne)continue;
            /* Pool 1's final 512 KiB is reserved for executor workspace.
             * The near slot can live there as long as it is above the
             * advertised writable boundary and disjoint from workspace. */
            if(i==1 && n>=b+c->pool1_writable_bytes && ne<=b+c->pools[i].bytes)continue;
            FAIL(E_MEMORY);
        }
    }
    /* Helper near slots: 128-byte aligned, disjoint from every pool, the
     * workspace, the call scratch, near and each other, else unused. */
    if(g->conv && g->conv->near && g->conv->par_near_bytes) {
        for(unsigned s=0;s<2;s++) {
            uintptr_t a=(uintptr_t)conv->par_near[s],e=a+conv->par_near_bytes;
            int ok=a && !(a&127u) && e>a;
            for(unsigned i=0;ok && i<7;i++)
                if(c->pools[i].bytes && a<(uintptr_t)c->pools[i].base+c->pools[i].bytes && (uintptr_t)c->pools[i].base<e) ok=0;
            if(ok && a<(uintptr_t)conv->workspace+conv->workspace_bytes && (uintptr_t)conv->workspace<e) ok=0;
            if(ok && a<(uintptr_t)conv->near+conv->near_bytes && (uintptr_t)conv->near<e) ok=0;
            if(ok && c->scratch && c->scratch_bytes && a<(uintptr_t)c->scratch+c->scratch_bytes && (uintptr_t)c->scratch<e) ok=0;
            if(ok && s && a<(uintptr_t)conv->par_near[0]+conv->par_near_bytes && (uintptr_t)conv->par_near[0]<e) ok=0;
            if(!ok) break;
            g->par_slots++;
        }
    }
    for(unsigned i=0;i<7;i++) {
        g->pool[i]=c->pools[i].bytes?c->pools[i].base:NULL;
        g->pool_bytes[i]=g->pool[i]?c->pools[i].bytes:0;
        if(g->conv && g->pool[i]) {
            uintptr_t b=(uintptr_t)g->pool[i],w=(uintptr_t)conv->workspace;
            if(w<b+g->pool_bytes[i] && b<w+conv->workspace_bytes) g->pool_bytes[i]=w>b?w-b:0;
            if(c->scratch && c->scratch_bytes) {
                uintptr_t s=(uintptr_t)c->scratch;
                if(w<s+c->scratch_bytes && s<w+conv->workspace_bytes) FAIL(E_MEMORY);
            }
        }
        g->write_limit[i]=g->pool_bytes[i];
    }
    if(c->pool1_writable_bytes<g->write_limit[1]) g->write_limit[1]=c->pool1_writable_bytes;
    /* The constant pool is immutable model state. */
    g->write_limit[5]=0;
    g->op=k->opcode;g->variant=k->variant;g->io=k->io_index;
    g->nin=k->input_count;g->nout=k->output_count;g->nfv=k->vector_count;g->ncalls=k->call_count;
    if(g->ncalls) {
        if(!c->bound->calls.data || k->call_first>c->bound->calls.bytes/CALL_BYTES ||
           g->ncalls>c->bound->calls.bytes/CALL_BYTES-k->call_first) FAIL(E_ARGS);
        g->calls=c->bound->calls.data+(size_t)k->call_first*CALL_BYTES;
    }
    if(g->nfv) {
        if(!c->bound->vectors.data || k->vector_first>c->bound->vectors.bytes/4u ||
           g->nfv>c->bound->vectors.bytes/4u-k->vector_first) FAIL(E_ARGS);
        g->fv=c->bound->vectors.data+(size_t)k->vector_first*4u;
    }
    if(g->nout>1) FAIL(E_ARGS);
    for(uint32_t i=0;i<g->nin && i<CACHED_INPUTS;i++)TRY(input_at(g,i,&g->in[i]));
    if(g->nout) {
        uint32_t index;Nprg2TensorView v;
        if(nprg2_exec_output(c,0,&index,&v)!=NPRG2_EXEC_OK || !decode(&v,&g->out)) FAIL(E_TENSOR);
    }
    return 0;
}

/* ctx is NULL or the Nprg2ReferenceConv given at conv registration. */
int nprg2_exec_reference_validate(const Nprg2ExecCall *c, void *ctx) {
    G g;
    if(setup(&g,c,1,0,ctx)) return 0;
    return run_kernel(&g)==0;
}
int nprg2_exec_reference_run(const Nprg2ExecCall *c, void *ctx) {
    G g;
    /* The dry pass measures the exact scratch need. A command that needs no
     * more than the near region (scratch region 1, L2-TCM on the Frame) runs
     * its temporaries there instead of in the DDR heap. Commands run one at
     * a time, so the region (also used by the elementwise handlers inside
     * their own commands) is free. Results do not depend on where scratch is.
     * Without a near region, a call the executor already preflighted skips
     * the repeated dry pass (same pass as nprg2_exec_reference_validate). */
    const Nprg2ExecBuffer *nr=&c->scratch_region[NPRG2_EXEC_SCRATCH_NEAR];
    int has_near=nr->data && nr->bytes;
    /* A fused conv takes no call scratch (the fused path lives in the workspace and near
     * memory), so its executor-validated run skips the repeated dry pass too. */
    if((!has_near || c->epilogue) && c->preflighted) {
        if(setup(&g,c,0,0,ctx)) return 0;
        return run_kernel(&g)==0;
    }
    if(setup(&g,c,1,0,ctx) || run_kernel(&g)) return 0;
    if(has_near && g.used && !((uintptr_t)nr->data&63u) && g.used+64u<=nr->bytes) {
        Nprg2ExecCall w=*c;
        w.scratch=nr->data;w.scratch_bytes=nr->bytes;
        if(setup(&g,&w,0,0,ctx)) return 0;
        return run_kernel(&g)==0;
    }
    if(setup(&g,c,0,0,ctx)) return 0;
    return run_kernel(&g)==0;
}
int nprg2_exec_reference_conv_measure(const Nprg2ExecCall *c, const Nprg2ReferenceConv *conv,
                                      uint32_t *scratch_bytes, uint32_t *workspace_bytes) {
    G g;
    if(!scratch_bytes || setup(&g,c,1,1,conv) || run_kernel(&g) || g.used>UINT32_MAX || g.wused>UINT32_MAX) return 0;
    *scratch_bytes=(uint32_t)g.used;
    if(workspace_bytes)*workspace_bytes=(uint32_t)g.wused;
    return 1;
}
int nprg2_exec_reference_scratch(const Nprg2ExecCall *c, uint32_t *bytes) {
    return nprg2_exec_reference_conv_measure(c,NULL,bytes,NULL);
}
uint32_t nprg2_exec_reference_error_line(void) {return reference_error_line;}
uint32_t nprg2_exec_reference_conv_path(void) {return conv_path;}

int nprg2_executor_register_reference(Nprg2Executor *e, uint8_t opcode, uint8_t variant) {
    if(!e || !variant_matches(opcode,variant)) return NPRG2_EXEC_ARGUMENT;
    return nprg2_executor_register(e,opcode,variant,nprg2_exec_reference_validate,nprg2_exec_reference_run,NULL);
}
int nprg2_executor_register_reference_missing(Nprg2Executor *e, uint32_t *added) {
    if(!e) return NPRG2_EXEC_ARGUMENT;
    uint32_t n=0;
    for(uint32_t v=1;v<=QDR_VARIANTS+1u;v++) {
        uint8_t op=v<=QDR_VARIANTS?qdr_variant_ops[v]:QDR_NOP,variant=v<=QDR_VARIANTS?(uint8_t)v:QDR_INPUT_SLICE_PAD;
        int r=nprg2_executor_register_reference(e,op,variant);
        if(r==NPRG2_EXEC_DUPLICATE) continue;
        if(r!=NPRG2_EXEC_OK){if(added)*added=n;return r;}
        n++;
    }
    if(added)*added=n;
    return NPRG2_EXEC_OK;
}
/* Fuse hook of the conv keys (Nprg2ExecFuse): dry-runs the conv with the
 * epilogue; accepted only when the fused near path would run. */
int nprg2_exec_reference_fuse(const Nprg2ExecCall *c, const Nprg2ExecCall *epi, int dead, void *ctx) {
    G g;
    if(!c || !epi || c->command.opcode!=QDR_HMX_CONV || c->epilogue) return 0;
    Nprg2ExecCall w=*c;
    w.epilogue=epi;w.epilogue_dead=dead?1:0;
    if(setup(&g,&w,1,0,ctx) || run_kernel(&g)) return 0;
    return g.fx_ok;
}
int nprg2_executor_register_reference_conv(Nprg2Executor *e, Nprg2ReferenceConv *conv, uint32_t *added) {
    if(!e || !conv || !conv->workspace || !conv->workspace_bytes) return NPRG2_EXEC_ARGUMENT;
    uint32_t n=0;
    for(uint32_t v=1;v<=QDR_VARIANTS;v++) {
        if(qdr_variant_ops[v]!=QDR_HMX_CONV) continue;
        int r=nprg2_executor_register(e,QDR_HMX_CONV,(uint8_t)v,nprg2_exec_reference_validate,nprg2_exec_reference_run,conv);
        if(r==NPRG2_EXEC_DUPLICATE) continue;
        if(r!=NPRG2_EXEC_OK){if(added)*added=n;return r;}
        nprg2_executor_set_fuse(e,QDR_HMX_CONV,(uint8_t)v,nprg2_exec_reference_fuse);
        n++;
    }
    if(added)*added=n;
    return NPRG2_EXEC_OK;
}

/* The call as the wrapped handler would get it unchained: its own scratch
 * region. 0 when that region is absent or below the registered demand. */
static int chain_call(const Nprg2ExecCall *c, const Nprg2ReferenceChain *k, Nprg2ExecCall *w) {
    if(k->scratch_region>=NPRG2_EXEC_SCRATCH_REGIONS) return 0;
    *w=*c;
    const Nprg2ExecBuffer *b=&c->scratch_region[k->scratch_region];
    w->scratch=b->bytes?b->data:NULL;w->scratch_bytes=b->bytes;
    return w->scratch_bytes>=k->scratch_bytes;
}
static int chain_validate(const Nprg2ExecCall *c, void *ctx) {
    const Nprg2ReferenceChain *k=ctx;Nprg2ExecCall w;
    return (chain_call(c,k,&w) && k->validate(&w,k->ctx)) || nprg2_exec_reference_validate(c,NULL);
}
static int chain_run(const Nprg2ExecCall *c, void *ctx) {
    const Nprg2ReferenceChain *k=ctx;Nprg2ExecCall w;
    if(chain_call(c,k,&w) && k->validate(&w,k->ctx)) return k->run(&w,k->ctx);
    return nprg2_exec_reference_run(c,NULL);
}
int nprg2_executor_chain_reference(Nprg2Executor *e, uint8_t opcode, uint8_t variant, Nprg2ReferenceChain *storage) {
    if(!e || !storage) return NPRG2_EXEC_ARGUMENT;
    for(uint32_t i=0;i<NPRG2_EXEC_REGISTRY_CAPACITY;i++) {
        Nprg2ExecHandler *h=&e->handlers[i];
        if(!h->used || h->opcode!=opcode || h->variant!=variant) continue;
        if(h->validate==chain_validate) return NPRG2_EXEC_DUPLICATE;
        if(h->scratch_bytes>UINT32_MAX-h->concurrent_worker_scratch_bytes) return NPRG2_EXEC_ARGUMENT;
        storage->validate=h->validate;storage->run=h->run;storage->ctx=h->ctx;
        storage->scratch_bytes=h->scratch_bytes+h->concurrent_worker_scratch_bytes;
        storage->scratch_region=h->scratch_region;
        /* The entry now gets the reference kernel's default region; the
         * reference validator checks its own scratch peak against it. */
        h->validate=chain_validate;h->run=chain_run;h->ctx=storage;
        h->scratch_region=NPRG2_EXEC_SCRATCH_DEFAULT;h->scratch_bytes=0;h->concurrent_worker_scratch_bytes=0;
        return NPRG2_EXEC_OK;
    }
    return NPRG2_EXEC_UNSUPPORTED;
}
