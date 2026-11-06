/* Persistent HVX convolution payload. Entered through qd_loader on each of the
 * three Valve workers; serves commands until QDC_OP_EXIT or an idle timeout.
 * QD_HOST_TEST compiles one-command dispatch with a scalar vrmpy substitute. */
#include "qd_protocol.h"
#include "qd_conv.h"
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
static uint32_t round_chunk(uint32_t n) { return (n+QDC_COPY_CHUNK-1) & ~(QDC_COPY_CHUNK-1); }

typedef struct { uint32_t xb, wb, cb; } Sizes;
static uint32_t validate(const QdConv *c, Sizes *s) {
    if(c->magic!=QDC_MAGIC || c->version!=QDC_VERSION || !c->sequence || c->arena_bytes!=QD_ARENA_BYTES)
        return QDC_BAD_HEADER;
    if(!c->g || !c->kh || !c->kw || !c->n || !c->oh || !c->owp || c->g>256 || c->kh>8 || c->kw>8 ||
       c->n>1024 || (c->n%QDC_NB) || (c->owp&31) || c->owp>2048 || c->oh+c->kh-1!=c->h ||
       c->wp<c->owp+c->kw-1 || c->wp>4096 || c->oh>64)
        return QDC_BAD_SHAPE;
    s->xb=c->h*c->g*c->wp*4; s->wb=c->kh*c->kw*c->g*c->n*4; s->cb=c->oh*c->n*c->owp*4;
    /* Over-read: the last unaligned vector of a row may extend 124 bytes. */
    if(round_chunk(s->xb+128)>QDC_VTCM_BYTES || round_chunk(s->wb)>QDC_L2TCM_BYTES) return QDC_BAD_SHAPE;
    if((c->x_offset&127) || (c->w_offset&127) || (c->c_offset&127) ||
       !in_range(c->x_offset,round_chunk(s->xb+128),QD_ARENA_BYTES) ||
       !in_range(c->w_offset,round_chunk(s->wb),QD_ARENA_BYTES) ||
       !in_range(c->c_offset,s->cb,QD_ARENA_BYTES) ||
       overlap(c->x_offset,round_chunk(s->xb+128),c->c_offset,s->cb) ||
       overlap(c->w_offset,round_chunk(s->wb),c->c_offset,s->cb))
        return QDC_BAD_RANGE;
    return QDC_OK;
}

#ifdef QD_HOST_TEST
typedef struct { int32_t w[32]; } Vec;
static Vec vzero(void) { Vec v; for(int i=0;i<32;i++) v.w[i]=0; return v; }
static Vec vload(const uint8_t *p) { Vec v; for(int i=0;i<32;i++) v.w[i]=(int32_t)(p[4*i]|p[4*i+1]<<8|p[4*i+2]<<16|(uint32_t)p[4*i+3]<<24); return v; }
static Vec vrmpyacc(Vec acc, Vec x, int32_t w) {
    for(int i=0;i<32;i++) for(int j=0;j<4;j++)
        acc.w[i]+=(int32_t)((uint32_t)x.w[i]>>(8*j)&255)*(int8_t)(w>>(8*j));
    return acc;
}
static void vstore(int32_t *p, Vec v) { for(int i=0;i<32;i++) p[i]=v.w[i]; }
#define VEC Vec
#define VZERO() vzero()
#define VLOADU(p) vload((const uint8_t *)(p))
#define VRMPY(a,x,w) a=vrmpyacc(a,x,w)
#define VSTORE(p,v) vstore((int32_t *)(p),v)
#else
#define VEC HVX_Vector
#define VZERO() Q6_V_vzero()
#define VLOADU(p) (*(const HVX_UVector *)(p))
#define VRMPY(a,x,w) a=Q6_Vw_vrmpyacc_VwVubRb(a,x,w)
#define VSTORE(p,v) (*(HVX_Vector *)(p)=(v))
#endif

static void barrier(void) {
#ifndef QD_HOST_TEST
    __asm__ volatile("barrier" ::: "memory");
#endif
}

/* Copy this worker's share of [src, src+bytes) in 1 KB chunks, 8 loads in flight. */
static void copy_share(uint8_t *dst, const uint8_t *src, uint32_t bytes, uint32_t tid) {
    for(uint32_t o=tid*QDC_COPY_CHUNK;o<bytes;o+=3*QDC_COPY_CHUNK) {
#ifdef QD_HOST_TEST
        for(uint32_t i=0;i<QDC_COPY_CHUNK;i++) dst[o+i]=src[o+i];
#else
        const HVX_Vector *s=(const HVX_Vector *)(src+o);
        HVX_Vector *d=(HVX_Vector *)(dst+o);
        HVX_Vector v0=s[0],v1=s[1],v2=s[2],v3=s[3],v4=s[4],v5=s[5],v6=s[6],v7=s[7];
        d[0]=v0; d[1]=v1; d[2]=v2; d[3]=v3; d[4]=v4; d[5]=v5; d[6]=v6; d[7]=v7;
#endif
    }
}

/* One unit: PB pixel vectors x NB output channels of output row oy. */
static void conv_unit(const QdConv *c, const uint8_t *x, const int32_t *w, int32_t *out,
                      uint32_t oy, uint32_t x0, uint32_t pb, uint32_t n0) {
    VEC a00=VZERO(),a01=VZERO(),a02=VZERO(),a03=VZERO(),a04=VZERO(),a05=VZERO(),a06=VZERO(),a07=VZERO();
    VEC a10=VZERO(),a11=VZERO(),a12=VZERO(),a13=VZERO(),a14=VZERO(),a15=VZERO(),a16=VZERO(),a17=VZERO();
    uint32_t g=c->g, n=c->n, wp=c->wp, row=g*wp*4;
    for(uint32_t ky=0;ky<c->kh;ky++)
        for(uint32_t kx=0;kx<c->kw;kx++) {
            const uint8_t *xp=x+(oy+ky)*row+(x0+kx)*4;
            const int32_t *wq=w+((ky*c->kw+kx)*g)*n+n0;
            for(uint32_t gi=0;gi<g;gi++,xp+=wp*4,wq+=n) {
                VEC v0=VLOADU(xp);
                int32_t w0=wq[0],w1=wq[1],w2=wq[2],w3=wq[3],w4=wq[4],w5=wq[5],w6=wq[6],w7=wq[7];
                VRMPY(a00,v0,w0); VRMPY(a01,v0,w1); VRMPY(a02,v0,w2); VRMPY(a03,v0,w3);
                VRMPY(a04,v0,w4); VRMPY(a05,v0,w5); VRMPY(a06,v0,w6); VRMPY(a07,v0,w7);
                if(pb>1) {
                    VEC v1=VLOADU(xp+128);
                    VRMPY(a10,v1,w0); VRMPY(a11,v1,w1); VRMPY(a12,v1,w2); VRMPY(a13,v1,w3);
                    VRMPY(a14,v1,w4); VRMPY(a15,v1,w5); VRMPY(a16,v1,w6); VRMPY(a17,v1,w7);
                }
            }
        }
    int32_t *o=out+(oy*n+n0)*c->owp+x0;
    uint32_t s=c->owp;
    VSTORE(o,a00); VSTORE(o+s,a01); VSTORE(o+2*s,a02); VSTORE(o+3*s,a03);
    VSTORE(o+4*s,a04); VSTORE(o+5*s,a05); VSTORE(o+6*s,a06); VSTORE(o+7*s,a07);
    if(pb>1) {
        o+=32;
        VSTORE(o,a10); VSTORE(o+s,a11); VSTORE(o+2*s,a12); VSTORE(o+3*s,a13);
        VSTORE(o+4*s,a14); VSTORE(o+5*s,a15); VSTORE(o+6*s,a16); VSTORE(o+7*s,a17);
    }
}

/* Units are dealt round-robin to the three workers (no hardware divide). */
static uint32_t compute(const QdConv *c, const uint8_t *x, const int32_t *w, int32_t *out, uint32_t tid) {
    uint32_t vecs=c->owp>>5, units=0, turn=0;
    for(uint32_t oy=0;oy<c->oh;oy++)
        for(uint32_t v=0;v<vecs;v+=QDC_PB)
            for(uint32_t n0=0;n0<c->n;n0+=QDC_NB) {
                if(turn==tid) {
                    conv_unit(c,x,w,out,oy,v*32,vecs-v>=QDC_PB ? QDC_PB : vecs-v,n0);
                    units++;
                }
                if(++turn==3) turn=0;
            }
    return units;
}

/* Word copy: a struct assignment could become a libc memcpy call. */
static void load_command(QdConv *dst, volatile QdConv *src) {
    volatile uint32_t *s=(volatile uint32_t *)src;
    uint32_t *d=(uint32_t *)dst;
    for(uint32_t i=0;i<sizeof(QdConv)/4;i++) d[i]=s[i];
}

static void publish(volatile QdStatus *st, uint32_t seq, uint32_t error, uint32_t units, uint32_t marker) {
    st->error=error; st->rows=units; st->reserved=marker;
    barrier(); st->done_sequence=seq; barrier();
}

#ifdef QD_HOST_TEST
/* Host test: arena doubles as TCM; tid loops are run sequentially, so the
 * copy phase of every tid must precede compute. */
void qdc_host_run(void *arena, void *vtcm, void *l2, uint32_t tid, uint32_t phase) {
    QdConv c; Sizes s;
    load_command(&c,(volatile QdConv *)arena);
    volatile QdStatus *st=(volatile QdStatus *)((uint8_t *)arena+QD_STATUS_OFFSET)+tid;
    uint32_t error=validate(&c,&s);
    if(error) { if(phase) publish(st,c.sequence,error,0,0); return; }
    if(!phase) {
        copy_share(vtcm,(uint8_t *)arena+c.x_offset,round_chunk(s.xb+128),tid);
        copy_share(l2,(uint8_t *)arena+c.w_offset,round_chunk(s.wb),tid);
        return;
    }
    uint32_t units=compute(&c,vtcm,(const int32_t *)l2,(int32_t *)((uint8_t *)arena+c.c_offset),tid);
    publish(st,c.sequence,QDC_OK,units,0);
}
#else
static inline uint64_t cycles(void) { uint64_t t; __asm__ volatile("%0 = upcycle" : "=r"(t)); return t; }

/* All three workers increment the counter, then wait until it reaches 3. */
static void arrive_and_wait(volatile uint32_t *counter) {
    uint32_t v;
    __asm__ volatile(
        "1: %0 = memw_locked(%1)\n"
        "   %0 = add(%0, #1)\n"
        "   memw_locked(%1, p0) = %0\n"
        "   if (!p0) jump 1b\n"
        : "=&r"(v) : "r"(counter) : "p0", "memory");
    while(*counter<3) {}
}

/* Executes one validated command; returns units completed by this worker. */
static uint32_t conv(const QdConv *c, const Sizes *s, uint8_t *arena, uint8_t *vtcm, uint8_t *l2,
                     volatile uint32_t *counter, uint32_t tid) {
    volatile uint32_t *timing=(volatile uint32_t *)(arena+QDC_TIMING_OFFSET)+tid*4;
    uint64_t t0=cycles();
    copy_share(vtcm, arena+c->x_offset, round_chunk(s->xb+128), tid);
    copy_share(l2, arena+c->w_offset, round_chunk(s->wb), tid);
    barrier();
    uint64_t t1=cycles();
    arrive_and_wait(counter);
    uint64_t t2=cycles();
    uint32_t units=compute(c,vtcm,(const int32_t *)l2,(int32_t *)(arena+c->c_offset),tid);
    barrier();
    uint64_t t3=cycles();
    timing[0]=(uint32_t)(t1-t0); timing[1]=(uint32_t)(t2-t1); timing[2]=(uint32_t)(t3-t2);
    return units;
}

__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    if(tid>=3) return;
    /* Uncached view: HVX bypasses L1 and the L2 is configured as TCM, so this
     * is as fast as the cached view and needs no cache maintenance. */
    uint8_t *arena=(uint8_t *)(0xf0000000u+QD_ARENA_OFFSET);
    volatile QdConv *cmd=(volatile QdConv *)(0xf0000000u+QD_ARENA_OFFSET);
    volatile QdStatus *st=(volatile QdStatus *)(0xf0000000u+QD_ARENA_OFFSET+QD_STATUS_OFFSET)+tid;
    volatile uint32_t *counter=(volatile uint32_t *)(0xf0000000u+QD_ARENA_OFFSET+QDC_BARRIER_OFFSET);
    uint32_t last=0;
    uint64_t idle_start=cycles();
    for(;;) {
        uint32_t seq=cmd->sequence;
        if(seq==last) {
            uint64_t limit=(uint64_t)(cmd->idle_mcycles ? cmd->idle_mcycles : 4000)*1000000u;
            if(cycles()-idle_start>limit) { publish(st,last,QDC_IDLE_EXIT,0,QDC_EXITED); return; }
            continue;
        }
        barrier();
        QdConv c; Sizes s;
        load_command(&c,cmd);
        last=seq;
        if(c.op==QDC_OP_EXIT && c.magic==QDC_MAGIC) { publish(st,seq,QDC_OK,0,QDC_EXITED); return; }
        uint32_t error=c.op==QDC_OP_CONV ? validate(&c,&s) : QDC_BAD_HEADER, units=0;
        if(!error) units=conv(&c,&s,arena,(uint8_t *)QDC_VTCM_BASE,(uint8_t *)QDC_L2TCM_BASE,counter,tid);
        publish(st,seq,error,units,0);
        idle_start=cycles();
    }
}
#endif
