/* SPDX-License-Identifier: LGPL-3.0-or-later */
#define QDG_DSP 1
#include "qd_glue_hvx.h"
#include "qd_pack_hvx.h"
#include "qd_graph.c"
#include "qd_graph_dsp.h"
#include "qd_hmx_layout.h"
#include "qd_direct_hmx.h"
#include "qd_protocol.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#define SERVICE ((uint8_t *)0xf1800000u)
static QdsWork *const work=(QdsWork *)QDS_WORK_L2;
static uint64_t metadata_cycles;
/* Serial tensor access cache. All numerical work is done on TCM bytes, with
 * complete DDR block transfers. It is private to one xfer: flush before any
 * subsequent operation/alias can observe memory. The top 512 KB of the
 * verified 0x2d0000 VTCM scratch region is reserved while this cache is active. */
typedef struct { uint32_t tags[256];uint8_t dirty[256];uint8_t *const *pools; } BlockCache;
static BlockCache *const block_cache=(BlockCache *)0x300b0000u;
#define CACHE_DATA ((uint8_t *)0x31250000u)
static void dsp_cache_begin(uint8_t *const *pools) {
    memset(block_cache->tags,0,sizeof block_cache->tags);
    memset(block_cache->dirty,0,sizeof block_cache->dirty);block_cache->pools=pools;
}
static void dsp_cache_flush(uint32_t slot) {
    uint32_t e=block_cache->tags[slot];
    if(e && block_cache->dirty[slot]) {
        memcpy(block_cache->pools[e&127]+(e&~127u),CACHE_DATA+slot*2048,2048);
        block_cache->dirty[slot]=0;
    }
}
static uint8_t *dsp_cache_block(uint32_t e,int store) {
    uint32_t slot=((e>>11)+(e&127)*17)&255;
    if(block_cache->tags[slot]!=e) {
        dsp_cache_flush(slot);
        memcpy(CACHE_DATA+slot*2048,block_cache->pools[e&127]+(e&~127u),2048);
        block_cache->tags[slot]=e;
    }
    if(store)block_cache->dirty[slot]=1;
    return CACHE_DATA+slot*2048;
}
static void dsp_cache_end(void){for(uint32_t i=0;i<256;i++)dsp_cache_flush(i);}

/* Bounded private heap, single master only. Free coalesces both neighbours;
 * dynamic weights must not accumulate across inferences. */
typedef struct Heap { size_t bytes;struct Heap *prev,*next;uint32_t used; } Heap;
/* Every block before heap_scan is in use, so first fit can start there and
 * still choose the same block as a walk from heap_first. Long-lived weights and
 * parameters fill the front of the heap; walking their headers in L1-only DDR
 * cost ~0.25 ms per allocation. */
static Heap *heap_first,*heap_scan;
static size_t heap_used,heap_peak,scratch_peak;
static uint64_t malloc_cycles,malloc_calls,malloc_steps;
static void dsp_clean_range(const void *p,size_t n) {
    uintptr_t start=(uintptr_t)p&~31u,end=(uintptr_t)p+n;
    for(uintptr_t a=start;a<end;a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    __asm__ volatile("barrier":::"memory");
}
/* Packet sources in the scalar heap are explicitly cleaned before this vector
 * copy. Its destination is uncached staging or TCM, never the scalar heap. */
static void dsp_packet_copy(void *dst,const void *src,size_t n) {
    uint8_t *d=dst;const uint8_t *s=src;
    /* HVX bypasses L1. Both C0 (non-L2-cacheable) and C7 permit scalar L1
     * caching. Publish heap metadata/values before vector reads. */
    if((uintptr_t)s>=QDO_BASE+0x2000000u &&
       (uintptr_t)s<QDO_BASE+0x4000000u)dsp_clean_range(s,n);
    while(n>=128){*(HVX_UVector *)d=*(const HVX_UVector *)s;d+=128;s+=128;n-=128;}
    while(n>=4){*(uint32_t *)d=*(const uint32_t *)s;d+=4;s+=4;n-=4;}
    while(n--)*d++=*s++;
}
typedef struct { uint32_t count,cursor,id[32];T tensor[32]; } TensorCache;
static TensorCache *const tensor_cache=(TensorCache *)0x300b4000u;
_Static_assert(sizeof(TensorCache)<=0x4000,"tensor cache overlaps metadata arena");
_Static_assert(sizeof(QdgGraph)<=0x2000,"graph state exceeds private L2 slot");
static void *dsp_metadata(const void *src,uint32_t bytes) {
    if(!bytes)return (void *)src;
    uint32_t at=(tensor_cache->cursor+127)&~127u;
    if(at>0x300ef000u || bytes>0x300ef000u-at)return (void *)src;
    dsp_packet_copy((void *)at,src,bytes);tensor_cache->cursor=at+bytes;return (void *)at;
}
static const T *dsp_tensor(QdgGraph *g,uint32_t index) {
    for(uint32_t i=0;i<tensor_cache->count;i++)if(tensor_cache->id[i]==index)return &tensor_cache->tensor[i];
    if(tensor_cache->count==32)return &g->t[index];
    uint64_t started=now_ns();
    uint32_t i=tensor_cache->count++;
    tensor_cache->id[i]=index;
    T *t=&tensor_cache->tensor[i];dsp_packet_copy(t,&g->t[index],sizeof *t);
    if(t->nblocks)t->blocks=dsp_metadata(t->blocks,t->nblocks*4);
    metadata_cycles+=now_ns()-started;
    return t;
}
static void dsp_command_begin(QdgGraph *g,uint32_t index,C *c) {
    uint64_t started=now_ns();
    tensor_cache->count=0;tensor_cache->cursor=0x300b8000u;
    dsp_packet_copy(c,&g->c[index],sizeof *c);
    c->in=dsp_metadata(c->in,c->nin*4);c->out=dsp_metadata(c->out,c->nout*4);
    c->fv=dsp_metadata(c->fv,c->nfv*4);c->calls=dsp_metadata(c->calls,c->ncalls*88);
    metadata_cycles+=now_ns()-started;
}
/* Only the master executes libc/math. It needs no QuRT TLS service. */
int *_Geterrno(void){static int e;return &e;}
static int resident_cancelled(void) {
    const volatile uint32_t *c=(const volatile uint32_t *)SERVICE;
    return c[3]==QDC_OP_EXIT || *(const volatile uint32_t *)(SERVICE+3840)!=0;
}
void *malloc(size_t n) {
    if(!n)n=1;if(n>QDO_BYTES-256)return NULL;n=(n+127)&~(size_t)127;
    uint64_t malloc_start;__asm__ volatile("%0=upcycle":"=r"(malloc_start));malloc_calls++;
    void *found=NULL;
    for(Heap *h=heap_scan;h;h=h->next,malloc_steps++)if(!h->used && h->bytes>=n) {
        if(h->bytes>=n+256) {
            Heap *tail=(Heap *)((uint8_t *)h+128+n);
            tail->bytes=h->bytes-n-128;tail->used=0;tail->prev=h;tail->next=h->next;
            if(tail->next)tail->next->prev=tail;h->next=tail;h->bytes=n;
        }
        h->used=1;heap_used+=h->bytes+128;if(heap_used>heap_peak)heap_peak=heap_used;
        while(heap_scan && heap_scan->used)heap_scan=heap_scan->next;
        found=(uint8_t *)h+128;break;
    }
    uint64_t malloc_end;__asm__ volatile("%0=upcycle":"=r"(malloc_end));malloc_cycles+=malloc_end-malloc_start;
    return found;
}
void free(void *p) {
    if(!p)return;Heap *h=(Heap *)((uint8_t *)p-128);heap_used-=h->bytes+128;h->used=0;
    if(h->next && !h->next->used){h->bytes+=h->next->bytes+128;h->next=h->next->next;if(h->next)h->next->prev=h;}
    if(h->prev && !h->prev->used){h->prev->bytes+=h->bytes+128;h->prev->next=h->next;if(h->next)h->next->prev=h->prev;h=h->prev;}
    /* h is the merged free block; any absorbed scan block lay above it. */
    if(!heap_scan || (uintptr_t)heap_scan>(uintptr_t)h)heap_scan=h;
}
void *calloc(size_t n,size_t size){if(size && n>SIZE_MAX/size)return NULL;size_t b=n*size;void *p=malloc(b);if(p)memset(p,0,b);return p;}
void *realloc(void *p,size_t n){if(!p)return malloc(n);Heap *h=(Heap *)((uint8_t *)p-128);if(n<=h->bytes)return p;void *q=malloc(n);if(q){memcpy(q,p,h->bytes);free(p);}return q;}
#if QDS_L2_CACHE
/* Set only inside the L2CFG=5 lease, where L2 keeps scalar L1 and HVX
 * coherent for C7 page 7. Outside it page 7 stays scalar-only. */
static uint32_t qds_l2_lease_active;
#define QDS_PAGE7(p) (!qds_l2_lease_active && (uintptr_t)(p)>=QDO_BASE+0x7000000u && (uintptr_t)(p)<QDO_BASE+0x8000000u)
#endif
void *memcpy(void *dst,const void *src,size_t n) {
    uint8_t *d=dst;const uint8_t *s=src;
    int cached=(((uintptr_t)d>=QDO_BASE && (uintptr_t)d<QDO_BASE+0x1000000u) ||
                ((uintptr_t)d>=QDO_BASE+0x2000000u && (uintptr_t)d<QDO_BASE+0x4000000u) ||
                ((uintptr_t)s>=QDO_BASE && (uintptr_t)s<QDO_BASE+0x1000000u) ||
                ((uintptr_t)s>=QDO_BASE+0x2000000u && (uintptr_t)s<QDO_BASE+0x4000000u)
#if QDS_L2_CACHE
                || QDS_PAGE7(d) || QDS_PAGE7(s)
#endif
                );
    if(!cached)while(n>=128){*(HVX_UVector *)d=*(const HVX_UVector *)s;d+=128;s+=128;n-=128;}
    if(!(((uintptr_t)d|(uintptr_t)s)&3)) {
        while(n>=4){*(uint32_t *)d=*(const uint32_t *)s;d+=4;s+=4;n-=4;}
    }
    while(n--)*d++=*s++;return dst;
}
void *memset(void *dst,int c,size_t n) {
    uint8_t *d=dst;HVX_Vector v=Q6_V_vsplat_R((uint32_t)(uint8_t)c*0x01010101u);
    int cached=(((uintptr_t)d>=QDO_BASE && (uintptr_t)d<QDO_BASE+0x1000000u) ||
                ((uintptr_t)d>=QDO_BASE+0x2000000u && (uintptr_t)d<QDO_BASE+0x4000000u)
#if QDS_L2_CACHE
                || QDS_PAGE7(d)
#endif
                );
    if(!cached)while(n>=128){*(HVX_UVector *)d=v;d+=128;n-=128;}
    if(!((uintptr_t)d&3))while(n>=4){*(uint32_t *)d=(uint32_t)(uint8_t)c*0x01010101u;d+=4;n-=4;}
    while(n--)*d++=(uint8_t)c;return dst;
}
static int send_work(uint32_t op,uint32_t sequence) {
    work->op=op;work->sequence=sequence;work->error=0;
    uint32_t gen=work->generation+1;
    __asm__ volatile("barrier":::"memory");work->generation=gen;
    uint32_t e=op==1?qdrc_execute(&work->conv,(uint8_t *)work->arena,0,work->prepared_a):0;
    uint64_t end=now_ns()+4000000000ull;
    for(uint32_t t=1;t<QDS_WORKER_SLOTS;t++)
        while(work->done[t]!=gen)if(now_ns()>end)return 90;
    __asm__ volatile("barrier":::"memory");return e?e:work->error;
}
static uint32_t send_work_start(uint32_t op,uint32_t sequence) {
    work->op=op;work->sequence=sequence;work->error=0;
    uint32_t gen=work->generation+1;
    __asm__ volatile("barrier":::"memory");work->generation=gen;
    return gen;
}
static int send_work_wait(uint32_t gen,uint64_t timeout_ns) {
    uint64_t end=now_ns()+timeout_ns;
    for(uint32_t t=1;t<QDS_WORKER_SLOTS;t++)
        while(work->done[t]!=gen)if(now_ns()>end)return 90;
    __asm__ volatile("barrier":::"memory");return work->error;
}
#if QDS_NATIVE_1X1
#include "kernels/adapters/qd_native_1x1_graph_bridge.h"
#endif
#if QDS_NATIVE_3X3
#include "kernels/convolution/qd_native_3x3_graph_bridge.h"
#endif
#if QDS_NATIVE_DIRECT_RANGE
#include "qd_graph_native_direct.h"
#endif
#if QDS_NATIVE_PREPARED
#include "qd_graph_prepared.h"
#endif
static int resident_request(QdgGraph *g,Layout *l,const A *x,Weights *w,const int32_t *params) {
    uint64_t started=now_ns();
    uint64_t whole_started=started;
    uint8_t *a=g->staging;
    if(x->dt!=DT_U8 && x->dt!=DT_U16)return QDG_ERR_UNSUPPORTED;
    size_t source_bytes=count4(x->n)*dsize(x->dt);
    CHECK(source_bytes<=0x2d0000u-512u,QDG_ERR_SHAPE);
    int direct=g->dsp_direct_tensor!=NULL;
    if(!direct && x->p!=(void *)QDC_VTCM_BASE)memcpy((void *)QDC_VTCM_BASE,x->p,source_bytes);
    const uint8_t *xp=(const uint8_t *)QDC_VTCM_BASE;
    uint32_t pack_start=(source_bytes+127)&~127u,packet_bytes=chunked(l->xb+128);
    int prepared=direct || (pack_start<=0x2d0000u-512u && packet_bytes<=0x2d0000u-512u-pack_start);
    uint8_t *packet=prepared?(uint8_t *)QDC_VTCM_BASE+pack_start:a+l->xo;
    if(direct) {
        QdConv shape={0};shape.h=l->h;shape.wp=l->wp;shape.g=l->g;shape.kh=l->kh;shape.kw=l->kw;
        shape.n=l->n;shape.oh=l->oh;shape.owp=l->owp;
        QdHmxLayout hl=hmx_layout(&shape,l->xb);
        CHECK(l->hmx && hl.bytes<=QDC_VTCM_BYTES,QDG_ERR_SHAPE);
        int cached=hl.bytes<=0x250000u;
        if(cached)dsp_cache_begin(g->pool);
        const T *t=g->dsp_direct_tensor;
        dsp_direct_hmx_a((uint8_t *)QDC_VTCM_BASE+hl.a,g->pool,t->blocks,t->nblocks,t->slice,t->m,
                         g->dsp_direct_origin,g->dsp_direct_h,g->dsp_direct_w,l->c,l->kh,l->kw,l->owp,l->oh==1,cached);
        if(cached)dsp_cache_end();
        g->resident_key=0;
        l->x32=1;g->dsp_direct_calls++;
    } else if(l->u16 && !(l->c&3)) {
        dsp_pack_u16_gather(packet,l->plane,(const uint16_t *)xp,l->h,l->width,l->c,l->wp,
                            (HVX_Vector *)(QDC_VTCM_BASE+0x2cff00u));
        memset(packet+l->xb,0,packet_bytes-l->xb);
    } else if(l->u16) {
        memset(packet,0,packet_bytes);
        const uint16_t *src=(const uint16_t *)xp;
        for(uint32_t h=0;h<l->h;h++)for(uint32_t group=0;group<l->g;group++)for(uint32_t w0=0;w0<l->width;w0++)
            for(uint32_t cc=0;cc<4 && group*4+cc<l->c;cc++) {
                uint16_t v=src[((size_t)h*l->width+w0)*l->c+group*4+cc];
                size_t at=((size_t)h*l->g+group)*l->wp*4+w0*4+cc;
                packet[at]=(uint8_t)v;packet[l->plane+at]=(uint8_t)(v>>8);
            }
    } else if(l->hmx && !(l->c&31)) {
        l->x32=1;
        dsp_pack_x32(packet,xp,l->h,l->width,l->c,l->wp);
        memset(packet+l->xb,0,packet_bytes-l->xb);
    } else pack_x(packet,xp,l->h,l->width,l->c,l->g,l->wp,packet_bytes);
    if(prepared && !direct) {
        /* Forward vector move is overlap-safe because packet starts above X. */
        for(uint32_t off=0;off<packet_bytes;off+=128)
            *(HVX_Vector *)(QDC_VTCM_BASE+off)=*(const HVX_Vector *)(packet+off);
        g->resident_key=0; /* the temporary packed packet can overlap old B */
    }
    g->dsp_pack_cycles[0]+=now_ns()-started;started=now_ns();
    int ws=l->hmx?1:0;
    uint8_t *packed=w->packed[ws];
    if(!packed || w->packed_ci[ws]!=l->c) {
        free(packed);w->packed[ws]=packed=pack_weights(w,l->c,l->g,l->kb,l->hmx,&w->packed_bytes[ws]);
        if(!packed)return QDG_ERR_MEMORY;w->packed_ci[ws]=l->c;
        dsp_clean_range(packed,w->packed_bytes[ws]);
    }
    if(w->packed_bytes[ws]!=l->wb)return QDG_ERR_SHAPE;
    g->dsp_pack_cycles[1]+=now_ns()-started;started=now_ns();
    l->key=l->hmx?w->key:0;
    l->keep=l->key && g->resident_key==l->key && g->resident_b==l->vb && g->resident_wb==l->wb;
    if(!l->keep)dsp_packet_copy(a+l->wo,packed,l->wb);
    if(params){uint32_t bytes=l->n*(l->u16?32:12);dsp_clean_range(params,bytes);memset(a+l->xo-l->pb,0,l->pb);dsp_packet_copy(a+l->xo-l->pb,params,bytes);}
    g->dsp_pack_cycles[2]+=now_ns()-started;
    uint32_t seq=++g->sequence;
    QdConv cmd={0x32564451u,6,seq,1,l->h,l->wp,l->g,l->kh,l->kw,l->n,l->oh,l->owp,
                l->xo,l->wo,l->co,ARENA_BYTES,20000,{1u|(l->quantized?2u:0u)|(l->hmx?4u:0u)|
                (prepared?128u:0u)|(l->u16?8u:0u)|(l->x32?32u:0u)|(l->keep?16u:0u)|(l->out_h?64u|(l->out_h<<8)|(l->out_w<<16):0),
                params?l->xo-l->pb:0,l->key}};
    work->conv=cmd;work->arena=(uint32_t)a;work->prepared_a=direct;
    g->stats.prepare_ns+=now_ns()-whole_started;started=now_ns();
    uint32_t e=send_work(1,seq);if(e){g->error_detail=e;return QDG_ERR_DSP;}
    g->stats.wait_ns+=now_ns()-started;
    for(int phase=0;phase<(l->hmx?4:3);phase++) {
        uint32_t max=0;
        for(int t=0;t<3;t++){uint32_t v=*(volatile uint32_t *)(a+2048+4*(phase+4*t));if(v>max)max=v;}
        if(l->hmx)g->stats.hmx_phase_cycles[phase]+=max;else g->stats.hvx_phase_cycles[l->u16][phase]+=max;
    }
    g->resident_key=l->key;g->resident_b=l->vb;g->resident_wb=l->wb;
    g->stats.calls++;if(l->hmx)g->stats.hmx_calls++;
    return 0;
}
static int resident_conv(QdgGraph *g,const C *c,uint32_t index,const A *x,Weights *w,
        const int32_t *mult,const int32_t *bias,int32_t oh,int32_t ow,int32_t co,int sixteen,const int32_t *origin) {
    uint64_t params_started=now_ns(),post_started=0;
    A xx=*x;
    if(w->kh==1 && w->kw==1){xx.n[1]=1;xx.n[2]=x->n[1]*x->n[2];}
    Layout l;
    int32_t *params=NULL;
    int quant=0,hmx=0;
    if(!sixteen) {
        ConvState *cs=&g->conv[index];
        if(cs->ready && cs->static_q){params=cs->params;quant=cs->quant_ok;}
        else {
            params=salloc(g,(size_t)co*12);CHECK(params,QDG_ERR_MEMORY);
            uint64_t quant_start=now_ns();
            quant=!quant_params(params,w,mult,bias,g->t[c->in[0]].zp);
            g->dsp_quant_cycles+=now_ns()-quant_start;
            if(w->key && g->static_fi[c->in[2]]) {
                cs->ready=cs->static_q=1;cs->quant_ok=quant;
                if(quant){cs->params=malloc((size_t)co*12);CHECK(cs->params,QDG_ERR_MEMORY);memcpy(cs->params,params,(size_t)co*12);params=cs->params;}
            }
        }
        /* Parameter proof scans scalar words. Keep its hot copy on the
         * private L2 stack, outside shared convolution scratch. */
        if(quant) {
            int32_t *hot=__builtin_alloca((size_t)co*12+127);
            hot=(int32_t *)(((uintptr_t)hot+127)&~127u);
            dsp_clean_range(params,(size_t)co*12);dsp_packet_copy(hot,params,(size_t)co*12);params=hot;
        }
        hmx=quant && co%32==0;
        for(int k=0;hmx && k<co;k++) {
            int64_t p=params[3*k+2],half=p>0?(1ll<<(p-1)):0;
            if(p>15 || (int64_t)params[3*k]-half<INT32_MIN)hmx=0;
        }
        if(hmx && conv_layout(&l,xx.n[1],xx.n[2],xx.n[3],w->kh,w->kw,co,1,1))hmx=0;
    }
    CHECK(!conv_layout(&l,xx.n[1],xx.n[2],xx.n[3],w->kh,w->kw,co,quant,hmx),QDG_ERR_SHAPE);
    if(g->dsp_direct_tensor && (!hmx || !quant))return QDG_DSP_DEFER;
    size_t values=(size_t)oh*ow*co;
    A r;int32_t dims[4]={1,oh,ow,co};TRY(anew(g,&r,dims,sixteen?DT_U16:DT_U8));
    if(sixteen) {
        int32_t *q16=salloc(g,(size_t)co*32);
        int ok=q16 && 255ll*128*w->kh*w->kw*w->ci<(1ll<<29);
        for(int k=0;ok && k<co;k++) {
            uint32_t h=(uint32_t)mult[k]&65535,shift=34-((h>>10)&31);
            int64_t post=(int64_t)shift-8,M=1024+(h&1023);
            int64_t corr=256*(int64_t)bias[k]-(int64_t)g->t[c->in[0]].zp*w->wsum[k],ch=corr>>8;
            if(post<1 || post>15 || ch<-(1ll<<29) || ch>(1ll<<29)){ok=0;break;}
            int64_t rd=1ll<<(post-1),top=((int64_t)65535<<post)-rd,pmax=(top+M-1)/M;
            int32_t *e=q16+8*k;e[0]=ch;e[1]=corr-256*ch;e[2]=pmax;e[3]=M;e[4]=post;e[5]=rd;e[6]=e[7]=0;
        }
        if(ok) {
            CHECK(!conv_layout_u16(&l,xx.n[1],xx.n[2],xx.n[3],w->kh,w->kw,co,0,0,1),QDG_ERR_SHAPE);
            g->dsp_conv_phase[5]+=now_ns()-params_started;
            TRY(resident_request(g,&l,&xx,w,q16));
            post_started=now_ns();
            const uint16_t *raw=(const uint16_t *)(g->staging+l.co);
            uint32_t rbytes=(values*2+127)&~127u;
            if(rbytes<=0x80000u && l.cb<=0x80000u-rbytes) {
                r.p=(void *)QDC_L2TCM_BASE;
                uint8_t *staged=(uint8_t *)QDC_L2TCM_BASE+rbytes;
                dsp_packet_copy(staged,raw,l.cb);raw=(const uint16_t *)staged;
            } else {
                r.p=(void *)QDC_VTCM_BASE;
                if(rbytes<=0x250000u && l.cb<=0x2d0000u-rbytes) {
                    uint8_t *staged=(uint8_t *)QDC_VTCM_BASE+rbytes;
                    dsp_packet_copy(staged,raw,l.cb);raw=(const uint16_t *)staged;
                }
            }
            for(uint32_t h=0;h<l.oh;h++)for(uint32_t n=0;n<l.n;n++)for(uint32_t ww=0;ww<l.ow;ww++)
                ((uint16_t *)r.p)[((size_t)h*l.ow+ww)*l.n+n]=raw[(((size_t)h*(l.n/2)+n/2)*(l.owp/32)+ww/32)*64+(n%2)*32+ww%32];
            int e=t_write_region(g,out(g,c,0),&r,origin);
            g->dsp_conv_phase[7]+=now_ns()-post_started;return e;
        }
    }
    if(quant) {
        const T *o=out(g,c,0);
        if(hmx && !(oh&7) && !(ow&7) && !(origin[1]&7) && !(origin[2]&7) &&
           (o->flags&QDR_BLOCKED) && !(o->flags&QDR_DIRECT)) {l.out_h=oh;l.out_w=ow;}
        g->dsp_conv_phase[4]+=now_ns()-params_started;
        TRY(resident_request(g,&l,&xx,w,params));
        post_started=now_ns();
        if(l.out_h) {
            Job job={0};job.cmd=index;job.oh=oh;job.ow=ow;job.co=co;memcpy(job.origin,origin,16);
            Req req={0};req.L=l;req.job=&job;
            HmxCollect a={g,&req,(T *)out(g,c,0)};
            g->arena=g->staging;hmx_collect_blocks(&a,0,1);g->arena=NULL;
            g->dsp_conv_phase[6]+=now_ns()-post_started;return 0;
        }
        const uint8_t *raw=g->staging+l.co;
        uint32_t rbytes=(values+127)&~127u;
        if(rbytes<=0x80000u && l.cb<=0x80000u-rbytes) {
            r.p=(void *)QDC_L2TCM_BASE;
            uint8_t *staged=(uint8_t *)QDC_L2TCM_BASE+rbytes;
            dsp_packet_copy(staged,raw,l.cb);raw=staged;
        } else {
            r.p=(void *)QDC_VTCM_BASE;
            if(rbytes<=0x250000u && l.cb<=0x2d0000u-rbytes) {
                uint8_t *staged=(uint8_t *)QDC_VTCM_BASE+rbytes;
                dsp_packet_copy(staged,raw,l.cb);raw=staged;
            }
        }
        for(uint32_t h=0;h<l.oh;h++)for(uint32_t ww=0;ww<l.ow;ww++)for(uint32_t n=0;n<l.n;n++) {
            size_t dst=((size_t)h*l.ow+ww)*l.n+n;
            size_t src=hmx?((((size_t)(h*l.owp+ww)/32*(l.n/32)+n/32)*32+(h*l.owp+ww)%32)*32+n%32):
                          ((((size_t)h*(l.n/4)+n/4)*(l.owp/32)+ww/32)*128+(n%4)*32+ww%32);
            ((uint8_t *)r.p)[dst]=raw[src];
        }
    } else {
        A preserved=*x;
        g->dsp_conv_phase[4+sixteen]+=now_ns()-params_started;
        uint64_t fallback_started=now_ns();
        int64_t *acc=NULL;uint8_t *local_part=NULL,*local_raw=NULL;
#if QDS_FALLBACK_TCM >= 2
        /* Prefer the master's lower 512 KB L2 scratch when the live arrays
         * fit above the worker's rounded weight copy. This branch has no HMX
         * tables or parameter packet. Raw output staging reuses the weight
         * area only after all workers have returned. */
        if(sixteen && !l.hmx && !l.pb) {
            size_t source_bytes=count4(xx.n);
            uint64_t low=chunked(l.wb);
            if(low<l.cb)low=l.cb;
            if(low<values*2)low=values*2;
            low=(low+127)&~127ull;
            uint64_t keep=(low+values*8+127)&~127ull;
            uint64_t part=(keep+source_bytes*2+127)&~127ull;
            if(part+source_bytes<=0x80000u) {
                local_raw=(uint8_t *)QDC_L2TCM_BASE;
                acc=(int64_t *)(local_raw+(uint32_t)low);
                preserved.p=local_raw+(uint32_t)keep;
                local_part=local_raw+(uint32_t)part;
                r.p=local_raw;g->dsp_fallback_tcm_calls++;g->dsp_fallback_l2_calls++;
            }
        }
#endif
#if QDS_FALLBACK_TCM
        /* A non-HMX request uses only the packed input at VTCM offset zero.
         * Pin the original u16 input and int64 accumulator above BOTH the
         * temporary packing span and the staged int32 output. The worker's
         * convolution and the next plane cannot overwrite either live array.
         * All storage remains inside the already verified 0x2d0000 scratch. */
        if(sixteen && !local_part) {
            size_t source_bytes=count4(xx.n);
            uint64_t low=((source_bytes+127)&~127u)+chunked(l.xb+128);
            if(low<l.cb)low=l.cb;
            low=(low+127)&~127ull;
            uint64_t keep=(low+values*8+127)&~127ull;
            uint64_t part=(keep+source_bytes*2+127)&~127ull;
            if(part+source_bytes<=0x2d0000u-512u && values*2<=low) {
                acc=(int64_t *)(QDC_VTCM_BASE+(uint32_t)low);
                preserved.p=(void *)(QDC_VTCM_BASE+(uint32_t)keep);
                local_part=(uint8_t *)(QDC_VTCM_BASE+(uint32_t)part);
                local_raw=(uint8_t *)QDC_VTCM_BASE;
                r.p=(void *)QDC_VTCM_BASE;g->dsp_fallback_tcm_calls++;
            }
        }
#endif
        if(sixteen) {
            if(!local_part)preserved.p=salloc(g,count4(x->n)*2);CHECK(preserved.p,QDG_ERR_MEMORY);
            memcpy(preserved.p,x->p,count4(x->n)*2);x=&preserved;xx.p=preserved.p;
        }
        if(!acc)acc=salloc(g,values*8);CHECK(acc,QDG_ERR_MEMORY);
        memset(acc,0,values*8);
        for(int plane=0;plane<(sixteen?2:1);plane++) {
            A part=xx;
            if(sixteen) {
                part.dt=DT_U8;part.p=local_part?local_part:salloc(g,values?count4(xx.n):0);CHECK(part.p,QDG_ERR_MEMORY);
                for(size_t j=0;j<count4(xx.n);j++)((uint8_t *)part.p)[j]=(uint8_t)(((const uint16_t *)x->p)[j]>>(plane?0:8));
            }
            g->dsp_fallback_phase[sixteen]+=now_ns()-fallback_started;
            TRY(resident_request(g,&l,&part,w,NULL));fallback_started=now_ns();
            const int32_t *raw=(const int32_t *)(g->staging+l.co);
            if(local_raw){dsp_packet_copy(local_raw,raw,l.cb);raw=(const int32_t *)local_raw;}
            for(uint32_t h=0;h<l.oh;h++)for(uint32_t ww=0;ww<l.ow;ww++)for(uint32_t n=0;n<l.n;n++) {
                size_t dst=((size_t)h*l.ow+ww)*l.n+n;
                acc[dst]+=(int64_t)raw[((size_t)h*l.n+n)*l.owp+ww]*(sixteen && !plane?256:1);
            }
            g->dsp_fallback_phase[2+sixteen]+=now_ns()-fallback_started;fallback_started=now_ns();
        }
        post_started=now_ns();
        int32_t zp=g->t[c->in[0]].zp;
        for(size_t p=0;p<(size_t)oh*ow;p++)for(int k=0;k<co;k++)acc[p*co+k]-=(int64_t)zp*w->wsum[k];
        requant(r.p,acc,mult,bias,(size_t)oh*ow,co,sixteen);
    }
    if(g->scratch_used>scratch_peak)scratch_peak=g->scratch_used;
    int e=t_write_region(g,out(g,c,0),&r,origin);
    g->dsp_conv_phase[6+sixteen]+=now_ns()-post_started;return e;
}
static void status(uint32_t tid,uint32_t seq) {
    volatile QdStatus *st=(volatile QdStatus *)(SERVICE+128)+tid;
    st->error=st->rows=st->reserved=0;__asm__ volatile("barrier":::"memory");st->done_sequence=seq;
}
#if QDS_PREPARED_THREADS>1 && (NPRG2_OVERLAP || NPRG2_QNN_RUNLISTS)
/* Follower 2 remains parked on the master generation but services one
 * independent nested job from the background QNN list when requested. */
static int follower_subjob(uint32_t tid,uint32_t *seen) {
#if QDS_PREPARED_THREADS==4 && QDS_QNN_CLAIM
#if QDS_QNN_LANES
    if(tid>=1 && qnn_claim_serve(tid))return 1;
#else
    if(tid>=2 && qnn_claim_serve(tid))return 1;
#endif
#endif
    if(tid==2 && work->sub_generation!=*seen) {
        __asm__ volatile("barrier":::"memory");
        uint32_t gen=work->sub_generation,parts=work->sub_parts;
        if(parts>1)work->sub_fn(work->sub_ctx,1,parts);
        __asm__ volatile("barrier":::"memory");work->sub_done=gen;*seen=gen;
        return 1;
    }
    if(tid==3 && work->msub_generation!=*seen) {
        __asm__ volatile("barrier":::"memory");
        uint32_t gen=work->msub_generation,parts=work->msub_parts;
        if(parts>1)work->msub_fn(work->msub_ctx,1,parts);
        __asm__ volatile("barrier":::"memory");work->msub_done=gen;*seen=gen;
        return 1;
    }
    return 0;
}
#endif
#if QDS_L2_CACHE && QDS_PREPARED_THREADS>1
/* op 5 (prepared_run, L2 mode): this follower runs here on its page-7 stack
 * for the whole cache lease, touching no L2-TCM address; op 6 returns to the
 * TCM stack, whose bytes the lease saved and restored. */
static void follower_leased(uint32_t tid) {
    tid=qds_logical_tid(tid);
    uint32_t seen=work->generation,sub_seen=tid==3?work->msub_generation:work->sub_generation;
    __asm__ volatile("barrier":::"memory");work->done[tid]=seen;
    for(;;) {
        while(work->generation==seen) {
#if NPRG2_OVERLAP || NPRG2_QNN_RUNLISTS
            if(!follower_subjob(tid,&sub_seen))
#endif
                __asm__ volatile("pause(#" QDS_PAUSE_STR(QDS_LANE_PAUSE) ")");
        }
        __asm__ volatile("barrier":::"memory");seen=work->generation;
        uint32_t op=work->op;
        if(op==6)return;
        if(op==4 && tid<work->job_parts) {
            work->job_started[tid]++;__asm__ volatile("barrier":::"memory");
#if QDS_PART_TIMING
            uint64_t pt0=now_ns();
#endif
            work->job_fn(work->job_ctx,tid,work->job_parts);
#if QDS_PART_TIMING
            if(nprg2_par_kind!=NPRG2_PAR_CONV)qd_part_cycles[tid]+=now_ns()-pt0;
#endif
            __asm__ volatile("barrier":::"memory");work->job_finished[tid]++;
        }
#if QDS_QNN_LANES
        if(op==8 && (tid==1||tid==2))work->job_fn(work->job_ctx,tid,work->job_parts);
#else
        if(op==8 && tid==1)work->job_fn(work->job_ctx,tid,work->job_parts);
#endif
        __asm__ volatile("barrier":::"memory");work->done[tid]=seen;
    }
}
#endif
static void follower(uint32_t tid) {
    uint32_t logical=qds_logical_tid(tid);
    uint32_t seen=0,sub_seen=logical==3?work->msub_generation:work->sub_generation;
    for(;;) {
        while(work->generation==seen) {
#if NPRG2_OVERLAP || NPRG2_QNN_RUNLISTS
            if(!follower_subjob(logical,&sub_seen))
#endif
                __asm__ volatile("pause(#32)");
        }
        __asm__ volatile("barrier":::"memory");seen=work->generation;
        uint32_t op=work->op;
#if QDS_L2_CACHE && QDS_PREPARED_THREADS>1
        if(op==5 && logical && logical<QDS_WORKER_SLOTS) {
            qds_on_stack(follower_leased,logical,QDS_L2_FSTACK(logical)+0x10000u);
            __asm__ volatile("barrier":::"memory");seen=work->generation; /* op 6 */
        }
#endif
        if(op==1){uint32_t e=qdrc_execute(&work->conv,(uint8_t *)work->arena,logical,work->prepared_a);if(e)work->error=e;}
        if(op==2)status(tid,work->sequence);
        if(op==4 && logical<work->job_parts) {
            work->job_started[logical]++;__asm__ volatile("barrier":::"memory");
#if QDS_PART_TIMING
            uint64_t pt0=now_ns();
#endif
            work->job_fn(work->job_ctx,logical,work->job_parts);
#if QDS_PART_TIMING
            if(nprg2_par_kind!=NPRG2_PAR_CONV)qd_part_cycles[logical]+=now_ns()-pt0;
#endif
            __asm__ volatile("barrier":::"memory");work->job_finished[logical]++;
        }
#if QDS_QNN_LANES
        if(op==8 && (logical==1||logical==2))work->job_fn(work->job_ctx,logical,work->job_parts);
#else
        if(op==8 && logical==1)work->job_fn(work->job_ctx,logical,work->job_parts);
#endif
        __asm__ volatile("barrier":::"memory");work->done[logical]=seen;
        if(op==3)return;
    }
}
static int span(uint32_t off,uint32_t bytes){return off>=0x100000 && off<=QDO_BYTES && bytes<=QDO_BYTES-off;}
__attribute__((section(".text.resident_entry"))) void resident_entry(uint32_t tid) {
    if(tid){follower(tid);return;}
    const QdsConfig *cfg=(const QdsConfig *)(SERVICE+QDS_CONFIG);
    volatile uint32_t *cmd=(volatile uint32_t *)SERVICE,*result=(volatile uint32_t *)(SERVICE+QDS_RESULT);
    QdgGraph *g=NULL;
#if QDS_NATIVE_PREPARED
    QdPreparedState *prepared=NULL;
#endif
    uint8_t *pools[7]={0};uint64_t sizes[7];
    const void *inputs[4];uint64_t input_bytes[4];void *outputs[2];uint64_t output_bytes[2];
    uint32_t init_error=!qds_config_valid(cfg);
    if(!init_error) {
        for(int p=0;p<7;p++){sizes[p]=cfg->pool_bytes[p];if(sizes[p]){if(!span(cfg->pool[p],sizes[p]))init_error=1;else pools[p]=(uint8_t *)(QDO_BASE+cfg->pool[p]);}}
        for(int i=0;i<4;i++){inputs[i]=(const void *)(QDO_BASE+cfg->input[i]);input_bytes[i]=cfg->input_bytes[i];if(!span(cfg->input[i],input_bytes[i]))init_error=1;}
        for(int i=0;i<2;i++){outputs[i]=(void *)(QDO_BASE+cfg->output[i]);output_bytes[i]=cfg->output_bytes[i];if(!span(cfg->output[i],output_bytes[i]))init_error=1;}
    }
    if(!init_error) {
        heap_first=(Heap *)(QDO_BASE+cfg->heap);heap_first->bytes=cfg->heap_bytes-128;heap_first->prev=heap_first->next=NULL;heap_first->used=0;heap_scan=heap_first;
        int e;g=qdg_create((const void *)(QDO_BASE+cfg->image),cfg->image_bytes,pools,sizes,&e);
        if(g) {
            dsp_clean_range(g->image,g->img.bytes);dsp_clean_range(g->t,g->ntensors*sizeof *g->t);
            dsp_clean_range(g->c,g->ncommands*sizeof *g->c);
#if !QDS_NATIVE_PREPARED
            QdgGraph *fast=(QdgGraph *)0x300b2000u;
            memcpy(fast,g,sizeof *g);g=fast;
#endif
        }
        init_error=g?qdrc_power(1):(uint32_t)e;
        if(g){g->backend=2;g->profile=1;}
#if QDS_NATIVE_1X1
        if(g)g->use_hmx=!init_error;
#endif
#if QDS_NATIVE_PREPARED
        if(g && !init_error) {
            prepared=prepared_open(g,cfg,inputs,input_bytes,outputs,output_bytes);
            if(!prepared)init_error=98;
        }
#endif
    }
    uint32_t last=0;
    for(;;) {
        uint64_t end=now_ns()+36864000000ull;
        uint32_t seq;
        while(!(seq=cmd[2]) || seq==last)if(now_ns()>end)goto exit;
        __asm__ volatile("barrier":::"memory");last=seq;
        if(cmd[3]==QDC_OP_EXIT)break;
        uint64_t start=now_ns();uint32_t e=init_error;
        if(!e && cmd[0]==QDC_MAGIC && cmd[1]==6 && cmd[3]==QDS_OP_RUN) {
            qdg_reset_stats(g);malloc_cycles=malloc_calls=malloc_steps=metadata_cycles=0;g->dsp_direct_enabled=cmd[6]&1;
            if(cmd[6]&~1u)e=99;
#if QDS_NATIVE_PREPARED
            else e=prepared_run(g,prepared,cmd[4],cmd[5]);
#elif QDS_NATIVE_DIRECT_RANGE
            else e=graph_direct_run(g,cmd[4],cmd[5]);
#else
            else e=qdg_run(g,inputs,input_bytes,outputs,output_bytes,cmd[4],cmd[5]);
#endif
        } else if(!e)e=99;
        uint64_t elapsed=now_ns()-start;
        result[0]=QDS_MAGIC;result[1]=e;result[2]=g?g->stats.error_command:0;result[3]=g?g->error_detail:0;
        result[4]=(uint32_t)elapsed;result[5]=(uint32_t)(elapsed>>32);
        result[6]=g?(uint32_t)g->stats.calls:0;result[7]=g?(uint32_t)g->stats.hmx_calls:0;
        result[8]=heap_used;result[9]=heap_peak;result[10]=scratch_peak;result[11]=g?g->dsp_l2_peak:0;result[12]=g?g->dsp_direct_calls:0;
#if QDS_NATIVE_DIRECT_RANGE
        result[15]=0x52444951u;
#endif
        if(g){memcpy((void *)(SERVICE+QDS_RESULT+64),&g->stats.prepare_ns,8);memcpy((void *)(SERVICE+QDS_RESULT+72),&g->stats.wait_ns,8);
#if QDS_NATIVE_1X1
              result[13]=0x31584e51u;
              memcpy((void *)(SERVICE+QDS_RESULT+520),g->dsp_native_1x1_diag,sizeof g->dsp_native_1x1_diag);
#endif
#if QDS_NATIVE_3X3
              result[14]=0x33584e51u;
              memcpy((void *)(SERVICE+QDS_RESULT+592),g->dsp_native_3x3_diag,sizeof g->dsp_native_3x3_diag);
#endif
              memcpy((void *)(SERVICE+QDS_RESULT+80),g->stats.hmx_phase_cycles,32);result[28]=g->dsp_scratch_peak;
              memcpy((void *)(SERVICE+QDS_RESULT+128),&g->dsp_read_cycles,8);memcpy((void *)(SERVICE+QDS_RESULT+136),&g->dsp_write_cycles,8);
              memcpy((void *)(SERVICE+QDS_RESULT+144),g->dsp_pack_cycles,24);
              memcpy((void *)(SERVICE+QDS_RESULT+168),&g->dsp_weights_cycles,8);memcpy((void *)(SERVICE+QDS_RESULT+176),&g->dsp_quant_cycles,8);
              memcpy((void *)(SERVICE+QDS_RESULT+184),&g->dsp_weights_miss_cycles,8);memcpy((void *)(SERVICE+QDS_RESULT+192),&g->dsp_weights_misses,8);
              memcpy((void *)(SERVICE+QDS_RESULT+200),&malloc_cycles,8);memcpy((void *)(SERVICE+QDS_RESULT+208),&malloc_calls,8);memcpy((void *)(SERVICE+QDS_RESULT+216),&malloc_steps,8);
              memcpy((void *)(SERVICE+QDS_RESULT+224),g->dsp_conv_phase,sizeof g->dsp_conv_phase);result[72]=0x34504643u;
              memcpy((void *)(SERVICE+QDS_RESULT+296),&metadata_cycles,8);memcpy((void *)(SERVICE+QDS_RESULT+304),&g->dsp_dynamic_tcm_calls,8);
              memcpy((void *)(SERVICE+QDS_RESULT+312),g->dsp_fallback_phase,sizeof g->dsp_fallback_phase);
              memcpy((void *)(SERVICE+QDS_RESULT+344),&g->dsp_fallback_tcm_calls,8);
              memcpy((void *)(SERVICE+QDS_RESULT+352),&g->dsp_fallback_l2_calls,8);
#if QDS_BLOCKED_GLUE
              result[72]=0x3b504643u;
              memcpy((void *)(SERVICE+QDS_RESULT+360),&g->dsp_blocked_lc_calls,8);
              memcpy((void *)(SERVICE+QDS_RESULT+368),&g->dsp_blocked_lc_prepare,8);
              memcpy((void *)(SERVICE+QDS_RESULT+376),&g->dsp_blocked_lc_apply,8);
              memcpy((void *)(SERVICE+QDS_RESULT+384),&g->dsp_blocked_lc_u8_calls,8);
              memcpy((void *)(SERVICE+QDS_RESULT+392),&g->dsp_blocked_sign_calls,8);
              memcpy((void *)(SERVICE+QDS_RESULT+400),&g->dsp_blocked_sign_prepare,8);
              memcpy((void *)(SERVICE+QDS_RESULT+408),&g->dsp_blocked_sign_apply,8);
              memcpy((void *)(SERVICE+QDS_RESULT+416),&g->dsp_blocked_add_calls,8);
              memcpy((void *)(SERVICE+QDS_RESULT+424),&g->dsp_blocked_add_prepare,8);
              memcpy((void *)(SERVICE+QDS_RESULT+432),&g->dsp_blocked_add_apply,8);
              memcpy((void *)(SERVICE+QDS_RESULT+440),&g->dsp_direct_input_pad_calls,8);
              memcpy((void *)(SERVICE+QDS_RESULT+448),&g->dsp_direct_im2col_calls,8);
              memcpy((void *)(SERVICE+QDS_RESULT+456),&g->dsp_direct_space_to_depth_calls,8);
#if QDS_BLOCKED_S2D
              result[72]=0x3c504643u;
              memcpy((void *)(SERVICE+QDS_RESULT+464),g->dsp_s2d_diag,sizeof g->dsp_s2d_diag);
#endif
#endif
        }
        if(g){memcpy(SERVICE+QDS_PROFILE,g->stats.variant_ns,80*8);memcpy(SERVICE+QDS_PROFILE+80*8,g->stats.variant_count,80*8);}
#if QDS_NATIVE_PREPARED && QDS_PREPARED_PROFILE
        if(prepared){prepared_export_profile(prepared,SERVICE+QDS_PROFILE2);
#if QDS_PREPARED_THREADS>1
            prepared_export_par(prepared,SERVICE+QDS_PROFILE2+4160u);
#endif
        }
#endif
        send_work(2,seq);status(0,seq);
    }
exit:
#if QDS_NATIVE_PREPARED
    if(prepared)prepared_state_dispose(prepared);
#endif
    send_work(3,last);
    uint32_t e=qdrc_power(0);if(e)result[1]=e;
}
