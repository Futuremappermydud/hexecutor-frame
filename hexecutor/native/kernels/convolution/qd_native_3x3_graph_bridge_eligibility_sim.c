/* Call the actual graph bridge on QDRX-generated tensor/command metadata.
 * Native ownership is deliberately false: supported cases must reach the
 * bridge's hard ownership error before staging; unsupported cases must defer. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "qd_native_3x3_adapter.h"

#define QDG_DSP 1
#define QDS_NATIVE_3X3 1
#define QDS_NATIVE_1X1 1
#define QDS_PRESERVE_VTCM 1
#define QDG_POOLS 7
#define QDG_DSP_DEFER 2001
#define QDG_ERR_DSP 8
#define QDR_HMX_CONV 6u
#define QDR_NONE 0xffffffffu
#define QDR_BLOCKED 2u
#define QDR_DIRECT 4u
#define DT_U8 1u
#define DT_I8 2u
#define DT_I32 5u

static uint8_t fake_vtcm[8u*1024u*1024u] __attribute__((aligned(128)));
#define QDVT_BASE ((uintptr_t)fake_vtcm)

typedef struct {
    uint32_t id,dtype,flags,esize,crouton,cls;
    int32_t d[4],m[4],p[4];
    uint32_t pool,offset,nblocks,slice;
    const uint32_t *blocks;
    int32_t zp,sv[4];
    uint32_t scalar,unique_blocks;
    double scale;
} T;
typedef struct {
    uint32_t seq,op,variant,io,nin,nout,nfv,ncalls;
    const uint32_t *in,*out,*fv;
    const uint8_t *calls;
} C;
typedef struct QdgGraph {
    uint32_t ntensors;
    T t[4];
    uint8_t *pool[QDG_POOLS];
    uint64_t pool_bytes[QDG_POOLS];
    uint32_t static_source[4];
    uint64_t dsp_native_3x3_diag[8];
    uint32_t use_hmx,resident_key,error_detail;
} QdgGraph;

#include "native3_bridge_cases_generated.h"

static uint32_t tensor_ids[3]={0,1,2},output_ids[1]={3},dummy_blocks[2];
static uint8_t *const unused_pool=NULL;
static uint32_t cache_calls,prepare_calls,issue_calls,drain_calls,finish_calls,invalidate_calls;
static uint64_t test_clock;

static uint64_t now_ns(void){return ++test_clock;}
static const T *dsp_tensor(QdgGraph*g,uint32_t i){return i<g->ntensors?&g->t[i]:NULL;}
static void dsp_clean_range(const void*p,size_t n){(void)p;(void)n;cache_calls++;}
static int is_class(const T*t,const char*a,const char*b){uint32_t x=0,y=0;memcpy(&x,a,strlen(a));memcpy(&y,b,strlen(b));return t->cls==x||t->cls==y;}
static int native_1x1_lease_valid(void){return 0;}
static int native_1x1_hmx_owned(void){return 0;}
static int native_1x1_workers_idle(void){return 0;}
static void qdrc_invalidate_native(void){invalidate_calls++;}
int qd_native_3x3_prepare(const QdN3x3Request*r,QdN3x3Prepared*p){(void)r;(void)p;prepare_calls++;return QD_N3X3_UNSUPPORTED;}
int qd_native_3x3_issue(QdN3x3Prepared*p){(void)p;issue_calls++;return QD_N3X3_UNSUPPORTED;}
int qd_native_3x3_drain(QdN3x3Prepared*p){(void)p;drain_calls++;return QD_N3X3_UNSUPPORTED;}
int qd_native_3x3_finish(const QdN3x3Prepared*p){(void)p;finish_calls++;return QD_N3X3_UNSUPPORTED;}

#include "qd_native_3x3_graph_bridge.h"

static void load_case(QdgGraph*g,C*c,uint32_t row){
    memset(g,0,sizeof *g);memset(c,0,sizeof *c);g->ntensors=4;g->use_hmx=0;
    for(uint32_t i=0;i<4;i++)g->static_source[i]=QDR_NONE;
    g->pool[1]=(uint8_t*)fake_vtcm;g->pool_bytes[1]=k_bridge_pool1_bytes;
    for(uint32_t i=0;i<4;i++){
        T*t=&g->t[i];const uint32_t*m=k_bridge_metadata[row][i];
        t->id=i;t->dtype=m[0];t->flags=m[1];t->esize=m[2];t->crouton=m[3];t->cls=m[4];
        for(uint32_t d=0;d<4;d++){t->d[d]=(int32_t)m[5+d];t->m[d]=(int32_t)m[9+d];t->p[d]=(int32_t)m[13+d];}
        t->pool=m[17];t->offset=m[18];t->nblocks=m[19];t->slice=m[20];t->zp=(int32_t)m[21];
        t->blocks=(t->flags&QDR_BLOCKED)?dummy_blocks:NULL;
    }
    c->seq=k_bridge_sequences[row];c->op=QDR_HMX_CONV;c->nin=3;c->nout=1;
    c->in=tensor_ids;c->out=output_ids;
}

static uint64_t fnv64(const uint8_t*p,uint32_t n){uint64_t h=UINT64_C(0xcbf29ce484222325);for(uint32_t i=0;i<n;i++)h=(h^p[i])*UINT64_C(0x100000001b3);return h;}

int main(void){
    memset(fake_vtcm,0x5a,sizeof fake_vtcm);uint64_t before=fnv64(fake_vtcm,sizeof fake_vtcm);
    for(uint32_t row=0;row<5;row++){
        QdgGraph g;C c;load_case(&g,&c,row);memset(g.dsp_native_3x3_diag,0,sizeof g.dsp_native_3x3_diag);
        g.error_detail=0;uint32_t e=(uint32_t)native_3x3_try(&g,&c,0);
        if(e!=QDG_ERR_DSP||g.dsp_native_3x3_diag[5]!=1||g.dsp_native_3x3_diag[4]||
           g.error_detail!=0x4e330003u){
            printf("FAIL supported sequence=%u result=%u diag4=%llu diag5=%llu error=%08x\n",c.seq,e,
              (unsigned long long)g.dsp_native_3x3_diag[4],(unsigned long long)g.dsp_native_3x3_diag[5],g.error_detail);return 1;
        }
        printf("PASS actual bridge accepts metadata seq=%u then declines missing ownership\n",c.seq);
        g.t[3].p[2]=1;g.error_detail=0;memset(g.dsp_native_3x3_diag,0,sizeof g.dsp_native_3x3_diag);
        e=(uint32_t)native_3x3_try(&g,&c,0);
        if(e!=QDG_DSP_DEFER||g.dsp_native_3x3_diag[4]!=1||g.dsp_native_3x3_diag[5]||g.error_detail){
            printf("FAIL unsupported sequence=%u result=%u diag4=%llu diag5=%llu error=%08x\n",c.seq,e,
              (unsigned long long)g.dsp_native_3x3_diag[4],(unsigned long long)g.dsp_native_3x3_diag[5],g.error_detail);return 2;
        }
        printf("PASS actual bridge defers unsupported geometry seq=%u\n",c.seq);
    }
    if(cache_calls||prepare_calls||issue_calls||drain_calls||finish_calls||invalidate_calls||
       fnv64(fake_vtcm,sizeof fake_vtcm)!=before){
        printf("FAIL side effects cache=%u prepare=%u issue=%u drain=%u finish=%u invalidate=%u\n",
          cache_calls,prepare_calls,issue_calls,drain_calls,finish_calls,invalidate_calls);return 3;
    }
    puts("PASS actual graph bridge metadata gate: 5 supported, 5 unsupported, no VTCM/cache/HMX side effects");
    (void)unused_pool;return 0;
}
