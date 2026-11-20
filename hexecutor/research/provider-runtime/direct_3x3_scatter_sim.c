/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "../../native/qd_resident.h"
#include "../../native/qd_graph.h"
#define QDG_DSP 1
enum { DT_U8=1, DT_I8=2, DT_I32=5 };
typedef struct {
    uint32_t id,dtype,flags,esize,crouton,cls;
    int32_t d[4],m[4],p[4];
    uint32_t pool,offset,nblocks,slice;
    const uint32_t *blocks;
    int32_t zp,sv[4];
    uint32_t scalar,unique_blocks;
    double scale;
} T;
typedef struct {uint32_t seq,op,variant,io,nin,nout,nfv,ncalls;const uint32_t *in,*out,*fv;const uint8_t *calls;} C;
struct QdgGraph {
    uint32_t ntensors,ncommands;T*t;C*c;uint8_t*pool[7];uint64_t pool_bytes[7];
    uint32_t resident_key;uint64_t dsp_read_cycles,dsp_write_cycles;
    uint64_t dsp_native_1x1_diag[8],dsp_native_3x3_diag[8];QdgStats stats;uint32_t error_detail;
};
#define SERVICE ((uintptr_t)0xf1800000u)
#include "qd_native_direct.h"

#ifndef QD_GRAPH_OUTPUT_PATH
#define QD_GRAPH_OUTPUT_PATH "/src/native_runtime/direct-3x3-scatter-output.bin"
#endif
#ifndef QD_GRAPH_MARKER_PATH
#define QD_GRAPH_MARKER_PATH "/src/native_runtime/direct-3x3-graph-caller.ok"
#endif
#define PAGE 2048u
#define POOL_CAP (4u*1024u*1024u)
#define FIX_CAP (6u*1024u*1024u)
#define SCATTER_MAGIC 0x324d4353u
#define L2_TABLES 0x30000000u
#define L2_ABI 0x30000800u
extern void qd_hmx_finish(void *,void *);
extern void qd_set_hmx_params_stride1(uint32_t*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
static uint8_t fixture[FIX_CAP] __attribute__((aligned(128)));
static uint8_t graph_pool[POOL_CAP] __attribute__((aligned(128)));
static uint8_t graph_snapshot[POOL_CAP];
static uint8_t reference_workspace[QD_N3X3_WORKSPACE_MAX] __attribute__((aligned(128)));
static uint8_t vtcm_snapshot[QD_DIRECT_ACTIVE_BYTES];
static uint8_t expected_graph_y[32u*PAGE];
static uint8_t y_mask[QD_DIRECT_ACTIVE_BYTES];
static uint32_t xblocks[2048],yblocks[2048],direct_xblocks[2048],direct_yblocks[2048];
static uint32_t graph_sizes[7]={0,POOL_CAP,0,0,0,0,0},direct_sizes[7];
static uint32_t service_backing[1024] __attribute__((aligned(4096)));
static uint32_t manual_tables[QD_DIRECT_TABLE_BYTES/4u],manual_abi[QD_DIRECT_ABI_BYTES/4u];
static uint32_t caller_abi_mismatches,caller_table_mismatches;
typedef struct {uint32_t source,dest;} PageMap;
static PageMap map_pages[256];

static const T *dsp_tensor(QdgGraph*g,uint32_t i){return i<g->ntensors?&g->t[i]:0;}
static void dsp_command_begin(QdgGraph*g,uint32_t i,C*out){*out=g->c[i];}
static int is_class(const T*t,const char*a,const char*b){(void)b;return t->cls==(!strcmp(a,"CB")?1u:!strcmp(a,"Xwb")?2u:!strcmp(a,"Fi")?3u:0u);}
static int native_1x1_pool_span(QdgGraph*g,const T*t,uint32_t n){return t->pool==1&&(uint64_t)t->offset+n<=g->pool_bytes[1];}
static int native_1x1_workers_idle(void){return 1;}
static int native_1x1_lease_valid(void){return 1;}
static int native_1x1_hmx_owned(void){return 1;}
static void dsp_clean_range(const void*p,uint32_t n){uintptr_t a=(uintptr_t)p,e=a+n;for(;a<e;a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");if(a==(uintptr_t)0x30000000u+n||a==(uintptr_t)0x30000800u+n){uint32_t d=0;for(uint32_t q=0;q<QD_DIRECT_TABLE_BYTES/4u;q++)d+=((uint32_t*)0x30000000u)[q]!=manual_tables[q];printf("CLEAN range %08x n=%u table mismatches after=%u\n",(uint32_t)(uintptr_t)p,n,d);fflush(stdout);}}
static void dsp_packet_copy(void*d,const void*s,uint32_t n){memcpy(d,s,n);}
static void qdrc_invalidate_native(void){}
static int resident_cancelled(void){return 0;}
static uint64_t now_ns(void){uint64_t n;__asm__ volatile("%0=upcycle":"=r"(n));return n;}
extern void qd_hmx_finish(void *,void *);
void qd_qnn_hmx_1x1(uint32_t*o,uint32_t*i,void*w,void*f,uint32_t*c,uint32_t*m){(void)o;(void)i;(void)w;(void)f;(void)c;(void)m;}
extern void graph_test_real_leaf(uint32_t*,uint32_t*,void*,void*,uint32_t*,uint32_t*) __asm__("qd_qnn_hmx_stride1_generic");
extern int graph_test_real_bind3(const QdN3x3Request*,void*,uint32_t,uint32_t*,uint32_t,uint32_t*,uint32_t,void*,uint32_t,QdDirectBinding*) __asm__("qd_direct_bind_3x3");
static void graph_test_stackcopy_leaf(uint32_t*,uint32_t*,void*,void*,uint32_t*,uint32_t*);
static int graph_test_bind3(const QdN3x3Request*,void*,uint32_t,uint32_t*,uint32_t,uint32_t*,uint32_t,void*,uint32_t,QdDirectBinding*);
#ifdef QD_GRAPH_STACKCOPY
#define qd_qnn_hmx_stride1_generic graph_test_stackcopy_leaf
#endif
#define qd_direct_bind_3x3 graph_test_bind3
#include "../../native/qd_graph_native_direct.h"
#ifdef QD_GRAPH_STACKCOPY
#undef qd_qnn_hmx_stride1_generic
#endif
#undef qd_direct_bind_3x3
static int graph_test_bind3(const QdN3x3Request*r,void*v,uint32_t active,uint32_t*t,uint32_t tb,uint32_t*a,uint32_t ab,void*tail,uint32_t tailb,QdDirectBinding*b){
    int e=graph_test_real_bind3(r,v,active,t,tb,a,ab,tail,tailb,b);
    if(!e&&b){uint32_t n=0;for(uint32_t q=0;q<QD_DIRECT_TABLE_BYTES/4u;q++)n+=t[q]!=manual_tables[q];printf("BIND COMMIT immediate table mismatches=%u\n",n);fflush(stdout);}
    return e;
}
static void graph_test_stackcopy_leaf(uint32_t*out,uint32_t*in,void*w,void*f,uint32_t*cfg,uint32_t*mode){
    uint32_t o[6],i[6],c[16],m[2];
    for(uint32_t q=0;q<6;q++){o[q]=out[q];i[q]=in[q];}
    for(uint32_t q=0;q<16;q++)c[q]=cfg[q];
    for(uint32_t q=0;q<2;q++)m[q]=mode[q];
    caller_abi_mismatches=0;
    for(uint32_t q=0;q<30;q++){uint32_t got=q<6?i[q]:q<12?o[q-6]:q<28?c[q-12]:m[q-28];if(got!=manual_abi[q])caller_abi_mismatches++;}
    caller_table_mismatches=0;
    for(uint32_t q=0;q<QD_DIRECT_TABLE_BYTES/4u;q++)if(((uint32_t*)0x30000000u)[q]!=manual_tables[q]){if(caller_table_mismatches<16)printf("table mismatch q=%u got=%08x manual=%08x\n",q,((uint32_t*)0x30000000u)[q],manual_tables[q]);caller_table_mismatches++;}
#ifdef QD_GRAPH_STACKCOPY
    printf("CALLER stackcopy comparison ABI mismatches=%u tables=%u cfg12=%08x\n",caller_abi_mismatches,caller_table_mismatches,c[12]);
    fflush(stdout);
    graph_test_real_leaf(o,i,w,f,c,m);
#else
    caller_abi_mismatches=caller_table_mismatches=0;
    graph_test_real_leaf(o,i,w,f,cfg,mode);
#endif
}

static uint32_t gcd32(uint32_t a,uint32_t b){while(b){uint32_t t=a%b;a=b;b=t;}return a;}
static uint32_t p32(const void *p){return (uint32_t)(uintptr_t)p;}
static uint32_t lookup_page(uint32_t off,uint32_t count){
    for(uint32_t i=0;i<count;i++)if(map_pages[i].source==off)return map_pages[i].dest;
    return UINT32_MAX;
}
static int map_page(uint32_t off,uint32_t count,uint32_t multiplier,uint32_t *next){
    if(off%PAGE)return 0;
    for(uint32_t i=0;i<*next;i++)if(map_pages[i].source==off)return 1;
    if(*next>=count)return 0;
    uint32_t slot=((*next)*multiplier)%count;
    map_pages[*next].source=off;map_pages[*next].dest=slot*PAGE;(*next)++;
    return 1;
}

int main(void){
#ifndef QD_GRAPH_DIRECT_CALLER
    remove("/src/native_runtime/direct-3x3-scatter-output.bin");
#endif
    FILE*f=fopen("/src/native_kernels/xwb_ci64_qdrx247_cpu_prefix_fixture.bin","rb");
    if(!f){puts("FAIL sequence247 fixture open");return 1;}
    uint32_t h[32];if(fread(h,sizeof h,1,f)!=1||h[1]!=247u){puts("FAIL fixture header");return 2;}
    uint32_t xb=h[7]*PAGE,yb=h[13]*PAGE,wb=h[17],fb=h[18];
    uint64_t total=(uint64_t)xb+yb+wb+fb;if(total>sizeof fixture||fread(fixture,(size_t)total,1,f)!=1||fgetc(f)!=EOF){puts("FAIL fixture payload");return 3;}fclose(f);
    const uint8_t*x=fixture,*y=x+xb,*w=y+yb,*fi=w+wb;
    memcpy(graph_pool,x,xb);memcpy(graph_pool+xb,y,yb);memcpy(graph_pool+xb+yb,w,wb);memcpy(graph_pool+xb+yb+wb,fi,fb);
    uint8_t*gw=graph_pool+xb+yb,*gfi=gw+wb;
    for(uint32_t i=0;i<h[7];i++)xblocks[i]=i*PAGE|1u;
    for(uint32_t i=0;i<h[13];i++)yblocks[i]=(xb+i*PAGE)|1u;
    memcpy(graph_snapshot,graph_pool,sizeof graph_pool);

    FILE*m=fopen("/src/native_runtime/direct-3x3-qdrx247-offsets.bin","rb");
    if(!m){puts("FAIL scatter map open; generate with make_direct_3x3_scatter_map.py");return 4;}
    uint32_t mh[8],xoff[2048],yoff[2048];
    if(fread(mh,sizeof mh,1,m)!=1||mh[0]!=SCATTER_MAGIC||mh[1]!=247||mh[2]!=h[7]||mh[3]!=h[13]||mh[6]!=wb||mh[7]!=fb||
       fread(xoff,sizeof(uint32_t),h[7],m)!=h[7]||fread(yoff,sizeof(uint32_t),h[13],m)!=h[13]||fgetc(m)!=EOF){puts("FAIL scatter map header");return 5;}fclose(m);

    uint32_t inpages=((h[4]+7u)/8u)*((h[5]+7u)/8u)*(h[14]/32u);
    uint32_t outpages=((h[10]+7u)/8u)*((h[11]+7u)/8u)*(h[15]/32u);
    if(inpages>h[7]||h[12]+outpages>h[13]){puts("FAIL active page counts");return 6;}
    /* Compact only the real QDRX pages this command touches, preserving each
     * exact duplicate offset. A coprime permutation makes table lookups jump
     * across physical pages instead of following tensor order. */
    uint32_t np=0;for(uint32_t i=0;i<inpages;i++)if(!map_page(xoff[i],256,73,&np)){puts("FAIL input page map");return 7;}
    for(uint32_t q=0;q<outpages;q++)if(!map_page(yoff[h[12]+q],256,73,&np)){puts("FAIL output page map");return 8;}
    uint32_t mul=73;while(gcd32(mul,np)!=1){if(mul>1000){puts("FAIL permutation gcd");return 9;}mul+=2;}
    for(uint32_t i=0;i<np;i++)map_pages[i].dest=((i*mul)%np)*PAGE;
    /* Copy X block contents according to the real QDRX offset identities. */
    for(uint32_t i=0;i<inpages;i++){
        uint32_t dst=lookup_page(xoff[i],np);if(dst==UINT32_MAX){puts("FAIL X offset lookup");return 10;}
        for(uint32_t j=0;j<i;j++)if(xoff[j]==xoff[i]&&memcmp(x+j*PAGE,x+i*PAGE,PAGE)){puts("FAIL QDRX duplicate X bytes differ");return 11;}
        direct_xblocks[i]=dst|1u;
    }
    uint8_t*vtcm=(uint8_t*)(__rdcfg(__vtcm_base)<<16);
    printf("SDK VTCM base=%08x direct base=%08x bytes=%x L2cfg=%08x\n",(uint32_t)(uintptr_t)vtcm,(uint32_t)QD_DIRECT_VTCM_BASE,QD_DIRECT_VTCM_BYTES,__rdcfg(__l2_cfg_base));
    for(uint32_t i=0;i<QD_DIRECT_VTCM_BYTES;i+=4096u)add_translation(vtcm+i,vtcm+i,6);
    for(uint32_t i=0;i<QD_DIRECT_VTCM_BYTES;i+=4096u)add_translation((void*)(QD_DIRECT_VTCM_BASE+i),vtcm+i,6);
    /* Re-copy X after establishing the simulator's VTCM translations. */
    memset(vtcm,0,QD_DIRECT_VTCM_BYTES);
    for(uint32_t i=0;i<inpages;i++)memcpy(vtcm+lookup_page(xoff[i],np),x+i*PAGE,PAGE);
    for(uint32_t i=0;i<h[13];i++)direct_yblocks[i]=0;
    for(uint32_t q=0;q<outpages;q++){
        uint32_t actual=h[12]+q,dst=lookup_page(yoff[actual],np);
        if(dst==UINT32_MAX){puts("FAIL Y offset lookup");return 12;}
        memcpy(vtcm+dst,y+actual*PAGE,PAGE);direct_yblocks[actual]=dst|1u;
    }
    uint32_t woff=(np*PAGE+127u)&~127u,foff=(woff+wb+127u)&~127u;
    if((uint64_t)foff+fb>QD_DIRECT_ACTIVE_BYTES||foff%128u){puts("FAIL compact W/FI bounds");return 13;}
    memcpy(vtcm+woff,w,wb);memcpy(vtcm+foff,fi,fb);
    direct_sizes[1]=QD_DIRECT_ACTIVE_BYTES;

    QdN3x3Request ref;memset(&ref,0,sizeof ref);ref.pool_count=7;ref.pool[1]=graph_pool;ref.pool_bytes=graph_sizes;
    ref.input=(QdN3x3Blocked){.dims={1,h[2],h[3],h[14]},.max_dims={1,h[4],h[5],h[14]},.padding={0,h[19],h[20],0},.slice_offset=h[6],.block_count=h[7],.blocks=xblocks};
    ref.output=(QdN3x3Blocked){.dims={1,h[8],h[9],h[24]?h[24]:h[15]},.max_dims={1,h[10],h[11],h[15]},.padding={0,h[21],h[22],0},.slice_offset=h[12],.block_count=h[13],.blocks=yblocks};
    ref.weights=gw;ref.weights_bytes=wb;ref.weight_ci=h[14];ref.weight_co=h[15];ref.weight_kernel_h=3;ref.weight_kernel_w=3;
    ref.input_dtype=1;ref.output_dtype=1;ref.weight_dtype=2;ref.fi_dtype=5;ref.weight_format=h[16];ref.fi=(const int32_t*)gfi;ref.fi_bytes=fb;
    ref.input_zero_point=h[23];ref.output_zero_point=h[25];ref.workspace=reference_workspace;ref.workspace_bytes=sizeof reference_workspace;
    QdN3x3Prepared p;int e=qd_native_3x3_prepare(&ref,&p);if(e){printf("FAIL reference ABI prepare=%d\n",e);return 14;}
    memcpy(vtcm_snapshot,vtcm,QD_DIRECT_ACTIVE_BYTES);
    QdN3x3Request direct=ref;direct.pool[1]=vtcm;direct.pool_bytes=direct_sizes;
    direct.input.blocks=direct_xblocks;direct.output.blocks=direct_yblocks;
    direct.weights=vtcm+woff;direct.fi=(const int32_t*)(vtcm+foff);direct.workspace=0;direct.workspace_bytes=0;
    for(uint32_t q=0;q<inpages;q++)if((direct_xblocks[q]&~127u)>=direct_sizes[1]||
        (direct_xblocks[q]&~127u)%PAGE||((direct_xblocks[q]&127u)!=1u)){
        printf("FAIL local X ref q=%u enc=%08x active=%x\n",q,direct_xblocks[q],direct_sizes[1]);return 26;}
    for(uint32_t q=0;q<outpages;q++){uint32_t ix=h[12]+q,enc=direct_yblocks[ix];if((enc&~127u)>=direct_sizes[1]||
        (enc&~127u)%PAGE||((enc&127u)!=1u)){
        printf("FAIL local Y ref q=%u ix=%u enc=%08x active=%x\n",q,ix,enc,direct_sizes[1]);return 27;}}
    /* Map fixed caller addresses to simulator L2 TCM itself. Mapping them to
     * a cached C global creates a CPU synonym: dccleaninva can then publish
     * stale alias lines over newly-built pointer-table entries. */
    for(uint32_t i=0;i<0x4000u;i+=4096u)add_translation((void*)(L2_TABLES+i),(void*)(0xd8000000u+i),6);
    uint32_t*tables=(uint32_t*)L2_TABLES,*abi=(uint32_t*)L2_ABI;
    QdDirectBinding b;e=qd_direct_validate_3x3(&direct,vtcm,QD_DIRECT_ACTIVE_BYTES,tables,QD_DIRECT_TABLE_BYTES,abi,QD_DIRECT_ABI_BYTES,
        vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,0);
    if(e){printf("FAIL scatter preflight=%d\n",e);return 15;}
    e=qd_direct_bind_3x3(&direct,vtcm,QD_DIRECT_ACTIVE_BYTES,tables,QD_DIRECT_TABLE_BYTES,abi,QD_DIRECT_ABI_BYTES,
        vtcm+QD_DIRECT_TAIL_OFFSET,QD_DIRECT_TAIL_BYTES,&b);
    if(e){printf("FAIL scatter bind=%d\n",e);return 16;}
    memcpy(manual_tables,tables,sizeof manual_tables);memcpy(manual_abi,abi,sizeof manual_abi);
    uint32_t mismatch=0,alias_count=0,nonmonotonic=0;
    for(uint32_t q=0;q<inpages;q++){
        uint32_t want=lookup_page(xoff[q],np);mismatch+=tables[q]!=p32(vtcm+want);
        if(q&&tables[q]<tables[q-1])nonmonotonic++;
        for(uint32_t j=0;j<q;j++)if(xoff[j]==xoff[q]){alias_count++;break;}
    }
    for(uint32_t q=0;q<outpages;q++){
        uint32_t want=lookup_page(yoff[h[12]+q],np);mismatch+=tables[QD_N3X3_TABLE_ENTRIES+q]!=p32(vtcm+want);
    }
    for(uint32_t i=1;i<6;i++)mismatch+=abi[i]!=p.input_desc[i];
    for(uint32_t i=1;i<6;i++)mismatch+=abi[6+i]!=p.output_desc[i];
    for(uint32_t i=0;i<16;i++)mismatch+=abi[12+i]!=p.cfg[i];
    mismatch+=abi[28]!=p.mode[0]||abi[29]!=p.mode[1];
    if(mismatch||alias_count==0||nonmonotonic==0){printf("FAIL scatter map/ABI mismatches=%u aliases=%u jumps=%u\n",mismatch,alias_count,nonmonotonic);return 17;}
    if(memcmp(vtcm,vtcm_snapshot,QD_DIRECT_ACTIVE_BYTES)){puts("FAIL binder wrote active pool data");return 18;}
#ifndef QD_GRAPH_DIRECT_CALLER
    for(uintptr_t a=(uintptr_t)tables;a<(uintptr_t)tables+QD_DIRECT_TABLE_BYTES;a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    for(uintptr_t a=(uintptr_t)abi;a<(uintptr_t)abi+QD_DIRECT_ABI_BYTES;a+=32)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
    __asm__ volatile("barrier":::"memory");
    uint32_t manual_clean_diff=0;for(uint32_t q=0;q<QD_DIRECT_TABLE_BYTES/4u;q++)manual_clean_diff+=tables[q]!=manual_tables[q];printf("MANUAL explicit clean table mismatches=%u\n",manual_clean_diff);fflush(stdout);
    uint32_t in_desc[6],out_desc[6],cfg[16],mode[2];
    memcpy(in_desc,b.in_desc,sizeof in_desc);memcpy(out_desc,b.out_desc,sizeof out_desc);memcpy(cfg,b.cfg,sizeof cfg);memcpy(mode,b.mode,sizeof mode);
    uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");
    __asm__ volatile("barrier\n mxclracc":::"memory");
    qd_qnn_hmx_stride1_generic(out_desc,in_desc,b.weights,b.fi,cfg,mode);
    *b.done=0xffffffffu;qd_hmx_finish(b.zero_fi,(void*)b.done);
    uint32_t tries=0;while(*b.done&&tries++<100000u)__asm__ volatile("nop");
    if(*b.done){puts("FAIL direct scatter HMX timeout");return 19;}
    if(memcmp(graph_pool,graph_snapshot,sizeof graph_pool)){puts("FAIL host graph pools changed");return 20;}
    memset(y_mask,0,sizeof y_mask);
    for(uint32_t q=0;q<outpages;q++){uint32_t off=lookup_page(yoff[h[12]+q],np);memset(y_mask+off,1,PAGE);}
    for(uint32_t i=0;i<QD_DIRECT_ACTIVE_BYTES;i++)if(!y_mask[i]&&vtcm[i]!=vtcm_snapshot[i]){printf("FAIL non-Y active byte changed %x\n",i);return 21;}
#endif
    for(uint32_t i=0;i<PAGE;i++)if(((uint8_t*)b.output_guard)[i]!=0xa5||((uint8_t*)b.input_guard)[i]){puts("FAIL direct tail guard");return 22;}
    for(uint32_t i=0;i<64;i++)if(((uint32_t*)b.zero_fi)[i]!=(i<32?0x404000u:0u)){puts("FAIL direct zero-Fi");return 23;}
#ifdef QD_GRAPH_DIRECT_CALLER
    /* Exercise the production inline caller with real fixed-address L2
     * tables/ABI and the same independently permuted seq247 page refs. */
#ifdef QD_GRAPH_DIRECT_CALLER
    FILE *expected_file=fopen("/src/native_runtime/direct-3x3-scatter-output.bin","rb");uint32_t expected_header[32];
    if(!expected_file||fread(expected_header,sizeof expected_header,1,expected_file)!=1||memcmp(expected_header,h,sizeof h)||fread(expected_graph_y,PAGE,outpages,expected_file)!=outpages){puts("FAIL graph caller expected output fixture");return 32;}fclose(expected_file);
    memset(y_mask,0,sizeof y_mask);for(uint32_t q=0;q<outpages;q++)memset(y_mask+(direct_yblocks[h[12]+q]&~127u),1,PAGE);
#else
    for(uint32_t q=0;q<outpages;q++)memcpy(expected_graph_y+q*PAGE,vtcm+(direct_yblocks[h[12]+q]&~127u),PAGE);
#endif
    memset(service_backing,0,sizeof service_backing);
    add_translation((void*)SERVICE,service_backing,6);
    memcpy(graph_pool,vtcm_snapshot,QD_DIRECT_ACTIVE_BYTES);
    T gt[4];memset(gt,0,sizeof gt);
    gt[0].dtype=DT_U8;gt[0].flags=QDR_BLOCKED;gt[0].crouton=1;gt[0].cls=1;gt[0].d[0]=1;gt[0].d[1]=h[2];gt[0].d[2]=h[3];gt[0].d[3]=h[14];gt[0].m[0]=1;gt[0].m[1]=h[4];gt[0].m[2]=h[5];gt[0].m[3]=h[14];gt[0].p[1]=h[19];gt[0].p[2]=h[20];gt[0].pool=1;gt[0].nblocks=h[7];gt[0].slice=h[6];gt[0].blocks=direct_xblocks;
    gt[1].dtype=DT_I8;gt[1].flags=QDR_DIRECT;gt[1].cls=2;gt[1].d[0]=gt[1].m[0]=3;gt[1].d[1]=gt[1].m[1]=3;gt[1].d[2]=gt[1].m[2]=h[14];gt[1].d[3]=gt[1].m[3]=h[15];gt[1].pool=1;gt[1].offset=woff;
    gt[2].dtype=DT_I32;gt[2].flags=QDR_DIRECT;gt[2].cls=3;gt[2].d[3]=gt[2].m[3]=2u*h[15];gt[2].d[0]=gt[2].m[0]=gt[2].d[1]=gt[2].m[1]=gt[2].d[2]=gt[2].m[2]=1;gt[2].pool=1;gt[2].offset=foff;
    gt[3].dtype=DT_U8;gt[3].flags=QDR_BLOCKED;gt[3].crouton=1;gt[3].cls=1;gt[3].d[0]=gt[3].m[0]=1;gt[3].d[1]=gt[3].m[1]=h[8];gt[3].d[2]=gt[3].m[2]=h[9];gt[3].d[3]=gt[3].m[3]=h[15];gt[3].pool=1;gt[3].nblocks=h[13];gt[3].slice=h[12];gt[3].blocks=direct_yblocks;
    uint32_t gin[3]={0,1,2},gout[1]={3};C gc={.seq=247,.op=QDR_HMX_CONV,.variant=33,.nin=3,.nout=1,.in=gin,.out=gout};
    QdgGraph gg;memset(&gg,0,sizeof gg);gg.ntensors=4;gg.ncommands=1;gg.t=gt;gg.c=&gc;gg.pool[1]=graph_pool;gg.pool_bytes[1]=QD_DIRECT_ACTIVE_BYTES;
    QdGraphDirectRequest gn;QdDirectBinding gb;gg.pool[1]=(uint8_t*)QD_DIRECT_VTCM_BASE;int gr=graph_direct_request(&gg,&gc,&gn);int gbr=gr?graph_direct_bind(&gn,0,&gb):-99;gg.pool[1]=graph_pool;
    if(gr!=1||gbr){uint32_t cc[16]={0};qd_set_hmx_params_stride1(cc,0x700u,3u,gn.three.input.max_dims[1]/8u,gn.three.weight_ci,7u,1u,12u);printf("FAIL actual graph preflight request=%d bind=%d cfg12=%08x dims=%d,%d/%d,%d pad=%d,%d slice=%u/%u blocks=%u/%u w=%u fi=%u\n",gr,gbr,cc[12],gt[0].d[1],gt[0].d[2],gt[3].d[1],gt[3].d[2],gt[0].p[1],gt[0].p[2],gt[0].slice,gt[3].slice,gt[0].nblocks,gt[3].nblocks,gn.three.weights_bytes,gn.three.fi_bytes);return 31;}
    gg.pool[1]=(uint8_t*)QD_DIRECT_VTCM_BASE;int cbe=graph_direct_bind(&gn,1,&gb);gg.pool[1]=graph_pool;uint32_t cbt=0;for(uint32_t q=0;q<QD_DIRECT_TABLE_BYTES/4u;q++)cbt+=((uint32_t*)0x30000000u)[q]!=manual_tables[q];printf("PREHEADER bind=%d table mismatches=%u woff=%u/%u fioff=%u/%u refwt=%u/%u ci/co=%u/%u\n",cbe,cbt,(uint32_t)(uintptr_t)(gn.three.weights-(uint8_t*)QD_DIRECT_VTCM_BASE),woff,(uint32_t)(uintptr_t)((uint8_t*)gn.three.fi-(uint8_t*)QD_DIRECT_VTCM_BASE),foff,direct.weights_bytes,gn.three.weights_bytes,gn.three.weight_ci,gn.three.weight_co);
    dsp_packet_copy((void*)QD_DIRECT_VTCM_BASE,graph_pool,QD_DIRECT_ACTIVE_BYTES);uint32_t copy_table_mismatch=0;for(uint32_t q=0;q<QD_DIRECT_TABLE_BYTES/4u;q++)copy_table_mismatch+=((uint32_t*)0x30000000u)[q]!=manual_tables[q];printf("BOUNDARY COPY table mismatches=%u\n",copy_table_mismatch);
    int ge=graph_direct_run(&gg,0,1);if(ge){printf("FAIL actual graph_direct_run error=%d detail=%08x trace=%u family=%u cfg12=%08x\n",ge,gg.error_detail,service_backing[256+1],service_backing[256+4],service_backing[256+10]);return 28;}
    for(uint32_t q=0;q<outpages;q++)if(memcmp(expected_graph_y+q*PAGE,vtcm+(direct_yblocks[h[12]+q]&~127u),PAGE)){printf("FAIL actual graph caller output page=%u\n",q);return 29;}
    if(gg.pool[1]!=graph_pool||memcmp(graph_pool,vtcm,QD_DIRECT_ACTIVE_BYTES)){puts("FAIL actual graph caller pool pointer/copyback mismatch");return 30;}
    for(uint32_t i=0;i<QD_DIRECT_ACTIVE_BYTES;i++)if(!y_mask[i]&&vtcm[i]!=vtcm_snapshot[i]){printf("FAIL actual graph caller non-Y byte changed %x\n",i);return 33;}
#endif
    FILE*out=fopen(QD_GRAPH_OUTPUT_PATH,"wb");if(!out||fwrite(h,sizeof h,1,out)!=1){puts("FAIL output header");return 24;}
    for(uint32_t q=0;q<outpages;q++)if(fwrite((const void*)(uintptr_t)b.out_table[q],PAGE,1,out)!=1){puts("FAIL output gather");return 25;}
    fclose(out);
#ifdef QD_GRAPH_DIRECT_CALLER
    {uint32_t proof[5]={0x33444347u,247u,caller_abi_mismatches,caller_table_mismatches,0u};
#ifdef QD_GRAPH_STACKCOPY
     proof[4]=1u;
#endif
#if QDS_DIRECT_COMPACT_INPUT
     proof[4]=1u+QDS_DIRECT_COMPACT_INPUT;
#endif
     FILE *pf=fopen(QD_GRAPH_MARKER_PATH,"wb");if(!pf||fwrite(proof,sizeof proof,1,pf)!=1){puts("FAIL graph caller proof marker");return 34;}fclose(pf);}
#endif
    printf("PASS real QDRX scatter S1 3x3 247: %u input pages, %u unique touched physical pages, %u repeated X aliases, %u nonmonotone transitions; L2 table ABI 0x30000000/0x30000800; output pages %u\n",inpages,np,alias_count,nonmonotonic,outpages);
    return 0;
}
