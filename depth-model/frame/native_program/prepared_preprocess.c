#include "prepared_preprocess.h"
#if defined(QDS_PREPARED_QUANTIZE) && QDS_PREPARED_QUANTIZE
#include "prepared_quantize.h"
#endif
#include "../layout_audit/im2col_u8_crouton_direct.h"
#include <limits.h>
#include <string.h>

enum { PP_SHAPE=1, PP_CROUTON=64, PP_SHAPE_VALUE=8, PP_DIRECT=1, PP_BLOCKED=2 };
typedef struct {const uint8_t *src;uint8_t *dst;uint32_t bytes;} SlicePlan;
typedef struct {const uint8_t *src;uint8_t *dst;} ActivationPlan;
typedef struct {const Nprg2ExecCall *call;const uint8_t *src;uint32_t src_bytes;const uint32_t *blocks;uint32_t nblocks,slice;uint8_t *pools[7];size_t sizes[7];int32_t in[4],out[4],max[4],kernel[4],stride[4];} Im2colPlan;
static uint16_t rd16p(const uint8_t *p){return (uint16_t)p[0]|(uint16_t)p[1]<<8;}
static uint32_t rd32p(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static int text4p(const uint8_t *g,const char *s){return !memcmp(g+2,s,strlen(s));}
static int dims_shape4p(const uint8_t *g,const int32_t dims[4],const int32_t max[4]){
    if(!g||g[0]!=1u||!(g[1]&PP_SHAPE))return 0;
    for(unsigned i=0;i<4;i++)if((int16_t)rd16p(g+6+2*i)!=dims[i]||
       (int16_t)rd16p(g+14+2*i)!=max[i]||(int16_t)rd16p(g+22+2*i))return 0;
    return 1;
}
static int block_pagep(const Nprg2TensorView *v,uint32_t index,uint32_t alignment,uint32_t *pool,uint32_t *offset){
    if(!v||!v->blocks||index>=v->block_count||(alignment!=128u&&alignment!=2048u))return 0;
    uint32_t e=v->blocks[index],p=e&127u,o=e&~127u;
    if(!p||(o&(alignment-1u)))return 0;*pool=p;*offset=o;return 1;
}
static int make_activation_plan(const Nprg2ExecCall *c,ActivationPlan *p){
    static const int32_t dims[4]={1,1,1,32},max[4]={1,8,8,32};
    if(!c||!p||c->command.opcode!=QDR_COPY_ACTIVATION||c->command.variant!=17u||
       (c->command.sequence!=67u&&c->command.sequence!=58u)||c->command.input_count!=1||
       c->command.output_count!=1||c->command.vector_count||c->command.call_count||
       c->command.duplicate_pad||!c->pools[1].base||!c->pools[5].base)return 0;
    uint32_t xi=0,yi=0,sp=0,so=0,dp=0,doff=0;Nprg2TensorView x,y;
    if(nprg2_exec_input(c,0,&xi,&x)!=NPRG2_EXEC_OK||nprg2_exec_output(c,0,&yi,&y)!=NPRG2_EXEC_OK)return 0;
    (void)xi;(void)yi;
    if(x.storage_kind!=PP_BLOCKED||y.storage_kind!=PP_BLOCKED||
       !text4p(x.geometry_record,"cB")||!text4p(y.geometry_record,"CB")||
       !dims_shape4p(x.geometry_record,dims,max)||!dims_shape4p(y.geometry_record,dims,max)||
       x.block_count!=1u||x.slice_origin!=0u)return 0;
    if((c->command.sequence==67u&&(y.slice_origin!=52u||y.block_count!=72u))||
       (c->command.sequence==58u&&(y.slice_origin!=0u||y.block_count!=20u)))return 0;
    if(!block_pagep(&x,x.slice_origin,128u,&sp,&so)||sp!=5u||
       !block_pagep(&y,y.slice_origin,2048u,&dp,&doff)||dp!=1u)return 0;
    if(so>c->pools[sp].bytes||2048u>c->pools[sp].bytes-so||
       doff>c->pools[dp].bytes||2048u>c->pools[dp].bytes-doff||
       doff>c->pool1_writable_bytes||2048u>c->pool1_writable_bytes-doff)return 0;
    uintptr_t sb=(uintptr_t)c->pools[sp].base,db=(uintptr_t)c->pools[dp].base;
    if(so>UINTPTR_MAX-sb||doff>UINTPTR_MAX-db)return 0;
    uintptr_t s=sb+so,d=db+doff;
    if(2048u>UINTPTR_MAX-s||2048u>UINTPTR_MAX-d||(s<d+2048u&&d<s+2048u))return 0;
    p->src=(const uint8_t *)s;p->dst=(uint8_t *)d;return 1;
}
static int activation_validate(const Nprg2ExecCall *c,void *ctx){(void)ctx;ActivationPlan p;return make_activation_plan(c,&p);}
static int activation_run(const Nprg2ExecCall *c,void *ctx){(void)ctx;ActivationPlan p;if(!make_activation_plan(c,&p))return 0;memmove(p.dst,p.src,2048u);return 1;}
static int shape_value4(const Nprg2TensorView *v,int32_t d[4]){
    if(!v||v->storage_kind||!v->geometry_record||v->geometry_record[0]||v->geometry_record[1]!=PP_SHAPE_VALUE||!text4p(v->geometry_record,"s4"))return 0;
    for(unsigned i=0;i<4;i++)d[i]=(int32_t)rd32p(v->geometry_record+6+i*4u);return 1;
}
static int make_slice_plan(const Nprg2ExecCall *c,SlicePlan *p){
    if(!c||!p||c->command.opcode!=QDR_INPUT_SLICE||c->command.variant!=1||c->command.io_index!=3||
       c->command.input_count!=6||c->command.output_count!=1||c->command.vector_count||c->command.call_count||
       c->command.duplicate_pad||!c->inputs||c->input_count<=3||!c->inputs[3].data)return 0;
    static const int32_t shape[6][4]={{0,0,0,0},{0,0,0,0},{1,2,4,4},{0,0,4,3},{1,2,4,4},{1,2,4,4}};
    for(uint32_t i=0;i<6;i++){uint32_t id;Nprg2TensorView v;int32_t got[4];if(nprg2_exec_input(c,i,&id,&v)!=NPRG2_EXEC_OK||!shape_value4(&v,got))return 0;for(unsigned j=0;j<4;j++)if(got[j]!=shape[i][j])return 0;}
    uint32_t id;Nprg2TensorView o;if(nprg2_exec_output(c,0,&id,&o)!=NPRG2_EXEC_OK)return 0;(void)id;
    const uint8_t *g=o.geometry_record;static const int32_t dims[4]={1,2,4,4};
    if(o.storage_kind!=PP_DIRECT||o.pool!=1||!c->pools[1].base||!g||g[0]!=6||g[1]!=PP_SHAPE||!text4p(g,"Ff"))return 0;
    for(unsigned i=0;i<4;i++)if((int16_t)rd16p(g+6+2*i)!=dims[i]||(int16_t)rd16p(g+14+2*i)!=dims[i]||(int16_t)rd16p(g+22+2*i))return 0;
    if(o.offset>c->pool1_writable_bytes||128u>c->pool1_writable_bytes-o.offset||o.offset>c->pools[1].bytes||128u>c->pools[1].bytes-o.offset||c->inputs[3].bytes!=128u)return 0;
    uintptr_t s=(uintptr_t)c->inputs[3].data,d=(uintptr_t)c->pools[1].base+o.offset;if(128u>UINTPTR_MAX-s||128u>UINTPTR_MAX-d||(s<d+128u&&d<s+128u))return 0;
    const uint8_t *src_bytes=(const uint8_t *)c->inputs[3].data;
    for(uint32_t i=0;i<32u;i++){uint32_t w=rd32p(src_bytes+i*4u);if((w&0x7f800000u)==0x7f800000u)return 0;}
    p->src=c->inputs[3].data;p->dst=(uint8_t *)d;p->bytes=128;return 1;
}
static int slice_validate(const Nprg2ExecCall *c,void *ctx){(void)ctx;SlicePlan p;return make_slice_plan(c,&p);}
static int slice_run(const Nprg2ExecCall *c,void *ctx){(void)ctx;SlicePlan p;if(!make_slice_plan(c,&p))return 0;memcpy(p.dst,p.src,p.bytes);return 1;}
static int make_im2col_plan(const Nprg2ExecCall *c,Im2colPlan *p){
    if(!c||!p||c->command.opcode!=QDR_IM2COL||c->command.variant!=24||c->command.input_count!=3||
       c->command.output_count!=1||c->command.vector_count||c->command.call_count||!c->pools[1].base)return 0;
    uint32_t ix;Nprg2TensorView x,k,s,o;int32_t kd[4],sd[4];
    if(nprg2_exec_input(c,0,&ix,&x)!=NPRG2_EXEC_OK||nprg2_exec_input(c,1,&ix,&k)!=NPRG2_EXEC_OK||
       nprg2_exec_input(c,2,&ix,&s)!=NPRG2_EXEC_OK||nprg2_exec_output(c,0,&ix,&o)!=NPRG2_EXEC_OK)return 0;
    const uint8_t *xg=x.geometry_record,*og=o.geometry_record;static const int32_t in[4]={1,10,258,1},out[4]={1,8,256,9},max[4]={1,8,256,32},kernel[4]={3,3,1,32},stride[4]={1,1,1,1};
    if(x.storage_kind!=PP_DIRECT||x.pool!=1||xg[0]!=1||xg[1]!=(PP_SHAPE|32u)||!text4p(xg,"FB")||
       o.storage_kind!=PP_BLOCKED||!o.blocks||!(og[1]&PP_CROUTON)||og[1]!=(PP_SHAPE|PP_CROUTON|32u)||og[0]!=1||!text4p(og,"CB")||
       !shape_value4(&k,kd)||!shape_value4(&s,sd))return 0;
    for(unsigned i=0;i<4;i++)if((int16_t)rd16p(xg+6+2*i)!=in[i]||(int16_t)rd16p(xg+14+2*i)!=in[i]||(int16_t)rd16p(xg+22+2*i)||
       (int16_t)rd16p(og+6+2*i)!=out[i]||(int16_t)rd16p(og+14+2*i)!=max[i]||(int16_t)rd16p(og+22+2*i)||kd[i]!=kernel[i]||sd[i]!=stride[i])return 0;
    if(x.offset>c->pools[1].bytes||2580u>c->pools[1].bytes-x.offset||x.offset>c->pool1_writable_bytes||2580u>c->pool1_writable_bytes-x.offset||
       o.slice_origin>o.block_count||o.block_count-o.slice_origin<32)return 0;
    memset(p,0,sizeof *p);p->call=c;p->src=c->pools[1].base+x.offset;p->src_bytes=2580;p->blocks=o.blocks;p->nblocks=o.block_count;p->slice=o.slice_origin;
    for(unsigned i=0;i<7;i++){p->pools[i]=c->pools[i].base;p->sizes[i]=c->pools[i].bytes;}
    memcpy(p->in,in,sizeof in);memcpy(p->out,out,sizeof out);memcpy(p->max,max,sizeof max);memcpy(p->kernel,kernel,sizeof kernel);memcpy(p->stride,stride,sizeof stride);
    uintptr_t src=(uintptr_t)p->src;if(p->src_bytes>UINTPTR_MAX-src)return 0;
    uintptr_t addresses[32];
    for(uint32_t i=0;i<32;i++){
        uint32_t e=p->blocks[p->slice+i],pool=e&127u,off=e&~127u;if(!pool||pool>=7||!p->pools[pool]||(off&2047u)||off>p->sizes[pool]||2048u>p->sizes[pool]-off)return 0;
        uint32_t limit=pool==1?c->pool1_writable_bytes:(uint32_t)p->sizes[pool];if(off>limit||2048u>limit-off)return 0;
        uintptr_t a=(uintptr_t)p->pools[pool];if(off>UINTPTR_MAX-a)return 0;a+=off;if(2048u>UINTPTR_MAX-a)return 0;
        if(src<a+2048u&&a<src+p->src_bytes)return 0;for(uint32_t j=0;j<i;j++)if(a<addresses[j]+2048u&&addresses[j]<a+2048u)return 0;addresses[i]=a;
    }
    return 1;
}
static uint8_t *im2col_block(uint32_t encoded,int store,void *ctx){(void)store;const Im2colPlan *p=ctx;uint32_t pool=encoded&127u,off=encoded&~127u;if(pool>=7||!p->pools[pool]||off>p->sizes[pool]||2048u>p->sizes[pool]-off)return NULL;return p->pools[pool]+off;}
static int im2col_validate(const Nprg2ExecCall *c,void *ctx){(void)ctx;Im2colPlan p;return make_im2col_plan(c,&p);}
static int im2col_run(const Nprg2ExecCall *c,void *ctx){(void)ctx;Im2colPlan p;if(!make_im2col_plan(c,&p))return 0;return qd_layout_im2col_u8_crouton_direct(p.src,p.src_bytes,p.in,p.out,p.max,p.kernel,p.stride,p.blocks,p.nblocks,p.slice,p.pools,p.sizes,im2col_block,&p);}
int nprg2_executor_register_preprocess(Nprg2Executor *e){
    if(!e)return NPRG2_EXEC_ARGUMENT;int r=nprg2_executor_register(e,QDR_INPUT_SLICE,1,slice_validate,slice_run,NULL);if(r)return r;
    r=nprg2_executor_register(e,QDR_COPY_ACTIVATION,17,activation_validate,activation_run,NULL);if(r)return r;
    r=nprg2_executor_register(e,QDR_IM2COL,24,im2col_validate,im2col_run,NULL);if(r)return r;
#if defined(QDS_PREPARED_QUANTIZE) && QDS_PREPARED_QUANTIZE
    return nprg2_executor_register_quantize(e);
#else
    return NPRG2_EXEC_OK;
#endif
}
