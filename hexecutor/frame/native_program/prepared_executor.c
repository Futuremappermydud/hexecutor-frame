#include "prepared_executor.h"
#include "prepared_vcopy.h"
#include "prepared_cost_volume_approx.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

static uint16_t rd16e(const uint8_t *p){return (uint16_t)p[0]|(uint16_t)p[1]<<8;}

void nprg2_executor_init(Nprg2Executor *e){if(e)memset(e,0,sizeof *e);}

static uint32_t handler_hash(uint8_t op,uint8_t variant){return ((uint32_t)op*37u+(uint32_t)variant*17u)%NPRG2_EXEC_REGISTRY_CAPACITY;}
static int handler_index(const Nprg2Executor *e,uint8_t op,uint8_t variant){
    uint32_t at=handler_hash(op,variant);
    for(uint32_t n=0;n<NPRG2_EXEC_REGISTRY_CAPACITY;n++,at=(at+1u)%NPRG2_EXEC_REGISTRY_CAPACITY){
        const Nprg2ExecHandler *h=&e->handlers[at];if(!h->used)return -1;if(h->opcode==op&&h->variant==variant)return (int)at;
    }
    return -1;
}

static int approx_cv_skip(const Nprg2ExecRequest *r,uint16_t sequence){
#if NPRG2_APPROX_COST_VOLUME_STEP == 2 || NPRG2_APPROX_COST_VOLUME_STEP == 4 || NPRG2_APPROX_COST_VOLUME_STEP == 8
    if(r && nprg2_approx_cv_primary_program(r->program)){
        return nprg2_approx_cv_primary_skip_seq(sequence);
    }
#else
    (void)r;(void)sequence;
#endif
    return 0;
}
static int approx_cv_concat_target(const Nprg2ExecRequest *r,const Nprg2Command *command){
#if NPRG2_APPROX_COST_VOLUME_STEP == 2 || NPRG2_APPROX_COST_VOLUME_STEP == 4 || NPRG2_APPROX_COST_VOLUME_STEP == 8
    if(!r || !command || command->opcode!=QDR_CONCAT || command->output_count!=1 ||
       !nprg2_approx_cv_primary_program(r->program))return 0;
    Nprg2ExecCall call={0};Nprg2TensorView view;uint32_t tensor_index;
    call.program=r->program;call.bound=r->bound;call.command=*command;
    if(nprg2_exec_output(&call,0,&tensor_index,&view)!=NPRG2_EXEC_OK || !view.geometry_record)return 0;
    const uint8_t *g=view.geometry_record;
    return g[0]==1u && (int16_t)(g[6]|(uint16_t)g[7]<<8)==1 &&
           (int16_t)(g[8]|(uint16_t)g[9]<<8)==80 &&
           (int16_t)(g[10]|(uint16_t)g[11]<<8)==64 &&
           (int16_t)(g[12]|(uint16_t)g[13]<<8)==64;
#else
    (void)r;(void)command;return 0;
#endif
}

int nprg2_executor_register(Nprg2Executor *e,uint8_t op,uint8_t variant,
                            Nprg2ExecValidate validate,Nprg2ExecRun run,void *ctx){
    return nprg2_executor_register_scratch(e,op,variant,validate,run,ctx,0,0);
}
int nprg2_executor_register_scratch(Nprg2Executor *e,uint8_t op,uint8_t variant,
                                    Nprg2ExecValidate validate,Nprg2ExecRun run,void *ctx,
                                    uint32_t scratch_bytes,uint32_t worker_scratch_bytes){
    return nprg2_executor_register_scratch_region(e,op,variant,validate,run,ctx,scratch_bytes,worker_scratch_bytes,
                                                  NPRG2_EXEC_SCRATCH_DEFAULT);
}
int nprg2_executor_register_scratch_region(Nprg2Executor *e,uint8_t op,uint8_t variant,
                                           Nprg2ExecValidate validate,Nprg2ExecRun run,void *ctx,
                                           uint32_t scratch_bytes,uint32_t worker_scratch_bytes,
                                           uint8_t region){
    if(!e||!op||op>QDR_OPCODES||!variant||variant>QDR_VARIANTS||!validate||!run||
       region>=NPRG2_EXEC_SCRATCH_REGIONS)return NPRG2_EXEC_ARGUMENT;
    if(e->count>=NPRG2_EXEC_REGISTRY_CAPACITY)return NPRG2_EXEC_REGISTRY_FULL;
    uint32_t at=handler_hash(op,variant);
    for(uint32_t n=0;n<NPRG2_EXEC_REGISTRY_CAPACITY;n++,at=(at+1u)%NPRG2_EXEC_REGISTRY_CAPACITY){
        Nprg2ExecHandler *h=&e->handlers[at];if(h->used){if(h->opcode==op&&h->variant==variant)return NPRG2_EXEC_DUPLICATE;continue;}
        h->opcode=op;h->variant=variant;h->used=1;h->validate=validate;h->run=run;h->ctx=ctx;
        h->scratch_bytes=scratch_bytes;h->concurrent_worker_scratch_bytes=worker_scratch_bytes;h->scratch_region=region;e->count++;return NPRG2_EXEC_OK;
    }
    return NPRG2_EXEC_REGISTRY_FULL;
}

int nprg2_executor_handler_slot(const Nprg2Executor *e,uint8_t op,uint8_t variant){return e?handler_index(e,op,variant):-1;}
int nprg2_executor_set_fuse(Nprg2Executor *e,uint8_t op,uint8_t variant,Nprg2ExecFuse fuse){
    if(!e)return NPRG2_EXEC_ARGUMENT;int hi=handler_index(e,op,variant);if(hi<0)return NPRG2_EXEC_UNSUPPORTED;
    e->handlers[hi].fuse=fuse;return NPRG2_EXEC_OK;
}
int nprg2_executor_mark_no_effect(Nprg2Executor *e,uint8_t op,uint8_t variant){
    if(!e)return NPRG2_EXEC_ARGUMENT;int hi=handler_index(e,op,variant);if(hi<0)return NPRG2_EXEC_UNSUPPORTED;
    e->handlers[hi].no_effect=1;return NPRG2_EXEC_OK;
}

static int get_ref(const Nprg2ExecCall *c,uint32_t slot,int output,uint32_t *index,Nprg2TensorView *view){
    if(!c||!c->program||!c->bound||!index||!view)return NPRG2_EXEC_ARGUMENT;
    uint32_t count=output?c->command.output_count:c->command.input_count;
    uint32_t first=output?c->command.output_first:c->command.input_first;
    if(slot>=count||first>c->bound->refs.bytes/2u||count>c->bound->refs.bytes/2u-first)return NPRG2_EXEC_ARGUMENT;
    *index=rd16e(c->bound->refs.data+(size_t)(first+slot)*2u);
    return nprg2_bound_tensor_at(c->program,c->bound,*index,view)==NPRG2_OK?NPRG2_EXEC_OK:NPRG2_EXEC_REJECTED;
}
int nprg2_exec_input(const Nprg2ExecCall *c,uint32_t slot,uint32_t *i,Nprg2TensorView *v){return get_ref(c,slot,0,i,v);}
int nprg2_exec_output(const Nprg2ExecCall *c,uint32_t slot,uint32_t *i,Nprg2TensorView *v){return get_ref(c,slot,1,i,v);}

static int pools_match(const Nprg2Program *p,const Nprg2Pool pools[7]){
    for(unsigned i=0;i<7;i++)if(pools[i].base!=p->session_pools[i].base||pools[i].bytes!=p->session_pools[i].bytes)return 0;
    return 1;
}
static int overlaps(const void *a,size_t an,const void *b,size_t bn){
    if(!an||!bn)return 0;if(!a||!b)return 1;uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(x>UINTPTR_MAX-an||y>UINTPTR_MAX-bn)return 1;return x<y+bn&&y<x+an;
}
/* Fills every field but call->command, which the caller has already decoded
 * in place (no memset or struct copy per command). */
static void fill_call(const Nprg2ExecRequest *r,uint32_t index,Nprg2ExecCall *call){
    call->command_index=index;call->program=r->program;call->bound=r->bound;
    call->pools=r->pools;call->pool1_writable_bytes=r->pool1_writable_bytes;
    call->inputs=r->inputs;call->input_count=r->input_count;call->outputs=r->outputs;call->output_count=r->output_count;
    call->user=r->user;call->weights_async=r->weights_async;call->scratch=r->scratch;call->scratch_bytes=r->scratch_bytes;
    call->scratch_region[0]=(Nprg2ExecBuffer){(uint8_t *)r->scratch,r->scratch_bytes};
    for(uint32_t k=1;k<NPRG2_EXEC_SCRATCH_REGIONS;k++)call->scratch_region[k]=r->extra_scratch[k-1u];
    call->preflighted=0;call->execution_lane=r->execution_lane;
    call->parallel=r->parallel;call->epilogue=NULL;call->epilogue_dead=0;
}
/* The plan pair this command is the conv of, for a request whose interval
 * contains both commands; NULL otherwise. */
static Nprg2FusePair *fuse_pair_of_conv(const Nprg2ExecRequest *r,uint32_t ix){
    const Nprg2FusePlan *p=r->fuse;if(!p||!p->role||ix>=p->commands)return NULL;
    uint16_t role=p->role[ix];if(!role||(role&NPRG2_FUSE_EPI))return NULL;
    Nprg2FusePair *pr=&p->pairs[role-1u];
    if(pr->epi<r->command_first||pr->epi-r->command_first>=r->command_count)return NULL;
    return pr;
}
static Nprg2FusePair *fuse_pair_of_epi(const Nprg2ExecRequest *r,uint32_t ix){
    const Nprg2FusePlan *p=r->fuse;if(!p||!p->role||ix>=p->commands)return NULL;
    uint16_t role=p->role[ix];if(!(role&NPRG2_FUSE_EPI))return NULL;
    Nprg2FusePair *pr=&p->pairs[(role&~NPRG2_FUSE_EPI)-1u];
    if(pr->conv<r->command_first||pr->conv-r->command_first>=r->command_count)return NULL;
    return pr;
}
/* Hands the handler the region it registered for. */
static void select_scratch(Nprg2ExecCall *call,const Nprg2ExecHandler *h){
    const Nprg2ExecBuffer *b=&call->scratch_region[h->scratch_region<NPRG2_EXEC_SCRATCH_REGIONS?h->scratch_region:0u];
    call->scratch=b->bytes?b->data:NULL;call->scratch_bytes=b->bytes;
}
/* Counters, effect bitmap and strip data are reset by a full preflight; the
 * run list (last member) is rewritten up to run_count instead of cleared. */
static void stats_reset(Nprg2ExecStats *s){memset(s,0,offsetof(Nprg2ExecStats,runs));s->failed_command=UINT32_MAX;}
/* Stand-in for a request without a stats object: its contents are never read
 * back, so it need not be reentrant (and the 48 KiB run list stays off the stack). */
static Nprg2ExecStats nostats;
static int no_graph_alias(const Nprg2Executor *e,const Nprg2ExecRequest *r,const void *ptr,size_t bytes){
    if(!bytes)return 1;
    for(unsigned i=0;i<7;i++)if(overlaps(ptr,bytes,r->pools[i].base,r->pools[i].bytes))return 0;
    const Nprg2Section *sections[6]={&r->bound->geometry,&r->bound->tensor,&r->bound->command,&r->bound->refs,&r->bound->vectors,&r->bound->calls};
    for(unsigned i=0;i<6;i++)if(overlaps(ptr,bytes,sections[i]->data,sections[i]->bytes))return 0;
    const Nprg2Program *p=r->program;
    if(overlaps(ptr,bytes,p->raw,p->raw_bytes)||overlaps(ptr,bytes,p->tensors,(size_t)p->tensor_count*sizeof *p->tensors)||
       overlaps(ptr,bytes,p->expanded_blocks,(size_t)p->expanded_block_count*sizeof *p->expanded_blocks))return 0;
    if(ptr!=(const void*)e&&overlaps(ptr,bytes,e,sizeof *e))return 0;
    return 1;
}
static int request_valid(const Nprg2Executor *e,const Nprg2ExecRequest *r){
    if(!r||!r->program||!r->bound||!r->bound->ready)return 0;
    if(r->weights_async){
        const Nprg2WeightsAsync *a=r->weights_async;
        if(a->command_count!=r->program->command_count||!a->timeout_cycles)return 0;
        if(a->ring_mode){if(!a->ring||a->first_hazard)return 0;}
        else if(!a->first_hazard||!a->start||!a->drain)return 0;
    }
    if(r->command_first>r->program->command_count||r->command_count>r->program->command_count-r->command_first)return 0;
    if(!pools_match(r->program,r->pools))return 0;
    if(!r->pool1_writable_bytes||r->pool1_writable_bytes>r->pools[1].bytes)return 0;
    if(r->scratch_bytes&&!r->scratch)return 0;
    if((r->input_count&&!r->inputs)||(r->output_count&&!r->outputs))return 0;
    for(uint32_t i=0;i<r->input_count;i++)if(r->inputs[i].bytes&&!r->inputs[i].data)return 0;
    for(uint32_t i=0;i<r->output_count;i++)if(r->outputs[i].bytes&&!r->outputs[i].data)return 0;
    for(unsigned i=0;i<7;i++)if(r->pools[i].bytes&&!r->pools[i].base)return 0;
    /* Control state, I/O and scratch stay disjoint from graph memory so a
     * kernel destination cannot corrupt the interpreter's own state. */
    if(!no_graph_alias(e,r,e,sizeof *e)||!no_graph_alias(e,r,r->scratch,r->scratch_bytes)||
       !no_graph_alias(e,r,r->stats,r->stats?sizeof *r->stats:0)||
       !no_graph_alias(e,r,r->inputs,(size_t)r->input_count*sizeof *r->inputs)||
       !no_graph_alias(e,r,r->outputs,(size_t)r->output_count*sizeof *r->outputs))return 0;
    for(uint32_t k=0;k+1u<NPRG2_EXEC_SCRATCH_REGIONS;k++){
        const Nprg2ExecBuffer *x=&r->extra_scratch[k];if(!x->bytes)continue;
        if(!x->data||!no_graph_alias(e,r,x->data,x->bytes)||overlaps(x->data,x->bytes,r->scratch,r->scratch_bytes)||
           overlaps(x->data,x->bytes,r->stats,r->stats?sizeof *r->stats:0)||
           overlaps(x->data,x->bytes,r->inputs,(size_t)r->input_count*sizeof *r->inputs)||
           overlaps(x->data,x->bytes,r->outputs,(size_t)r->output_count*sizeof *r->outputs))return 0;
        for(uint32_t j=0;j<r->input_count;j++)if(overlaps(x->data,x->bytes,r->inputs[j].data,r->inputs[j].bytes))return 0;
        for(uint32_t j=0;j<r->output_count;j++)if(overlaps(x->data,x->bytes,r->outputs[j].data,r->outputs[j].bytes))return 0;
        for(uint32_t j=0;j<k;j++)if(overlaps(x->data,x->bytes,r->extra_scratch[j].data,r->extra_scratch[j].bytes))return 0;
    }
    for(uint32_t j=0;j<r->input_count;j++)if(!no_graph_alias(e,r,r->inputs[j].data,r->inputs[j].bytes))return 0;
    for(uint32_t j=0;j<r->output_count;j++)if(!no_graph_alias(e,r,r->outputs[j].data,r->outputs[j].bytes))return 0;
    if(overlaps(r->scratch,r->scratch_bytes,r->stats,r->stats?sizeof *r->stats:0)||
       overlaps(r->scratch,r->scratch_bytes,r->inputs,(size_t)r->input_count*sizeof *r->inputs)||
       overlaps(r->scratch,r->scratch_bytes,r->outputs,(size_t)r->output_count*sizeof *r->outputs)||
       overlaps(r->stats,r->stats?sizeof *r->stats:0,r->inputs,(size_t)r->input_count*sizeof *r->inputs)||
       overlaps(r->stats,r->stats?sizeof *r->stats:0,r->outputs,(size_t)r->output_count*sizeof *r->outputs))return 0;
    return 1;
}
/* Identifies everything a preflight result depends on that the caller can
 * change between calls (a multiplicative hash over pointers, sizes and the registry; it guards
 * against accidental changes, not an adversary). */
static uint64_t sig_mix(uint64_t h,uint64_t v){h=(h^v)*0x9e3779b97f4a7c15ull;return h^(h>>29);}
static uint64_t request_signature(const Nprg2Executor *e,const Nprg2ExecRequest *r){
    uint64_t h=0x243f6a8885a308d3ull;
    h=sig_mix(h,(uintptr_t)e);h=sig_mix(h,(uintptr_t)r->program);h=sig_mix(h,(uintptr_t)r->bound);
    {
        const Nprg2Program *p=r->program;const Nprg2Bound *b=r->bound;
        h=sig_mix(h,(uintptr_t)p->raw);h=sig_mix(h,p->raw_bytes);h=sig_mix(h,p->command_count);
        h=sig_mix(h,(uintptr_t)p->tensors);h=sig_mix(h,p->tensor_count);
        h=sig_mix(h,(uintptr_t)p->expanded_blocks);h=sig_mix(h,p->expanded_block_count);
        const Nprg2Section *sec[6]={&b->geometry,&b->tensor,&b->command,&b->refs,&b->vectors,&b->calls};
        for(unsigned i=0;i<6;i++){h=sig_mix(h,(uintptr_t)sec[i]->data);h=sig_mix(h,sec[i]->bytes);}
        h=sig_mix(h,b->hot_bytes);
    }
    for(unsigned i=0;i<7;i++){h=sig_mix(h,(uintptr_t)r->pools[i].base);h=sig_mix(h,r->pools[i].bytes);}
    h=sig_mix(h,r->pool1_writable_bytes);h=sig_mix(h,(uintptr_t)r->user);h=sig_mix(h,r->execution_lane);
    if(r->weights_async){const Nprg2WeightsAsync *a=r->weights_async;
        h=sig_mix(h,(uintptr_t)a);h=sig_mix(h,(uintptr_t)a->first_hazard);h=sig_mix(h,a->command_count);
        h=sig_mix(h,(uintptr_t)a->start);h=sig_mix(h,(uintptr_t)a->drain);h=sig_mix(h,(uintptr_t)a->ring);h=sig_mix(h,a->ring_mode);
        h=sig_mix(h,(uintptr_t)a->ctx);h=sig_mix(h,a->timeout_cycles);
    }else h=sig_mix(h,0);
    h=sig_mix(h,(uintptr_t)r->scratch);h=sig_mix(h,r->scratch_bytes);
    for(unsigned k=0;k+1u<NPRG2_EXEC_SCRATCH_REGIONS;k++){h=sig_mix(h,(uintptr_t)r->extra_scratch[k].data);h=sig_mix(h,r->extra_scratch[k].bytes);}
    h=sig_mix(h,(uintptr_t)r->inputs);h=sig_mix(h,r->input_count);h=sig_mix(h,(uintptr_t)r->outputs);h=sig_mix(h,r->output_count);
    for(uint32_t i=0;i<r->input_count&&r->inputs;i++){h=sig_mix(h,(uintptr_t)r->inputs[i].data);h=sig_mix(h,r->inputs[i].bytes);}
    for(uint32_t i=0;i<r->output_count&&r->outputs;i++){h=sig_mix(h,(uintptr_t)r->outputs[i].data);h=sig_mix(h,r->outputs[i].bytes);}
    h=sig_mix(h,r->command_first);h=sig_mix(h,r->command_count);h=sig_mix(h,e->count);h=sig_mix(h,(uintptr_t)r->fuse);h=sig_mix(h,(uintptr_t)r->fuse);
    for(uint32_t i=0;i<NPRG2_EXEC_REGISTRY_CAPACITY;i++){
        const Nprg2ExecHandler *x=&e->handlers[i];if(!x->used)continue;
        h=sig_mix(h,(((uint64_t)x->opcode<<8|x->variant)<<8|x->scratch_region)<<1|x->no_effect);
        h=sig_mix(h,(uintptr_t)x->validate);h=sig_mix(h,(uintptr_t)x->fuse);h=sig_mix(h,(uintptr_t)x->ctx);
        h=sig_mix(h,((uint64_t)x->scratch_bytes<<32)|x->concurrent_worker_scratch_bytes);
    }
    return h|1u;
}
int nprg2_exec_weights_async_try(const Nprg2ExecCall *c,void *dst,const void *src,size_t bytes){
    if(!c||!c->weights_async||!c->weights_async->ring_mode||!c->weights_async->ring)return 0;
    /* The platform descriptor ring has one owner: the master/list-zero lane. */
    if(c->execution_lane!=0u)return 0;
    Nprg2WeightsAsync *a=c->weights_async;
    int queued=nprg2_dma_ring_enqueue(a->ring,dst,src,bytes);
    if(queued>0){a->starts++;return 1;}
    if(queued<0){a->failures++;return -1;}
    a->cpu_fallbacks++;return 0;
}
int nprg2_exec_weights_dma_checkpoint_set(Nprg2WeightsAsync *a,uint32_t event_id){
    if(!a||!a->ring_mode||!a->ring)return NPRG2_EXEC_ARGUMENT;
    return nprg2_dma_ring_checkpoint_set(a->ring,event_id)?NPRG2_EXEC_OK:NPRG2_EXEC_HANDLER_ERROR;
}
int nprg2_exec_weights_dma_checkpoint_wait(Nprg2WeightsAsync *a,uint32_t event_id){
    if(!a||!a->ring_mode||!a->ring)return NPRG2_EXEC_ARGUMENT;
    return nprg2_dma_ring_checkpoint_wait(a->ring,event_id)?NPRG2_EXEC_OK:NPRG2_EXEC_HANDLER_ERROR;
}
int nprg2_exec_weights_dma_drain_all(Nprg2WeightsAsync *a){
    if(!a||!a->ring_mode||!a->ring)return NPRG2_EXEC_ARGUMENT;
    int d=nprg2_dma_ring_drain_all(a->ring);
    if(!d){a->failures++;return NPRG2_EXEC_HANDLER_ERROR;}
    a->drains++;
    if(d<0){a->failures++;return NPRG2_EXEC_HANDLER_ERROR;}
    return NPRG2_EXEC_OK;
}
/* Digest of the contents every validator reads through the program and the
 * bound tables (geometry, tensor, command, refs, vectors, calls, and the
 * expanded block tables): lane sums over whole 128-byte chunks, then four
 * multiply-xor lanes over the tail; like the signature it guards against
 * accidental change. Leading whole 128-byte chunks: per 32-bit lane l, Fletcher sums
 * s1[l]+=w, s2[l]+=s1[l] (mod 2^32), so a change anywhere or a reordering of
 * chunks changes them. Scalar loads of the DDR-resident expanded block tables
 * are latency-bound on the Frame (~190 cycles per uncached line, 1.9-2.3 ms per
 * run for ~716 KB); HVX keeps eight loads in flight. Host builds compute the
 * same sums in C. Returns the bytes consumed (a multiple of 128). */
static size_t lane_sums(const uint8_t *b,size_t n,uint32_t *s1,uint32_t *s2){
    const size_t nv=n/128u;
#if defined(__HVX__) && __HVX_LENGTH__==128
    const HVX_UVector *v=(const HVX_UVector *)(const void *)b;
    HVX_Vector a=Q6_V_vzero(),c=Q6_V_vzero();size_t k=0;
    for(;k+8<=nv;k+=8){
        HVX_Vector x0=v[k],x1=v[k+1],x2=v[k+2],x3=v[k+3],x4=v[k+4],x5=v[k+5],x6=v[k+6],x7=v[k+7];
        a=Q6_Vw_vadd_VwVw(a,x0);c=Q6_Vw_vadd_VwVw(c,a);a=Q6_Vw_vadd_VwVw(a,x1);c=Q6_Vw_vadd_VwVw(c,a);
        a=Q6_Vw_vadd_VwVw(a,x2);c=Q6_Vw_vadd_VwVw(c,a);a=Q6_Vw_vadd_VwVw(a,x3);c=Q6_Vw_vadd_VwVw(c,a);
        a=Q6_Vw_vadd_VwVw(a,x4);c=Q6_Vw_vadd_VwVw(c,a);a=Q6_Vw_vadd_VwVw(a,x5);c=Q6_Vw_vadd_VwVw(c,a);
        a=Q6_Vw_vadd_VwVw(a,x6);c=Q6_Vw_vadd_VwVw(c,a);a=Q6_Vw_vadd_VwVw(a,x7);c=Q6_Vw_vadd_VwVw(c,a);
    }
    for(;k<nv;k++){a=Q6_Vw_vadd_VwVw(a,v[k]);c=Q6_Vw_vadd_VwVw(c,a);}
    *(HVX_UVector *)(void *)s1=a;*(HVX_UVector *)(void *)s2=c;
#else
    for(unsigned l=0;l<32u;l++)s1[l]=s2[l]=0;
    for(size_t k=0;k<nv;k++)for(unsigned l=0;l<32u;l++){
        uint32_t w;memcpy(&w,b+k*128u+l*4u,4);s1[l]+=w;s2[l]+=s1[l];
    }
#endif
    return nv*128u;
}
static uint64_t hash_bytes(uint64_t h,const void *p,size_t n){
    {
        uint32_t s1[32],s2[32];const size_t done=lane_sums(p,n,s1,s2);
        for(unsigned l=0;l<32u;l++)h=sig_mix(h,(uint64_t)s1[l]|(uint64_t)s2[l]<<32);
        p=(const uint8_t *)p+done;h=sig_mix(h,n);n-=done;
    }
    const uint8_t *b=p;uint32_t a0=0x811c9dc5u,a1=0x01000193u,a2=0x9e3779b1u,a3=0x85ebca6bu;size_t i=0;
    if(!((uintptr_t)b&3u)){ /* the tables are 4-byte aligned in practice: plain word loads */
        const uint32_t *w=(const uint32_t *)(const void *)b;
        for(;i+16<=n;i+=16,w+=4){
            a0=(a0^w[0])*0x9e3779b1u;a1=(a1^w[1])*0x85ebca6bu;a2=(a2^w[2])*0xc2b2ae35u;a3=(a3^w[3])*0x27d4eb2fu;
        }
    } else for(;i+16<=n;i+=16){
        uint32_t w0,w1,w2,w3;memcpy(&w0,b+i,4);memcpy(&w1,b+i+4,4);memcpy(&w2,b+i+8,4);memcpy(&w3,b+i+12,4);
        a0=(a0^w0)*0x9e3779b1u;a1=(a1^w1)*0x85ebca6bu;a2=(a2^w2)*0xc2b2ae35u;a3=(a3^w3)*0x27d4eb2fu;
    }
    for(;i<n;i++)a0=(a0^b[i])*0x01000193u;
    h=sig_mix(h,n);h=sig_mix(h,(uint64_t)a0|(uint64_t)a1<<32);return sig_mix(h,(uint64_t)a2|(uint64_t)a3<<32);
}
static uint64_t content_digest(const Nprg2ExecRequest *r){
    const Nprg2Bound *b=r->bound;const Nprg2Section *sec[6]={&b->geometry,&b->tensor,&b->command,&b->refs,&b->vectors,&b->calls};
    uint64_t h=0x13198a2e03707344ull;
    for(unsigned i=0;i<6;i++)h=hash_bytes(h,sec[i]->data,sec[i]->data?sec[i]->bytes:0);
    h=hash_bytes(h,r->program->expanded_blocks,r->program->expanded_blocks?(size_t)r->program->expanded_block_count*sizeof(uint32_t):0);
    return h|1u;
}
/* A stats object holding a successful preflight of this exact request (same
 * signature, same table contents, full interval validated). */
static int preflight_matches(const Nprg2Executor *e,const Nprg2ExecRequest *r,int fresh_ok){
    const Nprg2ExecStats *s=r->stats;
    if(!r->reuse_preflight||!s||!s->preflight_signature||s->validated_commands!=r->command_count||
       s->preflight_signature!=request_signature(e,r))return 0;
    /* digest 0: that preflight was made without reuse_preflight and recorded none; only the call that
     * follows it directly (digest_fresh) may take it. */
    return (fresh_ok&&s->digest_fresh)||(s->preflight_digest&&s->preflight_digest==content_digest(r));
}
int nprg2_executor_preflight(const Nprg2Executor *e,const Nprg2ExecRequest *r){
    if(!e||!request_valid(e,r))return NPRG2_EXEC_ARGUMENT;
    Nprg2ExecProfile *pr=r->profile&&r->profile->clock?r->profile:NULL;
    uint64_t p0=pr?pr->clock():0;
    if(preflight_matches(e,r,0)){
        /* Reused across calls: nothing to validate, but this call's counters restart. */
        Nprg2ExecStats *rs=r->stats;
        rs->executed_commands=0;rs->failed_command=UINT32_MAX;rs->failed_opcode=rs->failed_variant=0;rs->digest_fresh=1;
        if(pr)pr->preflight_cycles+=pr->clock()-p0;
        return NPRG2_EXEC_OK;
    }
    Nprg2ExecStats *s=r->stats?r->stats:&nostats;stats_reset(s);
    uint32_t nrun=0;
    /* Coverage, feature, bounds, alias, and scratch predicates are checked
     * without graph-pool writes. This public entry is also used before a
     * physical VTCM boundary copy by the DSP wrapper. */
    for(uint32_t n=0;n<r->command_count;n++){
        uint32_t ix=r->command_first+n;Nprg2ExecCall call;
        if(nprg2_command_at(r->bound,ix,&call.command)!=NPRG2_OK)return NPRG2_EXEC_REJECTED;
        fill_call(r,ix,&call);
        int hi=handler_index(e,call.command.opcode,call.command.variant);
        if(hi<0){s->failed_command=ix;s->failed_opcode=call.command.opcode;s->failed_variant=call.command.variant;return NPRG2_EXEC_UNSUPPORTED;}
        const Nprg2ExecHandler *h=&e->handlers[hi];select_scratch(&call,h);
        if(h->scratch_bytes>UINT32_MAX-h->concurrent_worker_scratch_bytes||
           h->scratch_bytes+h->concurrent_worker_scratch_bytes>call.scratch_bytes){
            s->failed_command=ix;s->failed_opcode=call.command.opcode;s->failed_variant=call.command.variant;return NPRG2_EXEC_REJECTED;
        }
        Nprg2ExecCall ecall;
        Nprg2FusePair *fp=fuse_pair_of_conv(r,ix);
        if(fp){
            fp->active=0;
            if(h->fuse&&nprg2_command_at(r->bound,fp->epi,&ecall.command)==NPRG2_OK){
                fill_call(r,fp->epi,&ecall);
                if(h->fuse(&call,&ecall,fp->dead,h->ctx)){fp->active=1;call.epilogue=&ecall;call.epilogue_dead=fp->dead;}
            }
        }
        uint64_t v0=pr?pr->clock():0;
        int valid=h->validate(&call,h->ctx);
        if(pr){uint64_t d=pr->clock()-v0;pr->validate_cycles+=d;if(pr->handler_validate_cycles)pr->handler_validate_cycles[hi]+=d;}
        if(!valid){s->failed_command=ix;s->failed_opcode=call.command.opcode;s->failed_variant=call.command.variant;return NPRG2_EXEC_REJECTED;}
        s->validated_commands++;
        if(ix<NPRG2_EXEC_STRIP_COMMANDS){
            if(h->no_effect)s->stripped_by_slot[hi]++;
            else{
                s->effect_bits[ix>>5]|=1u<<(ix&31u);
                Nprg2ExecRunRec *rec=&s->runs[nrun++];
                rec->ix=(uint16_t)ix;rec->hi=(uint8_t)hi;rec->fl=0;rec->epi=0;
                Nprg2FusePair *ep=fuse_pair_of_epi(r,ix);
                if(ep&&ep->active)rec->fl=NPRG2_RUN_SKIP;
                else if(fp&&fp->active){rec->fl=(uint8_t)(NPRG2_RUN_FUSED_CONV|(fp->dead?NPRG2_RUN_DEAD:0));rec->epi=(uint16_t)fp->epi;}
            }
        }
    }
    if(r->stats){
        s->preflight_signature=request_signature(e,r);s->digest_fresh=1;
        /* Hashing the tables costs about a millisecond on the Frame: only for a request that asks to be reused. */
        if(r->reuse_preflight)s->preflight_digest=content_digest(r);
        s->strip_valid=r->command_first+r->command_count<=NPRG2_EXEC_STRIP_COMMANDS;
        s->runs_valid=s->strip_valid;s->run_count=nrun;
    }
    if(pr){pr->preflight_cycles+=pr->clock()-p0;pr->preflight_calls++;}
    return NPRG2_EXEC_OK;
}
/* Run loop over the preflight's compact run list: the registry slot, the
 * fusion role and the epilogue index are in the record, and one call object
 * (all request-derived fields) is built once and only the per-command fields
 * are rewritten. Order, counters, cancellation points and profile are those
 * of the generic loop below. */
static int run_list(const Nprg2Executor *e,const Nprg2ExecRequest *r,Nprg2ExecStats *s,Nprg2ExecProfile *pr,uint64_t prev){
    Nprg2ExecCall call,ecall;
    call.command_index=0;fill_call(r,0,&call);call.preflighted=1;
    ecall=call;
    const uint32_t first=r->command_first;
    const Nprg2ExecRunRec *rec=s->runs,*end=rec+s->run_count;
    for(;rec<end;rec++){
        uint32_t ix=rec->ix;
        s->executed_commands=ix-first;
        if(nprg2_command_at(r->bound,ix,&call.command)!=NPRG2_OK)return NPRG2_EXEC_REJECTED;
        if(r->cancelled&&r->cancelled(r->cancel_ctx,ix)){s->failed_command=ix;s->failed_opcode=call.command.opcode;s->failed_variant=call.command.variant;return NPRG2_EXEC_CANCELLED;}
        if(approx_cv_skip(r,call.command.sequence)){
            s->approx_cost_volume_commands++;
            if(call.command.opcode==QDR_HMX_CONV)s->approx_cost_volume_conv_planes++;
            continue;
        }
        const uint32_t hi=rec->hi;const Nprg2ExecHandler *h=&e->handlers[hi];
        if(rec->fl&NPRG2_RUN_SKIP){   /* done by the conv's fused output step */
            if(pr&&pr->handler_run_count)pr->handler_run_count[hi]++;
            continue;
        }
        call.command_index=ix;
        {
            const Nprg2ExecBuffer *b=&call.scratch_region[h->scratch_region<NPRG2_EXEC_SCRATCH_REGIONS?h->scratch_region:0u];
            call.scratch=b->bytes?b->data:NULL;call.scratch_bytes=b->bytes;
        }
        call.epilogue=NULL;call.epilogue_dead=0;
        if(rec->fl&NPRG2_RUN_FUSED_CONV){
            if(nprg2_command_at(r->bound,rec->epi,&ecall.command)==NPRG2_OK){
                ecall.command_index=rec->epi;call.epilogue=&ecall;call.epilogue_dead=(rec->fl&NPRG2_RUN_DEAD)?1:0;
            }
        }
        uint32_t v=call.command.variant;
        int timed=pr&&pr->variant_cycles&&pr->variant_count&&v<NPRG2_EXEC_PROFILE_VARIANTS;
        uint64_t t0=pr?pr->clock():0;
        if(pr)pr->dispatch_cycles+=t0-prev;
        if(!h->run(&call,h->ctx)){s->failed_command=ix;s->failed_opcode=call.command.opcode;s->failed_variant=call.command.variant;return NPRG2_EXEC_HANDLER_ERROR;}
        if(approx_cv_concat_target(r,&call.command))s->approx_cost_volume_interpolated_planes=NPRG2_APPROX_CV_INTERPOLATED_PLANES;
        if(pr){
            prev=pr->clock();uint64_t d=prev-t0;pr->run_cycles+=d;
            if(timed){pr->variant_cycles[v]+=d;pr->variant_count[v]++;}
            if(pr->handler_run_cycles)pr->handler_run_cycles[hi]+=d;
            if(pr->handler_run_count)pr->handler_run_count[hi]++;
        }
    }
    s->executed_commands=r->command_count;
    if(pr)pr->dispatch_cycles+=pr->clock()-prev;
    return NPRG2_EXEC_OK;
}
int nprg2_executor_run(const Nprg2Executor *e,const Nprg2ExecRequest *r){
    Nprg2ExecProfile *pr=r&&r->profile&&r->profile->clock?r->profile:NULL;
    /* The execution API repeats preflight after any caller-side boundary
     * operations, unless the caller proves the identical request was just
     * preflighted; then it executes strictly in prepared schedule order. */
    if(r&&e&&r->reuse_preflight&&r->stats&&request_valid(e,r)&&preflight_matches(e,r,1)){
        /* A preflight call that has just verified the tables leaves digest_fresh
         * (consumed here); otherwise preflight_matches hashed them again. A reused
         * preflight is counted once per run (preflights_reused) whether it was made
         * moments ago or by an earlier call; preflight_calls counts full preflights. */
        Nprg2ExecStats *rs=r->stats;
        if(pr)pr->preflights_reused++;
        rs->digest_fresh=0;rs->executed_commands=0;rs->failed_command=UINT32_MAX;rs->failed_opcode=rs->failed_variant=0;
    } else {
        int preflight=nprg2_executor_preflight(e,r);if(preflight!=NPRG2_EXEC_OK)return preflight;
        if(r->stats)r->stats->digest_fresh=0;
    }
    Nprg2ExecStats *s=r->stats?r->stats:&nostats;
    s->approx_cost_volume_commands=0;s->approx_cost_volume_conv_planes=0;s->approx_cost_volume_interpolated_planes=0;
    const int strip=r->stats&&s->strip_valid;
    if(strip&&pr&&pr->handler_run_count)for(uint32_t k=0;k<NPRG2_EXEC_REGISTRY_CAPACITY;k++)pr->handler_run_count[k]+=s->stripped_by_slot[k];
    uint64_t prev=pr?pr->clock():0;
    if(strip&&s->runs_valid)return run_list(e,r,s,pr,prev);
    for(uint32_t n=0;n<r->command_count;n++){
        uint32_t ix=r->command_first+n;Nprg2ExecCall call;
        if(strip&&!(s->effect_bits[ix>>5]>>(ix&31u)&1u)){s->executed_commands++;continue;}
        if(nprg2_command_at(r->bound,ix,&call.command)!=NPRG2_OK)return NPRG2_EXEC_REJECTED;
        if(r->cancelled&&r->cancelled(r->cancel_ctx,ix)){s->failed_command=ix;s->failed_opcode=call.command.opcode;s->failed_variant=call.command.variant;return NPRG2_EXEC_CANCELLED;}
        if(approx_cv_skip(r,call.command.sequence)){
            s->approx_cost_volume_commands++;
            if(call.command.opcode==QDR_HMX_CONV)s->approx_cost_volume_conv_planes++;
            s->executed_commands++;continue;
        }
        int hi=handler_index(e,call.command.opcode,call.command.variant);if(hi<0)return NPRG2_EXEC_UNSUPPORTED;
        const Nprg2ExecHandler *h=&e->handlers[hi];
        {
            Nprg2FusePair *ep=fuse_pair_of_epi(r,ix);
            if(ep&&ep->active){   /* done by the conv's fused output step */
                if(pr&&pr->handler_run_count)pr->handler_run_count[hi]++;
                s->executed_commands++;continue;
            }
        }
        if(h->no_effect){
            if(pr&&pr->handler_run_count)pr->handler_run_count[hi]++;
            s->executed_commands++;continue;
        }
        fill_call(r,ix,&call);call.preflighted=1;select_scratch(&call,h);
        Nprg2ExecCall ecall;
        {
            Nprg2FusePair *fp=fuse_pair_of_conv(r,ix);
            if(fp&&fp->active&&nprg2_command_at(r->bound,fp->epi,&ecall.command)==NPRG2_OK){
                fill_call(r,fp->epi,&ecall);ecall.preflighted=1;call.epilogue=&ecall;call.epilogue_dead=fp->dead;
            }
        }
        uint32_t v=call.command.variant;
        int timed=pr&&pr->variant_cycles&&pr->variant_count&&v<NPRG2_EXEC_PROFILE_VARIANTS;
        uint64_t t0=pr?pr->clock():0;
        if(pr)pr->dispatch_cycles+=t0-prev;
        if(!h->run(&call,h->ctx)){s->failed_command=ix;s->failed_opcode=call.command.opcode;s->failed_variant=call.command.variant;return NPRG2_EXEC_HANDLER_ERROR;}
        if(approx_cv_concat_target(r,&call.command))s->approx_cost_volume_interpolated_planes=NPRG2_APPROX_CV_INTERPOLATED_PLANES;
        if(pr){
            prev=pr->clock();uint64_t d=prev-t0;pr->run_cycles+=d;
            if(timed){pr->variant_cycles[v]+=d;pr->variant_count[v]++;}
            if(pr->handler_run_cycles)pr->handler_run_cycles[hi]+=d;
            if(pr->handler_run_count)pr->handler_run_count[hi]++;
        }
        s->executed_commands++;
    }
    if(pr)pr->dispatch_cycles+=pr->clock()-prev;
    return NPRG2_EXEC_OK;
}

int nprg2_executor_run_preflighted_command(const Nprg2Executor *e,
                                           const Nprg2ExecRequest *full,
                                           uint32_t ix,uint8_t execution_lane,
                                           Nprg2Parallel *parallel_override,
                                           void *scratch,uint32_t scratch_bytes,
                                           void *near_scratch,uint32_t near_scratch_bytes) {
    Nprg2ExecRequest prepared;
    int rc=nprg2_executor_prepare_preflighted_request(e,full,execution_lane,
        parallel_override,scratch,scratch_bytes,near_scratch,near_scratch_bytes,&prepared);
    return rc?rc:nprg2_executor_run_preflighted_command_prepared(e,&prepared,ix);
}

int nprg2_executor_prepare_preflighted_request(const Nprg2Executor *e,
                                               const Nprg2ExecRequest *full,
                                               uint8_t execution_lane,
                                               Nprg2Parallel *parallel_override,
                                               void *scratch,uint32_t scratch_bytes,
                                               void *near_scratch,uint32_t near_scratch_bytes,
                                               Nprg2ExecRequest *prepared) {
    if(!e || !full || !full->stats || full->fuse ||
       !full->stats->runs_valid || !full->stats->strip_valid ||
       full->stats->failed_command!=UINT32_MAX ||
       full->stats->validated_commands!=full->command_count ||
       !prepared ||
       (scratch_bytes&&!scratch) || (near_scratch_bytes&&!near_scratch)) return NPRG2_EXEC_ARGUMENT;
    Nprg2ExecRequest local=*full;
    local.scratch=scratch;local.scratch_bytes=scratch_bytes;
    local.extra_scratch[0]=(Nprg2ExecBuffer){near_scratch,near_scratch_bytes};
    local.parallel=parallel_override;local.profile=NULL;local.execution_lane=execution_lane;
    /* Check caller-owned per-list arenas separately from the immutable
     * full-request preflight. The graph/pool mappings and command contents
     * remain covered by that preflight. */
    if(!request_valid(e,&local))return NPRG2_EXEC_ARGUMENT;
    *prepared=local;
    return NPRG2_EXEC_OK;
}

uint64_t (*nprg2_lane_clock)(void);
uint64_t nprg2_lane_prehandler[3];
int nprg2_executor_run_preflighted_command_prepared(const Nprg2Executor *e,
                                                     const Nprg2ExecRequest *full,
                                                     uint32_t ix) {
    if(!e || !full || !full->stats || full->fuse ||
       !full->stats->runs_valid || !full->stats->strip_valid ||
       full->stats->failed_command!=UINT32_MAX ||
       full->stats->validated_commands!=full->command_count ||
       ix<full->command_first || ix-full->command_first>=full->command_count)
        return NPRG2_EXEC_ARGUMENT;
    Nprg2ExecCall call;
#if defined(SETUP_FAST) && defined(__hexagon__)
    /* Preflight validated this record: halfword loads, no re-validation. */
    {
        const uint8_t *r=full->bound->command.data+(size_t)ix*NPRG2_COMMAND_BYTES;
        typedef uint16_t __attribute__((may_alias)) u16x;
        if((uintptr_t)r&1u){if(nprg2_command_at(full->bound,ix,&call.command)!=NPRG2_OK)return NPRG2_EXEC_REJECTED;}
        else {
            const u16x *h=(const u16x *)r;Nprg2Command *c=&call.command;
            c->sequence=h[0];c->opcode=r[2];c->variant=r[3];c->io_index=h[2];c->input_first=h[3];c->output_first=h[4];
            c->vector_first=h[5];c->call_first=h[6];c->input_count=r[14];c->output_count=r[15];c->vector_count=r[16];
            c->call_count=r[17];c->duplicate_pad=r[18];
        }
    }
#else
    if(nprg2_command_at(full->bound,ix,&call.command)!=NPRG2_OK)return NPRG2_EXEC_REJECTED;
#endif
    fill_call(full,ix,&call);call.preflighted=1;
    int hi=handler_index(e,call.command.opcode,call.command.variant);
    if(hi<0)return NPRG2_EXEC_UNSUPPORTED;
    const Nprg2ExecHandler *h=&e->handlers[hi];select_scratch(&call,h);
    if(h->scratch_bytes>UINT32_MAX-h->concurrent_worker_scratch_bytes ||
       h->scratch_bytes+h->concurrent_worker_scratch_bytes>call.scratch_bytes)return NPRG2_EXEC_REJECTED;
    if(full->cancelled && full->cancelled(full->cancel_ctx,ix))return NPRG2_EXEC_CANCELLED;
    if(nprg2_lane_clock && full->execution_lane<3u)nprg2_lane_prehandler[full->execution_lane]+=nprg2_lane_clock();
    if(h->no_effect)return NPRG2_EXEC_OK;
    return h->run(&call,h->ctx)?NPRG2_EXEC_OK:NPRG2_EXEC_HANDLER_ERROR;
}

static uint32_t dtype_bytes(uint8_t d){static const uint8_t sizes[8]={0,1,1,2,2,4,4,4};return d<8?sizes[d]:0;}
static uint32_t max_shape_bytes(const uint8_t *g){
    if(!g||!(g[1]&1u))return 0;uint64_t count=1;
    for(unsigned i=0;i<4;i++){int32_t d=(int16_t)rd16e(g+14+i*2u);if(d<0)return 0;count*=((uint32_t)d);if(count>UINT32_MAX)return 0;}
    uint32_t sz=dtype_bytes(g[0]);if(!sz||count>UINT32_MAX/sz)return 0;return (uint32_t)count*sz;
}
static int variant_classes_ok(uint8_t v,const uint8_t *s,const uint8_t *d){
    const char *a=NULL,*b=NULL;
    switch(v){case 27:a="fH";b="FH";break;case 28:a="fi";b="Fi";break;case 29:a="xpb";b="Xpb";break;case 30:a="xwb";b="Xwb";break;default:return 0;}
    return !memcmp(s,a,strlen(a))&&!memcmp(d,b,strlen(b));
}
int nprg2_exec_weights_validate(const Nprg2ExecCall *c,void *ctx){
    (void)ctx;if(!c||c->command.opcode!=QDR_WEIGHTS_TO_VTCM||c->command.input_count!=1||c->command.output_count!=1)return 0;
    uint32_t si,di;Nprg2TensorView s,d;if(nprg2_exec_input(c,0,&si,&s)||nprg2_exec_output(c,0,&di,&d))return 0;
    if(s.storage_kind!=1||d.storage_kind!=1||s.pool>=7||d.pool!=1||!c->pools[s.pool].base||!c->pools[d.pool].base)return 0;
    const uint8_t *sg=s.geometry_record,*dg=d.geometry_record;
    if(!variant_classes_ok(c->command.variant,sg+2,dg+2)||sg[0]!=dg[0]||!(sg[1]&1u)||!(dg[1]&1u))return 0;
    uint32_t sb=max_shape_bytes(sg),db=max_shape_bytes(dg);if(!sb||!db)return 0;uint32_t n=sb<db?sb:db;
    uint32_t dlimit=d.pool==1?c->pool1_writable_bytes:c->pools[d.pool].bytes;
    if(s.offset>c->pools[s.pool].bytes||n>c->pools[s.pool].bytes-s.offset||d.offset>dlimit||n>dlimit-d.offset)return 0;
    return 1;
}
#ifdef CMD_MEMO
/* Per-command memos: what a handler proved once for a program and pool table, it need not
 * prove again. Each command has one owner lane, and each lane has its own arena. */
#include <stdlib.h>
#define CMD_MEMO_COMMANDS 8192u
#define CMD_MEMO_ARENA 0x10000u
static void **cmd_memo;static uint8_t *cmd_arena[3];static uint32_t cmd_arena_used[3];
static const void *cmd_memo_program;static const uint8_t *cmd_memo_pool1;
void nprg2_exec_cmd_memo_begin(const void *program,const uint8_t *pool1){
    if(!cmd_memo)cmd_memo=malloc(sizeof(void *)*CMD_MEMO_COMMANDS);
    for(unsigned i=0;i<3;i++)if(!cmd_arena[i])cmd_arena[i]=malloc(CMD_MEMO_ARENA);
    if(!cmd_memo||!cmd_arena[0]||!cmd_arena[1]||!cmd_arena[2]){cmd_memo_program=NULL;return;}
    if(cmd_memo_program==program&&cmd_memo_pool1==pool1)return;
    memset(cmd_memo,0,sizeof(void *)*CMD_MEMO_COMMANDS);
    cmd_arena_used[0]=cmd_arena_used[1]=cmd_arena_used[2]=0;
    cmd_memo_program=program;cmd_memo_pool1=pool1;
}
void *nprg2_exec_cmd_memo(const Nprg2ExecCall *c){
    if(!c->preflighted||!cmd_memo_program||c->command_index>=CMD_MEMO_COMMANDS||
       (const void *)c->program!=cmd_memo_program||c->pools[1].base!=cmd_memo_pool1)return NULL;
    return cmd_memo[c->command_index];
}
void *nprg2_exec_cmd_memo_new(const Nprg2ExecCall *c,uint32_t bytes){
    uint32_t l=c->execution_lane;
    if(!c->preflighted||!cmd_memo_program||c->command_index>=CMD_MEMO_COMMANDS||l>2u||
       (const void *)c->program!=cmd_memo_program||c->pools[1].base!=cmd_memo_pool1)return NULL;
    bytes=(bytes+7u)&~7u;
    if(cmd_arena_used[l]+bytes>CMD_MEMO_ARENA)return NULL;
    void *p=cmd_arena[l]+cmd_arena_used[l];cmd_arena_used[l]+=bytes;return p;
}
void nprg2_exec_cmd_memo_commit(const Nprg2ExecCall *c,void *p){cmd_memo[c->command_index]=p;}
#endif
#ifdef WEIGHTS_MEMO
/* After preflight, a weight copy's addresses are fixed for a program and pool table: keep
 * them per command instead of revalidating on every run. WEIGHTS_CPU_BELOW: tensors under
 * this size are copied directly, which costs less than a DMA descriptor. */
#include <stdlib.h>
typedef struct {const void *program;const uint8_t *pool1;uint8_t *dst;const uint8_t *src;uint32_t n;} WeightsMemo;
static WeightsMemo *weights_memo;
void nprg2_exec_weights_memo_begin(void){if(!weights_memo)weights_memo=calloc(8192,sizeof(WeightsMemo));}
#ifndef WEIGHTS_CPU_BELOW
#define WEIGHTS_CPU_BELOW 0u
#endif
static int weights_copy(const Nprg2ExecCall *c,uint8_t *dst,const uint8_t *src,uint32_t n){
#ifdef WEIGHTS_CPU_VARIANTS
    /* These variants may run on any lane, so they must never use the DMA queue. */
    if(!((WEIGHTS_CPU_VARIANTS>>c->command.variant)&1u))
#endif
    if(n>=WEIGHTS_CPU_BELOW){
        int queued=nprg2_exec_weights_async_try(c,dst,src,n);
        if(queued>0)return 1;if(queued<0)return 0;
    }
    qd_vcopy(dst,src,n);return 1;
}
#endif
int nprg2_exec_weights_run(const Nprg2ExecCall *c,void *ctx){
    (void)ctx;
#ifdef WEIGHTS_MEMO
    WeightsMemo *wm=c&&c->preflighted&&weights_memo&&c->command_index<8192u?&weights_memo[c->command_index]:NULL;
    if(wm&&wm->program==(const void *)c->program&&wm->pool1==c->pools[1].base)return weights_copy(c,wm->dst,wm->src,wm->n);
#endif
    if(!nprg2_exec_weights_validate(c,NULL))return 0;uint32_t si,di;Nprg2TensorView s,d;
    if(nprg2_exec_input(c,0,&si,&s)||nprg2_exec_output(c,0,&di,&d))return 0;
    uint32_t sb=max_shape_bytes(s.geometry_record),db=max_shape_bytes(d.geometry_record),n=sb<db?sb:db;
#ifdef WEIGHTS_MEMO
    if(wm){wm->dst=c->pools[d.pool].base+d.offset;wm->src=c->pools[s.pool].base+s.offset;wm->n=n;wm->program=c->program;wm->pool1=c->pools[1].base;
           return weights_copy(c,wm->dst,wm->src,wm->n);}
#endif
    int queued=nprg2_exec_weights_async_try(c,c->pools[d.pool].base+d.offset,c->pools[s.pool].base+s.offset,n);
    if(queued>0)return 1;if(queued<0)return 0;
    qd_vcopy(c->pools[d.pool].base+d.offset,c->pools[s.pool].base+s.offset,n);return 1;
}
int nprg2_executor_register_weights(Nprg2Executor *e){
    if(!e)return NPRG2_EXEC_ARGUMENT;
    for(uint8_t v=27;v<=30;v++){int x=nprg2_executor_register(e,QDR_WEIGHTS_TO_VTCM,v,nprg2_exec_weights_validate,nprg2_exec_weights_run,NULL);if(x)return x;}
    return NPRG2_EXEC_OK;
}

/* ---------------------------------------------------------------- parallel */
volatile uint32_t nprg2_par_kind;
uint32_t nprg2_parallel_parts(Nprg2Parallel *p, uint32_t want) {
    if(!p || p->max_parts<2 || want<2) return 1;
    if(p->ready && !p->ready(p->impl)) { p->not_ready++; return 1; }
    return want<p->max_parts?want:p->max_parts;
}
void nprg2_parallel_for(Nprg2Parallel *p, uint32_t kind, uint32_t parts, Nprg2ParallelFn fn, void *ctx) {
    if(kind>=NPRG2_PAR_KINDS) kind=0;
    if(parts<2 || !p) { fn(ctx,0,parts<1?1:parts); return; }
    nprg2_par_kind=kind;
    if(p->run) {
        if(p->run(p->impl,parts,fn,ctx)) { p->jobs_parallel[kind]++; p->parts_run[kind]+=parts; return; }
        p->declined++;
    }
    /* Serial: all parts on the caller, in the requested order. */
    uint32_t idx[8],n=parts<8?parts:8;
    for(uint32_t i=0;i<n;i++) idx[i]=i;
    if(p->order==1) for(uint32_t i=0;i<n;i++) idx[i]=n-1u-i;
    else if(p->order==2) {
        for(uint32_t i=n;i>1;i--) {
            p->seed=p->seed*6364136223846793005ull+1442695040888963407ull;
            uint32_t j=(uint32_t)(p->seed>>33)%i,t=idx[i-1u];idx[i-1u]=idx[j];idx[j]=t;
        }
    }
    for(uint32_t i=0;i<n;i++) fn(ctx,idx[i],parts);
    p->jobs_serial[kind]++;p->parts_run[kind]+=parts;
}
