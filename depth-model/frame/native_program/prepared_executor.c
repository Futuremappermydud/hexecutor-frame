#include "prepared_executor.h"
#include "prepared_vcopy.h"

#include <limits.h>
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
    call->user=r->user;call->scratch=r->scratch;call->scratch_bytes=r->scratch_bytes;
    call->scratch_region[0]=(Nprg2ExecBuffer){(uint8_t *)r->scratch,r->scratch_bytes};
    for(uint32_t k=1;k<NPRG2_EXEC_SCRATCH_REGIONS;k++)call->scratch_region[k]=r->extra_scratch[k-1u];
    call->preflighted=0;call->epilogue=NULL;call->epilogue_dead=0;
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
    for(unsigned i=0;i<7;i++){h=sig_mix(h,(uintptr_t)r->pools[i].base);h=sig_mix(h,r->pools[i].bytes);}
    h=sig_mix(h,r->pool1_writable_bytes);h=sig_mix(h,(uintptr_t)r->user);
    h=sig_mix(h,(uintptr_t)r->scratch);h=sig_mix(h,r->scratch_bytes);
    for(unsigned k=0;k+1u<NPRG2_EXEC_SCRATCH_REGIONS;k++){h=sig_mix(h,(uintptr_t)r->extra_scratch[k].data);h=sig_mix(h,r->extra_scratch[k].bytes);}
    h=sig_mix(h,(uintptr_t)r->inputs);h=sig_mix(h,r->input_count);h=sig_mix(h,(uintptr_t)r->outputs);h=sig_mix(h,r->output_count);
    for(uint32_t i=0;i<r->input_count&&r->inputs;i++){h=sig_mix(h,(uintptr_t)r->inputs[i].data);h=sig_mix(h,r->inputs[i].bytes);}
    for(uint32_t i=0;i<r->output_count&&r->outputs;i++){h=sig_mix(h,(uintptr_t)r->outputs[i].data);h=sig_mix(h,r->outputs[i].bytes);}
    h=sig_mix(h,r->command_first);h=sig_mix(h,r->command_count);h=sig_mix(h,e->count);h=sig_mix(h,(uintptr_t)r->fuse);h=sig_mix(h,(uintptr_t)r->fuse);
    for(uint32_t i=0;i<NPRG2_EXEC_REGISTRY_CAPACITY;i++){
        const Nprg2ExecHandler *x=&e->handlers[i];if(!x->used)continue;
        h=sig_mix(h,((uint64_t)x->opcode<<8|x->variant)<<8|x->scratch_region);
        h=sig_mix(h,(uintptr_t)x->validate);h=sig_mix(h,(uintptr_t)x->fuse);h=sig_mix(h,(uintptr_t)x->ctx);
        h=sig_mix(h,((uint64_t)x->scratch_bytes<<32)|x->concurrent_worker_scratch_bytes);
    }
    return h|1u;
}
int nprg2_executor_preflight(const Nprg2Executor *e,const Nprg2ExecRequest *r){
    if(!e||!request_valid(e,r))return NPRG2_EXEC_ARGUMENT;
    Nprg2ExecStats local={0};Nprg2ExecStats *s=r->stats?r->stats:&local;memset(s,0,sizeof *s);s->failed_command=UINT32_MAX;
    Nprg2ExecProfile *pr=r->profile&&r->profile->clock?r->profile:NULL;
    uint64_t p0=pr?pr->clock():0;
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
    }
    if(r->stats)s->preflight_signature=request_signature(e,r);
    if(pr){pr->preflight_cycles+=pr->clock()-p0;pr->preflight_calls++;}
    return NPRG2_EXEC_OK;
}
int nprg2_executor_run(const Nprg2Executor *e,const Nprg2ExecRequest *r){
    Nprg2ExecProfile *pr=r&&r->profile&&r->profile->clock?r->profile:NULL;
    /* The execution API repeats preflight after any caller-side boundary
     * operations, unless the caller proves the identical request was just
     * preflighted; then it executes strictly in prepared schedule order. */
    if(r&&r->reuse_preflight&&e&&r->stats&&r->stats->preflight_signature&&request_valid(e,r)&&
       r->stats->preflight_signature==request_signature(e,r)&&r->stats->validated_commands==r->command_count){
        if(pr)pr->preflights_reused++;
    } else {
        int preflight=nprg2_executor_preflight(e,r);if(preflight!=NPRG2_EXEC_OK)return preflight;
    }
    Nprg2ExecStats local={0};Nprg2ExecStats *s=r->stats?r->stats:&local;
    uint64_t prev=pr?pr->clock():0;
    for(uint32_t n=0;n<r->command_count;n++){
        uint32_t ix=r->command_first+n;Nprg2ExecCall call;
        if(nprg2_command_at(r->bound,ix,&call.command)!=NPRG2_OK)return NPRG2_EXEC_REJECTED;
        if(r->cancelled&&r->cancelled(r->cancel_ctx,ix)){s->failed_command=ix;s->failed_opcode=call.command.opcode;s->failed_variant=call.command.variant;return NPRG2_EXEC_CANCELLED;}
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
int nprg2_exec_weights_run(const Nprg2ExecCall *c,void *ctx){
    (void)ctx;if(!nprg2_exec_weights_validate(c,NULL))return 0;uint32_t si,di;Nprg2TensorView s,d;
    if(nprg2_exec_input(c,0,&si,&s)||nprg2_exec_output(c,0,&di,&d))return 0;
    uint32_t sb=max_shape_bytes(s.geometry_record),db=max_shape_bytes(d.geometry_record),n=sb<db?sb:db;
    qd_vcopy(c->pools[d.pool].base+d.offset,c->pools[s.pool].base+s.offset,n);return 1;
}
int nprg2_executor_register_weights(Nprg2Executor *e){
    if(!e)return NPRG2_EXEC_ARGUMENT;
    for(uint8_t v=27;v<=30;v++){int x=nprg2_executor_register(e,QDR_WEIGHTS_TO_VTCM,v,nprg2_exec_weights_validate,nprg2_exec_weights_run,NULL);if(x)return x;}
    return NPRG2_EXEC_OK;
}
