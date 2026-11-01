#include "prepared_runner.h"

#include <stdint.h>
#include <string.h>

static int overlaps_runner(const void *a,size_t an,const void *b,size_t bn){
    if(!an||!bn)return 0;if(!a||!b)return 1;uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(x>UINTPTR_MAX-an||y>UINTPTR_MAX-bn)return 1;return x<y+bn&&y<x+an;
}
static int aliases_graph(const void *ptr,size_t bytes,const void *raw,uint32_t raw_bytes,
                         const Nprg2Pool pools[7],const Nprg2Program *p){
    if(overlaps_runner(ptr,bytes,raw,raw_bytes))return 1;
    for(unsigned i=0;i<7;i++)if(overlaps_runner(ptr,bytes,pools[i].base,pools[i].bytes))return 1;
    if(p&&(overlaps_runner(ptr,bytes,p->tensors,(size_t)p->tensor_count*sizeof *p->tensors)||
           overlaps_runner(ptr,bytes,p->expanded_blocks,(size_t)p->expanded_block_count*sizeof *p->expanded_blocks)))return 1;
    return 0;
}
static void close_restore(Nprg2PreparedRunner *r){
    if(r->program.raw){r->program.expanded_blocks=r->setup_expanded_blocks;nprg2_close(&r->program);}
    memset(&r->bound,0,sizeof r->bound);r->ready=0;
}

int nprg2_runner_open(Nprg2PreparedRunner *r,const void *raw,uint32_t raw_bytes,
                      const Nprg2ExpectedHashes *expected,const Nprg2Pool pools[7],
                      Nprg2Allocator allocator,void *l2_blocks,uint32_t l2_block_capacity,
                      void *l2_hot,uint32_t l2_hot_capacity,uint32_t l2_available,
                      uint8_t verified_vtcm){
    if(!r||!raw||!expected||!pools||!l2_hot||!verified_vtcm)return NPRG2_RUNNER_ARGUMENT;
    if(!pools[1].base||pools[1].bytes<NPRG2_RUNNER_POOL1_TOTAL_BYTES)return NPRG2_RUNNER_ARGUMENT;
    memset(r,0,sizeof *r);
    memcpy(r->pools,pools,sizeof r->pools);
    int e=nprg2_open(&r->program,raw,raw_bytes,expected,pools,allocator);if(e!=NPRG2_OK)return NPRG2_RUNNER_READER;
    r->setup_expanded_blocks=r->program.expanded_blocks;r->l2_hot=(uint8_t*)l2_hot;
    uint64_t block_bytes64=(uint64_t)r->program.expanded_block_count*sizeof(uint32_t);
    if(block_bytes64>UINT32_MAX||block_bytes64>l2_block_capacity||(!l2_blocks&&block_bytes64)||
       !l2_hot_capacity||l2_hot_capacity>l2_available){close_restore(r);return NPRG2_RUNNER_MEMORY;}
    uint32_t block_bytes=(uint32_t)block_bytes64;
    if(block_bytes&&((uintptr_t)l2_blocks&127u)){close_restore(r);return NPRG2_RUNNER_ARGUMENT;}
    if(overlaps_runner(l2_blocks,l2_block_capacity,l2_hot,l2_hot_capacity)||
       aliases_graph(l2_blocks,l2_block_capacity,raw,raw_bytes,pools,&r->program)||
       aliases_graph(l2_hot,l2_hot_capacity,raw,raw_bytes,pools,&r->program)){
        close_restore(r);return NPRG2_RUNNER_ALIAS;
    }
    if(block_bytes)memcpy(l2_blocks,r->setup_expanded_blocks,block_bytes);
    r->l2_expanded_blocks=(uint32_t*)l2_blocks;r->l2_expanded_bytes=block_bytes;
    r->program.expanded_blocks=r->l2_expanded_blocks;
    Nprg2BindConfig cfg={.hot_memory=r->l2_hot,.hot_capacity=l2_hot_capacity,
        .scratch_budget=0,.l2_available_bytes=l2_available,
        .pool1_is_verified_vtcm=verified_vtcm,.resolve=NULL,.resolver_ctx=NULL};
    e=nprg2_bind_metadata(&r->program,r->pools,&cfg,&r->bound);
    if(e!=NPRG2_OK){close_restore(r);return NPRG2_RUNNER_BIND;}
    r->l2_hot_bytes=r->bound.hot_bytes;r->ready=1;return NPRG2_RUNNER_OK;
}

static int runner_request(Nprg2PreparedRunner *r,uint32_t first,uint32_t count,uint32_t pool1_writable,
                          const Nprg2ExecBuffer *inputs,uint32_t input_count,
                          const Nprg2ExecBuffer *outputs,uint32_t output_count,
                          void *user,void *scratch,uint32_t scratch_bytes,
                          Nprg2ExecCancel cancelled,void *cancel_ctx,Nprg2ExecStats *stats,
                          Nprg2ExecRequest *q){
    if(!r||!r->ready||pool1_writable!=NPRG2_RUNNER_POOL1_ACTIVE_BYTES)return NPRG2_EXEC_ARGUMENT;
    memset(q,0,sizeof *q);*q=(Nprg2ExecRequest){.program=&r->program,.bound=&r->bound,.pool1_writable_bytes=pool1_writable,
        .inputs=inputs,.input_count=input_count,.outputs=outputs,.output_count=output_count,
        .user=user,.scratch=scratch,.scratch_bytes=scratch_bytes,.command_first=first,
        .command_count=count,.cancelled=cancelled,.cancel_ctx=cancel_ctx,.stats=stats};
    memcpy(q->pools,r->pools,sizeof q->pools);
    memcpy(q->extra_scratch,r->extra_scratch,sizeof q->extra_scratch);return NPRG2_EXEC_OK;
}
int nprg2_runner_set_scratch_region(Nprg2PreparedRunner *r,uint8_t region,void *scratch,uint32_t bytes){
    if(!r||!r->ready||!region||region>=NPRG2_EXEC_SCRATCH_REGIONS||(bytes&&!scratch))return NPRG2_RUNNER_ARGUMENT;
    r->extra_scratch[region-1u]=(Nprg2ExecBuffer){bytes?(uint8_t *)scratch:NULL,bytes};return NPRG2_RUNNER_OK;
}
int nprg2_runner_preflight(Nprg2PreparedRunner *r,const Nprg2Executor *e,
                           uint32_t first,uint32_t count,uint32_t pool1_writable,
                           const Nprg2ExecBuffer *inputs,uint32_t input_count,
                           const Nprg2ExecBuffer *outputs,uint32_t output_count,
                           void *user,void *scratch,uint32_t scratch_bytes,Nprg2ExecStats *stats){
    if(!e)return NPRG2_EXEC_ARGUMENT;Nprg2ExecRequest q;
    int x=runner_request(r,first,count,pool1_writable,inputs,input_count,outputs,output_count,user,scratch,scratch_bytes,NULL,NULL,stats,&q);
    return x?x:nprg2_executor_preflight(e,&q);
}
int nprg2_runner_execute(Nprg2PreparedRunner *r,const Nprg2Executor *e,
                         uint32_t first,uint32_t count,uint32_t pool1_writable,
                         const Nprg2ExecBuffer *inputs,uint32_t input_count,
                         const Nprg2ExecBuffer *outputs,uint32_t output_count,
                         void *user,void *scratch,uint32_t scratch_bytes,
                         Nprg2ExecCancel cancelled,void *cancel_ctx,Nprg2ExecStats *stats){
    if(!e)return NPRG2_EXEC_ARGUMENT;Nprg2ExecRequest q;
    int x=runner_request(r,first,count,pool1_writable,inputs,input_count,outputs,output_count,user,scratch,scratch_bytes,cancelled,cancel_ctx,stats,&q);
    return x?x:nprg2_executor_run(e,&q);
}

void nprg2_runner_close(Nprg2PreparedRunner *r){if(!r)return;close_restore(r);memset(r,0,sizeof *r);}
