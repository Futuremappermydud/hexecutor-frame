#include "prepared_runner.h"
#include "prepared_io.h"
#include "prepared_native.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);exit(1);}}while(0)
static uint32_t rd32(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint8_t *read_file(const char *path,uint32_t *n){FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long z=ftell(f);CHECK(z>0&&z<=(long)NPRG2_MAX_IMAGE_BYTES);CHECK(!fseek(f,0,SEEK_SET));uint8_t *p=malloc((size_t)z);CHECK(p&&fread(p,1,(size_t)z,f)==(size_t)z);fclose(f);*n=(uint32_t)z;return p;}
static void pool_alloc(const uint8_t *raw,Nprg2Pool p[7]){for(int i=0;i<7;i++){p[i].bytes=rd32(raw+144+i*4);p[i].base=p[i].bytes?(i==1?aligned_alloc(128,p[i].bytes):malloc(p[i].bytes)):NULL;CHECK(!p[i].bytes||p[i].base);if(p[i].bytes)memset(p[i].base,(int)(0x41+i*5),p[i].bytes);}}
static void pool_free(Nprg2Pool p[7]){for(int i=0;i<7;i++)free(p[i].base);}
static uint32_t product4(const int32_t d[4]){uint64_t n=1;for(int i=0;i<4;i++){CHECK(d[i]>=0);n*=(uint32_t)d[i];CHECK(n<=UINT32_MAX);}return (uint32_t)n;}
static uint32_t mock_hmx_calls;
static int mock_hmx_submit(void *ctx,const Nprg2ExecCall *call,const QdDirectBinding *binding,uint8_t family){(void)ctx;CHECK(call&&binding&&binding->in_desc&&binding->out_desc&&binding->weights&&binding->fi);CHECK(family==1||family==3);mock_hmx_calls++;return 1;}
void qd_set_hmx_params_stride1(uint32_t *c,uint32_t a,uint32_t b,uint32_t d,uint32_t e,uint32_t f,uint32_t g,uint32_t h){c[0]=a;c[1]=b;c[2]=d;c[3]=e;c[4]=f;c[5]=g;c[6]=h;c[12]=8;}
static void run_model(const char *qpath,const char *npath){
    uint32_t qn,nn;uint8_t *q=read_file(qpath,&qn),*raw=read_file(npath,&nn),qh[32],nh[32];Nprg2ExpectedHashes expected;
    nprg2_sha256(q,qn,qh);nprg2_sha256(raw,nn,nh);expected.context_sha256=q+32;expected.constants_sha256=q+64;expected.source_sha256=qh;expected.prepared_sha256=nh;
    Nprg2Pool pools[7]={0};pool_alloc(raw,pools);Nprg2PreparedRunner runner;uint32_t block_capacity=0x7e000u,hot_capacity=0x3f000u;
    uint32_t *blocks=aligned_alloc(128,block_capacity);uint8_t *hot=aligned_alloc(128,hot_capacity);CHECK(blocks&&hot);memset(blocks,0x5c,block_capacity);memset(hot,0xa7,hot_capacity);
    CHECK(nprg2_runner_open(&runner,raw,nn,&expected,pools,(Nprg2Allocator){0},blocks,block_capacity,hot,hot_capacity,hot_capacity,1)==NPRG2_RUNNER_OK);
    CHECK(runner.ready&&runner.program.expanded_blocks==blocks&&runner.l2_expanded_bytes==runner.program.expanded_block_count*4u);
    CHECK(runner.l2_hot_bytes<=hot_capacity&&runner.l2_hot_bytes>0);uint32_t hot_used=runner.l2_hot_bytes,expanded_used=runner.l2_expanded_bytes;
    int saw_block=0;for(uint32_t i=0;i<runner.program.tensor_count;i++){Nprg2TensorView t;CHECK(nprg2_bound_tensor_at(&runner.program,&runner.bound,i,&t)==NPRG2_OK);if(t.storage_kind==2){CHECK(t.blocks>=blocks&&t.blocks<blocks+runner.program.expanded_block_count);saw_block=1;break;}}CHECK(saw_block);
    Nprg2Executor e;nprg2_executor_init(&e);CHECK(nprg2_executor_register_core_safe_ops(&e)==NPRG2_EXEC_OK);
    uint8_t before[32];nprg2_sha256(pools[1].base,pools[1].bytes,before);
    CHECK(nprg2_runner_execute(&runner,&e,0,1,NPRG2_RUNNER_POOL1_ACTIVE_BYTES,NULL,0,NULL,0,NULL,NULL,0,NULL,NULL,NULL)==NPRG2_EXEC_UNSUPPORTED);
    uint8_t after[32];nprg2_sha256(pools[1].base,pools[1].bytes,after);CHECK(!memcmp(before,after,32));
    Nprg2ExecStats single_preflight={0};CHECK(nprg2_runner_preflight(&runner,&e,0,1,NPRG2_RUNNER_POOL1_ACTIVE_BYTES,NULL,0,NULL,0,NULL,NULL,0,&single_preflight)==NPRG2_EXEC_UNSUPPORTED);CHECK(single_preflight.validated_commands==0&&single_preflight.executed_commands==0&&single_preflight.failed_command==0);
    uint32_t first=UINT32_MAX;Nprg2Command c;
    for(uint32_t i=0;i<runner.program.command_count;i++){CHECK(nprg2_command_at(&runner.bound,i,&c)==NPRG2_OK);if(c.opcode==QDR_INPUT_SLICE_PAD){first=i;break;}}
    CHECK(first!=UINT32_MAX);CHECK(nprg2_command_at(&runner.bound,first,&c)==NPRG2_OK);
    Nprg2ExecCall call={.command_index=first,.command=c,.program=&runner.program,.bound=&runner.bound,.pools=pools,.pool1_writable_bytes=NPRG2_RUNNER_POOL1_ACTIVE_BYTES};
    uint32_t ti;Nprg2TensorView shape;CHECK(nprg2_exec_input(&call,6,&ti,&shape)==NPRG2_EXEC_OK);(void)ti;int32_t dims[4];for(int i=0;i<4;i++)dims[i]=(int32_t)rd32(shape.geometry_record+6+i*4u);uint32_t bytes=product4(dims);
    uint8_t *src=malloc(bytes);CHECK(src);for(uint32_t i=0;i<bytes;i++)src[i]=(uint8_t)(i*11u+0x23u);Nprg2ExecBuffer inputs[2]={{NULL,0},{NULL,0}};CHECK(c.io_index<2);inputs[c.io_index]=(Nprg2ExecBuffer){src,bytes};
    uint32_t first_unsupported=first;for(;first_unsupported<runner.program.command_count;first_unsupported++){CHECK(nprg2_command_at(&runner.bound,first_unsupported,&c)==NPRG2_OK);if(c.opcode!=QDR_NOP&&c.opcode!=QDR_INPUT_SLICE_PAD)break;}
    CHECK(first_unsupported>first&&first_unsupported<runner.program.command_count);nprg2_sha256(pools[1].base,pools[1].bytes,before);
    Nprg2ExecStats range_preflight={0},range_execute={0};
    CHECK(nprg2_runner_preflight(&runner,&e,first,first_unsupported-first+1,NPRG2_RUNNER_POOL1_ACTIVE_BYTES,inputs,2,NULL,0,NULL,NULL,0,&range_preflight)==NPRG2_EXEC_UNSUPPORTED);
    CHECK(range_preflight.validated_commands==first_unsupported-first&&range_preflight.executed_commands==0&&range_preflight.failed_command==first_unsupported);
    CHECK(nprg2_runner_execute(&runner,&e,first,first_unsupported-first+1,NPRG2_RUNNER_POOL1_ACTIVE_BYTES,inputs,2,NULL,0,NULL,NULL,0,NULL,NULL,&range_execute)==NPRG2_EXEC_UNSUPPORTED);
    CHECK(range_execute.validated_commands==range_preflight.validated_commands&&range_execute.executed_commands==0&&range_execute.failed_command==range_preflight.failed_command);
    nprg2_sha256(pools[1].base,pools[1].bytes,after);CHECK(!memcmp(before,after,32));
    uint8_t *pool_before=malloc(pools[1].bytes);CHECK(pool_before);memcpy(pool_before,pools[1].base,pools[1].bytes);
    CHECK(nprg2_runner_execute(&runner,&e,first,1,NPRG2_RUNNER_POOL1_ACTIVE_BYTES,inputs,2,NULL,0,NULL,NULL,0,NULL,NULL,NULL)==NPRG2_EXEC_OK);
    CHECK(memcmp(pool_before,pools[1].base,pools[1].bytes)!=0);CHECK(!memcmp(pool_before+NPRG2_RUNNER_POOL1_ACTIVE_BYTES,pools[1].base+NPRG2_RUNNER_POOL1_ACTIVE_BYTES,pools[1].bytes-NPRG2_RUNNER_POOL1_ACTIVE_BYTES));
    uint32_t *tables=aligned_alloc(128,2048),*abi=aligned_alloc(128,128);CHECK(tables&&abi);Nprg2NativeConfig native={.active_vtcm_bytes=NPRG2_RUNNER_POOL1_ACTIVE_BYTES,.tables=tables,.table_bytes=2048,.abi=abi,.abi_bytes=120,.tail=pools[1].base+NPRG2_RUNNER_POOL1_ACTIVE_BYTES,.tail_bytes=0x80000u,.submit_and_drain=mock_hmx_submit};
    CHECK(nprg2_executor_register_native_conv(&e,&native)==NPRG2_EXEC_OK);uint32_t hmx_index=UINT32_MAX;
    for(uint32_t i=0;i<runner.program.command_count;i++){CHECK(nprg2_command_at(&runner.bound,i,&c)==NPRG2_OK);if(c.opcode!=QDR_HMX_CONV||(c.variant!=32&&c.variant!=33))continue;int hi=-1;for(uint32_t j=0;j<NPRG2_EXEC_REGISTRY_CAPACITY;j++)if(e.handlers[j].used&&e.handlers[j].opcode==c.opcode&&e.handlers[j].variant==c.variant)hi=(int)j;CHECK(hi>=0);Nprg2ExecCall hc={.command_index=i,.command=c,.program=&runner.program,.bound=&runner.bound,.pools=pools,.pool1_writable_bytes=NPRG2_RUNNER_POOL1_ACTIVE_BYTES};if(e.handlers[hi].validate(&hc,e.handlers[hi].ctx)){hmx_index=i;break;}}
    CHECK(hmx_index!=UINT32_MAX);uint8_t *active_before=malloc(NPRG2_RUNNER_POOL1_ACTIVE_BYTES);CHECK(active_before);memcpy(active_before,pools[1].base,NPRG2_RUNNER_POOL1_ACTIVE_BYTES);mock_hmx_calls=0;
    CHECK(nprg2_runner_execute(&runner,&e,hmx_index,1,NPRG2_RUNNER_POOL1_ACTIVE_BYTES,NULL,0,NULL,0,NULL,NULL,0,NULL,NULL,NULL)==NPRG2_EXEC_OK);CHECK(mock_hmx_calls==1);CHECK(!memcmp(active_before,pools[1].base,NPRG2_RUNNER_POOL1_ACTIVE_BYTES));free(active_before);free(tables);free(abi);
    uint32_t first_block=blocks[0];nprg2_runner_close(&runner);CHECK(!runner.ready);CHECK(blocks[0]==first_block);
    /* The open path rejects aliased arenas before copying either table. */
    memset(hot,0xa7,hot_capacity);memset(blocks,0x5c,block_capacity);
    CHECK(nprg2_runner_open(&runner,raw,nn,&expected,pools,(Nprg2Allocator){0},blocks,block_capacity,blocks,hot_capacity,hot_capacity,1)==NPRG2_RUNNER_ALIAS);
    CHECK(((uint8_t*)blocks)[0]==0x5c&&((uint8_t*)hot)[0]==0xa7);
    printf("PASS bounded runner %s: hot=%u expanded=%u; no fallback\n",qpath,hot_used,expanded_used);
    free(pool_before);free(src);free(hot);free(blocks);pool_free(pools);free(q);free(raw);
}
int main(int argc,char **argv){CHECK(argc==5);run_model(argv[1],argv[2]);run_model(argv[3],argv[4]);return 0;}
