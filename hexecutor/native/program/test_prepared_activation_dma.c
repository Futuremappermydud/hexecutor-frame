/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_preprocess.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);return 1;}}while(0)
#define HEADER_BYTES 64u
#define ACTIVE_BYTES 0x780000u
static uint32_t rd32(const uint8_t*p){return(uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static int read_file(const char*path,uint8_t**data,uint32_t*bytes){
    FILE*f=fopen(path,"rb");if(!f)return 0;if(fseek(f,0,SEEK_END)){fclose(f);return 0;}
    long n=ftell(f);fclose(f);if(n<=0||(uint64_t)n>UINT32_MAX)return 0;
    f=fopen(path,"rb");if(!f)return 0;uint8_t*p=(uint8_t*)malloc((size_t)n);if(!p){fclose(f);return 0;}
    size_t got=0;while(got<(size_t)n){size_t chunk=(size_t)n-got;if(chunk>4096u)chunk=4096u;size_t step=fread(p+got,1,chunk,f);if(!step){free(p);fclose(f);return 0;}got+=step;}
    fclose(f);*data=p;*bytes=(uint32_t)n;return 1;
}
static int resolve_all(void*ctx,const Nprg2Program*p,uint32_t i,const Nprg2Command*c,Nprg2Capability*cap){(void)ctx;(void)p;(void)i;(void)c;cap->supported=1;cap->may_fallback=0;cap->scratch_bytes=0;cap->concurrent_worker_scratch_bytes=0;return 1;}
static int compare_before(Nprg2Pool pools[7],const uint8_t*before[7]){
    for(unsigned p=0;p<7;p++)if(pools[p].bytes&&memcmp(pools[p].base,before[p],pools[p].bytes)){
        uint32_t i=0;while(i<pools[p].bytes&&pools[p].base[i]==before[p][i])i++;
        fprintf(stderr,"preflight wrote pool=%u byte=%lu\n",p,(unsigned long)i);return 0;
    }
    return 1;
}
static int compare_after(Nprg2Pool pools[7],const uint8_t*before[7],uint32_t off,const uint8_t*patch,uint32_t bytes){
    for(unsigned p=0;p<7;p++)if(p!=1&&pools[p].bytes&&memcmp(pools[p].base,before[p],pools[p].bytes)){
        uint32_t i=0;while(i<pools[p].bytes&&pools[p].base[i]==before[p][i])i++;
        fprintf(stderr,"CPU-prefix pool mismatch pool=%u byte=%lu\n",p,(unsigned long)i);return 0;
    }
    if(memcmp(pools[1].base,before[1],off)){fprintf(stderr,"pool1 mismatch before target block\n");return 0;}
    if(memcmp(pools[1].base+off,patch,bytes)){fprintf(stderr,"pool1 target block mismatch\n");return 0;}
    uint32_t end=off+bytes;if(memcmp(pools[1].base+end,before[1]+end,pools[1].bytes-end)){fprintf(stderr,"pool1 mismatch after target block\n");return 0;}
    return 1;
}
static int run_case(const char*qpath,const char*npath,const char*fpath,uint32_t expected_cmd,uint32_t expected_seq){
    uint32_t qn,nn,fn;uint8_t*q=0,*raw=0,*fx=0,*qsha=malloc(32),*nsha=malloc(32);CHECK(qsha&&nsha);
    CHECK(read_file(qpath,&q,&qn)&&read_file(npath,&raw,&nn)&&read_file(fpath,&fx,&fn));
    CHECK(fn>=HEADER_BYTES&&!memcmp(fx,"PPFX",4)&&rd32(fx+4)==1u&&rd32(fx+8)==QDR_COPY_ACTIVATION&&rd32(fx+12)==17u);
    uint32_t seq=rd32(fx+16),cmd=rd32(fx+20),io=rd32(fx+24),io_bytes=rd32(fx+28),patches=rd32(fx+32);
    CHECK(cmd==expected_cmd&&seq==expected_seq&&io==UINT32_MAX&&io_bytes==0&&patches==1);
    Nprg2Pool pools[7]={0};for(unsigned p=0;p<7;p++){pools[p].bytes=rd32(raw+144+4*p);pools[p].base=pools[p].bytes?malloc(pools[p].bytes):0;CHECK(!pools[p].bytes||pools[p].base);}
    const uint8_t*cur=fx+HEADER_BYTES;const uint8_t*before[7]={0};
    for(unsigned p=0;p<7;p++){uint32_t z=rd32(fx+36+4*p);CHECK(z==pools[p].bytes&&(uint64_t)(cur-fx)+z<=fn);if(z){before[p]=cur;memcpy(pools[p].base,cur,z);}cur+=z;}
    CHECK(pools[1].bytes>=ACTIVE_BYTES);
    CHECK((uint64_t)(cur-fx)+8u<=fn);uint32_t off=rd32(cur),n=rd32(cur+4);cur+=8;CHECK(n==2048u&&off<=pools[1].bytes&&n<=pools[1].bytes-off&&(uint64_t)(cur-fx)+n==fn);const uint8_t*patch=cur;
    nprg2_sha256(q,qn,qsha);nprg2_sha256(raw,nn,nsha);Nprg2ExpectedHashes hashes={.context_sha256=q+32,.constants_sha256=q+64,.source_sha256=qsha,.prepared_sha256=nsha};
    Nprg2Program program;CHECK(nprg2_open(&program,raw,nn,&hashes,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t*hot=calloc(1,300000u);CHECK(hot);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000u,.l2_available_bytes=786432u,.pool1_is_verified_vtcm=1,.resolve=resolve_all};Nprg2Bound bound;CHECK(nprg2_bind_mock(&program,pools,&bc,&bound)==NPRG2_OK);
    Nprg2Command command;CHECK(nprg2_command_at(&bound,cmd,&command)==NPRG2_OK&&command.opcode==QDR_COPY_ACTIVATION&&command.variant==17u&&command.sequence==seq);
    Nprg2Executor executor;nprg2_executor_init(&executor);CHECK(nprg2_executor_register_preprocess(&executor)==NPRG2_EXEC_OK);
    Nprg2ExecRequest request={.program=&program,.bound=&bound,.pool1_writable_bytes=ACTIVE_BYTES,.command_first=cmd,.command_count=1};memcpy(request.pools,pools,sizeof pools);
    CHECK(nprg2_executor_preflight(&executor,&request)==NPRG2_EXEC_OK);CHECK(compare_before(pools,before));
    uint8_t*command_bytes=(uint8_t*)bound.command.data+(size_t)cmd*NPRG2_COMMAND_BYTES;uint8_t old_variant=command_bytes[3];command_bytes[3]=18u;
    CHECK(nprg2_executor_preflight(&executor,&request)==NPRG2_EXEC_UNSUPPORTED);CHECK(compare_before(pools,before));command_bytes[3]=old_variant;
    Nprg2ExecCall call={.command_index=cmd,.command=command,.program=&program,.bound=&bound,.pools=pools,.pool1_writable_bytes=ACTIVE_BYTES};uint32_t tid;Nprg2TensorView out;
    CHECK(nprg2_exec_output(&call,0,&tid,&out)==NPRG2_EXEC_OK&&out.slice_origin<out.block_count);uint32_t old_block=((uint32_t*)out.blocks)[out.slice_origin];((uint32_t*)out.blocks)[out.slice_origin]=0x780001u;
    CHECK(nprg2_executor_preflight(&executor,&request)==NPRG2_EXEC_REJECTED);CHECK(compare_before(pools,before));((uint32_t*)out.blocks)[out.slice_origin]=old_block;
    CHECK(nprg2_executor_preflight(&executor,&request)==NPRG2_EXEC_OK);CHECK(nprg2_executor_run(&executor,&request)==NPRG2_EXEC_OK);CHECK(compare_after(pools,before,off,patch,n));
    printf("PASS prepared activation-DMA CPU-prefix %s: opcode=17 variant=17 command=%lu seq=%lu destination=%lu; exact 2048-byte output and every non-output pool byte; variant/tail rejection before writes\n",qpath,(unsigned long)cmd,(unsigned long)seq,(unsigned long)off);
    for(unsigned p=0;p<7;p++)free(pools[p].base);free(hot);nprg2_close(&program);free(q);free(raw);free(fx);free(qsha);free(nsha);return 0;
}
int main(void){
    if(run_case("hexecutor/native/build/resident-primary.qdrx","depth-model/deployment/programs/primary-v2.nprg","hexecutor/native/program/fixtures/primary-smoke-activation-dma.ppcpu",5u,67u))return 1;
    if(run_case("hexecutor/native/build/resident-alternate.qdrx","depth-model/deployment/programs/alternate-v2.nprg","hexecutor/native/program/fixtures/alternate-smoke-activation-dma.ppcpu",7u,58u))return 1;
    return 0;
}
