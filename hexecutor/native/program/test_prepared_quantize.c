/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_preprocess.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);return 1;}}while(0)
#define POOL1_BYTES (8u*1024u*1024u)
#define ACTIVE_BYTES 0x780000u
static uint32_t rd32(const uint8_t*p){return(uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static void wr32(uint8_t*p,uint32_t x){p[0]=(uint8_t)x;p[1]=(uint8_t)(x>>8);p[2]=(uint8_t)(x>>16);p[3]=(uint8_t)(x>>24);}
static int read_exact(const char*path,void*data,uint32_t bytes){FILE*f=fopen(path,"rb");if(!f)return 0;size_t n=fread(data,1,bytes,f);int end=fgetc(f)==EOF;fclose(f);return n==bytes&&end;}
static int resolve_all(void*ctx,const Nprg2Program*p,uint32_t i,const Nprg2Command*c,Nprg2Capability*cap){(void)ctx;(void)p;(void)i;(void)c;cap->supported=1;cap->may_fallback=0;cap->scratch_bytes=0;cap->concurrent_worker_scratch_bytes=0;return 1;}
static int pools_unchanged(Nprg2Pool pools[7],const uint8_t*pool1_before){
    if(memcmp(pools[1].base,pool1_before,POOL1_BYTES)){fprintf(stderr,"pool1 changed during preflight\\n");return 0;}
    for(unsigned p=0;p<7;p++)if(p!=1&&pools[p].bytes)for(uint32_t i=0;i<pools[p].bytes;i++)if(pools[p].base[i]){fprintf(stderr,"unexpected write pool=%u byte=%lu\\n",p,(unsigned long)i);return 0;}
    return 1;
}
static int run_case(const char*model,uint32_t cmd,uint32_t seq,uint32_t src_off,uint32_t dst_off){
    char path[512];uint8_t*q=malloc(2u*1024u*1024u),*raw=malloc(512u*1024u),*source=malloc(128),*wanted=malloc(64),*pool1_before=malloc(POOL1_BYTES),*qsha=malloc(32),*nsha=malloc(32);
    CHECK(q&&raw&&source&&wanted&&pool1_before&&qsha&&nsha);
    snprintf(path,sizeof path,"hexecutor/native/build/resident-%s.qdrx",model);FILE*f=fopen(path,"rb");CHECK(f&&fread(q,1,2u*1024u*1024u,f)>256u);long qsize=ftell(f);CHECK(qsize>0&&qsize<=2L*1024L*1024L);fclose(f);
    snprintf(path,sizeof path,"depth-model/deployment/programs/%s-v2.nprg",model);f=fopen(path,"rb");CHECK(f&&fread(raw,1,512u*1024u,f)>256u);long nsize=ftell(f);CHECK(nsize>0&&nsize<=512L*1024L);fclose(f);
    snprintf(path,sizeof path,"frame/native_quantize/fixtures/%s-smoke.f32",model);CHECK(read_exact(path,source,128));
    snprintf(path,sizeof path,"frame/native_quantize/fixtures/%s-smoke.u16",model);CHECK(read_exact(path,wanted,64));
    snprintf(path,sizeof path,"frame/native_quantize/fixtures/%s-smoke.pool1",model);CHECK(read_exact(path,pool1_before,POOL1_BYTES));
    Nprg2Pool pools[7]={0};for(unsigned p=0;p<7;p++){pools[p].bytes=rd32(raw+144+4*p);if(!pools[p].bytes)continue;pools[p].base=calloc(1,pools[p].bytes);CHECK(pools[p].base);}
    CHECK(pools[1].bytes==POOL1_BYTES);memcpy(pools[1].base,pool1_before,POOL1_BYTES);CHECK(!memcmp(pools[1].base+src_off,source,128));
    nprg2_sha256(q,(uint32_t)qsize,qsha);nprg2_sha256(raw,(uint32_t)nsize,nsha);Nprg2ExpectedHashes hashes={.context_sha256=q+32,.constants_sha256=q+64,.source_sha256=qsha,.prepared_sha256=nsha};
    Nprg2Program program;CHECK(nprg2_open(&program,raw,(uint32_t)nsize,&hashes,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t*hot=calloc(1,300000u);CHECK(hot);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000u,.l2_available_bytes=786432u,.pool1_is_verified_vtcm=1,.resolve=resolve_all};Nprg2Bound bound;CHECK(nprg2_bind_mock(&program,pools,&bc,&bound)==NPRG2_OK);
    Nprg2Command command;CHECK(nprg2_command_at(&bound,cmd,&command)==NPRG2_OK&&command.opcode==QDR_QUANTIZE&&command.variant==48u&&command.sequence==seq);
    Nprg2Executor executor;nprg2_executor_init(&executor);CHECK(nprg2_executor_register_preprocess(&executor)==NPRG2_EXEC_OK);
    Nprg2ExecRequest request={.program=&program,.bound=&bound,.pool1_writable_bytes=ACTIVE_BYTES,.command_first=cmd,.command_count=1};memcpy(request.pools,pools,sizeof pools);
    CHECK(nprg2_executor_preflight(&executor,&request)==NPRG2_EXEC_OK);CHECK(pools_unchanged(pools,pool1_before));
    uint8_t*command_bytes=(uint8_t*)bound.command.data+(size_t)cmd*NPRG2_COMMAND_BYTES;uint8_t old_variant=command_bytes[3];command_bytes[3]=49u;CHECK(nprg2_executor_preflight(&executor,&request)==NPRG2_EXEC_UNSUPPORTED);CHECK(pools_unchanged(pools,pool1_before));command_bytes[3]=old_variant;
    uint8_t*call=(uint8_t*)bound.calls.data+(size_t)command.call_first*88u;uint64_t old_arg1=(uint64_t)call[16]|(uint64_t)call[17]<<8|(uint64_t)call[18]<<16|(uint64_t)call[19]<<24|(uint64_t)call[20]<<32|(uint64_t)call[21]<<40|(uint64_t)call[22]<<48|(uint64_t)call[23]<<56;wr32(call+16,32767u);memset(call+20,0,4);CHECK(nprg2_executor_preflight(&executor,&request)==NPRG2_EXEC_REJECTED);CHECK(pools_unchanged(pools,pool1_before));for(unsigned i=0;i<8;i++)call[16+i]=(uint8_t)(old_arg1>>(8*i));
    Nprg2ExecCall exec_call={.command_index=cmd,.command=command,.program=&program,.bound=&bound,.pools=pools,.pool1_writable_bytes=ACTIVE_BYTES};uint32_t xi,yi;Nprg2TensorView x,y;CHECK(nprg2_exec_input(&exec_call,0,&xi,&x)==NPRG2_EXEC_OK&&nprg2_exec_output(&exec_call,0,&yi,&y)==NPRG2_EXEC_OK);
    uint8_t*out_rec=(uint8_t*)bound.tensor.data+(size_t)yi*NPRG2_TENSOR_BYTES;uint32_t old_dst=rd32(out_rec+4);wr32(out_rec+4,src_off);CHECK(nprg2_executor_preflight(&executor,&request)==NPRG2_EXEC_REJECTED);CHECK(pools_unchanged(pools,pool1_before));wr32(out_rec+4,old_dst);
    uint32_t saved_source_word=rd32(pools[1].base+src_off);wr32(pools[1].base+src_off,0x7fc00000u);CHECK(nprg2_executor_preflight(&executor,&request)==NPRG2_EXEC_OK);CHECK(nprg2_executor_run(&executor,&request)==NPRG2_EXEC_HANDLER_ERROR);CHECK(!memcmp(pools[1].base,pool1_before,src_off));CHECK(!memcmp(pools[1].base+src_off+4u,pool1_before+src_off+4u,POOL1_BYTES-src_off-4u));wr32(pools[1].base+src_off,saved_source_word);
    CHECK(nprg2_executor_preflight(&executor,&request)==NPRG2_EXEC_OK);CHECK(nprg2_executor_run(&executor,&request)==NPRG2_EXEC_OK);CHECK(!memcmp(pools[1].base+dst_off,wanted,64));
    CHECK(!memcmp(pools[1].base,pool1_before,dst_off));CHECK(!memcmp(pools[1].base+dst_off+64,pool1_before+dst_off+64,POOL1_BYTES-dst_off-64));CHECK(!memcmp(pools[1].base+src_off,source,128));
    for(unsigned p=0;p<7;p++)if(p!=1&&pools[p].bytes)for(uint32_t i=0;i<pools[p].bytes;i++)CHECK(pools[p].base[i]==0);
    printf("PASS prepared Quantize CPU-prefix %s: opcode=28 variant=48 command=%lu seq=%lu exact 32-value output, pool1 outside output and all other pools preserved; variant/call/alias/nonfinite fail closed\n",model,(unsigned long)cmd,(unsigned long)seq);
    for(unsigned p=0;p<7;p++)free(pools[p].base);free(hot);nprg2_close(&program);free(q);free(raw);free(source);free(wanted);free(pool1_before);free(qsha);free(nsha);return 0;
}
int main(void){if(run_case("primary",60u,107u,1089536u,7589888u))return 1;return run_case("alternate",54u,100u,3997696u,2775040u);}
