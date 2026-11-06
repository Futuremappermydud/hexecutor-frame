#include "prepared_mul_flat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);exit(1);} }while(0)
static uint32_t rd32(const uint8_t*p){return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
static uint8_t*file(const char*p,uint32_t*n){FILE*f=fopen(p,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long z=ftell(f);CHECK(z>0);CHECK(!fseek(f,0,SEEK_SET));uint8_t*b=malloc((size_t)z);CHECK(b&&fread(b,1,(size_t)z,f)==(size_t)z);fclose(f);*n=(uint32_t)z;return b;}
static int resolve(void*x,const Nprg2Program*p,uint32_t i,const Nprg2Command*c,Nprg2Capability*k){(void)x;(void)p;(void)i;(void)c;k->supported=1;k->may_fallback=0;k->scratch_bytes=0;k->concurrent_worker_scratch_bytes=0;return 1;}
static void check_case(const char*model,const char*fixture){
 char qp[160],rp[160];CHECK(!strcmp(model,"primary")||!strcmp(model,"alternate"));snprintf(qp,sizeof qp,"frame/build/resident-%s.qdrx",model);snprintf(rp,sizeof rp,"frame/native_program/examples/%s-v2.nprg",model);
 uint32_t qn,pn,fn;uint8_t*q=file(qp,&qn),*raw=file(rp,&pn),*fx=file(fixture,&fn),qsha[32],psha[32];
 nprg2_sha256(q,qn,qsha);nprg2_sha256(raw,pn,psha);Nprg2ExpectedHashes x={q+32,q+64,qsha,psha};
 Nprg2Pool pools[7]={0};for(uint32_t i=0;i<7;i++){pools[i].bytes=rd32(raw+144+4*i);pools[i].base=pools[i].bytes?malloc(pools[i].bytes):NULL;CHECK(!pools[i].bytes||pools[i].base);if(pools[i].bytes)memset(pools[i].base,0,pools[i].bytes);}
 Nprg2Program prog;CHECK(nprg2_open(&prog,raw,pn,&x,pools,(Nprg2Allocator){0})==NPRG2_OK);uint8_t*hot=calloc(1,300000);CHECK(hot);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1,.resolve=resolve};Nprg2Bound bound;CHECK(nprg2_bind_mock(&prog,pools,&bc,&bound)==NPRG2_OK);
 CHECK(fn>=68&&!memcmp(fx,"EPCP",4)&&rd32(fx+4)==1&&rd32(fx+8)==41&&rd32(fx+12)==75);
 uint32_t command_index=rd32(fx+20);CHECK(command_index<prog.command_count);
 uint32_t pages=rd32(fx+36);CHECK(pages==1);uint64_t need=68ull+(uint64_t)pages*2052;for(uint32_t i=0;i<7;i++){CHECK(rd32(fx+40+4*i)==pools[i].bytes);need+=pools[i].bytes;}CHECK(need==fn);
 const uint8_t*cur=fx+68;uint8_t*want[7]={0};for(uint32_t i=0;i<7;i++){if(pools[i].bytes){want[i]=malloc(pools[i].bytes);CHECK(want[i]);memcpy(want[i],cur,pools[i].bytes);memcpy(pools[i].base,cur,pools[i].bytes);}cur+=pools[i].bytes;}
 uint32_t off=rd32(cur);cur+=4;CHECK(off+2048<=pools[1].bytes);memcpy(want[1]+off,cur,2048);
 Nprg2Executor e;nprg2_executor_init(&e);CHECK(nprg2_executor_register_mul_flat(&e)==NPRG2_EXEC_OK);Nprg2ExecRequest r={.program=&prog,.bound=&bound,.pool1_writable_bytes=0x780000,.command_first=command_index,.command_count=1};memcpy(r.pools,pools,sizeof pools);
 CHECK(nprg2_executor_preflight(&e,&r)==NPRG2_EXEC_OK);CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);
 for(uint32_t i=0;i<7;i++)if(pools[i].bytes)CHECK(!memcmp(pools[i].base,want[i],pools[i].bytes));
 /* Wrong scalar dtype is rejected without changing any physical pool. */
 Nprg2Command cmd;CHECK(nprg2_command_at(&bound,command_index,&cmd)==NPRG2_OK);Nprg2ExecCall call={.command_index=command_index,.command=cmd,.program=&prog,.bound=&bound,.pools=pools,.pool1_writable_bytes=0x780000};uint32_t ti;Nprg2TensorView scalar;CHECK(nprg2_exec_input(&call,1,&ti,&scalar)==NPRG2_EXEC_OK);uint8_t*sg=(uint8_t*)scalar.geometry_record;uint8_t dtype=sg[0];sg[0]=5;CHECK(nprg2_executor_preflight(&e,&r)==NPRG2_EXEC_REJECTED);sg[0]=dtype;for(uint32_t i=0;i<7;i++)if(pools[i].bytes)CHECK(!memcmp(pools[i].base,want[i],pools[i].bytes));
 printf("prepared_mul_flat %s %s CPU-prefix physical-pool comparison passed\n",model,fixture);
 for(int i=0;i<7;i++){free(pools[i].base);free(want[i]);}free(hot);nprg2_close(&prog);free(q);free(raw);free(fx);
}
int main(void){
 check_case("alternate","frame/native_program/fixtures/alternate-smoke-mul-flat-110.epcpu");
 check_case("alternate","frame/native_program/fixtures/alternate-smoke-mul-flat-129.epcpu");
 check_case("primary","frame/native_program/fixtures/primary-smoke-mul-flat-158.epcpu");
 check_case("primary","frame/native_program/fixtures/primary-smoke-mul-flat-181.epcpu");
 return 0;
}
