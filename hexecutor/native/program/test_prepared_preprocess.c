/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_preprocess.h"
#include <hexagon_standalone.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);exit(1);}}while(0)
static uint32_t rd32(const uint8_t*p){return(uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint8_t*read_file(const char*path,uint32_t*n){FILE*f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long z=ftell(f);CHECK(z>0&&z<=64L*1024L*1024L);CHECK(!fseek(f,0,SEEK_SET));uint8_t*p=malloc((size_t)z);CHECK(p&&fread(p,1,(size_t)z,f)==(size_t)z);fclose(f);*n=(uint32_t)z;return p;}
static int resolve_all(void*ctx,const Nprg2Program*p,uint32_t i,const Nprg2Command*c,Nprg2Capability*cap){(void)ctx;(void)p;(void)i;(void)c;cap->supported=1;cap->may_fallback=0;cap->scratch_bytes=0;cap->concurrent_worker_scratch_bytes=0;return 1;}
static void pool_alloc(const uint8_t*r,Nprg2Pool p[7]){for(unsigned i=0;i<7;i++){p[i].bytes=rd32(r+144+4*i);p[i].base=p[i].bytes?malloc(p[i].bytes):NULL;CHECK(!p[i].bytes||p[i].base);if(p[i].bytes)memset(p[i].base,0,p[i].bytes);}}
static void pool_free(Nprg2Pool p[7]){for(unsigned i=0;i<7;i++)free(p[i].base);}
static void compare_pools(Nprg2Pool p[7],uint8_t*want[7],const char*tag){for(unsigned i=0;i<7;i++)if(p[i].bytes&&memcmp(p[i].base,want[i],p[i].bytes)){uint32_t j=0;while(j<p[i].bytes&&p[i].base[j]==want[i][j])j++;fprintf(stderr,"pool mismatch %s pool=%u byte=%lu\n",tag,i,(unsigned long)j);exit(1);}}
static void run_case(const char*qpath,const char*npath,const char*slice_path,const char*im2_path,int im2_expected){
    uint32_t qn,nn;uint8_t*q=read_file(qpath,&qn),*raw=read_file(npath,&nn),qsha[32],nsha[32];nprg2_sha256(q,qn,qsha);nprg2_sha256(raw,nn,nsha);Nprg2ExpectedHashes ex={q+32,q+64,qsha,nsha};
    Nprg2Pool pools[7]={0};pool_alloc(raw,pools);Nprg2Program p;CHECK(nprg2_open(&p,raw,nn,&ex,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t*hot=calloc(1,300000);CHECK(hot);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1,.resolve=resolve_all};Nprg2Bound b;CHECK(nprg2_bind_mock(&p,pools,&bc,&b)==NPRG2_OK);
    Nprg2Executor e;nprg2_executor_init(&e);CHECK(nprg2_executor_register_preprocess(&e)==NPRG2_EXEC_OK);
    const char*paths[2]={slice_path,im2_path};const uint8_t ops[2]={QDR_INPUT_SLICE,QDR_IM2COL},vars[2]={1,24};
    for(unsigned which=0;which<2;which++){
        uint32_t fn;uint8_t*fx=read_file(paths[which],&fn);CHECK(fn>=64&&!memcmp(fx,"PPFX",4)&&rd32(fx+4)==1&&rd32(fx+8)==ops[which]&&rd32(fx+12)==vars[which]);
        uint32_t cmd=rd32(fx+20),io=rd32(fx+24),io_bytes=rd32(fx+28),patches=rd32(fx+32);Nprg2Command c;CHECK(nprg2_command_at(&b,cmd,&c)==NPRG2_OK);CHECK(c.opcode==ops[which]&&c.variant==vars[which]&&c.sequence==rd32(fx+16));
        uint8_t*want[7]={0};const uint8_t*cur=fx+64;
        for(unsigned i=0;i<7;i++){uint32_t z=rd32(fx+36+4*i);CHECK(z==pools[i].bytes);if(z){memcpy(pools[i].base,cur,z);want[i]=malloc(z);CHECK(want[i]);memcpy(want[i],cur,z);cur+=z;}}
        CHECK(io_bytes<=fn-(uint32_t)(cur-fx));Nprg2ExecBuffer inputs[4]={{0}};if(io_bytes){CHECK(io==3);inputs[3].data=(uint8_t*)cur;inputs[3].bytes=io_bytes;cur+=io_bytes;}
        for(uint32_t z=0;z<patches;z++){CHECK((uint64_t)(cur-fx)+8<=fn);uint32_t off=rd32(cur),n=rd32(cur+4);cur+=8;CHECK(n&&off<=pools[1].bytes&&n<=pools[1].bytes-off&&(uint64_t)(cur-fx)+n<=fn);memcpy(want[1]+off,cur,n);cur+=n;}
        CHECK((uint32_t)(cur-fx)==fn);
        Nprg2ExecRequest req={.program=&p,.bound=&b,.pool1_writable_bytes=0x780000,.inputs=inputs,.input_count=4,.command_first=cmd,.command_count=1};memcpy(req.pools,pools,sizeof pools);
        int pre=nprg2_executor_preflight(&e,&req);if(which==1&&!im2_expected){CHECK(pre==NPRG2_EXEC_REJECTED);compare_pools(pools,want,"im2col unsupported alt");printf("PASS Im2col fail-closed duplicate-page profile %s cmd=%lu\n",qpath,(unsigned long)cmd);}
        else {CHECK(pre==NPRG2_EXEC_OK);CHECK(nprg2_executor_run(&e,&req)==NPRG2_EXEC_OK);compare_pools(pools,want,which?"im2col CPU oracle":"input slice CPU oracle");printf("PASS prepared InputSlice/Im2col CPU-prefix %s opcode=%u cmd=%lu patches=%lu\n",qpath,ops[which],(unsigned long)cmd,(unsigned long)patches);}
        if(!which){Nprg2ExecCall call={.command_index=cmd,.command=c,.program=&p,.bound=&b,.pools=pools,.pool1_writable_bytes=req.pool1_writable_bytes,.inputs=inputs,.input_count=4};uint32_t tid;Nprg2TensorView o;CHECK(nprg2_exec_output(&call,0,&tid,&o)==NPRG2_EXEC_OK);(void)tid;inputs[3].data=pools[1].base+o.offset;inputs[3].bytes=io_bytes;memcpy(pools[1].base,want[1],pools[1].bytes);CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_ARGUMENT);compare_pools(pools,want,"InputSlice alias rejection");inputs[3].data=(uint8_t*)cur-io_bytes;}
        if(which==1&&im2_expected){Nprg2ExecCall call={.command_index=cmd,.command=c,.program=&p,.bound=&b,.pools=pools,.pool1_writable_bytes=req.pool1_writable_bytes};uint32_t tid;Nprg2TensorView o;CHECK(nprg2_exec_output(&call,0,&tid,&o)==NPRG2_EXEC_OK);(void)tid;uint32_t *blocks=(uint32_t*)o.blocks,old=blocks[o.slice_origin];blocks[o.slice_origin]=0x780001u;memcpy(pools[1].base,want[1],pools[1].bytes);CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_REJECTED);compare_pools(pools,want,"Im2col tail rejection");blocks[o.slice_origin]=old;}
        for(unsigned i=0;i<7;i++)free(want[i]);free(fx);
    }
    free(hot);nprg2_close(&p);pool_free(pools);free(q);free(raw);
}
int main(void){SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;run_case("hexecutor/native/build/resident-primary.qdrx","depth-model/deployment/programs/primary-v2.nprg","hexecutor/native/program/fixtures/primary-smoke-op3.ppcpu","hexecutor/native/program/fixtures/primary-smoke-op5.ppcpu",1);run_case("hexecutor/native/build/resident-alternate.qdrx","depth-model/deployment/programs/alternate-v2.nprg","hexecutor/native/program/fixtures/alternate-smoke-op3.ppcpu","hexecutor/native/program/fixtures/alternate-smoke-op5.ppcpu",1);SIM_RELEASE_HVX;return 0;}
