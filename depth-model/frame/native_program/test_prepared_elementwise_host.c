/* ASan/UBSan read-only preflight proof for real LinearClip and requant NPRG metadata.
 * Numeric HVX execution remains covered by test_prepared_elementwise.c. */
#include "prepared_elementwise.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); exit(1); } } while (0)
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24; }
static uint8_t *read_file(const char *path,uint32_t *n) {
    FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long z=ftell(f);CHECK(z>0&&z<64L*1024L*1024L);CHECK(!fseek(f,0,SEEK_SET));
    uint8_t *p=malloc((size_t)z);CHECK(p&&fread(p,1,(size_t)z,f)==(size_t)z);fclose(f);*n=(uint32_t)z;return p;
}
static int resolve_all(void *ctx,const Nprg2Program *p,uint32_t i,const Nprg2Command *c,Nprg2Capability *cap) {
    (void)ctx;(void)p;(void)i;(void)c;cap->supported=1;cap->may_fallback=0;cap->scratch_bytes=0;cap->concurrent_worker_scratch_bytes=0;return 1;
}
static void run_model(const char *qpath,const char *npath,const char *fixture_path,uint8_t opcode,uint8_t variant,const char *name) {
    uint32_t qn,nn,fn;uint8_t *q=read_file(qpath,&qn),*raw=read_file(npath,&nn),*fx=read_file(fixture_path,&fn);
    CHECK(fn>=68&&!memcmp(fx,"EPCP",4)&&rd32(fx+4)==1&&rd32(fx+8)==opcode&&rd32(fx+12)==variant);
    uint8_t qsha[32],nsha[32];nprg2_sha256(q,qn,qsha);nprg2_sha256(raw,nn,nsha);
    Nprg2ExpectedHashes x={q+32,q+64,qsha,nsha};Nprg2Pool pools[7]={0};
    uint64_t need=68;for(unsigned i=0;i<7;i++){uint32_t z=rd32(fx+40+4*i);CHECK(z==rd32(raw+144+4*i));need+=z;pools[i].bytes=z;pools[i].base=z?malloc(z):NULL;CHECK(!z||pools[i].base);}
    uint32_t pages=rd32(fx+36);need+=(uint64_t)pages*2052;CHECK(need==fn);const uint8_t *cur=fx+68;
    for(unsigned i=0;i<7;i++){if(pools[i].bytes){memcpy(pools[i].base,cur,pools[i].bytes);cur+=pools[i].bytes;}}
    /* The fixture appends output pages after the complete pool snapshots. */
    Nprg2Program p;CHECK(nprg2_open(&p,raw,nn,&x,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t *hot=calloc(1,300000);CHECK(hot);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1,.resolve=resolve_all};Nprg2Bound bound;
    CHECK(nprg2_bind_mock(&p,pools,&bc,&bound)==NPRG2_OK);
    uint32_t cmd=UINT32_MAX;for(uint32_t i=0;i<p.command_count;i++){Nprg2Command c;CHECK(nprg2_command_at(&bound,i,&c)==NPRG2_OK);if(c.opcode==opcode&&c.variant==variant){cmd=i;break;}}
    CHECK(cmd!=UINT32_MAX&&rd32(fx+20)==cmd);
    uint8_t *before[7]={0};for(unsigned i=0;i<7;i++)if(pools[i].bytes){before[i]=malloc(pools[i].bytes);CHECK(before[i]);memcpy(before[i],pools[i].base,pools[i].bytes);}
    Nprg2Executor e;nprg2_executor_init(&e);CHECK(nprg2_executor_register_elementwise(&e)==NPRG2_EXEC_OK);
    void *scratch_raw=malloc(NPRG2_ELEMENTWISE_SCRATCH_BYTES+sizeof(uintptr_t));CHECK(scratch_raw);void *scratch=(void *)(((uintptr_t)scratch_raw+sizeof(uintptr_t)-1)&~(uintptr_t)(sizeof(uintptr_t)-1));
    Nprg2ExecRequest req={.program=&p,.bound=&bound,.pool1_writable_bytes=0x780000,.scratch=scratch,.scratch_bytes=NPRG2_ELEMENTWISE_SCRATCH_BYTES,.command_first=cmd,.command_count=1};memcpy(req.pools,pools,sizeof pools);
    CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_OK);
    if(opcode==QDR_REQUANT_U8){
        Nprg2Command c;CHECK(nprg2_command_at(&bound,cmd,&c)==NPRG2_OK&&c.call_count==1);
        CHECK(c.call_first<bound.calls.bytes/88u);
        uint8_t *coefficient=(uint8_t *)bound.calls.data+(size_t)c.call_first*88u+40u;
        uint8_t saved=*coefficient;*coefficient=1u;
        CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_REJECTED);
        *coefficient=saved;
    }
    for(unsigned i=0;i<7;i++)if(pools[i].bytes)CHECK(!memcmp(before[i],pools[i].base,pools[i].bytes));
    printf("PASS ASan/UBSan %s actual-metadata preflight: %s command=%u\n",name,qpath,cmd);
    free(scratch_raw);for(unsigned i=0;i<7;i++){free(before[i]);free(pools[i].base);}free(hot);nprg2_close(&p);free(fx);free(q);free(raw);
}
int main(void) {
    run_model("frame/build/resident-primary.qdrx","frame/native_program/examples/primary-v2.nprg","frame/native_program/fixtures/primary-smoke-linearclip.epcpu",QDR_LINEARCLIP,68,"LinearClip");
    run_model("frame/build/resident-alternate.qdrx","frame/native_program/examples/alternate-v2.nprg","frame/native_program/fixtures/alternate-smoke-linearclip.epcpu",QDR_LINEARCLIP,68,"LinearClip");
    run_model("frame/build/resident-primary.qdrx","frame/native_program/examples/primary-v2.nprg","frame/native_program/fixtures/primary-smoke-requant.epcpu",QDR_REQUANT_U8,70,"requant u8");
    run_model("frame/build/resident-alternate.qdrx","frame/native_program/examples/alternate-v2.nprg","frame/native_program/fixtures/alternate-smoke-requant.epcpu",QDR_REQUANT_U8,70,"requant u8");
    return 0;
}
