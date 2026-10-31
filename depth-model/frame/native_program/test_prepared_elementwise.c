#include "prepared_elementwise.h"

#include <hexagon_standalone.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); exit(1); } } while (0)
static uint16_t rd16(const uint8_t *p){return (uint16_t)p[0]|(uint16_t)p[1]<<8;}
static uint32_t rd32(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static int32_t rdi16(const uint8_t *p){return (int16_t)rd16(p);}
static uint8_t *read_file(const char *path,uint32_t *n){FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long z=ftell(f);CHECK(z>0&&(uint64_t)z<=32u*1024u*1024u);CHECK(!fseek(f,0,SEEK_SET));uint8_t *p=malloc((size_t)z);CHECK(p&&fread(p,1,(size_t)z,f)==(size_t)z);fclose(f);*n=(uint32_t)z;return p;}
static void pool_alloc(const uint8_t *raw,Nprg2Pool p[7]){for(uint32_t i=0;i<7;i++){p[i].bytes=rd32(raw+144+i*4);p[i].base=!p[i].bytes?NULL:(i==1?aligned_alloc(128,p[i].bytes):malloc(p[i].bytes));CHECK(!p[i].bytes||p[i].base);if(p[i].bytes)memset(p[i].base,0,p[i].bytes);}}
static void pool_free(Nprg2Pool p[7]){for(int i=0;i<7;i++)free(p[i].base);}
static int resolve_all(void *ctx,const Nprg2Program *p,uint32_t i,const Nprg2Command *c,Nprg2Capability *cap){(void)ctx;(void)p;(void)i;(void)c;cap->supported=1;cap->may_fallback=0;cap->scratch_bytes=0;cap->concurrent_worker_scratch_bytes=0;return 1;}
static void shape(const Nprg2TensorView *v,int32_t d[4],int32_t m[4],int32_t o[4]){for(int i=0;i<4;i++){d[i]=rdi16(v->geometry_record+6+2*i);m[i]=rdi16(v->geometry_record+14+2*i);o[i]=rdi16(v->geometry_record+22+2*i);}}
static uint32_t first_touched_index(const Nprg2TensorView *v,int width){int32_t d[4],m[4],o[4];shape(v,d,m,o);uint32_t nbh=(uint32_t)(m[1]+7)/8u,nbw=(uint32_t)(m[2]+width-1)/width,nbd=(uint32_t)(m[3]+31)/32;return (uint32_t)((((uint64_t)o[0]*nbh+(uint32_t)o[1]/8u)*nbw+(uint32_t)o[2]/width)*nbd+v->slice_origin+(uint32_t)o[3]/32u);}
static int first_command(const Nprg2Program *p,const Nprg2Bound *b,uint8_t op,uint8_t variant){for(uint32_t i=0;i<p->command_count;i++){Nprg2Command c;CHECK(nprg2_command_at(b,i,&c)==NPRG2_OK);if(c.opcode==op&&c.variant==variant)return (int)i;}return -1;}
static void run_model(const char *qpath,const char *npath,const char *add_fixture,const char *clip_fixture){
    uint32_t qn,nn;uint8_t *q=read_file(qpath,&qn),*raw=read_file(npath,&nn),qsha[32],nsha[32];nprg2_sha256(q,qn,qsha);nprg2_sha256(raw,nn,nsha);Nprg2ExpectedHashes x={q+32,q+64,qsha,nsha};
    Nprg2Pool pools[7]={0};pool_alloc(raw,pools);Nprg2Program p;CHECK(nprg2_open(&p,raw,nn,&x,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t *hot=calloc(1,300000);CHECK(hot);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1,.resolve=resolve_all};Nprg2Bound b;CHECK(nprg2_bind_mock(&p,pools,&bc,&b)==NPRG2_OK);
    Nprg2Executor e;nprg2_executor_init(&e);CHECK(nprg2_executor_register_elementwise(&e)==NPRG2_EXEC_OK);
    uint8_t *scratch=malloc(NPRG2_ELEMENTWISE_SCRATCH_BYTES+sizeof(uintptr_t));CHECK(scratch);uintptr_t *aligned=(uintptr_t *)(((uintptr_t)scratch+sizeof(uintptr_t)-1u)&~(uintptr_t)(sizeof(uintptr_t)-1u));
    Nprg2ExecRequest r={.program=&p,.bound=&b,.pool1_writable_bytes=0x780000u,.scratch=aligned,.scratch_bytes=NPRG2_ELEMENTWISE_SCRATCH_BYTES};memcpy(r.pools,pools,sizeof pools);
    Nprg2ExecStats stats;r.stats=&stats;
    int cmds[2]={first_command(&p,&b,QDR_ADD_U8,13),first_command(&p,&b,QDR_LINEARCLIP,68)};CHECK(cmds[0]>=0&&cmds[1]>=0);
    for(int which=0;which<2;which++){
        uint32_t first=(uint32_t)cmds[which];Nprg2Command c;CHECK(nprg2_command_at(&b,first,&c)==NPRG2_OK);r.command_first=first;r.command_count=1;
        uint32_t fixture_bytes;uint8_t *fixture=read_file(which?clip_fixture:add_fixture,&fixture_bytes);
        CHECK(fixture_bytes>=68u&&!memcmp(fixture,"EPCP",4)&&rd32(fixture+4)==1u);
        CHECK(rd32(fixture+8)==c.opcode&&rd32(fixture+12)==c.variant&&rd32(fixture+16)==c.sequence&&rd32(fixture+20)==first);
        uint32_t fi_pool=rd32(fixture+24),fi_offset=rd32(fixture+28),fi_bytes=rd32(fixture+32),page_count=rd32(fixture+36);
        uint8_t *want_pool[7]={0};uint64_t need=68ull+(uint64_t)page_count*2052u;for(uint32_t pi=0;pi<7;pi++){uint32_t size=rd32(fixture+40+4u*pi);CHECK(size==pools[pi].bytes);need+=size;}
        CHECK(need==fixture_bytes);const uint8_t *cursor=fixture+68;
        for(uint32_t pi=0;pi<7;pi++){uint32_t size=pools[pi].bytes;if(size){memcpy(pools[pi].base,cursor,size);if(pi==1){want_pool[pi]=malloc(size);CHECK(want_pool[pi]);memcpy(want_pool[pi],cursor,size);}else want_pool[pi]=(uint8_t*)cursor;}cursor+=size;}
        uint32_t pool1_bytes=pools[1].bytes;uint8_t *want=want_pool[1];CHECK(want);
        if(fi_pool!=UINT32_MAX){CHECK(fi_pool<7&&pools[fi_pool].base&&fi_bytes==48u&&fi_offset<=pools[fi_pool].bytes&&fi_bytes<=pools[fi_pool].bytes-fi_offset);}else CHECK(!fi_bytes);
        for(uint32_t page=0;page<page_count;page++){uint32_t off=rd32(cursor);cursor+=4;CHECK(off<=pool1_bytes&&2048u<=pool1_bytes-off);memcpy(want+off,cursor,2048u);cursor+=2048u;}
        CHECK(nprg2_executor_preflight(&e,&r)==NPRG2_EXEC_OK);
        Nprg2ExecCall call={.command_index=first,.command=c,.program=&p,.bound=&b,.pools=pools,.pool1_writable_bytes=r.pool1_writable_bytes,.scratch=aligned,.scratch_bytes=NPRG2_ELEMENTWISE_SCRATCH_BYTES};
        CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);
        for(uint32_t pi=0;pi<7;pi++)if(pools[pi].bytes&&memcmp(pools[pi].base,want_pool[pi],pools[pi].bytes)){uint32_t i=0;while(i<pools[pi].bytes&&pools[pi].base[i]==want_pool[pi][i])i++;fprintf(stderr,"CPU-prefix physical pool mismatch model=%s which=%d pool=%lu cmd=%lu seq=%lu off=%lu got=%u cpu=%u\n",qpath,which,(unsigned long)pi,(unsigned long)first,(unsigned long)c.sequence,(unsigned long)i,pools[pi].base[i],want_pool[pi][i]);exit(1);}
        /* Corrupt the output's first encoded page to overlap an input page.
         * Full preflight must reject and leave the entire VTCM pool intact. */
        uint32_t oi,ii;Nprg2TensorView ov,iv;CHECK(!nprg2_exec_output(&call,0,&oi,&ov));CHECK(!nprg2_exec_input(&call,0,&ii,&iv));
        CHECK(ov.block_count&&iv.block_count);uint32_t oi0=first_touched_index(&ov,which?4:8),ii0=first_touched_index(&iv,8);CHECK(oi0<ov.block_count&&ii0<iv.block_count);uint32_t *out_blocks=(uint32_t *)ov.blocks;uint32_t old=out_blocks[oi0];out_blocks[oi0]=iv.blocks[ii0];
        CHECK(nprg2_executor_preflight(&e,&r)==NPRG2_EXEC_REJECTED);for(uint32_t pi=0;pi<7;pi++)if(pools[pi].bytes)CHECK(!memcmp(pools[pi].base,want_pool[pi],pools[pi].bytes));out_blocks[oi0]=old;
        out_blocks[oi0]=0x780001u;CHECK(nprg2_executor_preflight(&e,&r)==NPRG2_EXEC_REJECTED);for(uint32_t pi=0;pi<7;pi++)if(pools[pi].bytes)CHECK(!memcmp(pools[pi].base,want_pool[pi],pools[pi].bytes));out_blocks[oi0]=old;
        free(want_pool[1]);free(fixture);
    }
    /* Same opcode, unsupported LinearClip variant must fail closed. */
    int unsupported=first_command(&p,&b,QDR_LINEARCLIP,67);if(unsupported>=0){r.command_first=(uint32_t)unsupported;r.command_count=1;CHECK(nprg2_executor_preflight(&e,&r)==NPRG2_EXEC_UNSUPPORTED);}
    printf("PASS prepared blocked elementwise captured CPU prefix: %s\n",qpath);
    free(scratch);free(hot);nprg2_close(&p);pool_free(pools);free(q);free(raw);
}
int main(int argc,char **argv){
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    if(argc==1){run_model("frame/build/resident-primary.qdrx","frame/native_program/examples/primary-v2.nprg","frame/native_program/fixtures/primary-smoke-add.epcpu","frame/native_program/fixtures/primary-smoke-linearclip.epcpu");run_model("frame/build/resident-alternate.qdrx","frame/native_program/examples/alternate-v2.nprg","frame/native_program/fixtures/alternate-smoke-add.epcpu","frame/native_program/fixtures/alternate-smoke-linearclip.epcpu");SIM_RELEASE_HVX;return 0;}
    CHECK(argc==9);run_model(argv[1],argv[2],argv[3],argv[4]);run_model(argv[5],argv[6],argv[7],argv[8]);SIM_RELEASE_HVX;return 0;
}
