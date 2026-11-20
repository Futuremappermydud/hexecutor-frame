/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_spill_fill.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

static void wr32(uint8_t *p, uint32_t v) {
    p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); p[2]=(uint8_t)(v>>16); p[3]=(uint8_t)(v>>24);
}

static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}

static uint8_t *read_file(const char *path, uint32_t *bytes) {
    FILE *f=fopen(path,"rb"); CHECK(f);
    CHECK(!fseek(f,0,SEEK_END)); long n=ftell(f);
    CHECK(n>0 && (uint64_t)n<=NPRG2_MAX_IMAGE_BYTES && !fseek(f,0,SEEK_SET));
    uint8_t *p=malloc((size_t)n); CHECK(p && fread(p,1,(size_t)n,f)==(size_t)n);
    fclose(f); *bytes=(uint32_t)n; return p;
}

static int resolve_all(void *ctx,const Nprg2Program *p,uint32_t i,
                       const Nprg2Command *cmd,Nprg2Capability *cap) {
    (void)ctx;(void)p;(void)i;(void)cmd;
    cap->supported=1;cap->may_fallback=0;
    cap->scratch_bytes=cap->concurrent_worker_scratch_bytes=0;return 1;
}

static void oracle(uint8_t *const pool[7],const Nprg2Pool pools[7],
                   const Nprg2Command *cmd,const uint8_t *v) {
    uint64_t spill_offset=rd32(v+12);
    for(uint32_t i=4;i<cmd->vector_count;i+=2) {
        uint32_t encoded=rd32(v+(size_t)i*4u),n=rd32(v+(size_t)(i+1u)*4u);
        uint32_t p=encoded&0x7ffu,addr=encoded&~0x7ffu;
        if(cmd->variant==8)memmove(pool[p]+addr,pool[6]+(uint32_t)spill_offset,n);
        else memmove(pool[6]+(uint32_t)spill_offset,pool[p]+addr,n);
        spill_offset+=n;
    }
    (void)pools;
}

static int pools_equal(const Nprg2Pool pools[7],uint8_t *const expected[7]) {
    for(unsigned i=0;i<7;i++)if(pools[i].bytes&&memcmp(pools[i].base,expected[i],pools[i].bytes))return 0;
    return 1;
}

static void run_image(const char *qpath,const char *npath,uint32_t want_fill,uint32_t want_spill) {
    uint32_t qb,nb;uint8_t *q=read_file(qpath,&qb),*raw=read_file(npath,&nb);
    CHECK(qb>=96);uint8_t qsha[32],nsha[32];nprg2_sha256(q,qb,qsha);nprg2_sha256(raw,nb,nsha);
    Nprg2ExpectedHashes hashes={q+32,q+64,qsha,nsha};Nprg2Pool pools[7]={0};
    for(unsigned i=0;i<7;i++){pools[i].bytes=rd32(raw+144u+4u*i);pools[i].base=pools[i].bytes?malloc(pools[i].bytes):NULL;CHECK(!pools[i].bytes||pools[i].base);if(pools[i].bytes)memset(pools[i].base,(int)(i+29u),pools[i].bytes);}
    Nprg2Program program;CHECK(nprg2_open(&program,raw,nb,&hashes,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t *hot=calloc(1,300000u);CHECK(hot);Nprg2BindConfig cfg={.hot_memory=hot,.hot_capacity=300000u,.l2_available_bytes=786432u,.pool1_is_verified_vtcm=1,.resolve=resolve_all};
    Nprg2Bound bound;CHECK(nprg2_bind_mock(&program,pools,&cfg,&bound)==NPRG2_OK);
    Nprg2Executor e;nprg2_executor_init(&e);CHECK(nprg2_executor_register_spill_fill(&e)==NPRG2_EXEC_OK);
    uint8_t *expected[7]={0};for(unsigned i=0;i<7;i++)if(pools[i].bytes){expected[i]=malloc(pools[i].bytes);CHECK(expected[i]);for(uint32_t j=0;j<pools[i].bytes;j++)pools[i].base[j]=expected[i][j]=(uint8_t)(j*31u+i*17u+j/2048u);}
    CHECK(pools[1].bytes>=0x780000u);
    Nprg2ExecRequest req={.program=&program,.bound=&bound,.pool1_writable_bytes=0x780000u,.command_count=1};memcpy(req.pools,pools,sizeof pools);
    uint32_t fills=0,spills=0;
    for(uint32_t ci=0;ci<program.command_count;ci++) {
        Nprg2Command cmd;CHECK(nprg2_command_at(&bound,ci,&cmd)==NPRG2_OK);
        if(cmd.opcode!=26)continue;
        CHECK(cmd.variant==8||cmd.variant==9);if(cmd.variant==8)fills++;else spills++;
        req.command_first=ci;
        uint8_t *v=(uint8_t*)bound.vectors.data+(size_t)cmd.vector_first*4u;
        int preflight=nprg2_executor_preflight(&e,&req);
        if(preflight!=NPRG2_EXEC_OK)fprintf(stderr,"preflight reject %s command=%u variant=%u io=%u duplicate=%u vectors=%u rc=%d\\n",qpath,ci,cmd.variant,cmd.io_index,cmd.duplicate_pad,cmd.vector_count,preflight);
        CHECK(preflight==NPRG2_EXEC_OK);
        CHECK(pools_equal(pools,expected));
        if(cmd.vector_count>=8u) {
            uint32_t at=cmd.vector_count-2u,old=rd32(v+(size_t)at*4u),encoded=rd32(v+(size_t)at*4u);
            uint32_t pool=encoded&0x7ffu;uint32_t bad=((pools[pool].bytes+2047u)&~2047u)|pool;
            wr32(v+(size_t)at*4u,bad);
            CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_REJECTED);
            CHECK(pools_equal(pools,expected));wr32(v+(size_t)at*4u,old);
        }
        oracle(expected,pools,&cmd,v);
        CHECK(nprg2_executor_run(&e,&req)==NPRG2_EXEC_OK);
        CHECK(pools_equal(pools,expected));
    }
    CHECK(fills==want_fill&&spills==want_spill);
    printf("PASS prepared spill/fill %s: fill=%u spill=%u; full-pool oracle and preflight read-only\n",qpath,fills,spills);
    for(unsigned i=0;i<7;i++){free(expected[i]);free(pools[i].base);}free(hot);nprg2_close(&program);free(q);free(raw);
}

int main(void) {
    Nprg2Executor executor;
    nprg2_executor_init(&executor);
    CHECK(nprg2_executor_register_spill_fill(&executor) == NPRG2_EXEC_OK);

    uint8_t pool1[32], pool3[32], spill[32];
    memset(pool1, 0xa1, sizeof pool1);
    memset(pool3, 0xb2, sizeof pool3);
    memset(spill, 0xc3, sizeof spill);
    for (unsigned i=0;i<8;i++) pool1[i]=(uint8_t)(i+1);
    Nprg2Pool pools[7]={{0},{pool1,sizeof pool1},{0},{pool3,sizeof pool3},{0},{0},{spill,sizeof spill}};
    Nprg2ExecBuffer inputs[1]={{0}}, outputs[1]={{0}};
    Nprg2ExecCall c;
    memset(&c,0,sizeof c);
    c.command.opcode=26; c.command.variant=9; c.command.vector_count=8;
    /* QDR stores vectors as a command-relative slice; use four vectors for
       initial spill offset and then two ordered pool/address-size pairs. */
    uint8_t vec[8 * 4]; memset(vec,0,sizeof vec);
    wr32(vec+12,4); wr32(vec+16,1); wr32(vec+20,8);
    wr32(vec+24,3u); wr32(vec+28,4);
    c.bound=&(Nprg2Bound){0};
    Nprg2Bound bound={0}; bound.vectors.data=vec; bound.vectors.bytes=sizeof vec; bound.ready=1; c.bound=&bound;
    c.pools=pools; c.pool1_writable_bytes=sizeof pool1;
    c.inputs=inputs; c.outputs=outputs;
    Nprg2ExecHandler *spill_handler=NULL,*fill_handler=NULL;
    for(unsigned i=0;i<NPRG2_EXEC_REGISTRY_CAPACITY;i++) if(executor.handlers[i].used) {
        if(executor.handlers[i].variant==9) spill_handler=&executor.handlers[i];
        if(executor.handlers[i].variant==8) fill_handler=&executor.handlers[i];
    }
    CHECK(spill_handler && fill_handler);
    CHECK(spill_handler->validate(&c,spill_handler->ctx));
    CHECK(spill_handler->run(&c,spill_handler->ctx));
    CHECK(!memcmp(spill+4,pool1,8));
    CHECK(!memcmp(spill+12,pool3,4));

    c.command.variant=8;
    CHECK(fill_handler->validate(&c,fill_handler->ctx));
    memset(pool1,0,sizeof pool1); memset(pool3,0,sizeof pool3);
    CHECK(fill_handler->run(&c,fill_handler->ctx));
    CHECK(!memcmp(pool1,spill+4,8));
    CHECK(!memcmp(pool3,spill+12,4));

    /* Spill reads may include the preserved pool-1 tail; Fill writes may not. */
    c.pool1_writable_bytes=4;
    c.command.variant=9;
    CHECK(spill_handler->validate(&c,spill_handler->ctx));
    c.command.variant=8;
    CHECK(!fill_handler->validate(&c,fill_handler->ctx));
    c.pool1_writable_bytes=sizeof pool1;

    /* Invalid second span rejects the whole command before either copy. */
    memset(pool1,0x5a,sizeof pool1); memset(spill,0x6b,sizeof spill);
    wr32(vec+16,0x1001u); /* address 0x1000 in pool 1 */
    c.command.variant=9;
    CHECK(!spill_handler->validate(&c,spill_handler->ctx));
    CHECK(pool1[0]==0x5a && spill[4]==0x6b);

    puts("PASS prepared spill/fill variants=2; ordered pool spans, direction, fail-closed bounds");
    run_image("hexecutor/native/build/resident-primary.qdrx","depth-model/deployment/programs/primary-v2.nprg",62u,26u);
    run_image("hexecutor/native/build/resident-alternate.qdrx","depth-model/deployment/programs/alternate-v2.nprg",53u,20u);
    return 0;
}
