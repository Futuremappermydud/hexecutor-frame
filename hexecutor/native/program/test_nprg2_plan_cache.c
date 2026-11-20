/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "nprg2_plan_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);exit(1);}}while(0)
static uint32_t u32(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint8_t *read_file(const char *path,uint32_t *n){FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long z=ftell(f);CHECK(z>0&&(uint64_t)z<=NPRG2_MAX_IMAGE_BYTES);CHECK(!fseek(f,0,SEEK_SET));uint8_t *b=malloc((size_t)z);CHECK(b&&fread(b,1,(size_t)z,f)==(size_t)z);fclose(f);*n=(uint32_t)z;return b;}
static void pools_new(const uint8_t *nprg,Nprg2Pool p[7]){for(int i=0;i<7;i++){p[i].bytes=u32(nprg+144+i*4);p[i].base=p[i].bytes?calloc(1,p[i].bytes):NULL;CHECK(!p[i].bytes||p[i].base);}}
static void pools_free(Nprg2Pool p[7]){for(int i=0;i<7;i++)free(p[i].base);}
static void hashes(const uint8_t *q,uint32_t qn,const uint8_t *n,uint32_t nn,Nprg2ExpectedHashes *h,uint8_t qsha[32],uint8_t nsha[32]){CHECK(qn>=96);nprg2_sha256(q,qn,qsha);nprg2_sha256(n,nn,nsha);h->context_sha256=q+32;h->constants_sha256=q+64;h->source_sha256=qsha;h->prepared_sha256=nsha;}
static int bind_cover(void *ctx,const Nprg2Program *p,uint32_t i,const Nprg2Command *c,Nprg2Capability *cap){(void)ctx;(void)p;(void)i;(void)c;cap->supported=1;cap->may_fallback=0;cap->scratch_bytes=0;cap->concurrent_worker_scratch_bytes=0;return 1;}

typedef struct {uint32_t writes;const Nprg2ExecBuffer *inputs,*outputs;} MockState;
static int mock_validate(const Nprg2ExecCall *c,void *ctx){MockState *s=ctx;return c&&c->command.opcode&&c->command.variant&&c->inputs==s->inputs&&c->input_count==1&&c->outputs==s->outputs&&c->output_count==1;}
static int mock_run(const Nprg2ExecCall *c,void *ctx){MockState *s=ctx;if(c->inputs!=s->inputs||c->outputs!=s->outputs)return 0;s->writes++;return 1;}

static void test_plan_cache_model(const char *qpath,const char *npath){
    uint32_t qn,nn;uint8_t *q=read_file(qpath,&qn),*raw=read_file(npath,&nn),qsha[32],nsha[32];Nprg2ExpectedHashes h;hashes(q,qn,raw,nn,&h,qsha,nsha);
    Nprg2Pool pools[7];pools_new(raw,pools);Nprg2Program p;CHECK(nprg2_open(&p,raw,nn,&h,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t *hot=calloc(1,300000);CHECK(hot);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.scratch_budget=0,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1,.resolve=bind_cover};Nprg2Bound bound;CHECK(nprg2_bind_mock(&p,pools,&bc,&bound)==NPRG2_OK);
    uint8_t io_in_data[16]={0},io_out_data[16]={0};Nprg2ExecBuffer io_in={io_in_data,sizeof io_in_data},io_out={io_out_data,sizeof io_out_data};
    Nprg2ExecRequest r={.program=&p,.bound=&bound,.pool1_writable_bytes=0x780000u,.inputs=&io_in,.input_count=1,.outputs=&io_out,.output_count=1,.user=NULL,.command_first=0,.command_count=p.command_count};memcpy(r.pools,pools,sizeof pools);
    Nprg2Executor e;nprg2_executor_init(&e);MockState state={.writes=0,.inputs=&io_in,.outputs=&io_out};

    for(uint32_t i=0;i<p.command_count;i++){
        Nprg2Command c;CHECK(nprg2_command_at(&bound,i,&c)==NPRG2_OK);
        int seen=0;
        for(uint32_t j=0;j<NPRG2_EXEC_REGISTRY_CAPACITY;j++)
            if(e.handlers[j].used&&e.handlers[j].opcode==c.opcode&&e.handlers[j].variant==c.variant)seen=1;
        if(!seen)CHECK(nprg2_executor_register(&e,c.opcode,c.variant,mock_validate,mock_run,&state)==NPRG2_EXEC_OK);
    }

    size_t arena_sz = nprg2_plan_required_bytes(p.command_count);
    CHECK(arena_sz <= 5248);
    uint8_t *arena = aligned_alloc(128, 5248);
    CHECK(arena);

    /* Test build failures */
    Nprg2PlanCache cache;
    CHECK(nprg2_plan_build_from_request(NULL, &r, arena, arena_sz, &cache) == NPRG2_PLAN_ARGUMENT);
    CHECK(nprg2_plan_build_from_request(&e, &r, arena, arena_sz - 1, &cache) == NPRG2_PLAN_CAPACITY);

    int rc = nprg2_plan_build_from_request(&e, &r, arena, 5248, &cache);
    if (rc != NPRG2_PLAN_OK) {
        fprintf(stderr, "nprg2_plan_build_from_request returned %d\n", rc);
    }
    CHECK(rc == NPRG2_PLAN_OK);
    CHECK(cache.ready == 1);
    CHECK(cache.command_count == p.command_count);
    CHECK(cache.header->magic == NPRG2_PLAN_MAGIC);
    CHECK(cache.header->version == NPRG2_PLAN_VERSION);
    CHECK(cache.header->header_bytes == 128);
    CHECK(cache.header->total_bytes == (uint32_t)arena_sz);

    /* Run standard executor preflight and run */
    Nprg2ExecStats std_stats = {0};
    r.stats = &std_stats;
    state.writes = 0;
    CHECK(nprg2_executor_run(&e, &r) == NPRG2_EXEC_OK);
    CHECK(std_stats.executed_commands == p.command_count);
    CHECK(state.writes == p.command_count);

    /* Run cached executor preflight and run */
    Nprg2ExecStats cached_stats = {0};
    r.stats = &cached_stats;
    state.writes = 0;
    CHECK(nprg2_executor_preflight_cached(&e, &cache, &r) == NPRG2_EXEC_OK);
    CHECK(cached_stats.validated_commands == p.command_count);

    CHECK(nprg2_executor_run_cached(&e, &cache, &r) == NPRG2_EXEC_OK);
    CHECK(cached_stats.executed_commands == p.command_count);
    CHECK(state.writes == p.command_count);

    /* Fail-closed checks on mutated runtime request */
    Nprg2ExecRequest bad_r = r;
    bad_r.pool1_writable_bytes = 0x780001u; /* Exceeds plan ceiling */
    CHECK(nprg2_executor_preflight_cached(&e, &cache, &bad_r) == NPRG2_EXEC_ARGUMENT);

    bad_r = r;
    bad_r.pools[1].base = (uint8_t *)0x12345678; /* Shifted pool base */
    CHECK(nprg2_executor_preflight_cached(&e, &cache, &bad_r) == NPRG2_EXEC_ARGUMENT);

    free(arena);
    free(hot);
    nprg2_close(&p);
    pools_free(pools);
    free(q);
    free(raw);
    printf("PASS nprg2_plan_cache %s\n", qpath);
}

int main(int argc, char **argv) {
    CHECK(argc == 5);
    test_plan_cache_model(argv[1], argv[2]);
    test_plan_cache_model(argv[3], argv[4]);
    return 0;
}
