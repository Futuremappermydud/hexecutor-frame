/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_flat_vtcm.h"
#include "prepared_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)

static uint32_t rd32fvtest(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24; }
static uint8_t *read_file(const char *path, uint32_t *n) {
    FILE *f=fopen(path,"rb"); if(!f)return NULL;
    if(fseek(f,0,SEEK_END)){fclose(f);return NULL;} long z=ftell(f);
    if(z<=0 || (uint64_t)z>32u*1024u*1024u || fseek(f,0,SEEK_SET)){fclose(f);return NULL;}
    uint8_t *p=malloc((size_t)z); if(!p || fread(p,1,(size_t)z,f)!=(size_t)z){free(p);fclose(f);return NULL;}
    fclose(f);*n=(uint32_t)z;return p;
}

static int run_model(const char *model) {
    char qpath[96],npath[96];
    snprintf(qpath,sizeof qpath,"hexecutor/native/build/resident-%s.qdrx",model);
    snprintf(npath,sizeof npath,"samples/depth-model/deployment/programs/%s-v2.nprg",model);
    uint32_t qn,nn; uint8_t *q=read_file(qpath,&qn),*raw=read_file(npath,&nn);
    CHECK(q && raw);
    Nprg2Pool pools[7]={0};
    for(unsigned i=0;i<7;i++) {
        pools[i].bytes=rd32fvtest(raw+144+4*i);
        if(pools[i].bytes)pools[i].base=(uint8_t*)malloc(pools[i].bytes);
        CHECK(!pools[i].bytes || pools[i].base);
        if(pools[i].bytes)memset(pools[i].base,0,pools[i].bytes);
    }
    uint8_t qsha[32],nsha[32];nprg2_sha256(q,qn,qsha);nprg2_sha256(raw,nn,nsha);
    Nprg2ExpectedHashes exp={q+32,q+64,qsha,nsha};Nprg2Program program;
    CHECK(nprg2_open(&program,raw,nn,&exp,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t *hot=(uint8_t*)calloc(1,300000);CHECK(hot);
    Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1};
    Nprg2Bound bound;CHECK(nprg2_bind_metadata(&program,pools,&bc,&bound)==NPRG2_OK);
    Nprg2Executor exec;nprg2_executor_init(&exec);CHECK(nprg2_executor_register_flat_vtcm(&exec)==NPRG2_EXEC_OK);
    uint8_t *expected_pool1=(uint8_t*)malloc(pools[1].bytes);
    uint8_t *expected_pool3=(uint8_t*)malloc(pools[3].bytes);
    CHECK(expected_pool1 && expected_pool3);
    memcpy(expected_pool1,pools[1].base,pools[1].bytes);
    memset(expected_pool3,0,pools[3].bytes);
    uint32_t accepted=0;
    for(uint32_t i=0;i<program.command_count;i++) {
        Nprg2Command cmd;CHECK(nprg2_command_at(&bound,i,&cmd)==NPRG2_OK);
        if(cmd.opcode!=QDR_COPY_VALUES || cmd.variant!=65)continue;
        uint32_t si,di;Nprg2TensorView in,out;
        Nprg2ExecCall access={.command=cmd,.program=&program,.bound=&bound};
        CHECK(nprg2_exec_input(&access,0,&si,&in)==NPRG2_EXEC_OK);
        CHECK(nprg2_exec_output(&access,0,&di,&out)==NPRG2_EXEC_OK);
        const uint8_t *ig=in.geometry_record;uint32_t bytes=4;
        for(unsigned d=0;d<4;d++)bytes*=(uint32_t)(int16_t)((uint16_t)ig[14+2*d]|(uint16_t)ig[15+2*d]<<8);
        for(uint32_t j=0;j<bytes;j++) {
            uint8_t v=(uint8_t)(i+j*13u);
            pools[3].base[in.offset+j]=v;
            expected_pool3[in.offset+j]=v;
        }
        Nprg2ExecRequest req={.program=&program,.bound=&bound,.pool1_writable_bytes=NPRG2_RUNNER_POOL1_ACTIVE_BYTES,
            .command_first=i,.command_count=1};
        memcpy(req.pools,pools,sizeof pools);
        req.pool1_writable_bytes=out.offset+bytes-1u;
        CHECK(nprg2_executor_preflight(&exec,&req)==NPRG2_EXEC_REJECTED);
        req.pool1_writable_bytes=NPRG2_RUNNER_POOL1_ACTIVE_BYTES;
        CHECK(nprg2_executor_preflight(&exec,&req)==NPRG2_EXEC_OK);
        CHECK(nprg2_executor_run(&exec,&req)==NPRG2_EXEC_OK);
        CHECK(!memcmp(pools[1].base+out.offset,pools[3].base+in.offset,bytes));
        memcpy(expected_pool1+out.offset,pools[3].base+in.offset,bytes);
        accepted++;
    }
    CHECK(accepted==160);
    CHECK(!memcmp(pools[1].base,expected_pool1,pools[1].bytes));
    CHECK(!memcmp(pools[3].base,expected_pool3,pools[3].bytes));
    printf("prepared flat_to_vtcm accepted %u %s commands; complete pool1/pool3 comparisons passed\n",
           (unsigned)accepted,model);
    free(expected_pool1);free(expected_pool3);nprg2_close(&program);
    for(unsigned i=0;i<7;i++)free(pools[i].base);
    free(hot);free(raw);free(q);return 0;
}

int main(void) { return run_model("primary") || run_model("alternate"); }
