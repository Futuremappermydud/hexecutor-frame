/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_s2d.h"
#include "../kernels/layout/space_to_depth_u8_direct.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); exit(1); } } while (0)
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24; }
static uint8_t *read_file(const char *path, uint32_t *n) {
    FILE *f=fopen(path,"rb"); CHECK(f); CHECK(!fseek(f,0,SEEK_END)); long z=ftell(f);
    CHECK(z>=0 && (uint64_t)z<=32u*1024u*1024u); CHECK(!fseek(f,0,SEEK_SET));
    uint8_t *p=(uint8_t *)malloc((size_t)z); CHECK(p && fread(p,1,(size_t)z,f)==(size_t)z);
    fclose(f); *n=(uint32_t)z; return p;
}
static int resolve_all(void *ctx,const Nprg2Program *p,uint32_t i,const Nprg2Command *c,Nprg2Capability *cap) {
    (void)ctx;(void)p;(void)i;(void)c;cap->supported=1;cap->may_fallback=0;cap->scratch_bytes=0;cap->concurrent_worker_scratch_bytes=0;return 1;
}
static void set_block(Nprg2Pool pools[7],uint32_t id,const uint8_t *bytes) {
    uint32_t pool=id&127u,off=id&~127u; CHECK(pool>0&&pool<7&&pools[pool].base);
    CHECK(off<=pools[pool].bytes&&2048u<=pools[pool].bytes-off); memcpy(pools[pool].base+off,bytes,2048);
}
static void run_model(const char *model) {
    char npath[128],qpath[128]; snprintf(npath,sizeof npath,"samples/depth-model/deployment/programs/%s-v2.nprg",model);
    snprintf(qpath,sizeof qpath,"hexecutor/native/build/resident-%s.qdrx",model);
    uint32_t qn,nn; uint8_t *q=read_file(qpath,&qn),*raw=read_file(npath,&nn),qsha[32],nsha[32];
    nprg2_sha256(q,qn,qsha);nprg2_sha256(raw,nn,nsha);
    Nprg2ExpectedHashes x={q+32,q+64,qsha,nsha}; Nprg2Pool pools[7]={0};
    for(uint32_t i=0;i<7;i++){pools[i].bytes=rd32(raw+144+4*i); if(pools[i].bytes){
        pools[i].base=i==1?(uint8_t *)aligned_alloc(128,pools[i].bytes):(uint8_t *)malloc(pools[i].bytes);
        CHECK(pools[i].base);memset(pools[i].base,0,pools[i].bytes);}}
    Nprg2Program p;CHECK(nprg2_open(&p,raw,nn,&x,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t *hot=(uint8_t *)calloc(1,300000),*scratch=(uint8_t *)aligned_alloc(128,QD_LAYOUT_S2D_TILE_BYTES);CHECK(hot&&scratch);
    Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.l2_available_bytes=786432,
        .pool1_is_verified_vtcm=1,.resolve=resolve_all}; Nprg2Bound b;
    CHECK(nprg2_bind_mock(&p,pools,&bc,&b)==NPRG2_OK);
    Nprg2Executor e;nprg2_executor_init(&e);CHECK(nprg2_executor_register_s2d(&e)==NPRG2_EXEC_OK);
    Nprg2ExecRequest r={.program=&p,.bound=&b,.pool1_writable_bytes=0x780000u,
        .scratch=scratch,.scratch_bytes=QD_LAYOUT_S2D_TILE_BYTES};memcpy(r.pools,pools,sizeof pools);

    /* Read-only preflight every one of this model's 174 family commands. */
    uint32_t covered=0;
    for(uint32_t i=0;i<p.command_count;i++){Nprg2Command c;CHECK(nprg2_command_at(&b,i,&c)==NPRG2_OK);
        if(c.opcode==7&&c.variant==31){r.command_first=i;r.command_count=1;
            int pr=nprg2_executor_preflight(&e,&r);if(pr!=NPRG2_EXEC_OK)fprintf(stderr,"preflight reject %s index %lu seq %u rc %d\n",model,(unsigned long)i,c.sequence,pr);
            CHECK(pr==NPRG2_EXEC_OK);covered++;}}
    CHECK(covered==174);

    for(uint32_t fi=0;fi<10;fi++){
        char path[160];snprintf(path,sizeof path,"hexecutor/native/program/fixtures/%s-s2d-profile-%lu.s2pf",model,(unsigned long)fi);
        uint32_t fn;uint8_t *fx=read_file(path,&fn);CHECK(fn>=32&&!memcmp(fx,"S2PF",4));
        CHECK(rd32(fx+4)==1&&rd32(fx+8)==7&&rd32(fx+12)==31);
        uint32_t seq=rd32(fx+16),index=rd32(fx+20),ni=rd32(fx+24),no=rd32(fx+28);
        Nprg2Command c;CHECK(nprg2_command_at(&b,index,&c)==NPRG2_OK);
        CHECK(c.sequence==seq&&c.opcode==7&&c.variant==31);
        for(uint32_t i=0;i<7;i++)if(pools[i].base)memset(pools[i].base,0,pools[i].bytes);
        const uint8_t *cur=fx+32;
        for(uint32_t i=0;i<ni;i++){uint32_t id=rd32(cur);cur+=4;set_block(pools,id,cur);cur+=2048;}
        uint32_t *outids=(uint32_t *)malloc((size_t)no*sizeof(uint32_t));CHECK(outids);
        uint8_t *want=(uint8_t *)malloc((size_t)no*2048u);CHECK(want);
        for(uint32_t i=0;i<no;i++){
            outids[i]=rd32(cur);cur+=4;
            for(uint32_t j=0;j<i;j++)CHECK(outids[j]!=outids[i]);
            set_block(pools,outids[i],cur);cur+=2048;
            memcpy(want+(size_t)i*2048,cur,2048);cur+=2048;}
        CHECK(cur==fx+fn);r.command_first=index;r.command_count=1;
        uint8_t *want_pool1=(uint8_t *)malloc(pools[1].bytes);CHECK(want_pool1);
        memcpy(want_pool1,pools[1].base,pools[1].bytes);
        for(uint32_t i=0;i<no;i++){
            uint32_t off=outids[i]&~127u;
            memcpy(want_pool1+off,want+(size_t)i*2048,2048);
        }
        CHECK(nprg2_executor_preflight(&e,&r)==NPRG2_EXEC_OK);
        CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);
        CHECK(!memcmp(pools[1].base,want_pool1,pools[1].bytes));
        printf("PASS %s S2D profile %lu seq %lu, input blocks %lu output blocks %lu\n",model,(unsigned long)fi,(unsigned long)seq,(unsigned long)ni,(unsigned long)no);
        free(want_pool1);free(outids);free(want);free(fx);
    }
    free(scratch);free(hot);nprg2_close(&p);for(uint32_t i=0;i<7;i++)free(pools[i].base);free(raw);free(q);
}
int main(void){run_model("primary");run_model("alternate");puts("ALL S2D CPU-prefix and preflight checks passed");return 0;}
