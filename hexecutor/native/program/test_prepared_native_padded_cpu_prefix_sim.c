/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_native.h"
#include <hexagon_standalone.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NCPF_CASE
#error "compile with NCPF_CASE=1..4"
#endif
#if NCPF_CASE==1
#define QDRX_PATH "hexecutor/native/build/resident-primary.qdrx"
#define NPRG_PATH "depth-model/deployment/programs/primary-v2.nprg"
#define NCPF_PATH "hexecutor/native/program/fixtures/primary-smoke-first12.ncpfx"
#define CASE_NAME "primary CI12/logical9"
#elif NCPF_CASE==2
#define QDRX_PATH "hexecutor/native/build/resident-primary.qdrx"
#define NPRG_PATH "depth-model/deployment/programs/primary-v2.nprg"
#define NCPF_PATH "hexecutor/native/program/fixtures/primary-smoke-ci20.ncpfx"
#define CASE_NAME "primary CI20/logical18"
#elif NCPF_CASE==3
#define QDRX_PATH "hexecutor/native/build/resident-alternate.qdrx"
#define NPRG_PATH "depth-model/deployment/programs/alternate-v2.nprg"
#define NCPF_PATH "hexecutor/native/program/fixtures/alternate-smoke-first12.ncpfx"
#define CASE_NAME "alternate CI12/logical9"
#elif NCPF_CASE==4
#define QDRX_PATH "hexecutor/native/build/resident-alternate.qdrx"
#define NPRG_PATH "depth-model/deployment/programs/alternate-v2.nprg"
#define NCPF_PATH "hexecutor/native/program/fixtures/alternate-smoke-ci20.ncpfx"
#define CASE_NAME "alternate CI20/logical18"
#else
#error "unsupported NCPF_CASE"
#endif

#define ACTIVE_BYTES 0x780000u
#define TAIL_BYTES 0x80000u
#define PAGE_BYTES 2048u
#define NCPF_HEADER_BYTES 76u
#define MAX_PACKED_WEIGHT_BYTES 4096u

extern void qd_qnn_hmx_1x1(uint32_t *,uint32_t *,void *,void *,uint32_t *,uint32_t *);
extern void qd_hmx_finish(void *,void *);

typedef struct {uint32_t offset;uint8_t bytes[PAGE_BYTES];} PagePatch;
static uint32_t tables[512] __attribute__((aligned(128)));
static uint32_t abi[30] __attribute__((aligned(16)));
static uint8_t packed_expected[MAX_PACKED_WEIGHT_BYTES] __attribute__((aligned(128)));
static QdDirectBinding submitted_binding;
static uint8_t *vtcm;
static uint32_t padded_bytes;

#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);return 1;}}while(0)
static uint32_t rd32(const uint8_t*p){return(uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint16_t rd16(const uint8_t*p){return(uint16_t)p[0]|(uint16_t)p[1]<<8;}
static int read_file(const char*path,uint8_t**data,uint32_t*bytes){FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"reopen %s\n",path);return 0;}if(fseek(f,0,SEEK_END)){fprintf(stderr,"seekend %s\n",path);fclose(f);return 0;}long n=ftell(f);fclose(f);if(n<=0||((uint64_t)n)>UINT32_MAX)return 0;f=fopen(path,"rb");if(!f){fprintf(stderr,"reopen %s\n",path);return 0;}uint8_t*p=(uint8_t*)malloc((size_t)n);if(!p){fprintf(stderr,"malloc fail %s %ld\n",path,n);fclose(f);return 0;}size_t got=0;while(got<(size_t)n){size_t chunk=(size_t)n-got;if(chunk>4096u)chunk=4096u;size_t step=fread(p+got,1,chunk,f);if(!step){fprintf(stderr,"read fail %s got=%lu expected=%ld\n",path,(unsigned long)got,n);free(p);fclose(f);return 0;}got+=step;}fclose(f);*data=p;*bytes=(uint32_t)n;return 1;}
static void publish(const void*p,uint32_t n){uintptr_t a=(uintptr_t)p&~(uintptr_t)31u,e=((uintptr_t)p+n+31u)&~(uintptr_t)31u;for(;a<e;a+=32u)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");}
static void invalidate(const void*p,uint32_t n){uintptr_t a=(uintptr_t)p&~(uintptr_t)31u,e=((uintptr_t)p+n+31u)&~(uintptr_t)31u;for(;a<e;a+=32u)__asm__ volatile("dcinva(%0)"::"r"(a):"memory");}
static int bind_all(void*ctx,const Nprg2Program*p,uint32_t ix,const Nprg2Command*c,Nprg2Capability*cap){(void)ctx;(void)p;(void)ix;(void)c;cap->supported=1;cap->may_fallback=0;cap->scratch_bytes=0;cap->concurrent_worker_scratch_bytes=0;return 1;}
static int prepare_packed_expectation(const Nprg2ExecCall*call,const QdDirectBinding*b){
    uint32_t wi;Nprg2TensorView w;if(nprg2_exec_input(call,1,&wi,&w)!=NPRG2_EXEC_OK)return 0;(void)wi;
    uint32_t raw_ci=rd16(w.geometry_record+10);
    uint32_t co=rd16(w.geometry_record+12);
    /* Weight geometry is [KH,KW,CI,CO]; CI/CO are at offsets 10/12. */
    const uint8_t*raw=call->pools[w.pool].base+w.offset;uint32_t expected=32u*co;
    if((raw_ci!=12u&&raw_ci!=20u)||!b->weights||b->staged_weight_bytes!=expected||expected>sizeof packed_expected||
       b->tail_publish_bytes!=0x2000u+expected+128u||b->weights!=(uint8_t*)call->pools[1].base+0x780000u+0x2000u||!b->packed_weight_guard)return 0;
    memset(packed_expected,0,expected);uint32_t src_groups=raw_ci/4u;
    for(uint32_t ob=0;ob<co/32u;ob++)for(uint32_t ig=0;ig<src_groups;ig++)
        memcpy(packed_expected+((ob*8u+ig)*128u),raw+((ob*src_groups+ig)*128u),128u);
    padded_bytes=expected;return 1;
}
static int submit_and_drain(void*ctx,const Nprg2ExecCall*call,const QdDirectBinding*b,uint8_t family){
    (void)ctx;if(!call||!b||family!=1u||!b->in_desc||!b->out_desc||!b->weights||!b->fi||!b->done)return 0;
    if(!prepare_packed_expectation(call,b))return 0;
    submitted_binding=*b;
    publish(tables,QD_DIRECT_TABLE_BYTES);publish(abi,QD_DIRECT_ABI_BYTES);publish((const uint8_t*)call->pools[1].base+ACTIVE_BYTES,b->tail_publish_bytes);
    __asm__ volatile("barrier":::"memory");uint32_t ssr_before;__asm__ volatile("%0=ssr":"=r"(ssr_before));
    uint32_t ssr_native=ssr_before|(1u<<26);__asm__ volatile("ssr=%0\n isync"::"r"(ssr_native):"memory");
    __asm__ volatile("barrier\n mxclracc":::"memory");
    qd_qnn_hmx_1x1(b->out_desc,b->in_desc,b->weights,b->fi,b->cfg,b->mode);
    *b->done=0xffffffffu;qd_hmx_finish(b->zero_fi,(void*)b->done);
    uint32_t tries=0;while(*b->done&&tries++<1000000u)__asm__ volatile("nop");
    __asm__ volatile("ssr=%0\n isync"::"r"(ssr_before):"memory");
    if(*b->done||tries>=1000000u)return 0;
    if(memcmp(b->weights,packed_expected,padded_bytes))return 0;
    for(uint32_t i=0;i<128u;i++)if(((const uint8_t*)b->packed_weight_guard)[i]!=0xa5u)return 0;
    return 1;
}

int main(void){
    uint8_t *qraw=0,*nraw=0,*fixture=0;uint32_t qn=0,nn=0,fn=0;
    if(!read_file(QDRX_PATH,&qraw,&qn)){fprintf(stderr,"qdrx read fail\n");return 1;}if(!read_file(NPRG_PATH,&nraw,&nn)){fprintf(stderr,"nprg read fail\n");return 1;}if(!read_file(NCPF_PATH,&fixture,&fn)){fprintf(stderr,"ncpf read fail\n");return 1;}
    CHECK(fn>=NCPF_HEADER_BYTES&&!memcmp(fixture,"NCPF",4)&&rd32(fixture+4)==1u&&rd32(fixture+8)==6u&&rd32(fixture+12)==32u);
    uint32_t command_index=rd32(fixture+20),seq=rd32(fixture+16),xid=rd32(fixture+24),wid=rd32(fixture+28),fid=rd32(fixture+32),yid=rd32(fixture+36),page_count=rd32(fixture+40);
    CHECK(page_count>0&&page_count<256u);
    uint32_t sizes[7];uint64_t total=NCPF_HEADER_BYTES;for(uint32_t p=0;p<7;p++){sizes[p]=rd32(fixture+48+p*4u);total+=sizes[p];}
    uint64_t expected_size=total+(uint64_t)page_count*(4u+PAGE_BYTES);CHECK(expected_size==fn);
    vtcm=(uint8_t*)(__rdcfg(__vtcm_base)<<16);for(uint32_t i=0;i<QD_DIRECT_VTCM_BYTES;i+=4096u)add_translation(vtcm+i,vtcm+i,6);
    CHECK(sizes[1]==QD_DIRECT_VTCM_BYTES);Nprg2Pool pools[7]={0};uint8_t*pool_storage[7]={0};uint8_t*saved_other[7]={0};
    uint32_t cursor=NCPF_HEADER_BYTES;
    for(uint32_t p=0;p<7;p++){
        pools[p].bytes=sizes[p];if(!sizes[p])continue;
        if(p==1)pools[p].base=vtcm;
        else {pool_storage[p]=(uint8_t*)malloc(sizes[p]);CHECK(pool_storage[p]);pools[p].base=pool_storage[p];}
        memcpy(pools[p].base,fixture+cursor,sizes[p]);cursor+=sizes[p];
        if(p!=1){saved_other[p]=(uint8_t*)malloc(sizes[p]);CHECK(saved_other[p]);memcpy(saved_other[p],pools[p].base,sizes[p]);}
    }
    uint8_t*saved_active=(uint8_t*)malloc(ACTIVE_BYTES),*saved_tail=(uint8_t*)malloc(TAIL_BYTES);CHECK(saved_active&&saved_tail);
    memcpy(saved_active,vtcm,ACTIVE_BYTES);memcpy(saved_tail,vtcm+ACTIVE_BYTES,TAIL_BYTES);
    PagePatch*patch=(PagePatch*)calloc(page_count,sizeof *patch);CHECK(patch);
    for(uint32_t k=0;k<page_count;k++){patch[k].offset=rd32(fixture+cursor);cursor+=4;memcpy(patch[k].bytes,fixture+cursor,PAGE_BYTES);cursor+=PAGE_BYTES;CHECK(patch[k].offset<=ACTIVE_BYTES-PAGE_BYTES&&patch[k].offset%PAGE_BYTES==0);}
    for(uint32_t i=0;i<page_count;i++)for(uint32_t j=i+1;j<page_count;j++)CHECK(patch[i].offset!=patch[j].offset);
    for(uint32_t i=0;i<ACTIVE_BYTES;i+=32u)publish(vtcm+i,32u);__asm__ volatile("barrier":::"memory");

    uint8_t qhash[32],nhash[32];nprg2_sha256(qraw,qn,qhash);nprg2_sha256(nraw,nn,nhash);
    Nprg2ExpectedHashes hashes={.context_sha256=qraw+32,.constants_sha256=qraw+64,.source_sha256=qhash,.prepared_sha256=nhash};
    Nprg2Program program;CHECK(nprg2_open(&program,nraw,nn,&hashes,pools,(Nprg2Allocator){0})==NPRG2_OK);
    for(uint32_t p=0;p<7;p++)CHECK(program.pool_bytes[p]==sizes[p]);
    uint8_t*hot=(uint8_t*)malloc(300000u);CHECK(hot);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000u,.l2_available_bytes=786432u,.pool1_is_verified_vtcm=1,.resolve=bind_all};
    Nprg2Bound bound;CHECK(nprg2_bind_mock(&program,pools,&bc,&bound)==NPRG2_OK);
    Nprg2Command command;CHECK(nprg2_command_at(&bound,command_index,&command)==NPRG2_OK&&command.sequence==seq&&command.opcode==6u&&command.variant==32u);
    Nprg2ExecCall call={.command_index=command_index,.command=command,.program=&program,.bound=&bound,.pools=pools,.pool1_writable_bytes=ACTIVE_BYTES};
    uint32_t got_x,got_w,got_f,got_y;Nprg2TensorView vx,vw,vf,vy;
    CHECK(nprg2_exec_input(&call,0,&got_x,&vx)==NPRG2_EXEC_OK);
    CHECK(nprg2_exec_input(&call,1,&got_w,&vw)==NPRG2_EXEC_OK);
    CHECK(nprg2_exec_input(&call,2,&got_f,&vf)==NPRG2_EXEC_OK);
    CHECK(nprg2_exec_output(&call,0,&got_y,&vy)==NPRG2_EXEC_OK);
    uint32_t raw_ci=rd16(vw.geometry_record+10),logical_ci=rd16(vx.geometry_record+12),max_ci=rd16(vx.geometry_record+20);
    CHECK((raw_ci==12u&&logical_ci==9u)||(raw_ci==20u&&logical_ci==18u));CHECK(max_ci==32u);
    uint32_t *l2tables=tables,*l2abi=abi;Nprg2NativeConfig native={.active_vtcm_bytes=ACTIVE_BYTES,.tables=l2tables,.table_bytes=QD_DIRECT_TABLE_BYTES,.abi=l2abi,.abi_bytes=QD_DIRECT_ABI_BYTES,.tail=vtcm+ACTIVE_BYTES,.tail_bytes=TAIL_BYTES,.submit_and_drain=submit_and_drain,.family_mask=NPRG2_NATIVE_ALLOW_1X1};
    Nprg2Executor executor;nprg2_executor_init(&executor);CHECK(nprg2_executor_register_native_conv(&executor,&native)==NPRG2_EXEC_OK);
    Nprg2ExecRequest request={.program=&program,.bound=&bound,.pool1_writable_bytes=ACTIVE_BYTES,.command_first=command_index,.command_count=1};memcpy(request.pools,pools,sizeof pools);
    {int ps=nprg2_executor_preflight(&executor,&request);if(ps!=NPRG2_EXEC_OK){fprintf(stderr,"preflight=%d ci=%lu rawCI=%lu logicalCI=%lu maxCI=%lu\n",ps,(unsigned long)command_index,(unsigned long)raw_ci,(unsigned long)logical_ci,(unsigned long)max_ci);return 1;}}
    Nprg2ExecStats stats={0};request.stats=&stats;CHECK(nprg2_executor_run(&executor,&request)==NPRG2_EXEC_OK);
    CHECK(stats.executed_commands==1u&&stats.validated_commands==1u&&submitted_binding.out_pages==page_count&&submitted_binding.staged_weight_bytes>0u);
    uint64_t exact=0,plus1=0,minus1=0,other=0;uint32_t first_page=UINT32_MAX,first_off=0,first_got=0,first_expected=0;
    for(uint32_t i=0;i<page_count;i++){
        invalidate(vtcm+patch[i].offset,PAGE_BYTES);
        for(uint32_t z=0;z<PAGE_BYTES;z++){int d=(int)vtcm[patch[i].offset+z]-(int)patch[i].bytes[z];if(!d)exact++;else if(d==1)plus1++;else if(d==-1)minus1++;else other++;if(d&&first_page==UINT32_MAX){first_page=i;first_off=z;first_got=vtcm[patch[i].offset+z];first_expected=patch[i].bytes[z];}}
    }
    if(plus1||minus1||other)fprintf(stderr,"Y deltas %s pages=%lu exact=%llu +1=%llu -1=%llu other=%llu first_page=%lu page_offset=%lu got=%u expected=%u\n",CASE_NAME,(unsigned long)page_count,(unsigned long long)exact,(unsigned long long)plus1,(unsigned long long)minus1,(unsigned long long)other,(unsigned long)first_page,(unsigned long)first_off,(unsigned)first_got,(unsigned)first_expected);
    for(uint32_t j=0;j<submitted_binding.out_pages;j++){
        uintptr_t a=(uintptr_t)submitted_binding.out_table[j],base=(uintptr_t)vtcm;CHECK(a>=base&&a-base<ACTIVE_BYTES&&(a-base)%PAGE_BYTES==0);
        int found=0;for(uint32_t i=0;i<page_count;i++)if(patch[i].offset==(uint32_t)(a-base)){found=1;break;}CHECK(found);
    }
    uint8_t output_page[ACTIVE_BYTES/PAGE_BYTES];memset(output_page,0,sizeof output_page);
    for(uint32_t j=0;j<page_count;j++)output_page[patch[j].offset/PAGE_BYTES]=1;
    for(uint32_t i=0;i<ACTIVE_BYTES;i++)if(!output_page[i/PAGE_BYTES])CHECK(vtcm[i]==saved_active[i]);
    for(uint32_t i=submitted_binding.tail_publish_bytes;i<TAIL_BYTES;i++)CHECK(vtcm[ACTIVE_BYTES+i]==saved_tail[i]);
    for(uint32_t p=0;p<7;p++)if(p!=1&&sizes[p])CHECK(!memcmp(pools[p].base,saved_other[p],sizes[p]));
    printf("PASS padded native 1x1 CPU-prefix %s: command=%lu seq=%lu NPRG ids=[%lu,%lu,%lu,%lu] source QDRX ids=[%lu,%lu,%lu,%lu], rawCI=%lu logicalCI=%lu packed=%lu outputPages=%lu; Y deltas exact=%llu +1=%llu -1=%llu other=%llu and all outside-Y pools preserved\n",CASE_NAME,(unsigned long)command_index,(unsigned long)seq,(unsigned long)got_x,(unsigned long)got_w,(unsigned long)got_f,(unsigned long)got_y,(unsigned long)xid,(unsigned long)wid,(unsigned long)fid,(unsigned long)yid,(unsigned long)raw_ci,(unsigned long)logical_ci,(unsigned long)padded_bytes,(unsigned long)page_count,(unsigned long long)exact,(unsigned long long)plus1,(unsigned long long)minus1,(unsigned long long)other);
    nprg2_close(&program);free(hot);free(patch);free(saved_active);free(saved_tail);
    for(uint32_t p=0;p<7;p++){free(pool_storage[p]);free(saved_other[p]);}
    free(qraw);free(nraw);free(fixture);return 0;
}
