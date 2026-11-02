#include "prepared_executor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);exit(1);}}while(0)
static uint32_t u32(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint8_t *read_file(const char *path,uint32_t *n){FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long z=ftell(f);CHECK(z>0&&z<=NPRG2_MAX_IMAGE_BYTES);CHECK(!fseek(f,0,SEEK_SET));uint8_t *b=malloc((size_t)z);CHECK(b&&fread(b,1,(size_t)z,f)==(size_t)z);fclose(f);*n=(uint32_t)z;return b;}
static void pools_new(const uint8_t *nprg,Nprg2Pool p[7]){for(int i=0;i<7;i++){p[i].bytes=u32(nprg+144+i*4);p[i].base=p[i].bytes?calloc(1,p[i].bytes):NULL;CHECK(!p[i].bytes||p[i].base);}}
static void pools_free(Nprg2Pool p[7]){for(int i=0;i<7;i++)free(p[i].base);}
static void hashes(const uint8_t *q,uint32_t qn,const uint8_t *n,uint32_t nn,Nprg2ExpectedHashes *h,uint8_t qsha[32],uint8_t nsha[32]){CHECK(qn>=96);nprg2_sha256(q,qn,qsha);nprg2_sha256(n,nn,nsha);h->context_sha256=q+32;h->constants_sha256=q+64;h->source_sha256=qsha;h->prepared_sha256=nsha;}
static int bind_cover(void *ctx,const Nprg2Program *p,uint32_t i,const Nprg2Command *c,Nprg2Capability *cap){(void)ctx;(void)p;(void)i;(void)c;cap->supported=1;cap->may_fallback=0;cap->scratch_bytes=0;cap->concurrent_worker_scratch_bytes=0;return 1;}
typedef struct {uint32_t writes;const Nprg2ExecBuffer *inputs,*outputs;} MockState;
static int mock_validate(const Nprg2ExecCall *c,void *ctx){MockState *s=ctx;return c&&c->command.opcode&&c->command.variant&&c->inputs==s->inputs&&c->input_count==1&&c->outputs==s->outputs&&c->output_count==1;}
static int mock_run(const Nprg2ExecCall *c,void *ctx){MockState *s=ctx;if(c->inputs!=s->inputs||c->outputs!=s->outputs)return 0;s->writes++;return 1;}
static void mock_model(const char *qpath,const char *npath){
    uint32_t qn,nn;uint8_t *q=read_file(qpath,&qn),*raw=read_file(npath,&nn),qsha[32],nsha[32];Nprg2ExpectedHashes h;hashes(q,qn,raw,nn,&h,qsha,nsha);
    Nprg2Pool pools[7];pools_new(raw,pools);Nprg2Program p;CHECK(nprg2_open(&p,raw,nn,&h,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t *hot=calloc(1,300000);CHECK(hot);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.scratch_budget=0,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1,.resolve=bind_cover};Nprg2Bound bound;CHECK(nprg2_bind_mock(&p,pools,&bc,&bound)==NPRG2_OK);
    uint8_t io_in_data[16]={0},io_out_data[16]={0};Nprg2ExecBuffer io_in={io_in_data,sizeof io_in_data},io_out={io_out_data,sizeof io_out_data};
    Nprg2ExecRequest r={.program=&p,.bound=&bound,.pool1_writable_bytes=0x780000u,.inputs=&io_in,.input_count=1,.outputs=&io_out,.output_count=1,.user=NULL,.command_first=0,.command_count=p.command_count};memcpy(r.pools,pools,sizeof pools);
    Nprg2Executor e;nprg2_executor_init(&e);MockState state={.writes=0,.inputs=&io_in,.outputs=&io_out};
    Nprg2Command first;CHECK(nprg2_command_at(&bound,0,&first)==NPRG2_OK);Nprg2Executor scratch_e;nprg2_executor_init(&scratch_e);
    CHECK(nprg2_executor_register_scratch(&scratch_e,first.opcode,first.variant,mock_validate,mock_run,&state,1,0)==NPRG2_EXEC_OK);
    Nprg2ExecRequest scratch_r=r;scratch_r.command_count=1;CHECK(nprg2_executor_run(&scratch_e,&scratch_r)==NPRG2_EXEC_REJECTED);CHECK(state.writes==0);
    /* Missing one schedule key must reject before any registered callback runs. */
    Nprg2Command omitted={0};int have_omitted=0;
    for(uint32_t i=0;i<p.command_count&&!have_omitted;i++){Nprg2Command c;CHECK(nprg2_command_at(&bound,i,&c)==NPRG2_OK);uint32_t occurrences=0;for(uint32_t j=0;j<p.command_count;j++){Nprg2Command x;CHECK(nprg2_command_at(&bound,j,&x)==NPRG2_OK);if(x.opcode==c.opcode&&x.variant==c.variant)occurrences++;}if(occurrences==1){omitted=c;have_omitted=1;}}
    CHECK(have_omitted);
    for(uint32_t i=0;i<p.command_count;i++){Nprg2Command c;CHECK(nprg2_command_at(&bound,i,&c)==NPRG2_OK);if(c.opcode==omitted.opcode&&c.variant==omitted.variant)continue;int seen=0;for(uint32_t j=0;j<NPRG2_EXEC_REGISTRY_CAPACITY;j++)if(e.handlers[j].used&&e.handlers[j].opcode==c.opcode&&e.handlers[j].variant==c.variant)seen=1;if(!seen)CHECK(nprg2_executor_register(&e,c.opcode,c.variant,mock_validate,mock_run,&state)==NPRG2_EXEC_OK);}
    CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_UNSUPPORTED);CHECK(state.writes==0);
    CHECK(nprg2_executor_register(&e,omitted.opcode,omitted.variant,mock_validate,mock_run,&state)==NPRG2_EXEC_OK);
    Nprg2ExecStats stats;r.stats=&stats;CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);CHECK(stats.validated_commands==p.command_count&&stats.executed_commands==p.command_count&&state.writes==p.command_count);
    free(hot);nprg2_close(&p);pools_free(pools);free(q);free(raw);printf("PASS mock schedule %s\n",qpath);
}
static uint64_t fake_clock_now;
static uint64_t fake_clock(void){return fake_clock_now+=10;}
typedef struct {uint32_t validates,runs,preflighted_runs;} CountState;
static int count_validate(const Nprg2ExecCall *c,void *ctx){(void)c;((CountState *)ctx)->validates++;return 1;}
static int count_run(const Nprg2ExecCall *c,void *ctx){CountState *s=ctx;s->runs++;s->preflighted_runs+=c->preflighted;return 1;}
/* Profile plumbing, preflight reuse and effect-free handlers on a real schedule. */
static void profile_model(const char *qpath,const char *npath){
    uint32_t qn,nn;uint8_t *q=read_file(qpath,&qn),*raw=read_file(npath,&nn),qsha[32],nsha[32];Nprg2ExpectedHashes h;hashes(q,qn,raw,nn,&h,qsha,nsha);
    Nprg2Pool pools[7];pools_new(raw,pools);Nprg2Program p;CHECK(nprg2_open(&p,raw,nn,&h,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t *hot=calloc(1,300000);CHECK(hot);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1,.resolve=bind_cover};Nprg2Bound bound;CHECK(nprg2_bind_mock(&p,pools,&bc,&bound)==NPRG2_OK);
    Nprg2ExecRequest r={.program=&p,.bound=&bound,.pool1_writable_bytes=0x780000u,.command_count=p.command_count};memcpy(r.pools,pools,sizeof pools);
    CountState cs={0};Nprg2Executor e;nprg2_executor_init(&e);uint32_t nops=0;
    for(uint32_t i=0;i<p.command_count;i++){Nprg2Command c;CHECK(nprg2_command_at(&bound,i,&c)==NPRG2_OK);if(c.opcode==QDR_NOP)nops++;
        int seen=0;for(uint32_t j=0;j<NPRG2_EXEC_REGISTRY_CAPACITY;j++)if(e.handlers[j].used&&e.handlers[j].opcode==c.opcode&&e.handlers[j].variant==c.variant)seen=1;
        if(!seen)CHECK(nprg2_executor_register(&e,c.opcode,c.variant,count_validate,count_run,&cs)==NPRG2_EXEC_OK);}
    CHECK(nprg2_executor_mark_no_effect(&e,255,1)==NPRG2_EXEC_UNSUPPORTED);
    uint64_t vc[80]={0},vn[80]={0},hrc[NPRG2_EXEC_REGISTRY_CAPACITY]={0},hrn[NPRG2_EXEC_REGISTRY_CAPACITY]={0},hvc[NPRG2_EXEC_REGISTRY_CAPACITY]={0};
    Nprg2ExecProfile prof={.clock=fake_clock,.variant_cycles=vc,.variant_count=vn,.handler_run_cycles=hrc,.handler_run_count=hrn,.handler_validate_cycles=hvc};
    Nprg2ExecStats stats;memset(&stats,0xa5,sizeof stats);r.stats=&stats;r.profile=&prof;
    /* Garbage in the stats must not matter: preflight clears them, the profile lives in the request. */
    CHECK(nprg2_executor_preflight(&e,&r)==NPRG2_EXEC_OK);
    CHECK(cs.validates==p.command_count&&stats.validated_commands==p.command_count&&prof.preflight_calls==1);
    CHECK(prof.validate_cycles==10u*p.command_count&&prof.preflight_cycles>=prof.validate_cycles);
    uint64_t hv=0;for(unsigned i=0;i<NPRG2_EXEC_REGISTRY_CAPACITY;i++)hv+=hvc[i];CHECK(hv==prof.validate_cycles);
    /* reuse: identical request skips the second preflight, and calls are marked preflighted */
    r.reuse_preflight=1;uint32_t before=cs.validates;
    CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);
    CHECK(cs.validates==before&&prof.preflights_reused==1&&cs.runs==p.command_count&&cs.preflighted_runs==p.command_count);
    CHECK(stats.validated_commands==p.command_count&&stats.executed_commands==p.command_count);
    uint64_t rc=0,rn=0,vcs=0,vns=0;for(unsigned i=0;i<NPRG2_EXEC_REGISTRY_CAPACITY;i++){rc+=hrc[i];rn+=hrn[i];}for(unsigned i=0;i<80;i++){vcs+=vc[i];vns+=vn[i];}
    CHECK(rn==p.command_count&&vns==p.command_count&&rc==prof.run_cycles&&vcs==prof.run_cycles&&rc==10u*p.command_count);
    CHECK(prof.dispatch_cycles>0);
    /* a changed request, or a different stats object, preflights again */
    uint8_t scratch[64];before=cs.validates;r.scratch=scratch;r.scratch_bytes=sizeof scratch;
    CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);CHECK(cs.validates==before+p.command_count&&prof.preflights_reused==1);
    r.scratch=NULL;r.scratch_bytes=0;Nprg2ExecStats other={0};r.stats=&other;before=cs.validates;
    CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);CHECK(cs.validates==before+p.command_count&&prof.preflights_reused==1);
    r.stats=&stats;r.reuse_preflight=0;before=cs.validates;CHECK(nprg2_executor_preflight(&e,&r)==NPRG2_EXEC_OK);
    CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);CHECK(cs.validates==before+2u*p.command_count);
    /* effect-free handlers are counted but their run is not called */
    for(uint32_t j=0;j<NPRG2_EXEC_REGISTRY_CAPACITY;j++)if(e.handlers[j].used&&e.handlers[j].opcode==QDR_NOP)CHECK(nprg2_executor_mark_no_effect(&e,QDR_NOP,e.handlers[j].variant)==NPRG2_EXEC_OK);
    memset(vc,0,sizeof vc);memset(vn,0,sizeof vn);memset(hrn,0,sizeof hrn);memset(hrc,0,sizeof hrc);memset(&cs,0,sizeof cs);
    CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);
    CHECK(cs.runs==p.command_count-nops&&stats.executed_commands==p.command_count);
    rn=0;for(unsigned i=0;i<NPRG2_EXEC_REGISTRY_CAPACITY;i++)rn+=hrn[i];CHECK(rn==p.command_count);
    free(hot);nprg2_close(&p);pools_free(pools);free(q);free(raw);printf("PASS profile/reuse/no-effect %s (nops %u)\n",qpath,(unsigned)nops);
}
static void weights_model(const char *qpath,const char *npath){
    uint32_t qn,nn;uint8_t *q=read_file(qpath,&qn),*raw=read_file(npath,&nn),qsha[32],nsha[32];Nprg2ExpectedHashes h;hashes(q,qn,raw,nn,&h,qsha,nsha);
    Nprg2Pool pools[7];pools_new(raw,pools);Nprg2Program p;CHECK(nprg2_open(&p,raw,nn,&h,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t *hot=calloc(1,300000);CHECK(hot);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1,.resolve=bind_cover};Nprg2Bound bound;CHECK(nprg2_bind_mock(&p,pools,&bc,&bound)==NPRG2_OK);
    Nprg2Executor e;nprg2_executor_init(&e);CHECK(nprg2_executor_register_weights(&e)==NPRG2_EXEC_OK);
    static const uint8_t sz[8]={0,1,1,2,2,4,4,4};uint32_t tested=0,last_dest_pool=0,last_dest_offset=0,last_bytes=0;Nprg2ExecRequest r={.program=&p,.bound=&bound,.pool1_writable_bytes=0x780000u,.command_count=1};memcpy(r.pools,pools,sizeof pools);
    for(uint8_t variant=27;variant<=30;variant++){
        uint32_t ix=UINT32_MAX;Nprg2Command cmd;for(uint32_t i=0;i<p.command_count;i++){CHECK(nprg2_command_at(&bound,i,&cmd)==NPRG2_OK);if(cmd.opcode==QDR_WEIGHTS_TO_VTCM&&cmd.variant==variant){ix=i;break;}}CHECK(ix!=UINT32_MAX);
        CHECK(nprg2_command_at(&bound,ix,&cmd)==NPRG2_OK);Nprg2ExecCall call={.command_index=ix,.command=cmd,.program=&p,.bound=&bound,.pools=pools};uint32_t si,di;Nprg2TensorView s,d;CHECK(!nprg2_exec_input(&call,0,&si,&s));CHECK(!nprg2_exec_output(&call,0,&di,&d));CHECK(s.storage_kind==1&&d.storage_kind==1);
        const uint8_t *sg=s.geometry_record,*dg=d.geometry_record;uint32_t sb=1,db=1;for(int k=0;k<4;k++){sb*=(uint16_t)sg[14+k*2]|(uint16_t)sg[15+k*2]<<8;db*=(uint16_t)dg[14+k*2]|(uint16_t)dg[15+k*2]<<8;}sb*=sz[sg[0]];db*=sz[dg[0]];uint32_t source_bytes=sb<db?sb:db;CHECK(source_bytes);
        memset(pools[s.pool].base+s.offset,0x5a,source_bytes);memset(pools[d.pool].base+d.offset,0xa5,source_bytes);uint8_t *expected=malloc(source_bytes);CHECK(expected);memcpy(expected,pools[s.pool].base+s.offset,source_bytes);
        r.command_first=ix;CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);CHECK(!memcmp(expected,pools[d.pool].base+d.offset,source_bytes));
        free(expected);tested++;last_dest_pool=d.pool;last_dest_offset=d.offset;last_bytes=source_bytes;
    }
    CHECK(tested==4);
    /* CPU semantic oracle above is memmove(min(max-shape bytes)); aliasing the
     * supplied scratch with a graph destination rejects before any copy. */
    memset(pools[last_dest_pool].base+last_dest_offset,0xa5,last_bytes);r.scratch=pools[last_dest_pool].base+last_dest_offset;r.scratch_bytes=16;
    CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_ARGUMENT);for(uint32_t i=0;i<last_bytes;i++)CHECK(pools[last_dest_pool].base[last_dest_offset+i]==0xa5);
    free(hot);nprg2_close(&p);pools_free(pools);free(q);free(raw);printf("PASS weights-to-VTCM variants 27-30 %s\n",qpath);
}
int main(int argc,char **argv){CHECK(argc==5);mock_model(argv[1],argv[2]);mock_model(argv[3],argv[4]);weights_model(argv[1],argv[2]);weights_model(argv[3],argv[4]);profile_model(argv[1],argv[2]);profile_model(argv[3],argv[4]);return 0;}
