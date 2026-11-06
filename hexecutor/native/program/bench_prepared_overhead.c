/* Executor control-loop overhead on the real schedules, handlers stubbed.
 *
 *   bench_prepared_overhead <qdrx> <nprg> [reps]    (several pairs may follow)
 *
 * Every (opcode, variant) in the schedule gets a stub. "trivial" stubs
 * return 1 at once, so the numbers are the executor's own cost: command
 * decode, registry lookup, scratch check and the per-call setup. "decode"
 * stubs additionally read every input and output tensor view like a real
 * validator, so they include the block-table lookups. Ticks are simulator
 * pcycles on Hexagon (link -lhexagon) and clock() ticks on the host. */
#include "prepared_executor.h"
#ifdef BENCH_REAL
#include "prepared_reference.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __hexagon__
#include <hexagon_sim_timer.h>
#else
#include <time.h>
#endif

#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);exit(1);}}while(0)
static uint64_t ticks(void){
#ifdef __hexagon__
    return hexagon_sim_read_pcycles();
#else
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000ull+(uint64_t)t.tv_nsec;
#endif
}
#ifdef __hexagon__
#define UNIT "pcycles"
#else
#define UNIT "ns"
#endif
static uint32_t u32(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint8_t *read_file(const char *path,uint32_t *n){FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long z=ftell(f);CHECK(z>0);CHECK(!fseek(f,0,SEEK_SET));uint8_t *b=malloc((size_t)z);CHECK(b&&fread(b,1,(size_t)z,f)==(size_t)z);fclose(f);*n=(uint32_t)z;return b;}
static int cover(void *ctx,const Nprg2Program *p,uint32_t i,const Nprg2Command *c,Nprg2Capability *cap){(void)ctx;(void)p;(void)i;(void)c;cap->supported=1;cap->may_fallback=0;cap->scratch_bytes=0;cap->concurrent_worker_scratch_bytes=0;return 1;}
static int ok_trivial(const Nprg2ExecCall *c,void *ctx){(void)c;(void)ctx;return 1;}
static volatile uint32_t sink;
static int ok_decode(const Nprg2ExecCall *c,void *ctx){
    (void)ctx;uint32_t x=0;
    for(uint32_t k=0;k<c->command.input_count;k++){uint32_t i;Nprg2TensorView v;if(nprg2_exec_input(c,k,&i,&v))return 0;x+=v.offset+v.block_count;}
    for(uint32_t k=0;k<c->command.output_count;k++){uint32_t i;Nprg2TensorView v;if(nprg2_exec_output(c,k,&i,&v))return 0;x+=v.offset+v.block_count;}
    sink=x;return 1;
}
static void model(const char *qpath,const char *npath,uint32_t reps){
    uint32_t qn,nn;uint8_t *q=read_file(qpath,&qn),*raw=read_file(npath,&nn),qsha[32],nsha[32];
    nprg2_sha256(q,qn,qsha);nprg2_sha256(raw,nn,nsha);
    Nprg2ExpectedHashes h={q+32,q+64,qsha,nsha};
    Nprg2Pool pools[7];for(int i=0;i<7;i++){pools[i].bytes=u32(raw+144+i*4);pools[i].base=pools[i].bytes?malloc(pools[i].bytes):NULL;CHECK(!pools[i].bytes||pools[i].base);}
    Nprg2Program p;CHECK(nprg2_open(&p,raw,nn,&h,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t *hot=malloc(300000);CHECK(hot);
    Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1,.resolve=cover};
    Nprg2Bound bound;CHECK(nprg2_bind_mock(&p,pools,&bc,&bound)==NPRG2_OK);
    uint8_t ib[16],ob[16];Nprg2ExecBuffer in={ib,16},out={ob,16};
    Nprg2ExecRequest r={.program=&p,.bound=&bound,.pool1_writable_bytes=0x780000u,.inputs=&in,.input_count=1,.outputs=&out,.output_count=1,.command_count=p.command_count};
    memcpy(r.pools,pools,sizeof pools);
    uint32_t nops=0,distinct=0;
    static const char *const names[3]={"trivial","decode","trivial+noeffect-nops"};
    for(int pass=0;pass<3;pass++){
        Nprg2Executor e;nprg2_executor_init(&e);
        for(uint32_t i=0;i<p.command_count;i++){
            Nprg2Command c;CHECK(nprg2_command_at(&bound,i,&c)==NPRG2_OK);
            if(!pass&&c.opcode==QDR_NOP)nops++;
            int seen=0;for(uint32_t j=0;j<NPRG2_EXEC_REGISTRY_CAPACITY;j++)if(e.handlers[j].used&&e.handlers[j].opcode==c.opcode&&e.handlers[j].variant==c.variant)seen=1;
            if(!seen){CHECK(nprg2_executor_register(&e,c.opcode,c.variant,pass==1?ok_decode:ok_trivial,ok_trivial,NULL)==NPRG2_EXEC_OK);if(!pass)distinct++;
                if(pass==2&&c.opcode==QDR_NOP)CHECK(nprg2_executor_mark_no_effect(&e,c.opcode,c.variant)==NPRG2_EXEC_OK);}
        }
        Nprg2ExecStats st;r.stats=&st;
        uint64_t pre=~0ull,full=~0ull,reuse=~0ull,prof=~0ull,hit=~0ull,steady=~0ull,t;
        uint64_t hrc[NPRG2_EXEC_REGISTRY_CAPACITY],hrn[NPRG2_EXEC_REGISTRY_CAPACITY],hvc[NPRG2_EXEC_REGISTRY_CAPACITY];
        Nprg2ExecProfile pf;
        for(uint32_t k=0;k<reps;k++){
            r.profile=NULL;r.reuse_preflight=0;
            t=ticks();CHECK(nprg2_executor_preflight(&e,&r)==NPRG2_EXEC_OK);t=ticks()-t;if(t<pre)pre=t;
            t=ticks();CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);t=ticks()-t;if(t<full)full=t;
            CHECK(st.executed_commands==p.command_count);
            CHECK(nprg2_executor_preflight(&e,&r)==NPRG2_EXEC_OK);r.reuse_preflight=1;
            t=ticks();CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);t=ticks()-t;if(t<reuse)reuse=t;
            CHECK(st.executed_commands==p.command_count);
            memset(hrc,0,sizeof hrc);memset(hrn,0,sizeof hrn);memset(hvc,0,sizeof hvc);memset(&pf,0,sizeof pf);
            pf.clock=ticks;pf.handler_run_cycles=hrc;pf.handler_run_count=hrn;pf.handler_validate_cycles=hvc;r.profile=&pf;r.reuse_preflight=0;
            t=ticks();CHECK(nprg2_executor_preflight(&e,&r)==NPRG2_EXEC_OK);r.reuse_preflight=1;
            CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);t=ticks()-t;if(t<prof)prof=t;
            CHECK(pf.preflights_reused==1);
            /* Steady state of an unchanged request across inferences: the preflight is a cache hit (signature and
             * table digest only), then the run takes the verified preflight. Older executors preflight in full here. */
            r.profile=NULL;r.reuse_preflight=1;
            CHECK(nprg2_executor_preflight(&e,&r)==NPRG2_EXEC_OK); /* the first inference: full, records the digest */
            t=ticks();CHECK(nprg2_executor_preflight(&e,&r)==NPRG2_EXEC_OK);uint64_t h=ticks()-t;if(h<hit)hit=h;
            t=ticks();CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);h+=ticks()-t;if(h<steady)steady=h;
            CHECK(st.executed_commands==p.command_count);
        }
        r.profile=NULL;r.reuse_preflight=0;
        unsigned long long uc=p.command_count;
        printf("BENCH %s %-22s commands=%u nops=%u keys=%u %s: preflight=%llu run(preflight+dispatch)=%llu run(reused preflight)=%llu preflight+run(reuse)+profile=%llu | per cmd: preflight=%llu run_full=%llu run_reuse=%llu\n",
               npath,names[pass],(unsigned)p.command_count,(unsigned)nops,(unsigned)distinct,UNIT,(unsigned long long)pre,(unsigned long long)full,
               (unsigned long long)reuse,(unsigned long long)prof,(unsigned long long)pre/uc,(unsigned long long)full/uc,(unsigned long long)reuse/uc);
        printf("BENCH %s %-22s %s: steady-state inference control = preflight(cached)=%llu + run=%llu -> %llu\n",npath,names[pass],UNIT,
               (unsigned long long)hit,(unsigned long long)(steady-hit),(unsigned long long)steady);
    }
#ifdef BENCH_REAL
    {   /* The real scalar reference validators (they dry-run each kernel, no
         * pool bytes are read) over the whole schedule: the preflight cost the
         * handlers add to the executor's own. Pools stay unwritten. */
        Nprg2Executor e;nprg2_executor_init(&e);uint32_t added=0;
        CHECK(nprg2_executor_register_reference_missing(&e,&added)==NPRG2_EXEC_OK);
        uint32_t sb=1800000u;uint8_t *scr=malloc(sb);CHECK(scr);
        /* Real model I/O sizes: x, y, hidden, t and the two outputs. */
        static const uint32_t ins[4]={81920,81920,655360,128},outs[2]={163840,655360};
        Nprg2ExecBuffer rin[4],rout[2];
        for(int k=0;k<4;k++){rin[k]=(Nprg2ExecBuffer){malloc(ins[k]),ins[k]};CHECK(rin[k].data);}
        for(int k=0;k<2;k++){rout[k]=(Nprg2ExecBuffer){malloc(outs[k]),outs[k]};CHECK(rout[k].data);}
        r.inputs=rin;r.input_count=4;r.outputs=rout;r.output_count=2;
        {   /* Dry-run validation reads constant values (permutations, shapes). */
            char cp[256];uint32_t cn;snprintf(cp,sizeof cp,"recovered/%s/constants/constant_pool_5.bin",strstr(npath,"alternate")?"alternate":"primary");
            uint8_t *cb=read_file(cp,&cn);CHECK(cn<=pools[5].bytes);memcpy(pools[5].base,cb,cn);free(cb);
        }
        r.scratch=scr;r.scratch_bytes=sb;Nprg2ExecStats st;r.stats=&st;
        uint64_t pre=~0ull,t;
        for(uint32_t k=0;k<reps;k++){
            t=ticks();int x=nprg2_executor_preflight(&e,&r);t=ticks()-t;
            if(x){fprintf(stderr,"real preflight %d at %u (op %u v %u) line %u\n",x,(unsigned)st.failed_command,st.failed_opcode,st.failed_variant,(unsigned)nprg2_exec_reference_error_line());exit(1);}
            if(t<pre)pre=t;
        }
        printf("BENCH %s real-validate commands=%u preflight=%llu %s per_cmd=%llu\n",npath,(unsigned)p.command_count,
               (unsigned long long)pre,UNIT,(unsigned long long)(pre/p.command_count));
        free(scr);for(int k=0;k<4;k++)free(rin[k].data);for(int k=0;k<2;k++)free(rout[k].data);
    }
#endif
    nprg2_close(&p);free(hot);for(int i=0;i<7;i++)free(pools[i].base);free(q);free(raw);
}
int main(int argc,char **argv){
    CHECK(argc>=3);uint32_t reps=(argc-1)%2?(uint32_t)atoi(argv[argc-1]):1u;
    for(int a=1;a+1<argc;a+=2)model(argv[a],argv[a+1],reps);
    return 0;
}
