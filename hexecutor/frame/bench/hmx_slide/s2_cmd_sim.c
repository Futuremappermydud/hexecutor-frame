/* Real 6/33 commands through the executor with the simulated HMX hooks: for each
 * distinct shape of the variant, the HVX fast conv (no hmx) vs the HMX slide path,
 * hash of the command's output blocks only, path taken and conv phase cycles.
 * Build via slide_cmd_sim.sh (include prepared_reference.c, qd_hmx_ops_encoded.S).
 * usage: elf resident.qdrx program.nprg variant constant_pool_5.bin [maxcmds] */
#include "../../native_program/prepared_reference.c"
#include <stdio.h>
#include <stdlib.h>
#ifdef __hexagon__
#include <hexagon_standalone.h>
#include <hexagon_sim_timer.h>
#endif
#define CK(x) do{if(!(x)){fprintf(stderr,"FAIL %d %s\n",__LINE__,#x);exit(1);}}while(0)
#ifdef __hexagon__
extern void qd_hmx_wide(void *, void *, void *, void *, uint32_t);
extern void qd_hmx_finish(void *, void *);
extern void qd_hmx_wide_n(void *, void *, void *, void *, uint32_t, uint32_t);
extern void qd_hmx_native_n(void *, void *, void *, void *const *, uint32_t, uint32_t);
extern void qd_hmx_native16_n(void *, void *, void *, void *const *, uint32_t, uint32_t);
extern void qd_hmx_slide3_n(const uint32_t *, const void *, const void *, const uint32_t *, uint32_t, uint32_t);
static uint32_t en(void){uint32_t s;__asm__ volatile("%0=ssr":"=r"(s));uint32_t on=s|(1u<<26);__asm__ volatile("ssr=%0\n isync\n barrier"::"r"(on):"memory");return s;}
static void lv(uint32_t s){__asm__ volatile("ssr=%0\n isync\n barrier"::"r"(s):"memory");}
static uint64_t clk(void){return hexagon_sim_read_pcycles();}
static const Nprg2HmxOps hops={.enter=en,.leave=lv,.wide=qd_hmx_wide,.finish=qd_hmx_finish,.wide_n=qd_hmx_wide_n,.native_n=qd_hmx_native_n,
    .native16_n=qd_hmx_native16_n,.slide3_n=qd_hmx_slide3_n};
#else
static uint64_t clk(void){return 0;}
static void dfin(void *a,void *b){(void)a;(void)b;}
static const Nprg2HmxOps hops={.finish=dfin,.native_n=NULL};
#endif
static uint8_t *rf(const char *p,uint32_t *n){FILE *f=fopen(p,"rb");CK(f);fseek(f,0,SEEK_END);long z=ftell(f);fseek(f,0,SEEK_SET);uint8_t *b=malloc(z);CK(fread(b,1,z,f)==(size_t)z);fclose(f);*n=z;return b;}
static int cover(void *x,const Nprg2Program *p,uint32_t i,const Nprg2Command *c,Nprg2Capability *cap){(void)x;(void)p;(void)i;(void)c;cap->supported=1;return 1;}
static uint64_t ohash(G *g){
    const T *o=&g->out;uint32_t onbw=(o->m[2]+7)/8u,onbd=(o->m[3]+31)/32u;uint64_t h=1469598103934665603ull;
    for(int by=0;by<(o->d[1]+7)/8;by++)for(int bx=0;bx<(o->d[2]+7)/8;bx++)for(int j=0;j<o->d[3]/32;j++){
        uint32_t idx=(((uint32_t)(o->p[1]>>3)+by)*onbw+(uint32_t)(o->p[2]>>3)+bx)*onbd+o->slice+j;
        if(idx>=o->nblocks)continue;uint32_t e=o->blocks[idx],pool=e&127u;if(!pool||!g->pool[pool])continue;
        const uint32_t *w=(const uint32_t *)(g->pool[pool]+(e&~127u));
        for(int i=0;i<512;i++){h^=w[i];h*=1099511628211ull;}}
    return h;}
int main(int argc,char **argv){
#ifdef __hexagon__
    uint32_t base=__rdcfg(__vtcm_base)<<16,pn=base>>12;
    {uint64_t entry=((uint64_t)(0xc0000000u|pn)<<32)|(0xf6000000u|(pn<<1)|0x40u);__asm__ volatile("tlbw(%0,%1)\n isync"::"r"(entry),"r"(120):"memory");}
    {uint32_t ssr;__asm__ volatile("%0=ssr":"=r"(ssr));ssr|=1u<<26;__asm__ volatile("ssr=%0\n isync"::"r"(ssr):"memory");}
#else
    uintptr_t base=(uintptr_t)aligned_alloc(2048,0x800000);
#endif
    uint32_t qn,nn,cn;uint8_t *q=rf(argv[1],&qn),*raw=rf(argv[2],&nn),qs[32],ns[32];uint32_t want=atoi(argv[3]);uint8_t *cst=rf(argv[4],&cn);
    uint32_t maxc=argc>5?atoi(argv[5]):100;uint32_t wantpath=argc>6?atoi(argv[6]):4;
    nprg2_sha256(q,qn,qs);nprg2_sha256(raw,nn,ns);
    Nprg2ExpectedHashes h={q+32,q+64,qs,ns};Nprg2Pool pools[7];
    for(int i=0;i<7;i++){pools[i].bytes=rd32(raw+144+i*4);pools[i].base=NULL;if(!pools[i].bytes)continue;
        pools[i].base=i==1?(uint8_t *)(uintptr_t)base:aligned_alloc(2048,(pools[i].bytes+2047)&~2047u);CK(pools[i].base);memset(pools[i].base,0,pools[i].bytes);}
    CK(pools[1].bytes<=0x800000u);
    memcpy(pools[5].base,cst,cn<pools[5].bytes?cn:pools[5].bytes);
    {uint32_t *w=(uint32_t *)pools[1].base,x=2463534242u;for(uint32_t i=0;i<0x780000u/4;i++){x^=x<<13;x^=x>>17;x^=x<<5;w[i]=x;}}
    Nprg2Program p;CK(nprg2_open(&p,raw,nn,&h,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t *hot=calloc(1,300000);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1,.resolve=cover};
    Nprg2Bound b;CK(nprg2_bind_mock(&p,pools,&bc,&b)==NPRG2_OK);
    static uint64_t phase[8];
    Nprg2ReferenceConv conv;memset(&conv,0,sizeof conv);
    conv.workspace=(uint8_t *)(uintptr_t)base+0x782000u;conv.workspace_bytes=0x7e000;conv.near=aligned_alloc(128,0x8000);conv.near_bytes=0x8000;
    conv.clock=clk;conv.phase=phase;
    Nprg2Executor e;nprg2_executor_init(&e);uint32_t added=0;
    CK(nprg2_executor_register_reference_conv(&e,&conv,&added)==NPRG2_EXEC_OK);
    CK(nprg2_executor_register_reference_missing(&e,&added)==NPRG2_EXEC_OK);
    uint32_t scr_bytes=0x100000;uint8_t *scr=aligned_alloc(128,scr_bytes);
    Nprg2ExecBuffer in[4]={{0}},ob[2]={{0}};Nprg2ExecStats st;memset(&st,0,sizeof st);
    Nprg2ExecRequest pr={.program=&p,.bound=&b,.pool1_writable_bytes=0x780000u,.inputs=in,.input_count=4,.outputs=ob,.output_count=2,
        .scratch=scr,.scratch_bytes=scr_bytes,.command_count=1,.stats=&st};
    memcpy(pr.pools,pools,sizeof pools);
    uint32_t cursor=0,rok=0,rfail=0;
    /* weights and Fi reach pool 1 through the schedule's weight-prep commands (pool 5 -> pool 1); pool 1 is reused over time,
     * so replay, in order, only those before the command under test */
#define REPLAY(upto) do{conv.hmx=NULL;for(;cursor<(upto);cursor++){Nprg2Command pc;CK(nprg2_command_at(&b,cursor,&pc)==NPRG2_OK);\
        if(!(pc.variant>=19&&pc.variant<=30&&pc.variant!=24&&pc.variant!=25))continue;pr.command_first=cursor;\
        if(nprg2_executor_preflight(&e,&pr)!=NPRG2_EXEC_OK||nprg2_executor_run(&e,&pr)!=NPRG2_EXEC_OK){rfail++;continue;}rok++;}}while(0)
    uint32_t seen_n=0;uint32_t seen[64][4];int bad=0;
    for(uint32_t cmd=0;cmd<p.command_count && seen_n<maxc;cmd++){
        Nprg2Command cm;CK(nprg2_command_at(&b,cmd,&cm)==NPRG2_OK);
        if(cm.variant!=want)continue;
        REPLAY(cmd);
        Nprg2ExecCall c;memset(&c,0,sizeof c);c.command=cm;c.command_index=cmd;c.program=&p;c.bound=&b;c.pools=pools;c.pool1_writable_bytes=0x780000u;
        c.scratch=scr;c.scratch_bytes=scr_bytes;c.scratch_region[0]=(Nprg2ExecBuffer){scr,scr_bytes};c.preflighted=1;if(want==36&&0)continue;
        G g;{int sr=setup(&g,&c,0,0,NULL);if(sr){printf("setup rc %d line %lu cmd %u\n",sr,(unsigned long)nprg2_exec_reference_error_line(),cmd);return 1;}}
        uint32_t key[4]={g.out.d[1],g.out.d[2],g.out.d[3],g.in[0].d[3]};int dup=0;
        for(uint32_t k=0;k<seen_n;k++)if(!memcmp(seen[k],key,16))dup=1;if(dup&&!getenv("S2_ALL"))continue;memcpy(seen[seen_n++],key,16);
        Nprg2ExecRequest r={.program=&p,.bound=&b,.pool1_writable_bytes=0x780000u,.inputs=in,.input_count=4,.outputs=ob,.output_count=2,
            .scratch=scr,.scratch_bytes=scr_bytes,.command_first=cmd,.command_count=1,.stats=&st};
        memcpy(r.pools,pools,sizeof pools);
        uint64_t hv[2]={0,0},cyc[2]={0,0},ph[2][4]={{0}};
        for(int mode=0;mode<2;mode++){ /* 0: HVX fast conv, 1: HMX slide (second run timed) */
            conv.hmx=(mode&&hops.finish)?&hops:NULL;
            for(int rep=0;rep<2;rep++){
                for(int k=0;k<8;k++)phase[k]=0;
                CK(nprg2_executor_preflight(&e,&r)==NPRG2_EXEC_OK);
                uint64_t t=clk();CK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);t=clk()-t;
                cyc[mode]=t;for(int k=0;k<4;k++)ph[mode][k]=phase[k];
            }
            G g2;{int sr=setup(&g2,&c,0,0,NULL);if(sr){printf("setup2 rc %d line %lu\n",sr,(unsigned long)nprg2_exec_reference_error_line());return 1;}}hv[mode]=ohash(&g2);
            if(mode==0){ /* clear the output blocks so the HMX run must write them */
                const T *o=&g2.out;uint32_t onbw=(o->m[2]+7)/8u,onbd=(o->m[3]+31)/32u;
                for(int by=0;by<(o->d[1]+7)/8;by++)for(int bx=0;bx<(o->d[2]+7)/8;bx++)for(int j=0;j<o->d[3]/32;j++){
                    uint32_t idx=(((uint32_t)(o->p[1]>>3)+by)*onbw+(uint32_t)(o->p[2]>>3)+bx)*onbd+o->slice+j;
                    if(idx>=o->nblocks)continue;uint32_t en2=o->blocks[idx],pool=en2&127u;if(pool&&g2.pool[pool])memset(g2.pool[pool]+(en2&~127u),0xa5,2048);}
            }
        }
        int ok=hv[0]==hv[1]&&(nprg2_exec_reference_conv_path()==(int)wantpath||nprg2_exec_reference_conv_path()==3);if(!ok)bad++; /* path 4 = hmx_native */
        printf("cmd %u out %ux%ux%u cin %u path=%u exact=%d hash=%016llx hvx_cyc=%llu hmx_cyc=%llu hmx_phase[stage,mac,setup,out]=%llu,%llu,%llu,%llu\n",
               cmd,key[0],key[1],key[2],key[3],nprg2_exec_reference_conv_path(),ok,(unsigned long long)hv[1],(unsigned long long)cyc[0],(unsigned long long)cyc[1],
               (unsigned long long)ph[1][0],(unsigned long long)ph[1][1],(unsigned long long)ph[1][2],(unsigned long long)ph[1][3]);
    }
    printf("s2 in-place calls %u, bcopy calls %u\n",s2_ip_taken,s2_bc_taken);
    printf("replay %u ran %u failed\n",rok,rfail);
    printf("%s bad=%d\n",bad?"FAIL":"PASS",bad);
    return bad;
}
