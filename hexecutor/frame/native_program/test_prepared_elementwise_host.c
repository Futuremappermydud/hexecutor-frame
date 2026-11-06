/* ASan/UBSan read-only preflight proof for real LinearClip and requant NPRG metadata.
 * Numeric HVX execution remains covered by test_prepared_elementwise.c. */
#include "prepared_elementwise.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); exit(1); } } while (0)
static uint16_t rd16le(const uint8_t *p){return (uint16_t)(p[0]|p[1]<<8);}
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
    {
        /* Handlers registered for scratch region 1: region 0 too small to
         * matter, region 1 absent -> rejected, too small -> rejected,
         * aliasing region 0 -> argument error, sized -> accepted. */
        Nprg2Executor e1;nprg2_executor_init(&e1);
        CHECK(nprg2_executor_register_elementwise_region(&e1,NPRG2_EXEC_SCRATCH_NEAR)==NPRG2_EXEC_OK);
        CHECK(nprg2_executor_register_elementwise_region(&e1,NPRG2_EXEC_SCRATCH_REGIONS)!=NPRG2_EXEC_OK);
        void *r0_raw=malloc(64);CHECK(r0_raw);
        Nprg2ExecRequest q1=req;q1.scratch=r0_raw;q1.scratch_bytes=64;
        CHECK(nprg2_executor_preflight(&e1,&q1)==NPRG2_EXEC_REJECTED);
        q1.extra_scratch[0]=(Nprg2ExecBuffer){(uint8_t *)scratch,NPRG2_ELEMENTWISE_SCRATCH_BYTES-1u};
        CHECK(nprg2_executor_preflight(&e1,&q1)==NPRG2_EXEC_REJECTED);
        q1.extra_scratch[0]=(Nprg2ExecBuffer){(uint8_t *)r0_raw,64};
        CHECK(nprg2_executor_preflight(&e1,&q1)==NPRG2_EXEC_ARGUMENT);
        q1.extra_scratch[0]=(Nprg2ExecBuffer){(uint8_t *)scratch,NPRG2_ELEMENTWISE_SCRATCH_BYTES};
        CHECK(nprg2_executor_preflight(&e1,&q1)==NPRG2_EXEC_OK);
        free(r0_raw);
    }
    for(unsigned i=0;i<7;i++)if(pools[i].bytes)CHECK(!memcmp(before[i],pools[i].base,pools[i].bytes));
    printf("PASS ASan/UBSan %s actual-metadata preflight: %s command=%u\n",name,qpath,cmd);
    free(scratch_raw);for(unsigned i=0;i<7;i++){free(before[i]);free(pools[i].base);}free(hot);nprg2_close(&p);free(fx);free(q);free(raw);
}

/* Every command of the given opcode/variant in the real schedule must be
 * accepted by its HVX validator (so none falls back to the reference kernel);
 * then three mutations of the real metadata must be rejected. Pools are zero:
 * validation reads metadata and the call table only. */
static int view_shape_t(const Nprg2TensorView *v,int32_t d[4],int32_t m[4],int32_t o[4]){
    for(int i=0;i<4;i++){d[i]=(int16_t)rd16le(v->geometry_record+6+2*i);m[i]=(int16_t)rd16le(v->geometry_record+14+2*i);o[i]=(int16_t)rd16le(v->geometry_record+22+2*i);}return 1;}
static void run_all_u16(const char *qpath,const char *npath,uint8_t opcode,uint8_t variant,const char *name,uint32_t expect) {
    uint32_t qn,nn;uint8_t *q=read_file(qpath,&qn),*raw=read_file(npath,&nn);
    uint8_t qsha[32],nsha[32];nprg2_sha256(q,qn,qsha);nprg2_sha256(raw,nn,nsha);
    Nprg2ExpectedHashes x={q+32,q+64,qsha,nsha};Nprg2Pool pools[7]={0};
    for(unsigned i=0;i<7;i++){uint32_t z=rd32(raw+144+4*i);pools[i].bytes=z;pools[i].base=z?calloc(1,z):NULL;CHECK(!z||pools[i].base);}
    Nprg2Program p;CHECK(nprg2_open(&p,raw,nn,&x,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t *hot=calloc(1,300000);CHECK(hot);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1,.resolve=resolve_all};Nprg2Bound bound;
    CHECK(nprg2_bind_mock(&p,pools,&bc,&bound)==NPRG2_OK);
    Nprg2Executor e;nprg2_executor_init(&e);CHECK(nprg2_executor_register_elementwise(&e)==NPRG2_EXEC_OK);
    void *scratch_raw=malloc(NPRG2_ELEMENTWISE_SCRATCH_BYTES+sizeof(uintptr_t));CHECK(scratch_raw);void *scratch=(void *)(((uintptr_t)scratch_raw+sizeof(uintptr_t)-1)&~(uintptr_t)(sizeof(uintptr_t)-1));
    uint32_t n=0,first=UINT32_MAX;
    for(uint32_t i=0;i<p.command_count;i++){
        Nprg2Command c;CHECK(nprg2_command_at(&bound,i,&c)==NPRG2_OK);
        if(c.opcode!=opcode||c.variant!=variant)continue;
        Nprg2ExecRequest req={.program=&p,.bound=&bound,.pool1_writable_bytes=0x780000,.scratch=scratch,.scratch_bytes=NPRG2_ELEMENTWISE_SCRATCH_BYTES,.command_first=i,.command_count=1};
        memcpy(req.pools,pools,sizeof pools);
        int rc=nprg2_executor_preflight(&e,&req);
        if(rc!=NPRG2_EXEC_OK){fprintf(stderr,"%s %s cmd %u rejected rc=%d\n",qpath,name,i,rc);exit(1);}
        if(first==UINT32_MAX)first=i;
        n++;
    }
    CHECK(n==expect);
    /* mutations on the first command: a call-argument change outside the proven ranges, a
     * wrong call table function id, and an output block overlapping an input block. */
    Nprg2Command c;CHECK(nprg2_command_at(&bound,first,&c)==NPRG2_OK&&c.call_count==1);
    uint8_t *rec=(uint8_t *)bound.calls.data+(size_t)c.call_first*88u;
    Nprg2ExecRequest req={.program=&p,.bound=&bound,.pool1_writable_bytes=0x780000,.scratch=scratch,.scratch_bytes=NPRG2_ELEMENTWISE_SCRATCH_BYTES,.command_first=first,.command_count=1};
    memcpy(req.pools,pools,sizeof pools);
    /* a command rejected here is still accepted by the reference kernel registered behind it in real use;
     * with only the HVX handler registered it is REJECTED. */
    uint8_t saved[88];memcpy(saved,rec,88);
    rec[0]^=0x40;CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_REJECTED);memcpy(rec,saved,88);
    for(unsigned arg=3;arg<9;arg++){ /* make argument `arg` huge (>= 2^40) */
        rec[8+8*arg+5]=0xff;
        int rc=nprg2_executor_preflight(&e,&req);
        CHECK(rc==NPRG2_EXEC_OK||rc==NPRG2_EXEC_REJECTED);
        memcpy(rec,saved,88);
    }
    CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_OK);
    uint32_t oti,iti;Nprg2TensorView ov,iv;Nprg2ExecCall call={.command_index=first,.command=c,.program=&p,.bound=&bound,.pools=pools,.pool1_writable_bytes=0x780000};
    CHECK(!nprg2_exec_output(&call,0,&oti,&ov)&&!nprg2_exec_input(&call,0,&iti,&iv)&&ov.block_count&&iv.block_count);
    /* first block each tensor touches (slice offset + crop origin), as in the CPU-prefix tests */
    int32_t od[4],om[4],oo[4],id[4],im[4],io[4];
    CHECK(view_shape_t(&ov,od,om,oo)&&view_shape_t(&iv,id,im,io));
    uint32_t oi0=(uint32_t)((((uint64_t)oo[0]*((im[1]+7)/8)+oo[1]/8)*((om[2]+3)/4)+oo[2]/4)*((om[3]+31)/32)+ov.slice_origin+oo[3]/32);
    uint32_t ii0=(uint32_t)((((uint64_t)io[0]*((im[1]+7)/8)+io[1]/8)*((im[2]+3)/4)+io[2]/4)*((im[3]+31)/32)+iv.slice_origin+io[3]/32);
    CHECK(oi0<ov.block_count&&ii0<iv.block_count);
    uint32_t *ob=(uint32_t *)ov.blocks;uint32_t oldv=ob[oi0];
    ob[oi0]=iv.blocks[ii0];CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_REJECTED);ob[oi0]=oldv;
    ob[oi0]=0x780001u;CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_REJECTED);ob[oi0]=oldv;
    CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_OK);
    printf("PASS ASan/UBSan %s: all %u commands in %s accepted by the HVX validator\n",name,n,qpath);
    free(scratch_raw);for(unsigned i=0;i<7;i++)free(pools[i].base);free(hot);nprg2_close(&p);free(q);free(raw);
}

/* SIGN/39, LINEARCLIP/67 (CH u16) and LINEARCLIP/69 (flat u8): every command of the real schedule, with the real
 * constant pool (the fi operands live in pool 5), must be accepted by the HVX validator, and mutated metadata or
 * constants must be rejected. */
static void run_all_new(const char *model,uint8_t opcode,uint8_t variant,const char *name,uint32_t expect) {
    char qpath[256],npath[256],cpath[256];
    snprintf(qpath,sizeof qpath,"frame/build/resident-%s.qdrx",model);
    snprintf(npath,sizeof npath,"frame/native_program/examples/%s-v2.nprg",model);
    snprintf(cpath,sizeof cpath,"recovered/%s/constants/constant_pool_5.bin",model);
    uint32_t qn,nn,cn;uint8_t *q=read_file(qpath,&qn),*raw=read_file(npath,&nn),*consts=read_file(cpath,&cn);
    uint8_t qsha[32],nsha[32];nprg2_sha256(q,qn,qsha);nprg2_sha256(raw,nn,nsha);
    Nprg2ExpectedHashes x={q+32,q+64,qsha,nsha};Nprg2Pool pools[7]={0};
    for(unsigned i=0;i<7;i++){uint32_t z=rd32(raw+144+4*i);pools[i].bytes=z;pools[i].base=z?calloc(1,z):NULL;CHECK(!z||pools[i].base);}
    CHECK(pools[5].bytes==cn);memcpy(pools[5].base,consts,cn);
    Nprg2Program p;CHECK(nprg2_open(&p,raw,nn,&x,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t *hot=calloc(1,300000);CHECK(hot);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1,.resolve=resolve_all};Nprg2Bound bound;
    CHECK(nprg2_bind_mock(&p,pools,&bc,&bound)==NPRG2_OK);
    Nprg2Executor e;nprg2_executor_init(&e);CHECK(nprg2_executor_register_elementwise(&e)==NPRG2_EXEC_OK);
    void *scratch_raw=malloc(NPRG2_ELEMENTWISE_SCRATCH_BYTES+sizeof(uintptr_t));CHECK(scratch_raw);void *scratch=(void *)(((uintptr_t)scratch_raw+sizeof(uintptr_t)-1)&~(uintptr_t)(sizeof(uintptr_t)-1));
    uint32_t n=0,first=UINT32_MAX;
    Nprg2ExecRequest req={.program=&p,.bound=&bound,.pool1_writable_bytes=0x780000,.scratch=scratch,.scratch_bytes=NPRG2_ELEMENTWISE_SCRATCH_BYTES,.command_count=1};
    memcpy(req.pools,pools,sizeof pools);
    for(uint32_t i=0;i<p.command_count;i++){
        Nprg2Command c;CHECK(nprg2_command_at(&bound,i,&c)==NPRG2_OK);
        if(c.opcode!=opcode||c.variant!=variant)continue;
        req.command_first=i;
        int rc=nprg2_executor_preflight(&e,&req);
        if(rc!=NPRG2_EXEC_OK){fprintf(stderr,"%s %s cmd %u rejected rc=%d\n",model,name,i,rc);exit(1);}
        if(first==UINT32_MAX)first=i;
        n++;
    }
    CHECK(n==expect);
    req.command_first=first;
    Nprg2Command c;CHECK(nprg2_command_at(&bound,first,&c)==NPRG2_OK);
    uint32_t oti,iti;Nprg2TensorView ov,iv;Nprg2ExecCall call={.command_index=first,.command=c,.program=&p,.bound=&bound,.pools=pools,.pool1_writable_bytes=0x780000};
    CHECK(!nprg2_exec_output(&call,0,&oti,&ov)&&!nprg2_exec_input(&call,0,&iti,&iv));
    if(opcode==QDR_LINEARCLIP){
        /* the fi operand: shift out of range, a negative-multiplier sign for the flat form, a non-zero zero point */
        uint32_t fti;Nprg2TensorView fv;CHECK(!nprg2_exec_input(&call,1,&fti,&fv)&&fv.pool==5);
        uint8_t *fi=pools[5].base+fv.offset,saved[48];memcpy(saved,fi,48);
        fi[43]=0x7f;CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_REJECTED);memcpy(fi,saved,48);
        if(variant==69){
            fi[27]^=0x80;CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_REJECTED);memcpy(fi,saved,48);
            fi[44]=1;CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_REJECTED);memcpy(fi,saved,48);
        }
        /* a fi operand outside pool 5 is not a constant */
        uint8_t *ft=(uint8_t *)bound.tensor.data+(size_t)fti*NPRG2_TENSOR_BYTES+3u,pool=ft[0];ft[0]=1;
        CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_REJECTED);ft[0]=pool;
    }
    if(opcode==QDR_SIGN){
        uint8_t *g=(uint8_t *)ov.geometry_record+34,saved[8];memcpy(saved,g,8);memset(g,0,8);
        CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_REJECTED);memcpy(g,saved,8);
        uint8_t *ig=(uint8_t *)iv.geometry_record+30,isaved[4];memcpy(isaved,ig,4);ig[2]=1;ig[3]=0;
        CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_REJECTED);memcpy(ig,isaved,4);
    }
    if(variant==69){
        /* identical in place is fine, a shifted overlap or an output past the pool is not */
        uint8_t *ot=(uint8_t *)bound.tensor.data+(size_t)oti*NPRG2_TENSOR_BYTES+4u;
        uint32_t off=rd32(ot),ioff=rd32((uint8_t *)bound.tensor.data+(size_t)iti*NPRG2_TENSOR_BYTES+4u);
        uint32_t t1[3]={ioff,ioff+64u,0x780000u-16u};
        for(int k=0;k<3;k++){
            memcpy(ot,&t1[k],4);
            CHECK(nprg2_executor_preflight(&e,&req)==(k?NPRG2_EXEC_REJECTED:NPRG2_EXEC_OK));
        }
        memcpy(ot,&off,4);
    } else {
        int32_t od[4],om[4],oo[4],id[4],im[4],io[4];
        CHECK(view_shape_t(&ov,od,om,oo)&&view_shape_t(&iv,id,im,io));
        uint32_t oi0=(uint32_t)((((uint64_t)oo[0]*((om[1]+7)/8)+oo[1]/8)*((om[2]+3)/4)+oo[2]/4)*((om[3]+31)/32)+ov.slice_origin+oo[3]/32);
        uint32_t ii0=(uint32_t)((((uint64_t)io[0]*((im[1]+7)/8)+io[1]/8)*((im[2]+3)/4)+io[2]/4)*((im[3]+31)/32)+iv.slice_origin+io[3]/32);
        CHECK(oi0<ov.block_count&&ii0<iv.block_count);
        uint32_t *ob=(uint32_t *)ov.blocks;uint32_t oldv=ob[oi0];
        ob[oi0]=iv.blocks[ii0];CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_REJECTED);ob[oi0]=oldv;
        ob[oi0]=0x780001u;CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_REJECTED);ob[oi0]=oldv;
    }
    CHECK(nprg2_executor_preflight(&e,&req)==NPRG2_EXEC_OK);
    printf("PASS ASan/UBSan %s: all %u commands in %s accepted by the HVX validator; mutations rejected\n",name,n,qpath);
    free(scratch_raw);for(unsigned i=0;i<7;i++)free(pools[i].base);free(hot);nprg2_close(&p);free(q);free(raw);free(consts);
}
int main(void) {
    run_model("frame/build/resident-primary.qdrx","frame/native_program/examples/primary-v2.nprg","frame/native_program/fixtures/primary-smoke-linearclip.epcpu",QDR_LINEARCLIP,68,"LinearClip");
    run_model("frame/build/resident-alternate.qdrx","frame/native_program/examples/alternate-v2.nprg","frame/native_program/fixtures/alternate-smoke-linearclip.epcpu",QDR_LINEARCLIP,68,"LinearClip");
    run_model("frame/build/resident-primary.qdrx","frame/native_program/examples/primary-v2.nprg","frame/native_program/fixtures/primary-smoke-requant.epcpu",QDR_REQUANT_U8,70,"requant u8");
    run_model("frame/build/resident-alternate.qdrx","frame/native_program/examples/alternate-v2.nprg","frame/native_program/fixtures/alternate-smoke-requant.epcpu",QDR_REQUANT_U8,70,"requant u8");
    static const struct { uint8_t op,var;const char *name;uint32_t pn,an; } u16[]={
        {QDR_REQUANT_U16,71,"requant u16",54,64},{QDR_ADD1111_U16,59,"add1111 u16",60,60},
        {QDR_MUL_U16,74,"mul u16",50,50},{QDR_ADD_U16,14,"add u16",20,20},
        {QDR_ADDSUB_U16,60,"addsub u16",12,12},{QDR_MUL1111_U16,73,"mul1111 u16",10,10}};
    for(unsigned i=0;i<sizeof u16/sizeof u16[0];i++){
        run_all_u16("frame/build/resident-primary.qdrx","frame/native_program/examples/primary-v2.nprg",u16[i].op,u16[i].var,u16[i].name,u16[i].pn);
        run_all_u16("frame/build/resident-alternate.qdrx","frame/native_program/examples/alternate-v2.nprg",u16[i].op,u16[i].var,u16[i].name,u16[i].an);
    }
    run_all_new("primary",QDR_SIGN,39,"sign",12);run_all_new("alternate",QDR_SIGN,39,"sign",12);
    run_all_new("primary",QDR_LINEARCLIP,67,"linearclip u16",30);run_all_new("alternate",QDR_LINEARCLIP,67,"linearclip u16",30);
    run_all_new("primary",QDR_LINEARCLIP,69,"linearclip flat u8",9);run_all_new("alternate",QDR_LINEARCLIP,69,"linearclip flat u8",9);
    return 0;
}
