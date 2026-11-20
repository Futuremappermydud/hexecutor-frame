/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_native.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);exit(1);}}while(0)
static uint32_t rd32(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint16_t rd16(const uint8_t *p){return (uint16_t)p[0]|(uint16_t)p[1]<<8;}
static void wr16(uint8_t *p,uint16_t x){p[0]=(uint8_t)x;p[1]=(uint8_t)(x>>8);}
static uint8_t *read_file(const char *path,uint32_t *n){FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long z=ftell(f);CHECK(z>0&&(uint64_t)z<=NPRG2_MAX_IMAGE_BYTES);CHECK(!fseek(f,0,SEEK_SET));uint8_t *p=malloc((size_t)z);CHECK(p&&fread(p,1,(size_t)z,f)==(size_t)z);fclose(f);*n=(uint32_t)z;return p;}
static void pool_alloc(const uint8_t *raw,Nprg2Pool p[7]){for(uint32_t i=0;i<7;i++){p[i].bytes=rd32(raw+144+i*4);if(i==1&&p[i].bytes)p[i].base=aligned_alloc(128,p[i].bytes);else p[i].base=p[i].bytes?calloc(1,p[i].bytes):NULL;CHECK(!p[i].bytes||p[i].base);if(p[i].bytes)memset(p[i].base,0,p[i].bytes);}}
static void pool_free(Nprg2Pool p[7]){for(int i=0;i<7;i++)free(p[i].base);}
static int bind_all(void *ctx,const Nprg2Program *p,uint32_t ix,const Nprg2Command *c,Nprg2Capability *cap){(void)ctx;(void)p;(void)ix;(void)c;cap->supported=1;cap->may_fallback=0;cap->scratch_bytes=0;cap->concurrent_worker_scratch_bytes=0;return 1;}
static uint32_t submitted,submitted_1x1,submitted_3x3;
static int no_kernel(void *ctx,const Nprg2ExecCall *call,const QdDirectBinding *b,uint8_t family){(void)ctx;CHECK(call&&b&&b->in_desc&&b->out_desc&&b->weights&&b->fi);CHECK(family==1||family==3);submitted++;if(family==1)submitted_1x1++;else submitted_3x3++;return 1;}
void qd_set_hmx_params_stride1(uint32_t *c,uint32_t a,uint32_t b,uint32_t d,uint32_t e,uint32_t f,uint32_t g,uint32_t h){c[0]=a;c[1]=b;c[2]=d;c[3]=e;c[4]=f;c[5]=g;c[6]=h;c[12]=8;}
static int find_handler(const Nprg2Executor *e,uint8_t op,uint8_t variant){for(uint32_t i=0;i<NPRG2_EXEC_REGISTRY_CAPACITY;i++)if(e->handlers[i].used&&e->handlers[i].opcode==op&&e->handlers[i].variant==variant)return (int)i;return -1;}
static void run_model(const char *qpath,const char *npath){
    submitted=0;submitted_1x1=0;submitted_3x3=0;
    uint32_t qn,nn;uint8_t *q=read_file(qpath,&qn),*raw=read_file(npath,&nn),qhash[32],nhash[32];Nprg2ExpectedHashes expected;
    nprg2_sha256(q,qn,qhash);nprg2_sha256(raw,nn,nhash);expected.context_sha256=q+32;expected.constants_sha256=q+64;expected.source_sha256=qhash;expected.prepared_sha256=nhash;
    Nprg2Pool pools[7]={0};pool_alloc(raw,pools);Nprg2Program p;CHECK(nprg2_open(&p,raw,nn,&expected,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t *hot=calloc(1,300000);CHECK(hot);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1,.resolve=bind_all};Nprg2Bound bound;CHECK(nprg2_bind_mock(&p,pools,&bc,&bound)==NPRG2_OK);
    uint32_t *tables=aligned_alloc(128,2048);uint32_t *abi=aligned_alloc(128,128);CHECK(tables&&abi);uint8_t *tail=pools[1].base+0x780000u;
    memset(tables,0x39,2048);memset(abi,0x39,128);memset(tail,0x39,0x80000u);uint8_t *tail_before=malloc(0x80000u);CHECK(tail_before);memset(tail_before,0x39,0x80000u);
    Nprg2NativeConfig native={.active_vtcm_bytes=0x780000u,.tables=tables,.table_bytes=2048,.abi=abi,.abi_bytes=120,.tail=tail,.tail_bytes=0x80000u,.submit_and_drain=no_kernel};
    Nprg2Executor e;nprg2_executor_init(&e);CHECK(nprg2_executor_register_native_conv(&e,&native)==NPRG2_EXEC_OK);
    Nprg2ExecRequest full={.program=&p,.bound=&bound,.pool1_writable_bytes=0x780000u,.command_first=0,.command_count=p.command_count};memcpy(full.pools,pools,sizeof pools);
    CHECK(nprg2_executor_run(&e,&full)==NPRG2_EXEC_UNSUPPORTED);CHECK(submitted==0);CHECK(!memcmp(tables,tail_before,2048)&&!memcmp(abi,tail_before,120)&&!memcmp(tail,tail_before,0x80000u));
    uint32_t accepted=0,rejected=0,first_1x1=UINT32_MAX,first_3x3=UINT32_MAX;
    uint32_t padded_ci12=UINT32_MAX,padded_ci20=UINT32_MAX;int tested_same_opcode_bad_shape=0;
    for(uint32_t i=0;i<p.command_count;i++){
        Nprg2Command cmd;CHECK(nprg2_command_at(&bound,i,&cmd)==NPRG2_OK);
        if(cmd.opcode!=QDR_HMX_CONV||(cmd.variant!=32&&cmd.variant!=33))continue;
        int hi=find_handler(&e,cmd.opcode,cmd.variant);CHECK(hi>=0);
        Nprg2ExecCall call={.command_index=i,.command=cmd,.program=&p,.bound=&bound,.pools=pools,.pool1_writable_bytes=0x780000u};
        memset(tables,0x39,2048);memset(abi,0x39,128);memset(tail,0x39,0x80000u);
        int eligible=e.handlers[hi].validate(&call,e.handlers[hi].ctx);
        CHECK(!memcmp(tables,tail_before,2048)&&!memcmp(abi,tail_before,120)&&!memcmp(tail,tail_before,0x80000u));
        if(!eligible){rejected++;continue;}
        if(cmd.variant==32&&first_1x1==UINT32_MAX)first_1x1=i;
        if(cmd.variant==32){
            uint32_t xi,wi;Nprg2TensorView xv,wv;
            CHECK(nprg2_exec_input(&call,0,&xi,&xv)==NPRG2_EXEC_OK);
            CHECK(nprg2_exec_input(&call,1,&wi,&wv)==NPRG2_EXEC_OK);(void)xi;(void)wi;
            uint32_t raw_ci=(uint16_t)rd16(wv.geometry_record+10);
            uint32_t logical_ci=(uint16_t)rd16(xv.geometry_record+12);
            uint32_t max_ci=(uint16_t)rd16(xv.geometry_record+20);
            if(raw_ci==12u&&logical_ci==9u&&max_ci==32u&&padded_ci12==UINT32_MAX)padded_ci12=i;
            if(raw_ci==20u&&logical_ci==18u&&max_ci==32u&&padded_ci20==UINT32_MAX)padded_ci20=i;
        }
        if(cmd.variant==33&&first_3x3==UINT32_MAX){uint32_t wi;Nprg2TensorView wv;CHECK(nprg2_exec_input(&call,1,&wi,&wv)==NPRG2_EXEC_OK);if((int16_t)rd16(wv.geometry_record+14)==3&&(int16_t)rd16(wv.geometry_record+16)==3)first_3x3=i;}
        if(!tested_same_opcode_bad_shape){
            uint32_t xi;Nprg2TensorView xv;CHECK(nprg2_exec_input(&call,0,&xi,&xv)==NPRG2_EXEC_OK);(void)xi;
            uint8_t *geom=(uint8_t*)xv.geometry_record;uint16_t old=rd16(geom+18);CHECK(old>0);wr16(geom+18,(uint16_t)(old-1u));
            CHECK(!e.handlers[hi].validate(&call,e.handlers[hi].ctx));
            CHECK(!memcmp(tables,tail_before,2048)&&!memcmp(abi,tail_before,120)&&!memcmp(tail,tail_before,0x80000u));
            wr16(geom+18,old);tested_same_opcode_bad_shape=1;
            uint32_t *safe_tables=native.tables;native.tables=(uint32_t*)bound.command.data;
            CHECK(!e.handlers[hi].validate(&call,e.handlers[hi].ctx));native.tables=safe_tables;
            CHECK(!memcmp(tables,tail_before,2048)&&!memcmp(abi,tail_before,120)&&!memcmp(tail,tail_before,0x80000u));
        }
        Nprg2ExecRequest one={.program=&p,.bound=&bound,.pool1_writable_bytes=0x780000u,.command_first=i,.command_count=1};memcpy(one.pools,pools,sizeof pools);
        CHECK(nprg2_executor_run(&e,&one)==NPRG2_EXEC_OK);accepted++;CHECK(submitted==accepted);
    }
    CHECK(accepted>0&&tested_same_opcode_bad_shape);printf("PASS native direct preflight %s: 1x1=%lu 3x3=%lu rejected=%lu; mock backend only\n",qpath,(unsigned long)submitted_1x1,(unsigned long)submitted_3x3,(unsigned long)rejected);
#if defined(QDS_NATIVE_PADDED_1X1) && QDS_NATIVE_PADDED_1X1
    CHECK(padded_ci12!=UINT32_MAX&&padded_ci20!=UINT32_MAX);
    {
        uint32_t candidates[2]={padded_ci12,padded_ci20};
        uint8_t gate_tables[2048],gate_abi[120],*gate_tail=malloc(0x80000u);CHECK(gate_tail);
        for(uint32_t k=0;k<2;k++){
            Nprg2Command pc;CHECK(nprg2_command_at(&bound,candidates[k],&pc)==NPRG2_OK);
            Nprg2ExecCall pcall={.command_index=candidates[k],.command=pc,.program=&p,.bound=&bound,.pools=pools,.pool1_writable_bytes=0x780000u};
            memcpy(gate_tables,tables,sizeof gate_tables);memcpy(gate_abi,abi,sizeof gate_abi);memcpy(gate_tail,tail,0x80000u);
            int ph=find_handler(&e,QDR_HMX_CONV,pc.variant);CHECK(ph>=0&&e.handlers[ph].validate(&pcall,e.handlers[ph].ctx));
            CHECK(!memcmp(gate_tables,tables,sizeof gate_tables)&&!memcmp(gate_abi,abi,sizeof gate_abi)&&!memcmp(gate_tail,tail,0x80000u));
            uint32_t xi;Nprg2TensorView xv;CHECK(nprg2_exec_input(&pcall,0,&xi,&xv)==NPRG2_EXEC_OK);(void)xi;
            uint8_t *g=(uint8_t*)xv.geometry_record;uint16_t logical=rd16(g+12);CHECK(logical==9u||logical==18u);
            wr16(g+12,(uint16_t)(logical-1u));
            CHECK(!e.handlers[ph].validate(&pcall,e.handlers[ph].ctx));
            CHECK(!memcmp(gate_tables,tables,sizeof gate_tables)&&!memcmp(gate_abi,abi,sizeof gate_abi)&&!memcmp(gate_tail,tail,0x80000u));
            wr16(g+12,logical);
        }
        free(gate_tail);
    }
    printf("PASS padded Xpb metadata gates %s: CI12/logical9 and CI20/logical18 accepted; neighboring logical-CI metadata rejected read-only\n",qpath);
#else
    (void)padded_ci12;(void)padded_ci20;
#endif
    CHECK(first_1x1!=UINT32_MAX&&first_3x3!=UINT32_MAX);
    native.family_mask=NPRG2_NATIVE_ALLOW_1X1;
    Nprg2Command c1,c3;CHECK(nprg2_command_at(&bound,first_1x1,&c1)==NPRG2_OK);CHECK(nprg2_command_at(&bound,first_3x3,&c3)==NPRG2_OK);
    Nprg2ExecCall call1={.command_index=first_1x1,.command=c1,.program=&p,.bound=&bound,.pools=pools,.pool1_writable_bytes=0x780000u};
    Nprg2ExecCall call3={.command_index=first_3x3,.command=c3,.program=&p,.bound=&bound,.pools=pools,.pool1_writable_bytes=0x780000u};
    int h1=find_handler(&e,QDR_HMX_CONV,c1.variant),h3=find_handler(&e,QDR_HMX_CONV,c3.variant);CHECK(h1>=0&&h3>=0);
    CHECK(e.handlers[h1].validate(&call1,e.handlers[h1].ctx));CHECK(!e.handlers[h3].validate(&call3,e.handlers[h3].ctx));
    Nprg2ExecRequest one1={.program=&p,.bound=&bound,.pool1_writable_bytes=0x780000u,.command_first=first_1x1,.command_count=1};memcpy(one1.pools,pools,sizeof pools);
    Nprg2ExecRequest one3=one1;one3.command_first=first_3x3;
    uint8_t preflight_tables[2048],preflight_abi[120],*preflight_tail=malloc(0x80000u);CHECK(preflight_tail);memcpy(preflight_tables,tables,sizeof preflight_tables);memcpy(preflight_abi,abi,sizeof preflight_abi);memcpy(preflight_tail,tail,0x80000u);
    CHECK(nprg2_executor_preflight(&e,&one1)==NPRG2_EXEC_OK);CHECK(nprg2_executor_preflight(&e,&one3)==NPRG2_EXEC_REJECTED);
    CHECK(!memcmp(preflight_tables,tables,sizeof preflight_tables)&&!memcmp(preflight_abi,abi,sizeof preflight_abi)&&!memcmp(preflight_tail,tail,0x80000u));free(preflight_tail);
    printf("PASS native family mask %s: selected 1x1 accepted; same 3x3 family opcode rejected in preflight\n",qpath);
    /* The optional compaction cap is checked before the direct binder can
     * publish tables/ABI/tail or touch any graph pool. */
    native.family_mask=0;Nprg2ExecCall cap_call={.command_index=first_3x3,.command=c3,.program=&p,.bound=&bound,.pools=pools,.pool1_writable_bytes=0x780000u};
    uint32_t cap_xi;Nprg2TensorView cap_x;CHECK(nprg2_exec_input(&cap_call,0,&cap_xi,&cap_x)==NPRG2_EXEC_OK);(void)cap_xi;
    uint64_t input_pages=(uint64_t)((uint16_t)rd16(cap_x.geometry_record+16)/8u)*
                         (uint64_t)((uint16_t)rd16(cap_x.geometry_record+18)/8u)*
                         (uint64_t)((uint16_t)rd16(cap_x.geometry_record+20)/32u);
    CHECK(input_pages>1&&input_pages<=UINT32_MAX);native.max_3x3_input_pages=(uint32_t)input_pages-1u;
    uint8_t cap_tables[2048],cap_abi[120],*cap_tail=malloc(0x80000u),*pool_snap[7]={0};CHECK(cap_tail);
    memcpy(cap_tables,tables,sizeof cap_tables);memcpy(cap_abi,abi,sizeof cap_abi);memcpy(cap_tail,tail,0x80000u);
    for(unsigned i=0;i<7;i++)if(pools[i].bytes){pool_snap[i]=malloc(pools[i].bytes);CHECK(pool_snap[i]);memcpy(pool_snap[i],pools[i].base,pools[i].bytes);}
    CHECK(nprg2_executor_preflight(&e,&one3)==NPRG2_EXEC_REJECTED);
    CHECK(!memcmp(cap_tables,tables,sizeof cap_tables)&&!memcmp(cap_abi,abi,sizeof cap_abi)&&!memcmp(cap_tail,tail,0x80000u));
    for(unsigned i=0;i<7;i++)if(pools[i].bytes)CHECK(!memcmp(pool_snap[i],pools[i].base,pools[i].bytes));
    native.max_3x3_input_pages=(uint32_t)input_pages;CHECK(nprg2_executor_preflight(&e,&one3)==NPRG2_EXEC_OK);
    CHECK(!memcmp(cap_tables,tables,sizeof cap_tables)&&!memcmp(cap_abi,abi,sizeof cap_abi)&&!memcmp(cap_tail,tail,0x80000u));
    for(unsigned i=0;i<7;i++){if(pools[i].bytes)CHECK(!memcmp(pool_snap[i],pools[i].base,pools[i].bytes));free(pool_snap[i]);}free(cap_tail);
    printf("PASS native 3x3 input-page cap %s: pages=%lu cap=%lu rejected before any pool/table/ABI/tail writes\n",qpath,(unsigned long)input_pages,(unsigned long)(input_pages-1u));
    free(tail_before);free(tables);free(abi);free(hot);nprg2_close(&p);pool_free(pools);free(q);free(raw);
}
int main(int argc,char **argv){CHECK(argc==5);run_model(argv[1],argv[2]);run_model(argv[3],argv[4]);return 0;}
