/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);exit(1);}}while(0)
static uint16_t rd16(const uint8_t *p){return (uint16_t)p[0]|(uint16_t)p[1]<<8;}
static uint32_t rd32(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static void wr32(uint8_t *p,uint32_t x){p[0]=(uint8_t)x;p[1]=(uint8_t)(x>>8);p[2]=(uint8_t)(x>>16);p[3]=(uint8_t)(x>>24);}
static int32_t rdi32(const uint8_t *p){return (int32_t)rd32(p);}
static uint8_t *read_file(const char *path,uint32_t *n){FILE *f=fopen(path,"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long z=ftell(f);CHECK(z>0&&z<=NPRG2_MAX_IMAGE_BYTES);CHECK(!fseek(f,0,SEEK_SET));uint8_t *p=malloc((size_t)z);CHECK(p&&fread(p,1,(size_t)z,f)==(size_t)z);fclose(f);*n=(uint32_t)z;return p;}
static void pool_alloc(const uint8_t *raw,Nprg2Pool p[7]){for(uint32_t i=0;i<7;i++){p[i].bytes=rd32(raw+144+i*4);p[i].base=p[i].bytes?malloc(p[i].bytes):NULL;CHECK(!p[i].bytes||p[i].base);if(p[i].bytes)memset(p[i].base,(int)(0x31u+i*7u),p[i].bytes);}}
static void pool_free(Nprg2Pool p[7]){for(int i=0;i<7;i++)free(p[i].base);}
static int bind_all(void *ctx,const Nprg2Program *p,uint32_t i,const Nprg2Command *c,Nprg2Capability *cap){(void)ctx;(void)p;(void)i;(void)c;cap->supported=1;cap->may_fallback=0;cap->scratch_bytes=0;cap->concurrent_worker_scratch_bytes=0;return 1;}
static int32_t shape_value(const Nprg2TensorView *v,uint32_t i){return rdi32(v->geometry_record+6+i*4u);}
static uint32_t count_shape(const int32_t d[4]){uint64_t n=1;for(int i=0;i<4;i++){CHECK(d[i]>=0);n*=(uint32_t)d[i];CHECK(n<=UINT32_MAX);}return (uint32_t)n;}
static uint32_t input_shape_from_command(const Nprg2Bound *b,const Nprg2Program *p,uint32_t ix,int32_t dims[4],Nprg2TensorView *out){
    Nprg2Command c;CHECK(nprg2_command_at(b,ix,&c)==NPRG2_OK);Nprg2ExecCall call={.command_index=ix,.command=c,.program=p,.bound=b};uint32_t ti;CHECK(nprg2_exec_input(&call,6,&ti,out)==NPRG2_EXEC_OK);(void)ti;for(int i=0;i<4;i++)dims[i]=shape_value(out,(uint32_t)i);return count_shape(dims);
}
static void ref_slice_pad(const Nprg2ExecCall *call,Nprg2Pool pools[7],const Nprg2ExecBuffer io[2],uint8_t *dst_pool){
    Nprg2TensorView v[7],o;uint32_t ix;for(int i=0;i<7;i++)CHECK(nprg2_exec_input(call,(uint32_t)i,&ix,&v[i])==NPRG2_EXEC_OK);CHECK(nprg2_exec_output(call,0,&ix,&o)==NPRG2_EXEC_OK);
    int32_t pad[4],start[4],out[4],in[4];for(int i=0;i<4;i++){pad[i]=shape_value(&v[0],(uint32_t)i);start[i]=shape_value(&v[1],(uint32_t)i);out[i]=shape_value(&v[2],(uint32_t)i);in[i]=shape_value(&v[6],(uint32_t)i);}
    int32_t fill=(int32_t)rd32(v[3].geometry_record+6);uint32_t dst_bytes=count_shape(out);uint8_t *dst=dst_pool+o.offset;memset(dst,(uint8_t)fill,dst_bytes);
    int32_t lo[4],s0[4],s1[4];for(int i=0;i<4;i++){lo[i]=start[i]-pad[i];s0[i]=lo[i]>0?lo[i]:0;s1[i]=lo[i]+out[i]<in[i]?lo[i]+out[i]:in[i];if(s1[i]<s0[i])s1[i]=s0[i];}
    const uint8_t *src=io[call->command.io_index].data;
    for(int32_t b=s0[0];b<s1[0];b++)for(int32_t h=s0[1];h<s1[1];h++)for(int32_t w=s0[2];w<s1[2];w++){
        size_t si=(((size_t)b*in[1]+h)*in[2]+w)*in[3]+s0[3];
        size_t di=(((size_t)(b-lo[0])*out[1]+(h-lo[1]))*out[2]+(w-lo[2]))*out[3]+(s0[3]-lo[3]);
        memcpy(dst+di,src+si,(size_t)(s1[3]-s0[3]));
    }
    (void)pools;
}
static void run_model(const char *qpath,const char *npath){
    uint32_t qn,nn;uint8_t *q=read_file(qpath,&qn),*raw=read_file(npath,&nn),qsha[32],nsha[32];Nprg2ExpectedHashes expected;nprg2_sha256(q,qn,qsha);nprg2_sha256(raw,nn,nsha);
    expected.context_sha256=q+32;expected.constants_sha256=q+64;expected.source_sha256=qsha;expected.prepared_sha256=nsha;
    Nprg2Pool pools[7]={0};pool_alloc(raw,pools);Nprg2Program p;CHECK(nprg2_open(&p,raw,nn,&expected,pools,(Nprg2Allocator){0})==NPRG2_OK);
    uint8_t *hot=calloc(1,300000);CHECK(hot);Nprg2BindConfig bc={.hot_memory=hot,.hot_capacity=300000,.l2_available_bytes=786432,.pool1_is_verified_vtcm=1,.resolve=bind_all};Nprg2Bound bound;CHECK(nprg2_bind_mock(&p,pools,&bc,&bound)==NPRG2_OK);
    Nprg2Executor e;nprg2_executor_init(&e);CHECK(nprg2_executor_register_core_safe_ops(&e)==NPRG2_EXEC_OK);
    uint32_t first=UINT32_MAX,last=0,io_bytes[2]={0};
    for(uint32_t i=0;i<p.command_count;i++){Nprg2Command c;CHECK(nprg2_command_at(&bound,i,&c)==NPRG2_OK);if(c.opcode==QDR_INPUT_SLICE_PAD){first=i;break;}}
    CHECK(first!=UINT32_MAX);for(uint32_t i=first;i<p.command_count;i++){Nprg2Command c;CHECK(nprg2_command_at(&bound,i,&c)==NPRG2_OK);if(c.opcode!=QDR_INPUT_SLICE_PAD)break;last=i;
        int32_t d[4];Nprg2TensorView sh;uint32_t b=input_shape_from_command(&bound,&p,i,d,&sh);CHECK(c.io_index<2);if(io_bytes[c.io_index])CHECK(io_bytes[c.io_index]==b);else io_bytes[c.io_index]=b;
    }
    uint8_t *i0=malloc(io_bytes[0]?io_bytes[0]:1),*i1=malloc(io_bytes[1]?io_bytes[1]:1);CHECK(i0&&i1);
    for(uint32_t i=0;i<io_bytes[0];i++)i0[i]=(uint8_t)(i*29u+7u);for(uint32_t i=0;i<io_bytes[1];i++)i1[i]=(uint8_t)(i*41u+13u);
    Nprg2ExecBuffer inputs[2]={{i0,io_bytes[0]},{i1,io_bytes[1]}};Nprg2ExecRequest r={.program=&p,.bound=&bound,.pool1_writable_bytes=0x780000u,.inputs=inputs,.input_count=2,.command_first=first,.command_count=last-first+1};memcpy(r.pools,pools,sizeof pools);
    uint8_t *before[7]={0};for(int i=0;i<7;i++)if(pools[i].bytes){before[i]=malloc(pools[i].bytes);CHECK(before[i]);memcpy(before[i],pools[i].base,pools[i].bytes);}
    uint8_t *want=malloc(pools[1].bytes);CHECK(want);memcpy(want,pools[1].base,pools[1].bytes);
    for(uint32_t i=first;i<=last;i++){Nprg2Command c;CHECK(nprg2_command_at(&bound,i,&c)==NPRG2_OK);Nprg2ExecCall call={.command_index=i,.command=c,.program=&p,.bound=&bound,.pools=pools,.pool1_writable_bytes=r.pool1_writable_bytes,.inputs=inputs,.input_count=2};ref_slice_pad(&call,pools,inputs,want);}
    CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);CHECK(!memcmp(pools[1].base,want,pools[1].bytes));
    for(int i=0;i<7;i++)if(i!=1&&pools[i].bytes)CHECK(!memcmp(pools[i].base,before[i],pools[i].bytes));
    /* Direct offsets in the preserved tail reject during preflight, leaving
     * the entire 8 MiB pool unchanged. */
    Nprg2Command c;CHECK(nprg2_command_at(&bound,first,&c)==NPRG2_OK);Nprg2ExecCall call={.command_index=first,.command=c,.program=&p,.bound=&bound,.pools=pools,.pool1_writable_bytes=r.pool1_writable_bytes,.inputs=inputs,.input_count=2};uint32_t out_id;Nprg2TensorView out;CHECK(nprg2_exec_output(&call,0,&out_id,&out)==NPRG2_EXEC_OK);
    uint8_t *row=(uint8_t*)bound.tensor.data+(size_t)out_id*NPRG2_TENSOR_BYTES;uint32_t old=rd32(row+4),bytes=out.geometry_record?count_shape((int32_t[4]){(int16_t)rd16(out.geometry_record+14),(int16_t)rd16(out.geometry_record+16),(int16_t)rd16(out.geometry_record+18),(int16_t)rd16(out.geometry_record+20)}):0;CHECK(bytes&&bytes<0x780000u);wr32(row+4,0x780000u-bytes+1u);
    memcpy(pools[1].base,want,pools[1].bytes);Nprg2ExecRequest one=r;one.command_first=first;one.command_count=1;CHECK(nprg2_executor_run(&e,&one)==NPRG2_EXEC_REJECTED);wr32(row+4,old);CHECK(!memcmp(pools[1].base,want,pools[1].bytes));
    /* A caller input that aliases pool1 is rejected before any output store. */
    Nprg2ExecBuffer alias_inputs[2];alias_inputs[0].data=pools[1].base+out.offset;alias_inputs[0].bytes=inputs[0].bytes;alias_inputs[1]=inputs[1];
    r.inputs=alias_inputs;memcpy(pools[1].base,want,pools[1].bytes);CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_ARGUMENT);CHECK(!memcmp(pools[1].base,want,pools[1].bytes));
    /* Real duplicate-pad NOPs and framework NOP variants touch no pool bytes. */
    r.inputs=inputs;for(uint8_t v=2;v<=10;v++){int found=0;for(uint32_t i=0;i<p.command_count;i++){CHECK(nprg2_command_at(&bound,i,&c)==NPRG2_OK);if(c.opcode==QDR_NOP&&c.variant==v){r.command_first=i;r.command_count=1;CHECK(nprg2_executor_run(&e,&r)==NPRG2_EXEC_OK);found=1;CHECK(!memcmp(pools[1].base,want,pools[1].bytes));for(int k=0;k<7;k++)if(k!=1&&pools[k].bytes)CHECK(!memcmp(pools[k].base,before[k],pools[k].bytes));break;}}if(v==2||v==6||v==7||v==10)CHECK(found);}
    printf("PASS InputSlicePad/NOP %s: commands=%u full-pool preservation\n",qpath,last-first+1);
    for(int i=0;i<7;i++)free(before[i]);free(want);free(i0);free(i1);free(hot);nprg2_close(&p);pool_free(pools);free(q);free(raw);
}
int main(int argc,char **argv){CHECK(argc==5);run_model(argv[1],argv[2]);run_model(argv[3],argv[4]);return 0;}
