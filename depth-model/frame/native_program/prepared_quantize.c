#include "prepared_quantize.h"
#include "prepared_quantize_core.h"
#include <limits.h>
#include <string.h>

enum { PQ_SHAPE=1u, PQ_QUANT=32u, PQ_DIRECT=1u, PQ_CALL_BYTES=88u };
typedef struct {const float *src;uint16_t *dst;uint32_t src_bytes,dst_bytes,count,zp;float scale;} QuantizePlan;
static uint16_t rd16q(const uint8_t*p){return(uint16_t)p[0]|(uint16_t)p[1]<<8;}
static uint32_t rd32q(const uint8_t*p){return(uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint64_t rd64q(const uint8_t*p){return(uint64_t)rd32q(p)|(uint64_t)rd32q(p+4)<<32;}
static int class4q(const uint8_t*g,const char*s){return g&&!memcmp(g+2,s,strlen(s));}
static int shape4q(const uint8_t*g,const int16_t dims[4]){
    if(!g||!(g[1]&PQ_SHAPE))return 0;
    for(unsigned i=0;i<4;i++)if((int16_t)rd16q(g+6+2*i)!=dims[i]||
       (int16_t)rd16q(g+14+2*i)!=dims[i]||(int16_t)rd16q(g+22+2*i))return 0;
    return 1;
}
static int make_quantize_plan(const Nprg2ExecCall*c,QuantizePlan*p){
    static const int16_t dims[4]={1,2,4,4};
    if(!c||!p||c->command.opcode!=QDR_QUANTIZE||c->command.variant!=48u||
       c->command.input_count!=1u||c->command.output_count!=1u||c->command.vector_count||
       c->command.call_count!=1u||c->command.duplicate_pad||c->command.io_index!=0xffffu||!c->pools[1].base)return 0;
    uint32_t expected_x,expected_y;uint64_t expected_arg9;
    if(c->command_index==60u&&c->command.sequence==107u){expected_x=1089536u;expected_y=7589888u;expected_arg9=UINT64_C(0x5555560d0500);}
    else if(c->command_index==54u&&c->command.sequence==100u){expected_x=3997696u;expected_y=2775040u;expected_arg9=UINT64_C(0x5555560d0700);}
    else return 0;
    uint32_t xi,yi;Nprg2TensorView x,y;
    if(nprg2_exec_input(c,0,&xi,&x)!=NPRG2_EXEC_OK||nprg2_exec_output(c,0,&yi,&y)!=NPRG2_EXEC_OK)return 0;
    (void)xi;(void)yi;
    const uint8_t*xg=x.geometry_record,*yg=y.geometry_record;
    if(!xg||!yg||x.storage_kind!=PQ_DIRECT||y.storage_kind!=PQ_DIRECT||x.pool!=1u||y.pool!=1u||
       x.offset!=expected_x||y.offset!=expected_y||xg[0]!=6u||yg[0]!=3u||
       !class4q(xg,"Ff")||!class4q(yg,"FH")||!shape4q(xg,dims)||!shape4q(yg,dims)||
       !(yg[1]&PQ_QUANT)||rd32q(yg+30)!=32768u)return 0;
    uint64_t scale_bits=rd64q(yg+34);double scale_double;memcpy(&scale_double,&scale_bits,sizeof scale_double);
    float scale=(float)scale_double;uint32_t scale_f32_bits;memcpy(&scale_f32_bits,&scale,sizeof scale_f32_bits);
    if(scale_f32_bits!=0x39a000a0u)return 0;
    if(x.offset>c->pools[1].bytes||128u>c->pools[1].bytes-x.offset||
       y.offset>c->pools[1].bytes||64u>c->pools[1].bytes-y.offset||
       x.offset>c->pool1_writable_bytes||128u>c->pool1_writable_bytes-x.offset||
       y.offset>c->pool1_writable_bytes||64u>c->pool1_writable_bytes-y.offset)return 0;
    uintptr_t base=(uintptr_t)c->pools[1].base;if(x.offset>UINTPTR_MAX-base||y.offset>UINTPTR_MAX-base)return 0;
    uintptr_t src=base+x.offset,dst=base+y.offset;
    if(128u>UINTPTR_MAX-src||64u>UINTPTR_MAX-dst||(src<dst+64u&&dst<src+128u))return 0;
    if(!c->bound->calls.data||(uint64_t)c->command.call_first+1u>c->bound->calls.bytes/PQ_CALL_BYTES)return 0;
    const uint8_t*call=c->bound->calls.data+(size_t)c->command.call_first*PQ_CALL_BYTES;
    if(rd32q(call)!=12u||rd32q(call+4)!=133u||rd64q(call+8)!=0u||rd64q(call+16)!=32768u||
       rd64q(call+24)!=0u||rd64q(call+32)!=32u||rd64q(call+40)!=0u||rd64q(call+48)!=0u||
       rd64q(call+56)!=1u||rd64q(call+64)!=0u||rd64q(call+72)!=UINT64_C(0x454ccc00ff577600)||
       rd64q(call+80)!=expected_arg9)return 0;
    p->src=(const float*)src;p->dst=(uint16_t*)dst;p->src_bytes=128u;p->dst_bytes=64u;p->count=32u;p->zp=32768u;p->scale=scale;return 1;
}
static int quantize_validate(const Nprg2ExecCall*c,void*ctx){(void)ctx;QuantizePlan p;return make_quantize_plan(c,&p);}
static int quantize_run(const Nprg2ExecCall*c,void*ctx){(void)ctx;QuantizePlan p;if(!make_quantize_plan(c,&p))return 0;return nprg2_quantize_f32_to_u16(p.src,p.src_bytes,p.dst,p.dst_bytes,p.count,(uint16_t)p.zp,p.scale);}
int nprg2_executor_register_quantize(Nprg2Executor*e){if(!e)return NPRG2_EXEC_ARGUMENT;return nprg2_executor_register(e,QDR_QUANTIZE,48,quantize_validate,quantize_run,NULL);}
