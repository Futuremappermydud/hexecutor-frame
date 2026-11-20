/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_addsub_flat.h"
#include <stdint.h>
#include <string.h>

enum { DIRECT=1, SHAPE=1, QUANT=32, CALL_BYTES=88, DTYPE_U16=3, DTYPE_I32=5 };
static uint16_t rd16a(const uint8_t*p){return (uint16_t)p[0]|(uint16_t)p[1]<<8;}
static int32_t rdi16a(const uint8_t*p){return (int16_t)rd16a(p);}
static uint32_t rd32a(const uint8_t*p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint64_t rd64a(const uint8_t*p){return (uint64_t)rd32a(p)|(uint64_t)rd32a(p+4)<<32;}
static int32_t s32a(uint64_t x){return (int32_t)(uint32_t)x;}
static int scalar(const Nprg2TensorView*v,const char*cls,uint8_t dt){
    const uint8_t*g=v->geometry_record;
    if(!g||g[0]!=dt||!(g[1]&SHAPE)||!(g[1]&QUANT)||memcmp(g+2,cls,2))return 0;
    for(unsigned i=0;i<4;i++)if(rdi16a(g+6+2*i)!=1||rdi16a(g+14+2*i)!=1||rdi16a(g+22+2*i))return 0;
    return 1;
}
static int direct_span(const Nprg2ExecCall*c,const Nprg2TensorView*v,uint32_t n){
    return v->storage_kind==DIRECT&&v->pool<7&&c->pools[v->pool].base&&v->offset<=c->pools[v->pool].bytes&&n<=c->pools[v->pool].bytes-v->offset;
}
static int plan(const Nprg2ExecCall*c,Nprg2TensorView*x,Nprg2TensorView*y,Nprg2TensorView*fi,Nprg2TensorView*out,int64_t*k){
    if(!c||c->command.opcode!=QDR_ADDSUB_FLAT||c->command.variant!=62||
       c->command.input_count!=3||c->command.output_count!=1||c->command.vector_count||c->command.call_count!=1||
       c->command.duplicate_pad||c->command.io_index!=0xffffu||!c->bound||!c->bound->calls.data)return 0;
    uint32_t xi,yi,fii,oi;
    if(nprg2_exec_input(c,0,&xi,x)!=NPRG2_EXEC_OK||nprg2_exec_input(c,1,&yi,y)!=NPRG2_EXEC_OK||
       nprg2_exec_input(c,2,&fii,fi)!=NPRG2_EXEC_OK||nprg2_exec_output(c,0,&oi,out)!=NPRG2_EXEC_OK)return 0;
    (void)xi;(void)yi;(void)fii;(void)oi;
    if(!scalar(x,"FH",DTYPE_U16)||!scalar(y,"FH",DTYPE_U16)||!scalar(out,"FH",DTYPE_U16))return 0;
    const uint8_t*fg=fi->geometry_record;
    if(!fg||fg[0]!=DTYPE_I32||!(fg[1]&SHAPE)||memcmp(fg+2,"fi",2))return 0;
    static const int16_t dims[4]={1,1,1,5};
    for(unsigned i=0;i<4;i++)if(rdi16a(fg+6+2*i)!=dims[i]||rdi16a(fg+14+2*i)!=dims[i]||rdi16a(fg+22+2*i))return 0;
    /* These operand roles are backed by graph-writable pool 1 and immutable pool 5. */
    if(x->pool!=1||y->pool!=1||out->pool!=1||fi->pool!=5)return 0;
    if(!direct_span(c,x,2)||!direct_span(c,y,2)||!direct_span(c,out,2)||!direct_span(c,fi,20))return 0;
    if(out->offset>c->pool1_writable_bytes||2u>c->pool1_writable_bytes-out->offset)return 0;
    if((out->pool==x->pool&&out->offset<x->offset+2&&x->offset<out->offset+2)||
       (out->pool==y->pool&&out->offset<y->offset+2&&y->offset<out->offset+2)||
       (out->pool==fi->pool&&out->offset<fi->offset+20&&fi->offset<out->offset+2))return 0;
    if((uint64_t)c->command.call_first+1u>c->bound->calls.bytes/CALL_BYTES)return 0;
    const uint8_t*r=c->bound->calls.data+(size_t)c->command.call_first*CALL_BYTES;
    uint32_t fn=rd32a(r),mask=rd32a(r+4);
    if((fn!=QDR_CALL_ELEMENTWISE_SUB_FLAT_U16&&fn!=QDR_CALL_ELEMENTWISE_ADD_FLAT_U16)||(mask&0xf0u))return 0;
    uint64_t v[10];for(unsigned i=0;i<10;i++)v[i]=rd64a(r+8+8*i);
    for(unsigned i=0;i<10;i++)if((0x70u&(1u<<i))&&v[i]>=(UINT64_C(1)<<40))return 0;
    if(v[6]>=48u)return 0; /* qd_graph requires shift_ok(r[6]+1). */
    k[0]=(int64_t)v[4];k[1]=(int64_t)v[5];k[2]=(int64_t)v[6];k[3]=s32a(v[7]);
    k[4]=fn==QDR_CALL_ELEMENTWISE_SUB_FLAT_U16?-1:1;
    return 1;
}
int nprg2_exec_addsub_flat_validate(const Nprg2ExecCall*c,void*ctx){(void)ctx;Nprg2TensorView x,y,fi,o;int64_t k[5];return plan(c,&x,&y,&fi,&o,k);}
int nprg2_exec_addsub_flat_run(const Nprg2ExecCall*c,void*ctx){(void)ctx;Nprg2TensorView x,y,fi,o;int64_t k[5];if(!plan(c,&x,&y,&fi,&o,k))return 0;
    const uint16_t*a=(const uint16_t*)(c->pools[x.pool].base+x.offset),*b=(const uint16_t*)(c->pools[y.pool].base+y.offset);uint16_t*z=(uint16_t*)(c->pools[o.pool].base+o.offset);
    int64_t value=((int64_t)a[0]*k[0]+k[4]*(int64_t)b[0]*k[1]+2*k[3]+((int64_t)1<<k[2]))>>(k[2]+1);
    if(value<0)value=0;if(value>65535)value=65535;z[0]=(uint16_t)value;return 1;}
int nprg2_executor_register_addsub_flat(Nprg2Executor*e){if(!e)return NPRG2_EXEC_ARGUMENT;return nprg2_executor_register(e,QDR_ADDSUB_FLAT,62,nprg2_exec_addsub_flat_validate,nprg2_exec_addsub_flat_run,NULL);}
