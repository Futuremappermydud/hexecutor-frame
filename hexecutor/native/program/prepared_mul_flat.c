/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_mul_flat.h"
#include <stdint.h>
#include <string.h>

enum { OPCODE_MUL_INLINE=41, CALL_MUL1111_FLAT_U16=9, DTYPE_U16=3, DTYPE_I32=5, DIRECT=1, WORDS=6, CALL_BYTES=88 };
static uint16_t rd16(const uint8_t*p){return (uint16_t)p[0]|((uint16_t)p[1]<<8);}
static int32_t rdi16(const uint8_t*p){return (int16_t)rd16(p);}
static uint32_t rd32(const uint8_t*p){return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
static int32_t s32(uint32_t x){return (int32_t)x;}
static int zeroes(const uint8_t*p,uint32_t n){for(uint32_t i=0;i<n;i++)if(p[i])return 0;return 1;}
static uint32_t count(const uint8_t*g){uint32_t n=1;for(int i=0;i<4;i++){int32_t d=rdi16(g+6+2*i);if(d<=0||n>UINT32_MAX/(uint32_t)d)return 0;n*=(uint32_t)d;}return n;}
static int shape(const uint8_t*g,int b,int h,int w,int d){return rdi16(g+6)==b&&rdi16(g+8)==h&&rdi16(g+10)==w&&rdi16(g+12)==d&&rdi16(g+14)==b&&rdi16(g+16)==h&&rdi16(g+18)==w&&rdi16(g+20)==d&&zeroes(g+22,8);}
static int tensor(const Nprg2ExecCall*c,uint32_t slot,int output,Nprg2TensorView*v,uint32_t*idx){return (output?nprg2_exec_output(c,slot,idx,v):nprg2_exec_input(c,slot,idx,v))==NPRG2_EXEC_OK;}
static int flat_direct(const Nprg2ExecCall*c,const Nprg2TensorView*v,uint32_t bytes){return v->storage_kind==DIRECT&&v->pool<7&&c->pools[v->pool].base&&v->offset<=c->pools[v->pool].bytes&&bytes<=c->pools[v->pool].bytes-v->offset;}
static int call_signature(const Nprg2ExecCall*c){
 if(!c->bound||!c->bound->calls.data||c->command.call_count!=1||c->command.call_first>c->bound->calls.bytes/CALL_BYTES||CALL_BYTES>c->bound->calls.bytes-(uint32_t)c->command.call_first*CALL_BYTES)return 0;
 const uint8_t*p=c->bound->calls.data+(uint32_t)c->command.call_first*CALL_BYTES;return rd32(p)==CALL_MUL1111_FLAT_U16;}
int nprg2_exec_mul_flat_validate(const Nprg2ExecCall*c,void*ctx){(void)ctx;if(!c||c->command.opcode!=OPCODE_MUL_INLINE||c->command.variant!=75||c->command.input_count!=3||c->command.output_count!=1||c->command.vector_count||c->command.duplicate_pad||c->command.io_index!=UINT16_MAX||!call_signature(c))return 0;
 Nprg2TensorView a,b,p,o;uint32_t ai,bi,pi,oi;if(!tensor(c,0,0,&a,&ai)||!tensor(c,1,0,&b,&bi)||!tensor(c,2,0,&p,&pi)||!tensor(c,0,1,&o,&oi))return 0;
 const uint8_t*ag=a.geometry_record,*bg=b.geometry_record,*pg=p.geometry_record,*og=o.geometry_record;if(!ag||!bg||!pg||!og)return 0;
 if(ag[0]!=DTYPE_U16||og[0]!=DTYPE_U16||bg[0]!=DTYPE_U16||pg[0]!=DTYPE_I32||memcmp(ag+2,"FH",2)||memcmp(bg+2,"fH",2)||memcmp(pg+2,"fi",2)||memcmp(og+2,"FH",2))return 0;
 /* Bounded to the sole observed command-110 signature and its tensor metadata. */
 if(!shape(ag,1,1,1,1)||!shape(bg,1,1,1,1)||!shape(pg,1,1,1,WORDS)||!shape(og,1,1,1,1))return 0;
 uint32_t n=count(ag);if(n!=1||count(bg)!=1||count(pg)!=WORDS||count(og)!=1||!flat_direct(c,&a,n*2u)||!flat_direct(c,&o,n*2u)||!flat_direct(c,&b,2u)||!flat_direct(c,&p,WORDS*4u))return 0;
 /* Two observed quantization profiles share the same proven scalar algorithm. */
 static const uint8_t in0[8]={0,0,0,0,0x20,0,0,0x3f}, sc0[8]={0,0,0,0,0x10,0,0xe0,0x3e}, out0[8]={0,0,0,0,0x10,0,0xf0,0x3e};
 static const uint8_t in1[8]={0,0,0,0,0x18,0,0xf0,0x3e}, sc1[8]={0,0,0,0x80,0x46,0x33,0xd3,0x3e}, out1[8]={0,0,0,0x80,0x46,0x33,0xd3,0x3e};
 int profile0=rd32(ag+30)==32768u&&!memcmp(ag+34,in0,8)&&rd32(bg+30)==0&&!memcmp(bg+34,sc0,8)&&rd32(og+30)==32768u&&!memcmp(og+34,out0,8);
 int profile1=rd32(ag+30)==0&&!memcmp(ag+34,in1,8)&&rd32(bg+30)==0&&!memcmp(bg+34,sc1,8)&&rd32(og+30)==0&&!memcmp(og+34,out1,8);
 if(!profile0&&!profile1)return 0;
 /* The parameter tensor is six signed words and carries no quantization record. */
 if(!zeroes(pg+30,12))return 0;
 static const uint32_t params0[6]={787202u,0u,262152u,32768u,32768u,262140u},params1[6]={787202u,0u,262150u,0u,2147549184u,262142u};
 for(int k=0;k<6;k++){uint32_t v=rd32(c->pools[p.pool].base+p.offset+4u*(uint32_t)k);if(v!=(profile0?params0[k]:params1[k]))return 0;}
 if(o.pool==1&&(o.offset>c->pool1_writable_bytes||n*2u>c->pool1_writable_bytes-o.offset))return 0;
 if((a.pool==o.pool&&a.offset<o.offset+n*2u&&o.offset<a.offset+n*2u)||(b.pool==o.pool&&b.offset<o.offset+n*2u&&o.offset<b.offset+2u)||(p.pool==o.pool&&p.offset<o.offset+n*2u&&o.offset<p.offset+WORDS*4u))return 0;
 for(int k=0;k<4;k++)if(rdi16(ag+6+2*k)!=rdi16(og+6+2*k))return 0;
 return 1;}
int nprg2_exec_mul_flat_run(const Nprg2ExecCall*c,void*ctx){if(!nprg2_exec_mul_flat_validate(c,ctx))return 0;Nprg2TensorView a,b,p,o;uint32_t i;nprg2_exec_input(c,0,&i,&a);nprg2_exec_input(c,1,&i,&b);nprg2_exec_input(c,2,&i,&p);nprg2_exec_output(c,0,&i,&o);
 uint32_t n=count(a.geometry_record);const uint8_t*ap=c->pools[a.pool].base+a.offset,*bp=c->pools[b.pool].base+b.offset,*pp=c->pools[p.pool].base+p.offset;uint8_t*op=c->pools[o.pool].base+o.offset;
 uint32_t pw[6];for(int j=0;j<6;j++)pw[j]=rd32(pp+4*j);int64_t scalar=rd16(bp),P1=s32(pw[2]),z=pw[3],zb=z&65535u,za=z>>16,K=s32(pw[4]),P2=s32(pw[5]);
 for(uint32_t j=0;j<n;j++){int64_t x=rd16(ap+2*j);int32_t d=s32((uint32_t)(x*scalar+K-(scalar*zb+x*za)));int64_t e=((int64_t)d*(P1&65535))>>16;int64_t q=((2*e+2*(int64_t)d*(P1>>16))>>16);if(q>INT32_MAX)q=INT32_MAX;if(q<INT32_MIN)q=INT32_MIN;int64_t s=q+P2;if(s>INT32_MAX)s=INT32_MAX;if(s<INT32_MIN)s=INT32_MIN;int64_t y=(s+4)>>3;if(y<0)y=0;if(y>65535)y=65535;op[2*j]=(uint8_t)y;op[2*j+1]=(uint8_t)(y>>8);}return 1;}
int nprg2_executor_register_mul_flat(Nprg2Executor*e){if(!e)return NPRG2_EXEC_ARGUMENT;return nprg2_executor_register(e,OPCODE_MUL_INLINE,75,nprg2_exec_mul_flat_validate,nprg2_exec_mul_flat_run,NULL);}
