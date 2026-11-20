/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_sign.h"
#include <math.h>

#include <limits.h>
#include <stdint.h>
#include <string.h>

enum { SIGN_FH = 40, SIGN_CH = 39, DIRECT = 1, BLOCKED = 2,
       SHAPE = 1, QUANT = 32, CROUTON = 64, BLOCK_BYTES = 2048 };
typedef struct { int64_t neg,zero,pos; } SignLevels;
static uint32_t rd32(const uint8_t *p);
static int levels_prepare(const uint8_t *g,SignLevels *v) {
    if(!g || !v || !(g[1]&QUANT))return 0;
    double scale;memcpy(&scale,g+34,8);int32_t zp=(int32_t)rd32(g+30);
    if(!isfinite(scale)||scale<=0)return 0;
    double a=floor(-1.0/scale+zp+0.5),b=floor((double)zp+0.5),d=floor(1.0/scale+zp+0.5);
    if(a<0)a=0;if(a>65535)a=65535;if(b<0)b=0;if(b>65535)b=65535;if(d<0)d=0;if(d>65535)d=65535;
    v->neg=(int64_t)a;v->zero=(int64_t)b;v->pos=(int64_t)d;return 1;
}
static int64_t select_level(uint16_t x,int32_t zp,const SignLevels *v){return x<zp?v->neg:x>zp?v->pos:v->zero;}

static uint16_t rd16(const uint8_t *p) { return (uint16_t)p[0] | (uint16_t)p[1] << 8; }
static int32_t rdi16(const uint8_t *p) { return (int16_t)rd16(p); }
static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static int tensor_shape(const Nprg2TensorView *v, int32_t d[4], int32_t m[4], int32_t o[4]) {
    const uint8_t *g=v->geometry_record;
    if (!g || !(g[1]&SHAPE)) return 0;
    for (unsigned i=0;i<4;i++) {
        d[i]=rdi16(g+6+2*i); m[i]=rdi16(g+14+2*i); o[i]=rdi16(g+22+2*i);
        if (d[i]<=0 || m[i]<=0 || o[i]<0 || (int64_t)d[i]+o[i]>m[i]) return 0;
    }
    return 1;
}

static int sign_levels(const Nprg2TensorView *in,const Nprg2TensorView *out,
                       int32_t *input_zero,SignLevels *levels) {
    if (!in->geometry_record || !out->geometry_record ||
        !(in->geometry_record[1]&QUANT) || !(out->geometry_record[1]&QUANT)) return 0;
    *input_zero=(int32_t)rd32(in->geometry_record+30);
    return levels_prepare(out->geometry_record,levels);
}

static int block_ptr(const Nprg2ExecCall *c,const Nprg2TensorView *v,uint32_t index,
                     int writing,uint8_t **p) {
    if (index>=v->block_count || !v->blocks) return 0;
    uint32_t x=v->blocks[index], pool=x&127u, off=x&~127u;
    if (!pool || pool>=7 || !c->pools[pool].base || (off&(BLOCK_BYTES-1u)) ||
        off>c->pools[pool].bytes || BLOCK_BYTES>c->pools[pool].bytes-off) return 0;
    uint32_t lim=pool==1 ? (writing?c->pool1_writable_bytes:c->pools[pool].bytes) : c->pools[pool].bytes;
    if (off>lim || BLOCK_BYTES>lim-off) return 0;
    *p=c->pools[pool].base+off; return 1;
}

/* Exact CH physical layout audited by blocked_glue/sign_blocked.c. */
static int elem_ptr(const Nprg2ExecCall *c,const Nprg2TensorView *v,
                    const int32_t *m,uint32_t b,uint32_t h,uint32_t w,uint32_t d,
                    int writing,uint8_t **p) {
    uint32_t H=h+(uint32_t)rdi16(v->geometry_record+24);
    uint32_t W=w+(uint32_t)rdi16(v->geometry_record+26);
    uint32_t D=d+(uint32_t)rdi16(v->geometry_record+28);
    uint32_t nbh=((uint32_t)m[1]+7u)/8u, nbw=((uint32_t)m[2]+3u)/4u, nbd=((uint32_t)m[3]+31u)/32u;
    uint64_t ix=(((uint64_t)b*nbh+H/8u)*nbw+W/4u)*nbd+v->slice_origin+D/32u;
    if(ix>UINT32_MAX)return 0;
    uint8_t *base;if(!block_ptr(c,v,(uint32_t)ix,writing,&base))return 0;
    uint32_t byte=(H%8u)*256u+((W%4u)/2u)*128u+(D%32u)*4u+(W%2u)*2u;
    *p=base+byte;return 1;
}

static int validate_tensor(const Nprg2ExecCall *c,const Nprg2TensorView *v,
                           int is_ch,int writing,int32_t d[4],int32_t m[4],int32_t o[4]) {
    if(!tensor_shape(v,d,m,o)||v->geometry_record[0]!=3)return 0;
    if(is_ch) {
        if(v->storage_kind!=BLOCKED || !(v->geometry_record[1]&CROUTON) || memcmp(v->geometry_record+2,"CH",2) || !v->blocks) return 0;
        uint32_t nbh=((uint32_t)m[1]+7u)/8u, nbw=((uint32_t)m[2]+3u)/4u, nbd=((uint32_t)m[3]+31u)/32u;
        uint64_t count=(uint64_t)m[0]*nbh*nbw*nbd;
        if(count>v->block_count)return 0;
        for(uint32_t b=(uint32_t)o[0];b<(uint32_t)(o[0]+d[0]);b++)
         for(uint32_t h=(uint32_t)o[1]/8;h<=((uint32_t)(o[1]+d[1]-1))/8;h++)
          for(uint32_t w=(uint32_t)o[2]/4;w<=((uint32_t)(o[2]+d[2]-1))/4;w++)
           for(uint32_t z=(uint32_t)o[3]/32;z<=((uint32_t)(o[3]+d[3]-1))/32;z++) {
            uint64_t ix=(((uint64_t)b*nbh+h)*nbw+w)*nbd+v->slice_origin+z;
            if(ix>=v->block_count)return 0;
            uint8_t *unused;
            if((v->blocks[ix]&127u) && !block_ptr(c,v,(uint32_t)ix,writing,&unused))return 0;
           }
    } else {
        if(v->storage_kind!=DIRECT || memcmp(v->geometry_record+2,"FH",2) || v->pool>=7 || !c->pools[v->pool].base)return 0;
        uint64_t n=(uint64_t)d[0]*d[1]*d[2]*d[3], bytes=n*2u;
        if(bytes>UINT32_MAX || v->offset>c->pools[v->pool].bytes || bytes>c->pools[v->pool].bytes-v->offset)return 0;
        if(writing && v->pool==1 && (v->offset>c->pool1_writable_bytes || bytes>c->pool1_writable_bytes-v->offset))return 0;
    }
    return 1;
}

static int collect_pages(const Nprg2ExecCall *c,const Nprg2TensorView *v,
                         const int32_t *m,const int32_t *d,const int32_t *o,
                         int writing,uintptr_t pages[2560],size_t *count) {
    uint32_t nh=((uint32_t)(o[1]+d[1]-1)/8u)-(uint32_t)o[1]/8u+1u;
    uint32_t nw=((uint32_t)(o[2]+d[2]-1)/4u)-(uint32_t)o[2]/4u+1u;
    uint32_t nd=((uint32_t)(o[3]+d[3]-1)/32u)-(uint32_t)o[3]/32u+1u;
    uint64_t n=(uint64_t)d[0]*nh*nw*nd;if(n>2560)return 0;*count=0;
    uint32_t nbh=((uint32_t)m[1]+7u)/8u,nbw=((uint32_t)m[2]+3u)/4u,nbd=((uint32_t)m[3]+31u)/32u;
    for(uint32_t b=(uint32_t)o[0];b<(uint32_t)(o[0]+d[0]);b++)
     for(uint32_t h=(uint32_t)o[1]/8;h<=((uint32_t)(o[1]+d[1]-1))/8;h++)
      for(uint32_t w=(uint32_t)o[2]/4;w<=((uint32_t)(o[2]+d[2]-1))/4;w++)
       for(uint32_t z=(uint32_t)o[3]/32;z<=((uint32_t)(o[3]+d[3]-1))/32;z++) {
        uint64_t ix=(((uint64_t)b*nbh+h)*nbw+w)*nbd+v->slice_origin+z;uint8_t *p;
        if(ix>=v->block_count)return 0;
        if(!(v->blocks[ix]&127u)){if(writing)return 0;continue;}
        if(!block_ptr(c,v,(uint32_t)ix,writing,&p))return 0;
        pages[(*count)++]=(uintptr_t)p;
       }
    return 1;
}

int nprg2_exec_sign_validate(const Nprg2ExecCall *c,void *ctx) {
    (void)ctx;
    if(!c || c->command.opcode!=QDR_SIGN || (c->command.variant!=SIGN_FH && c->command.variant!=SIGN_CH) ||
       c->command.input_count!=1 || c->command.output_count!=1 || c->command.vector_count ||
       c->command.call_count || c->command.duplicate_pad || c->command.io_index!=UINT16_MAX || !c->pools)return 0;
    int ch=c->command.variant==SIGN_CH;
    uint32_t ii,oi;Nprg2TensorView in,out;int32_t id[4],im[4],io[4],od[4],om[4],oo[4];
    if(nprg2_exec_input(c,0,&ii,&in)!=NPRG2_EXEC_OK || nprg2_exec_output(c,0,&oi,&out)!=NPRG2_EXEC_OK ||
       !validate_tensor(c,&in,ch,0,id,im,io)||!validate_tensor(c,&out,ch,1,od,om,oo))return 0;
    (void)ii;(void)oi;
    for(unsigned k=0;k<4;k++)if(id[k]!=od[k]||im[k]!=om[k]||io[k]!=oo[k])return 0;
    if(ch && ((io[2]&3)!=(oo[2]&3)||(io[3]&31)!=(oo[3]&31)))return 0;
    int32_t iz;SignLevels lv;if(!sign_levels(&in,&out,&iz,&lv))return 0;
    if(ch) {
        /* Bounded page census proves pool bounds, output uniqueness, and
         * read/write disjointness before execution starts. */
        uintptr_t ip[2560],op[2560];size_t ni,no;
        if(!collect_pages(c,&in,im,id,io,0,ip,&ni)||!collect_pages(c,&out,om,od,oo,1,op,&no))return 0;
        for(size_t a=0;a<no;a++)for(size_t b=0;b<ni;b++)
            if(op[a]<ip[b]+BLOCK_BYTES&&ip[b]<op[a]+BLOCK_BYTES)return 0;
        for(size_t a=0;a<no;a++)for(size_t b=0;b<a;b++)
            if(op[a]<op[b]+BLOCK_BYTES&&op[b]<op[a]+BLOCK_BYTES)return 0;
    } else {
        uint64_t bytes=(uint64_t)id[0]*id[1]*id[2]*id[3]*2u;
        if(in.pool==out.pool && in.offset<out.offset+bytes && out.offset<in.offset+bytes)return 0;
    }
    return 1;
}

int nprg2_exec_sign_run(const Nprg2ExecCall *c,void *ctx) {
    if(!nprg2_exec_sign_validate(c,ctx))return 0;
    uint32_t ii,oi;Nprg2TensorView in,out;nprg2_exec_input(c,0,&ii,&in);nprg2_exec_output(c,0,&oi,&out);(void)ii;(void)oi;
    int32_t d[4],m[4],o[4],tmpd[4],tmpm[4],tmpo[4],iz;SignLevels lv;
    tensor_shape(&in,d,m,o);tensor_shape(&out,tmpd,tmpm,tmpo);sign_levels(&in,&out,&iz,&lv);
    if(c->command.variant==SIGN_FH) {
        uint64_t n=(uint64_t)d[0]*d[1]*d[2]*d[3];uint8_t *src=c->pools[in.pool].base+in.offset,*dst=c->pools[out.pool].base+out.offset;
        for(uint64_t i=0;i<n;i++){uint16_t x=rd16(src+2*i);int64_t y=select_level(x,iz,&lv);dst[2*i]=(uint8_t)y;dst[2*i+1]=(uint8_t)(y>>8);}
        return 1;
    }
    for(uint32_t b=0;b<(uint32_t)d[0];b++)for(uint32_t h=0;h<(uint32_t)d[1];h++)for(uint32_t w=0;w<(uint32_t)d[2];w++)for(uint32_t z=0;z<(uint32_t)d[3];z++){
        uint8_t *a,*q;if(!elem_ptr(c,&in,m,b,h,w,z,0,&a)||!elem_ptr(c,&out,m,b,h,w,z,1,&q))return 0;
        int64_t y=select_level(rd16(a),iz,&lv);q[0]=(uint8_t)y;q[1]=(uint8_t)(y>>8);
    }
    return 1;
}

int nprg2_executor_register_sign(Nprg2Executor *e) {
    if(!e)return NPRG2_EXEC_ARGUMENT;
    int rc=nprg2_executor_register(e,QDR_SIGN,SIGN_FH,nprg2_exec_sign_validate,nprg2_exec_sign_run,NULL);
    if(rc!=NPRG2_EXEC_OK)return rc;
    return nprg2_executor_register(e,QDR_SIGN,SIGN_CH,nprg2_exec_sign_validate,nprg2_exec_sign_run,NULL);
}
