#include "qd_native_direct.h"
#include <stdint.h>

extern void qd_set_hmx_params_stride1(uint32_t *,uint32_t,uint32_t,uint32_t,
                                      uint32_t,uint32_t,uint32_t,uint32_t);

static uint32_t p32(const void *p){return (uint32_t)(uintptr_t)p;}
static int overlap(const void *a,uint64_t an,const void *b,uint64_t bn){
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(an>UINTPTR_MAX-x||bn>UINTPTR_MAX-y)return 1;
    return x<y+bn&&y<x+an;
}
static int in_range(const void *p,uint64_t n,const void *base,uint64_t cap){
    uintptr_t x=(uintptr_t)p,b=(uintptr_t)base;
    return x>=b&&n<=cap&&x-b<=cap-n;
}
static void zero_words(uint32_t *p,uint32_t n){for(uint32_t i=0;i<n;i++)p[i]=0;}
static void tail_init(uint8_t *tail,uint32_t n){
    for(uint32_t i=0;i<n;i++)tail[i]=0xa5;
    for(uint32_t i=0;i<QD_DIRECT_PAGE_BYTES;i++)tail[QD_DIRECT_PAGE_BYTES+i]=0;
    volatile uint32_t *done=(volatile uint32_t *)(tail+0x1000u);*done=0;
    uint32_t *z=(uint32_t *)(tail+0x1800u);
    for(uint32_t i=0;i<64;i++)z[i]=i<32?0x404000u:0u;
}
static int base_check(void *v,uint32_t active,uint32_t *tables,uint32_t tb,
                      uint32_t *abi,uint32_t ab,void *tail,uint32_t tailb){
    if(!v||active!=QD_DIRECT_ACTIVE_BYTES||!tables||tb<QD_DIRECT_TABLE_BYTES||
       !abi||ab<QD_DIRECT_ABI_BYTES||!tail||tailb!=QD_DIRECT_TAIL_BYTES||tailb<0x2000u||
       ((uintptr_t)v&127u)||((uintptr_t)tables&127u)||((uintptr_t)abi&3u)||
       ((uintptr_t)tail&127u)||p32((uint8_t*)v+QD_DIRECT_TAIL_OFFSET)!=p32(tail))
        return QD_DIRECT_BAD_ARGUMENT;
#if defined(__hexagon__) && !defined(QD_DIRECT_ALLOW_MOCK_BASE)
    if(p32(v)!=QD_DIRECT_VTCM_BASE)return QD_DIRECT_RANGE;
#endif
    if(overlap(tables,tb,v,QD_DIRECT_VTCM_BYTES)||overlap(abi,ab,v,QD_DIRECT_VTCM_BYTES))
        return QD_DIRECT_ALIAS;
    return QD_DIRECT_OK;
}
static int block_page(const uint32_t *pool_bytes,const uint8_t *pool1,
                      const uint32_t *blocks,uint32_t count,uint32_t ix,
                      uint32_t active,const uint8_t **out){
    if(!blocks||ix>=count)return QD_DIRECT_RANGE;
    uint32_t enc=blocks[ix],pool=enc&127u,off=enc&~127u;
    if(pool!=1u||!pool_bytes||off%QD_DIRECT_PAGE_BYTES||
       off>active||QD_DIRECT_PAGE_BYTES>active-off||off>pool_bytes[1]||
       QD_DIRECT_PAGE_BYTES>pool_bytes[1]-off)return QD_DIRECT_RANGE;
    *out=pool1+off;return QD_DIRECT_OK;
}
static int block_ix(const uint32_t *m,uint32_t slice,uint32_t nblocks,
                    uint32_t y,uint32_t x,uint32_t cb,uint32_t *out){
    uint32_t cbn=m[3]/32u,nbw=m[2]/8u;
    uint64_t i=((uint64_t)(y/8u)*nbw+x/8u)*cbn+slice+cb;
    if(cb>=cbn||i>=nblocks||i>UINT32_MAX)return QD_DIRECT_RANGE;
    *out=(uint32_t)i;return QD_DIRECT_OK;
}
static int distinct_page(const uint8_t **refs,uint32_t n,const uint8_t *p){
    for(uint32_t i=0;i<n;i++)if(refs[i]==p)return 0;
    return 1;
}
static int table_and_data_alias(const void *tables,uint32_t tb,const void *abi,uint32_t ab,
                                const void *tail,uint32_t tailb,const uint8_t **in,uint32_t ni,
                                const uint8_t **out,uint32_t no,const void *w,uint32_t wb,
                                const void *fi,uint32_t fb){
    if(overlap(tables,tb,abi,ab)||overlap(tables,tb,tail,tailb)||overlap(abi,ab,tail,tailb)||
       overlap(w,wb,fi,fb)||overlap(w,wb,tail,tailb)||overlap(fi,fb,tail,tailb)||
       overlap(w,wb,tables,tb)||overlap(w,wb,abi,ab)||overlap(fi,fb,tables,tb)||overlap(fi,fb,abi,ab))return 0;
    for(uint32_t i=0;i<ni;i++){
        if(overlap(in[i],QD_DIRECT_PAGE_BYTES,tables,tb)||overlap(in[i],QD_DIRECT_PAGE_BYTES,abi,ab)||
           overlap(in[i],QD_DIRECT_PAGE_BYTES,tail,tailb)||overlap(in[i],QD_DIRECT_PAGE_BYTES,w,wb)||
           overlap(in[i],QD_DIRECT_PAGE_BYTES,fi,fb))return 0;
        for(uint32_t j=0;j<no;j++)if(in[i]==out[j])return 0;
    }
    for(uint32_t i=0;i<no;i++){
        if(overlap(out[i],QD_DIRECT_PAGE_BYTES,tables,tb)||overlap(out[i],QD_DIRECT_PAGE_BYTES,abi,ab)||
           overlap(out[i],QD_DIRECT_PAGE_BYTES,tail,tailb)||overlap(out[i],QD_DIRECT_PAGE_BYTES,w,wb)||
           overlap(out[i],QD_DIRECT_PAGE_BYTES,fi,fb)||!distinct_page(out,i,out[i]))return 0;
    }
    return 1;
}

static int bind_1x1(const QdN1x1Request*r,void*v,uint32_t active,
                    uint32_t*t,uint32_t tb,uint32_t*a,uint32_t ab,
                    void*tail,uint32_t tailb,QdDirectBinding*b,int commit){
    if(!r||(!b&&commit)||!r->pool_bytes||r->pool_count<2||!r->pool[1])return QD_DIRECT_BAD_ARGUMENT;
    int e=base_check(v,active,t,tb,a,ab,tail,tailb);if(e)return e;
    if(r->pool[1]!=(const uint8_t*)v)return QD_DIRECT_RANGE;
    const QdN1x1Blocked*x=&r->input,*y=&r->output;
    uint32_t native_ci=r->weight_ci;int padded_ci=0;
#if QDS_NATIVE_PADDED_1X1
    if(r->weight_native_ci)native_ci=r->weight_native_ci;
    padded_ci=native_ci==32u && r->weight_format==QD_N1X1_WEIGHT_XPB &&
        x->max_dims[3]==32u && !x->padding[3] &&
        ((r->weight_ci==12u && x->dims[3]==9u) ||
         (r->weight_ci==20u && x->dims[3]==18u));
#endif
    int channel_shape_ok=padded_ci || (native_ci==r->weight_ci &&
        x->max_dims[3]==r->weight_ci && x->dims[3]==r->weight_ci);
    uint64_t packed_bytes=(uint64_t)native_ci*r->weight_co;
    if(padded_ci && packed_bytes>tailb-0x2000u-128u)return QD_DIRECT_UNSUPPORTED;
    if(x->dims[0]!=1||y->dims[0]!=1||x->max_dims[0]!=1||y->max_dims[0]!=1||
       x->dims[1]!=y->dims[1]||x->dims[2]!=y->dims[2]||
       x->dims[1]!=x->max_dims[1]||x->dims[2]!=x->max_dims[2]||
       y->dims[1]!=y->max_dims[1]||y->dims[2]!=y->max_dims[2]||
       x->padding[0]||x->padding[1]||x->padding[2]||x->padding[3]||
       y->padding[0]||y->padding[1]||y->padding[2]||y->padding[3]||!channel_shape_ok||y->max_dims[3]!=r->weight_co||
       y->dims[3]!=r->weight_co||!r->weight_ci||!r->weight_co||
       (native_ci&31u)||(r->weight_ci&3u)||(r->weight_co&31u)||r->weight_kernel_h!=1||
       r->weight_kernel_w!=1||r->input_dtype!=QD_N1X1_DTYPE_U8||
       r->output_dtype!=QD_N1X1_DTYPE_U8||r->weight_dtype!=QD_N1X1_DTYPE_I8||
       r->fi_dtype!=QD_N1X1_DTYPE_I32||
       (r->weight_format!=QD_N1X1_WEIGHT_XPB&&r->weight_format!=QD_N1X1_WEIGHT_XWB)||
       !r->weights||!r->fi||r->weights_bytes!=(uint64_t)r->weight_ci*r->weight_co||
       r->fi_bytes!=(uint64_t)r->weight_co*8u||((uintptr_t)r->weights&127u)||
       ((uintptr_t)r->fi&127u)||!in_range(r->weights,r->weights_bytes,v,active)||
       !in_range(r->fi,r->fi_bytes,v,active)||!x->max_dims[1]||!x->max_dims[2]||
       !y->max_dims[1]||!y->max_dims[2]||(x->max_dims[1]&7u)||
       (x->max_dims[2]&7u)||(y->max_dims[1]&7u)||(y->max_dims[2]&7u))return QD_DIRECT_UNSUPPORTED;
    uint32_t icb=native_ci/32u,ocb=r->weight_co/32u;
    uint64_t ni=(uint64_t)(x->max_dims[1]/8u)*(x->max_dims[2]/8u)*icb;
    uint64_t no=(uint64_t)(y->max_dims[1]/8u)*(y->max_dims[2]/8u)*ocb;
    if(!ni||!no||ni>QD_N1X1_TABLE_ENTRIES||no>QD_N1X1_TABLE_ENTRIES)return QD_DIRECT_UNSUPPORTED;
    const uint8_t *ir[QD_N1X1_TABLE_ENTRIES],*or[QD_N1X1_TABLE_ENTRIES];
    for(uint32_t ty=0;ty<x->max_dims[1]/8u;ty++)for(uint32_t tx=0;tx<x->max_dims[2]/8u;tx++)for(uint32_t cb=0;cb<icb;cb++){
        uint32_t q=((ty*(x->max_dims[2]/8u)+tx)*icb+cb),ix;
        if((e=block_ix(x->max_dims,x->slice_offset,x->block_count,ty*8u,tx*8u,cb,&ix))||
           (e=block_page(r->pool_bytes,r->pool[1],x->blocks,x->block_count,ix,active,&ir[q])))return e;
    }
    for(uint32_t ty=0;ty<y->max_dims[1]/8u;ty++)for(uint32_t tx=0;tx<y->max_dims[2]/8u;tx++)for(uint32_t cb=0;cb<ocb;cb++){
        uint32_t q=((ty*(y->max_dims[2]/8u)+tx)*ocb+cb),ix;
        if((e=block_ix(y->max_dims,y->slice_offset,y->block_count,ty*8u,tx*8u,cb,&ix))||
           (e=block_page(r->pool_bytes,r->pool[1],y->blocks,y->block_count,ix,active,&or[q])))return e;
    }
    if(!table_and_data_alias(t,tb,a,ab,tail,tailb,ir,(uint32_t)ni,or,(uint32_t)no,
                             r->weights,r->weights_bytes,r->fi,r->fi_bytes))return QD_DIRECT_ALIAS;
    uint64_t xb=(uint64_t)x->block_count*4u,yb=(uint64_t)y->block_count*4u;
    if(xb>UINT32_MAX||yb>UINT32_MAX||
       overlap(r,sizeof *r,t,tb)||overlap(r,sizeof *r,a,ab)||overlap(r,sizeof *r,tail,tailb)||
       overlap(r->pool_bytes,7u*sizeof(uint32_t),t,tb)||overlap(r->pool_bytes,7u*sizeof(uint32_t),a,ab)||
       overlap(r->pool_bytes,7u*sizeof(uint32_t),tail,tailb)||
       overlap(x->blocks,xb,t,tb)||overlap(x->blocks,xb,a,ab)||overlap(x->blocks,xb,tail,tailb)||
       overlap(y->blocks,yb,t,tb)||overlap(y->blocks,yb,a,ab)||overlap(y->blocks,yb,tail,tailb)||
       (b&&(overlap(b,sizeof *b,t,tb)||overlap(b,sizeof *b,a,ab)||overlap(b,sizeof *b,tail,tailb)||
            overlap(b,sizeof *b,r,sizeof *r)||overlap(b,sizeof *b,r->pool_bytes,7u*sizeof(uint32_t))||
            overlap(b,sizeof *b,x->blocks,xb)||overlap(b,sizeof *b,y->blocks,yb)||
            overlap(b,sizeof *b,r->weights,r->weights_bytes)||overlap(b,sizeof *b,r->fi,r->fi_bytes))))return QD_DIRECT_ALIAS;
    if(b){for(uint32_t i=0;i<(uint32_t)ni;i++)if(overlap(b,sizeof *b,ir[i],QD_DIRECT_PAGE_BYTES))return QD_DIRECT_ALIAS;
          for(uint32_t i=0;i<(uint32_t)no;i++)if(overlap(b,sizeof *b,or[i],QD_DIRECT_PAGE_BYTES))return QD_DIRECT_ALIAS;}
    /* The extracted generic body branches on cfg[12].bit3 to an aligned
     * implementation that is not linked into this image. Match the original
     * adapter's preflight so validation rejects before touching L2/VTCM. */
    uint32_t cfgcheck[16]={0};
    qd_set_hmx_params_stride1(cfgcheck,0x700u,3u,x->max_dims[1]/8u,
                              native_ci,7u,1u,12u);
    if(!(cfgcheck[12]&8u))return QD_DIRECT_UNSUPPORTED;
    if(!commit)return QD_DIRECT_OK;
    tail_init((uint8_t*)tail,0x2000u);
    uint32_t *it=t,*ot=t+QD_N1X1_TABLE_ENTRIES;
    for(uint32_t i=0;i<(uint32_t)ni;i++)it[i]=p32(ir[i]);
    for(uint32_t i=(uint32_t)ni;i<QD_N1X1_TABLE_ENTRIES;i++)it[i]=p32((uint8_t*)tail+0x800u);
    for(uint32_t i=0;i<(uint32_t)no;i++)ot[i]=p32(or[i]);
    for(uint32_t i=(uint32_t)no;i<QD_N1X1_TABLE_ENTRIES;i++)ot[i]=p32(tail);
    zero_words(a,30);uint32_t *od=a+6,*cfg=od+6,*mode=cfg+16;
    a[0]=p32(it);a[1]=icb;a[2]=(uint32_t)ni;a[3]=(x->max_dims[1]/8u)*(x->max_dims[2]/8u);a[4]=64;a[5]=native_ci;
    od[0]=p32(ot);od[1]=ocb;od[2]=(uint32_t)no;od[3]=(y->max_dims[1]/8u)*(y->max_dims[2]/8u);od[4]=64;od[5]=r->weight_co;
    cfg[3]=2047;cfg[6]=1023;mode[0]=1;
    b->in_desc=a;b->out_desc=od;b->cfg=cfg;b->mode=mode;b->in_table=it;b->out_table=ot;
    b->weights=(void*)r->weights;b->tail_publish_bytes=0x2000u;
    b->staged_weight_bytes=0;b->packed_weight_guard=0;
#if QDS_NATIVE_PADDED_1X1
    if(padded_ci){
        uint8_t *dst=(uint8_t *)tail+0x2000u;
        /* Same proven Xpb [O/32][I/4][O%32][I%4] expansion as the staged
         * adapter. Only synthetic input-group weights become zero. */
#ifdef __hexagon__
        typedef unsigned char V128 __attribute__((vector_size(128),aligned(128)));
        V128 zero={0};for(uint32_t q=0;q<(uint32_t)packed_bytes/128u;q++)((V128*)dst)[q]=zero;
        for(uint32_t ob=0;ob<r->weight_co/32u;ob++)for(uint32_t ig=0;ig<r->weight_ci/4u;ig++)
            ((V128*)dst)[ob*8u+ig]=((const V128*)r->weights)[ob*(r->weight_ci/4u)+ig];
#else
        for(uint32_t q=0;q<(uint32_t)packed_bytes;q++)dst[q]=0;
        for(uint32_t ob=0;ob<r->weight_co/32u;ob++)for(uint32_t ig=0;ig<r->weight_ci/4u;ig++)
            for(uint32_t q=0;q<128u;q++)dst[(ob*8u+ig)*128u+q]=r->weights[(ob*(r->weight_ci/4u)+ig)*128u+q];
#endif
        uint32_t *guard=(uint32_t *)(dst+(uint32_t)packed_bytes);
        for(uint32_t q=0;q<32;q++)guard[q]=0xa5a5a5a5u;
        b->weights=dst;b->staged_weight_bytes=(uint32_t)packed_bytes;
        b->packed_weight_guard=guard;b->tail_publish_bytes=0x2000u+(uint32_t)packed_bytes+128u;
    }
#endif
    b->fi=(void*)r->fi;b->zero_fi=(uint8_t*)tail+0x1800u;
    b->done=(volatile uint32_t*)((uint8_t*)tail+0x1000u);b->input_guard=(uint8_t*)tail+0x800u;b->output_guard=tail;
    b->in_pages=(uint32_t)ni;b->out_pages=(uint32_t)no;
    return QD_DIRECT_OK;
}

int qd_direct_validate_1x1(const QdN1x1Request*r,void*v,uint32_t active,uint32_t*t,uint32_t tb,
                           uint32_t*a,uint32_t ab,void*tail,uint32_t tailb,QdDirectBinding*b){
    return bind_1x1(r,v,active,t,tb,a,ab,tail,tailb,b,0);
}
int qd_direct_bind_1x1(const QdN1x1Request*r,void*v,uint32_t active,uint32_t*t,uint32_t tb,
                       uint32_t*a,uint32_t ab,void*tail,uint32_t tailb,QdDirectBinding*b){
    return bind_1x1(r,v,active,t,tb,a,ab,tail,tailb,b,1);
}

static int bind_3x3(const QdN3x3Request*r,void*v,uint32_t active,
                    uint32_t*t,uint32_t tb,uint32_t*a,uint32_t ab,
                    void*tail,uint32_t tailb,QdDirectBinding*b,int commit){
    if(!r||(!b&&commit)||!r->pool_bytes||r->pool_count<2||!r->pool[1])return QD_DIRECT_BAD_ARGUMENT;
    int e=base_check(v,active,t,tb,a,ab,tail,tailb);if(e)return e;
    if(r->pool[1]!=(const uint8_t*)v)return QD_DIRECT_RANGE;
    const QdN3x3Blocked*x=&r->input,*y=&r->output;
    if(x->dims[0]!=1||y->dims[0]!=1||x->max_dims[0]!=1||y->max_dims[0]!=1||
       x->max_dims[1]!=24||x->dims[1]!=10||x->padding[1]!=7||x->padding[2]!=7||x->padding[0]||x->padding[3]||
       y->padding[0]||y->padding[1]||y->padding[2]||y->padding[3]||
       x->dims[1]!=(uint64_t)y->dims[1]+2u||x->dims[2]!=(uint64_t)y->dims[2]+2u||
       !x->max_dims[2]||x->max_dims[2]>272u||(x->max_dims[2]&7u)||
       !y->max_dims[1]||!y->max_dims[2]||y->max_dims[1]>8u||
       (y->max_dims[1]&7u)||y->max_dims[2]>256u||(y->max_dims[2]&7u)||
       y->dims[1]!=y->max_dims[1]||y->dims[2]!=y->max_dims[2]||
       x->dims[1]>x->max_dims[1]||x->dims[2]>x->max_dims[2]||
       (uint64_t)x->dims[1]+x->padding[1]+2u>x->max_dims[1]||
       (uint64_t)x->dims[2]+x->padding[2]+2u>x->max_dims[2]||
       r->weight_kernel_h!=3||r->weight_kernel_w!=3||!r->weight_ci||!r->weight_co||
       (r->weight_ci&31u)||(r->weight_co&31u)||x->max_dims[3]!=r->weight_ci||
       y->max_dims[3]!=r->weight_co||x->dims[3]!=r->weight_ci||
       (y->dims[3]!=r->weight_co&&!(r->weight_ci==32u&&r->weight_co==32u&&y->dims[3]==2u&&
        x->dims[1]==10u&&x->dims[2]==258u&&x->max_dims[2]==272u&&
        y->dims[1]==8u&&y->dims[2]==256u&&y->max_dims[1]==8u&&y->max_dims[2]==256u))||
       r->input_dtype!=QD_N3X3_DTYPE_U8||
       r->output_dtype!=QD_N3X3_DTYPE_U8||r->weight_dtype!=QD_N3X3_DTYPE_I8||
       r->fi_dtype!=QD_N3X3_DTYPE_I32||r->weight_format!=QD_N3X3_WEIGHT_XWB||
       r->input_zero_point!=0||r->weights_bytes!=(uint64_t)9*r->weight_ci*r->weight_co||
       r->fi_bytes!=(uint64_t)r->weight_co*8u||!r->weights||!r->fi||
       ((uintptr_t)r->weights&127u)||((uintptr_t)r->fi&127u)||
       !in_range(r->weights,r->weights_bytes,v,active)||!in_range(r->fi,r->fi_bytes,v,active))return QD_DIRECT_UNSUPPORTED;
    uint32_t icb=r->weight_ci/32u,ocb=r->weight_co/32u;
    uint32_t inrows=x->max_dims[1]/8u,incols=x->max_dims[2]/8u;
    uint32_t outrows=y->max_dims[1]/8u,outcols=y->max_dims[2]/8u;
    uint64_t ni=(uint64_t)inrows*incols*icb,no=(uint64_t)outrows*outcols*ocb;
    if(!ni||!no||ni>QD_N3X3_TABLE_ENTRIES||no>QD_N3X3_TABLE_ENTRIES||
       tb<2u*QD_N3X3_TABLE_ENTRIES*4u)return QD_DIRECT_UNSUPPORTED;
    const uint8_t *ir[QD_N3X3_TABLE_ENTRIES],*or[QD_N3X3_TABLE_ENTRIES];
    for(uint32_t ty=0;ty<inrows;ty++)for(uint32_t tx=0;tx<incols;tx++)for(uint32_t cb=0;cb<icb;cb++){
        uint32_t q=((ty*incols+tx)*icb+cb),ix;
        if((e=block_ix(x->max_dims,x->slice_offset,x->block_count,ty*8u,tx*8u,cb,&ix))||
           (e=block_page(r->pool_bytes,r->pool[1],x->blocks,x->block_count,ix,active,&ir[q])))return e;
    }
    for(uint32_t ty=0;ty<outrows;ty++)for(uint32_t tx=0;tx<outcols;tx++)for(uint32_t cb=0;cb<ocb;cb++){
        uint32_t q=((ty*outcols+tx)*ocb+cb),ix;
        if((e=block_ix(y->max_dims,y->slice_offset,y->block_count,ty*8u,tx*8u,cb,&ix))||
           (e=block_page(r->pool_bytes,r->pool[1],y->blocks,y->block_count,ix,active,&or[q])))return e;
    }
    if(!table_and_data_alias(t,tb,a,ab,tail,tailb,ir,(uint32_t)ni,or,(uint32_t)no,
                             r->weights,r->weights_bytes,r->fi,r->fi_bytes))return QD_DIRECT_ALIAS;
    uint64_t xb=(uint64_t)x->block_count*4u,yb=(uint64_t)y->block_count*4u;
    if(xb>UINT32_MAX||yb>UINT32_MAX||
       overlap(r,sizeof *r,t,tb)||overlap(r,sizeof *r,a,ab)||overlap(r,sizeof *r,tail,tailb)||
       overlap(r->pool_bytes,7u*sizeof(uint32_t),t,tb)||overlap(r->pool_bytes,7u*sizeof(uint32_t),a,ab)||
       overlap(r->pool_bytes,7u*sizeof(uint32_t),tail,tailb)||
       overlap(x->blocks,xb,t,tb)||overlap(x->blocks,xb,a,ab)||overlap(x->blocks,xb,tail,tailb)||
       overlap(y->blocks,yb,t,tb)||overlap(y->blocks,yb,a,ab)||overlap(y->blocks,yb,tail,tailb)||
       (b&&(overlap(b,sizeof *b,t,tb)||overlap(b,sizeof *b,a,ab)||overlap(b,sizeof *b,tail,tailb)||
            overlap(b,sizeof *b,r,sizeof *r)||overlap(b,sizeof *b,r->pool_bytes,7u*sizeof(uint32_t))||
            overlap(b,sizeof *b,x->blocks,xb)||overlap(b,sizeof *b,y->blocks,yb)||
            overlap(b,sizeof *b,r->weights,r->weights_bytes)||overlap(b,sizeof *b,r->fi,r->fi_bytes))))return QD_DIRECT_ALIAS;
    if(b){for(uint32_t i=0;i<(uint32_t)ni;i++)if(overlap(b,sizeof *b,ir[i],QD_DIRECT_PAGE_BYTES))return QD_DIRECT_ALIAS;
          for(uint32_t i=0;i<(uint32_t)no;i++)if(overlap(b,sizeof *b,or[i],QD_DIRECT_PAGE_BYTES))return QD_DIRECT_ALIAS;}
    if(!commit)return QD_DIRECT_OK;
    tail_init((uint8_t*)tail,0x2000u);
    uint32_t *it=t,*ot=t+QD_N3X3_TABLE_ENTRIES;
    for(uint32_t i=0;i<(uint32_t)ni;i++)it[i]=p32(ir[i]);
    for(uint32_t i=(uint32_t)ni;i<QD_N3X3_TABLE_ENTRIES;i++)it[i]=p32((uint8_t*)tail+0x800u);
    for(uint32_t i=0;i<(uint32_t)no;i++)ot[i]=p32(or[i]);
    for(uint32_t i=(uint32_t)no;i<QD_N3X3_TABLE_ENTRIES;i++)ot[i]=p32(tail);
    zero_words(a,30);uint32_t *od=a+6,*cfg=od+6,*mode=cfg+16;
    a[0]=p32(it);a[1]=icb;a[2]=incols*icb;a[3]=x->dims[2]+x->padding[2];a[4]=x->dims[1]+x->padding[1];a[5]=r->weight_ci;
    od[0]=p32(ot);od[1]=ocb;od[2]=outcols*ocb;od[3]=y->dims[2];od[4]=y->dims[1];od[5]=r->weight_co;
    qd_set_hmx_params_stride1(cfg,0x700u,3u,x->max_dims[1]/8u,r->weight_ci,7u,1u,12u);mode[0]=1;
    b->in_desc=a;b->out_desc=od;b->cfg=cfg;b->mode=mode;b->in_table=it;b->out_table=ot;
    b->tail_publish_bytes=0x2000u;b->staged_weight_bytes=0;b->packed_weight_guard=0;
    b->weights=(void*)r->weights;b->fi=(void*)r->fi;b->zero_fi=(uint8_t*)tail+0x1800u;
    b->done=(volatile uint32_t*)((uint8_t*)tail+0x1000u);b->input_guard=(uint8_t*)tail+0x800u;b->output_guard=tail;
    b->in_pages=(uint32_t)ni;b->out_pages=(uint32_t)no;
    return QD_DIRECT_OK;
}

int qd_direct_validate_3x3(const QdN3x3Request*r,void*v,uint32_t active,uint32_t*t,uint32_t tb,
                           uint32_t*a,uint32_t ab,void*tail,uint32_t tailb,QdDirectBinding*b){
    return bind_3x3(r,v,active,t,tb,a,ab,tail,tailb,b,0);
}
int qd_direct_bind_3x3(const QdN3x3Request*r,void*v,uint32_t active,uint32_t*t,uint32_t tb,
                       uint32_t*a,uint32_t ab,void*tail,uint32_t tailb,QdDirectBinding*b){
    return bind_3x3(r,v,active,t,tb,a,ab,tail,tailb,b,1);
}
