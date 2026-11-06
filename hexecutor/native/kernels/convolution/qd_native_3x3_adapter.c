#include "qd_native_3x3_adapter.h"
#include <string.h>

#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
typedef unsigned char QdVec128 __attribute__((vector_size(128), aligned(128)));
typedef unsigned char QdVec32 __attribute__((vector_size(32), aligned(32)));
static void copy128(void *d,const void *s,uint32_t n){QdVec128*vd=(QdVec128*)d;const QdVec128*vs=(const QdVec128*)s;for(uint32_t i=0;i<n;i+=128)vd[i/128]=vs[i/128];}
static void copy32(void*d,const void*s){*(QdVec32*)d=*(const QdVec32*)s;}
static int bytes_match(const uint8_t*p,uint32_t n,uint8_t value){
    HVX_Vector expected=Q6_V_vsplat_R((uint32_t)value*0x01010101u);
    HVX_Vector diff=Q6_V_vzero();uint32_t aligned=n&~127u;
    for(uint32_t i=0;i<aligned;i+=128)
        diff=Q6_V_vor_VV(diff,Q6_V_vxor_VV(*(const HVX_Vector*)(p+i),expected));
    uint32_t any=0;
    /* Vector-to-scalar extraction is coherent here; spilling a reduction
     * vector into the scalar-cached stack is not. */
    for(uint32_t lane=0;lane<32;lane++)any|=(uint32_t)Q6_R_vextract_VR(diff,lane*4u);
    if(any)return 0;
    for(uint32_t i=aligned;i<n;i++)if(p[i]!=value)return 0;
    return 1;
}
#else
static void copy128(void*d,const void*s,uint32_t n){memcpy(d,s,n);}
static void copy32(void*d,const void*s){memcpy(d,s,32);}
static int bytes_match(const uint8_t*p,uint32_t n,uint8_t value){for(uint32_t i=0;i<n;i++)if(p[i]!=value)return 0;return 1;}
#endif

extern void qd_hmx_finish(void *,void *);
extern void qd_set_hmx_params_stride1(uint32_t *,uint32_t,uint32_t,uint32_t,
                                      uint32_t,uint32_t,uint32_t,uint32_t);

static uint32_t up128(uint32_t x){return (x+127u)&~127u;}
static uint32_t ptr32(const void*p){return (uint32_t)(uintptr_t)p;}
static int range_overlap(const void*a,uint64_t an,const void*b,uint64_t bn){
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(an>UINTPTR_MAX-x||bn>UINTPTR_MAX-y)return 1;
    return x<y+bn&&y<x+an;
}
static int workspace_alias(const QdN3x3Request*r,const void*p,uint64_t n){
    return range_overlap(r->workspace,r->workspace_bytes,p,n);
}
static int resolve(const QdN3x3Request*r,const QdN3x3Blocked*t,uint32_t index,const uint8_t**out){
    if(index>=t->block_count||!t->blocks)return QD_N3X3_RANGE;
    uint32_t e=t->blocks[index],pool=e&127u,off=e&~127u;
    if(pool>=r->pool_count||pool>=7||!r->pool[pool]||!r->pool_bytes||
       off%QD_N3X3_PAGE_BYTES||off>r->pool_bytes[pool]||
       QD_N3X3_PAGE_BYTES>r->pool_bytes[pool]-off)return QD_N3X3_RANGE;
    *out=r->pool[pool]+off;
    return ((uintptr_t)*out&127u)?QD_N3X3_RANGE:QD_N3X3_OK;
}
static int block_index(const QdN3x3Blocked*t,uint32_t y,uint32_t x,uint32_t cb,uint32_t*out){
    uint64_t icb=t->max_dims[3]/32u,nbw=t->max_dims[2]/8u;
    uint64_t i=((uint64_t)(y/8u)*nbw+x/8u)*icb+t->slice_offset+cb;
    if(cb>=icb||i>=t->block_count||i>UINT32_MAX)return QD_N3X3_RANGE;
    *out=(uint32_t)i;return QD_N3X3_OK;
}

int qd_native_3x3_prepare(const QdN3x3Request*r,QdN3x3Prepared*p){
    if(!r||!p||!r->workspace||!r->pool_bytes||r->pool_count<7||
       ((uintptr_t)r->workspace&127u)||r->workspace_bytes>QD_N3X3_WORKSPACE_MAX||
       r->workspace_bytes<4096||!r->weights||!r->fi||!r->input.blocks||!r->output.blocks)
        return QD_N3X3_UNSUPPORTED;
    if(workspace_alias(r,r,sizeof *r)||workspace_alias(r,p,sizeof *p))return QD_N3X3_ALIAS;
    const QdN3x3Blocked*x=&r->input,*y=&r->output;
    if(x->dims[0]!=1||y->dims[0]!=1||x->max_dims[0]!=1||y->max_dims[0]!=1||
       r->weight_kernel_h!=3||r->weight_kernel_w!=3||
       !r->weight_ci||!r->weight_co||(r->weight_ci&31u)||(r->weight_co&31u)||
       x->dims[3]!=r->weight_ci||x->max_dims[3]!=r->weight_ci||
       !y->dims[3]||y->dims[3]>r->weight_co||y->max_dims[3]!=r->weight_co||
       r->input_dtype!=QD_N3X3_DTYPE_U8||r->output_dtype!=QD_N3X3_DTYPE_U8||
       r->weight_dtype!=QD_N3X3_DTYPE_I8||r->fi_dtype!=QD_N3X3_DTYPE_I32||
       r->weight_format!=QD_N3X3_WEIGHT_XWB||r->input_zero_point!=0||
       r->weights_bytes!=(uint64_t)9u*r->weight_ci*r->weight_co||
       r->fi_bytes!=(uint64_t)r->weight_co*8u||
       ((uintptr_t)r->weights&127u)||((uintptr_t)r->fi&127u))return QD_N3X3_UNSUPPORTED;
    if(y->dims[3]!=r->weight_co &&
       !(r->weight_ci==32u&&r->weight_co==32u&&y->dims[3]==2u&&
         x->dims[1]==10u&&x->dims[2]==258u&&x->max_dims[2]==272u&&
         y->dims[1]==8u&&y->dims[2]==256u&&y->max_dims[1]==8u&&y->max_dims[2]==256u))
        return QD_N3X3_UNSUPPORTED;
    /* This is the currently proven wrapper geometry/configuration family. */
    if(!x->dims[1]||!x->dims[2]||!y->dims[1]||!y->dims[2]||
       x->max_dims[1]!=24||x->padding[1]!=7||x->padding[2]!=7||
       x->padding[0]||x->padding[3]||y->padding[0]||y->padding[1]||y->padding[2]||y->padding[3]||
       x->max_dims[1]%8u||x->max_dims[2]%8u||y->max_dims[1]%8u||y->max_dims[2]%8u||
       x->dims[1]>x->max_dims[1]||x->dims[2]>x->max_dims[2]||
       y->dims[1]>y->max_dims[1]||y->dims[2]>y->max_dims[2]||
       x->dims[1]+x->padding[1]+2u>x->max_dims[1]||
       x->dims[2]+x->padding[2]+2u>x->max_dims[2])return QD_N3X3_UNSUPPORTED;

    uint32_t icb=r->weight_ci/32u,ocb=r->weight_co/32u;
    uint32_t ihr=x->max_dims[1]/8u,ihc=x->max_dims[2]/8u;
    uint32_t ohr=y->max_dims[1]/8u,ohc=y->max_dims[2]/8u;
    uint64_t inpages=(uint64_t)ihr*ihc*icb,outpages=(uint64_t)ohr*ohc*ocb;
    if(!inpages||!outpages||inpages>QD_N3X3_TABLE_ENTRIES||outpages>QD_N3X3_TABLE_ENTRIES||
       (uint64_t)x->slice_offset+inpages>x->block_count||
       (uint64_t)y->slice_offset+outpages>y->block_count||
       x->slice_offset%icb
#if !QDS_NATIVE_3X3_SLICES
       ||y->slice_offset%ocb
#endif
       )return QD_N3X3_RANGE;

    uint64_t inbytes=inpages*QD_N3X3_PAGE_BYTES,outbytes=outpages*QD_N3X3_PAGE_BYTES;
    uint64_t itbytes=(uint64_t)x->block_count*sizeof(uint32_t),otbytes=(uint64_t)y->block_count*sizeof(uint32_t);
    if(workspace_alias(r,r->weights,r->weights_bytes)||workspace_alias(r,r->fi,r->fi_bytes)||
       workspace_alias(r,x->blocks,itbytes)||workspace_alias(r,y->blocks,otbytes)||
       range_overlap(r->weights,r->weights_bytes,r->fi,r->fi_bytes))return QD_N3X3_ALIAS;

    const uint8_t*inrefs[QD_N3X3_TABLE_ENTRIES],*outrefs[QD_N3X3_TABLE_ENTRIES];
    uint32_t e;
    for(uint32_t ty=0;ty<ihr;ty++)for(uint32_t tx=0;tx<ihc;tx++)for(uint32_t cb=0;cb<icb;cb++){
        uint32_t q=((ty*ihc+tx)*icb+cb),ix;
        if((e=block_index(x,ty*8u,tx*8u,cb,&ix))||
           (e=resolve(r,x,ix,&inrefs[q])))return e;
        if(workspace_alias(r,inrefs[q],QD_N3X3_PAGE_BYTES))return QD_N3X3_ALIAS;
    }
    for(uint32_t ty=0;ty<ohr;ty++)for(uint32_t tx=0;tx<ohc;tx++)for(uint32_t cb=0;cb<ocb;cb++){
        uint32_t q=((ty*ohc+tx)*ocb+cb),ix;
        if((e=block_index(y,ty*8u,tx*8u,cb,&ix))||
           (e=resolve(r,y,ix,&outrefs[q])))return e;
        if(workspace_alias(r,outrefs[q],QD_N3X3_PAGE_BYTES))return QD_N3X3_ALIAS;
        for(uint32_t k=0;k<q;k++)if(outrefs[k]==outrefs[q])return QD_N3X3_ALIAS;
    }

    uint32_t cfgcheck[16]={0};
    qd_set_hmx_params_stride1(cfgcheck,0x700u,3u,x->max_dims[1]/8u,
                              r->weight_ci,7u,1u,12u);
    if(!(cfgcheck[12]&8u))return QD_N3X3_UNSUPPORTED;
    uint32_t cursor=0;
#define TAKE(bytes,field) do{cursor=up128(cursor);(field)=cursor;if((uint64_t)cursor+(bytes)>UINT32_MAX)return QD_N3X3_WORKSPACE;cursor+=(uint32_t)(bytes);}while(0)
    uint32_t oi,oo,ow,ofi,oz,og,ogin,owg,oc;
    TAKE(inbytes,oi);TAKE(outbytes,oo);TAKE(r->weights_bytes,ow);TAKE(r->fi_bytes,ofi);
    TAKE(64u*sizeof(uint32_t),oz);TAKE(QD_N3X3_PAGE_BYTES,oc);TAKE(QD_N3X3_PAGE_BYTES,ogin);
    TAKE(QD_N3X3_PAGE_BYTES,og);TAKE(128u,owg);cursor=up128(cursor);
#undef TAKE
    if(cursor>r->workspace_bytes)return QD_N3X3_WORKSPACE;

    memset(p,0,sizeof *p);p->request=r;p->input_pages=(uint32_t)inpages;p->output_pages=(uint32_t)outpages;
    p->input_page_rows=ihr;p->input_page_cols=ihc;p->output_page_rows=ohr;p->output_page_cols=ohc;
    p->logical_h=y->dims[1];p->logical_w=y->dims[2];p->physical_h=y->max_dims[1];p->physical_w=y->max_dims[2];
    p->workspace_used=cursor;uint8_t*base=(uint8_t*)r->workspace;
    p->input_table=p->tables;p->output_table=p->tables+QD_N3X3_TABLE_ENTRIES;
    p->input_data=base+oi;p->output_data=base+oo;p->weights=base+ow;p->fi=(int32_t*)(base+ofi);
    p->zero_fi=(int32_t*)(base+oz);p->done=(volatile uint32_t*)(base+oc);
    p->input_guard=base+ogin;p->output_guard=base+og;p->weight_guard=base+owg;
    p->input_desc=p->abi;p->output_desc=p->input_desc+6;p->cfg=p->output_desc+6;p->mode=p->cfg+16;
    memset(p->input_guard,0xa5,QD_N3X3_PAGE_BYTES);memset(p->output_guard,0xa5,QD_N3X3_PAGE_BYTES);
    memset(p->weight_guard,0xa5,128u);memset((void*)p->done,0,QD_N3X3_PAGE_BYTES);
    for(uint32_t q=0;q<p->input_pages;q++){uint8_t*d=p->input_data+q*QD_N3X3_PAGE_BYTES;copy128(d,inrefs[q],QD_N3X3_PAGE_BYTES);p->input_table[q]=ptr32(d);}
    for(uint32_t q=p->input_pages;q<QD_N3X3_TABLE_ENTRIES;q++)p->input_table[q]=ptr32(p->input_guard);
    for(uint32_t q=0;q<p->output_pages;q++){uint8_t*d=p->output_data+q*QD_N3X3_PAGE_BYTES;copy128(d,outrefs[q],QD_N3X3_PAGE_BYTES);p->output_table[q]=ptr32(d);}
    for(uint32_t q=p->output_pages;q<QD_N3X3_TABLE_ENTRIES;q++)p->output_table[q]=ptr32(p->output_guard);
    copy128(p->weights,r->weights,r->weights_bytes);copy128(p->fi,r->fi,r->fi_bytes);
    for(uint32_t i=0;i<64u;i++)((uint32_t*)p->zero_fi)[i]=i<32u?0x404000u:0u;
    p->input_desc[0]=ptr32(p->input_table);p->input_desc[1]=icb;p->input_desc[2]=ihc*icb;
    p->input_desc[3]=x->dims[2]+x->padding[2];p->input_desc[4]=x->dims[1]+x->padding[1];p->input_desc[5]=r->weight_ci;
    p->output_desc[0]=ptr32(p->output_table);p->output_desc[1]=ocb;p->output_desc[2]=ohc*ocb;
    p->output_desc[3]=y->dims[2];p->output_desc[4]=y->dims[1];p->output_desc[5]=r->weight_co;
    memset(p->cfg,0,16u*sizeof(uint32_t));qd_set_hmx_params_stride1(p->cfg,0x700u,3u,x->max_dims[1]/8u,
                                                                        r->weight_ci,7u,1u,12u);
    /* The generic Xwb path consumes zero as its second mode word. Output
     * quantization offset is already represented by the Fi rows. */
    p->mode[0]=1u;p->mode[1]=0u;
    return QD_N3X3_OK;
}

int qd_native_3x3_issue(QdN3x3Prepared*p){
    if(!p||!p->request||p->issued||p->executed||!p->done)return QD_N3X3_UNSUPPORTED;
    const uint8_t *base=(const uint8_t*)p->request->workspace;
    for(uintptr_t q=(uintptr_t)base;q<(uintptr_t)base+p->workspace_used;q+=32)
        __asm__ volatile("dccleaninva(%0)"::"r"(q):"memory");
    for(uintptr_t q=(uintptr_t)p->tables;q<(uintptr_t)p->tables+sizeof p->tables;q+=32)
        __asm__ volatile("dccleaninva(%0)"::"r"(q):"memory");
    for(uintptr_t q=(uintptr_t)p->abi;q<(uintptr_t)p->abi+sizeof p->abi;q+=32)
        __asm__ volatile("dccleaninva(%0)"::"r"(q):"memory");
    __asm__ volatile("barrier":::"memory");
    uint32_t old_ssr;__asm__ volatile("%0=ssr":"=r"(old_ssr));p->saved_ssr=old_ssr;
    uint32_t hmx_ssr=old_ssr|(1u<<26);
    __asm__ volatile("ssr=%0\n isync"::"r"(hmx_ssr):"memory");
    *p->done=0xffffffffu;__asm__ volatile("mxclracc":::"memory");
    qd_qnn_hmx_stride1_generic(p->output_desc,p->input_desc,p->weights,p->fi,p->cfg,p->mode);
    p->issued=1u;return QD_N3X3_OK;
}

int qd_native_3x3_drain(QdN3x3Prepared*p){
    if(!p||!p->request||!p->issued||p->executed||!p->done)return QD_N3X3_UNSUPPORTED;
    qd_hmx_finish(p->zero_fi,(void*)p->done);
    uint32_t n=0;while(*p->done&&n++<100001u)__asm__ volatile("nop");
    if(*p->done)return QD_N3X3_TIMEOUT;
    __asm__ volatile("ssr=%0\n isync"::"r"(p->saved_ssr):"memory");
    p->issued=0u;p->executed=1u;return QD_N3X3_OK;
}

int qd_native_3x3_execute(QdN3x3Prepared*p){
    int e=qd_native_3x3_issue(p);if(e)return e;
    return qd_native_3x3_drain(p);
}

int qd_native_3x3_finish(const QdN3x3Prepared*p){
    if(!p||!p->request||!p->executed)return QD_N3X3_UNSUPPORTED;
    const QdN3x3Request*r=p->request;const QdN3x3Blocked*y=&r->output;
    if(!bytes_match(p->input_guard,QD_N3X3_PAGE_BYTES,0xa5u)||
       !bytes_match(p->output_guard,QD_N3X3_PAGE_BYTES,0xa5u)||
       !bytes_match(p->weight_guard,128u,0xa5u))return QD_N3X3_GUARD;
    for(uint32_t i=0;i<p->input_pages;i++)if(p->input_table[i]!=ptr32(p->input_data+(uint64_t)i*QD_N3X3_PAGE_BYTES))return QD_N3X3_GUARD;
    for(uint32_t i=p->input_pages;i<QD_N3X3_TABLE_ENTRIES;i++)if(p->input_table[i]!=ptr32(p->input_guard))return QD_N3X3_GUARD;
    for(uint32_t i=0;i<p->output_pages;i++)if(p->output_table[i]!=ptr32(p->output_data+(uint64_t)i*QD_N3X3_PAGE_BYTES))return QD_N3X3_GUARD;
    for(uint32_t i=p->output_pages;i<QD_N3X3_TABLE_ENTRIES;i++)if(p->output_table[i]!=ptr32(p->output_guard))return QD_N3X3_GUARD;
    uint32_t ocb=r->weight_co/32u;
    for(uint32_t ty=0;ty<p->output_page_rows;ty++)for(uint32_t tx=0;tx<p->output_page_cols;tx++)for(uint32_t cb=0;cb<ocb;cb++){
        uint32_t q=((ty*p->output_page_cols+tx)*ocb+cb),block;
        const uint8_t*src=(const uint8_t*)(uintptr_t)p->output_table[q];
        int e=block_index(y,ty*8u,tx*8u,cb,&block);if(e)return e;
        const uint8_t*target;if((e=resolve(r,y,block,&target)))return e;
        if(ty*8u+8u<=p->logical_h&&tx*8u+8u<=p->logical_w){
            /* A complete physical crouton is the common graph case. The
             * CO2/maxCO32 profile is intentionally full-tile too because the
             * old CPU engine writes all 32 physical lanes there. */
            copy128((uint8_t*)target,src,QD_N3X3_PAGE_BYTES);
            continue;
        }
        for(uint32_t ry=0;ry<8u;ry++)if(ty*8u+ry<p->logical_h)
            for(uint32_t rx=0;rx<8u;rx++)if(tx*8u+rx<p->logical_w){
                copy32((uint8_t*)target+ry*256u+rx*32u,
                       src+ry*256u+rx*32u);
            }
    }
    return QD_N3X3_OK;
}
