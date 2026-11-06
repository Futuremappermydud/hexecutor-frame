#include "prepared_native.h"

#include <stdint.h>
#include <string.h>

enum { NATIVE_FAMILY_1X1=1, NATIVE_FAMILY_3X3=3 };
enum { G_SHAPE=1, G_CROUTON=64, G_DIRECT=4, G_BLOCKED=2 };

typedef struct {
    uint32_t pool_bytes[7];
    QdN1x1Request one;
    QdN3x3Request three;
    uint8_t family;
} NativeRequest;

static int ranges_overlap(const void *a,size_t an,const void *b,size_t bn){
    if(!an||!bn)return 0;if(!a||!b)return 1;uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(x>UINTPTR_MAX-an||y>UINTPTR_MAX-bn)return 1;return x<y+bn&&y<x+an;
}
static int native_config_ranges_ok(const Nprg2ExecCall *call,const Nprg2NativeConfig *c){
    if(!call||!c||!call->program||!call->bound||!call->pools||!c->tables||!c->abi||!c->tail||
       c->table_bytes<QD_DIRECT_TABLE_BYTES||c->abi_bytes<QD_DIRECT_ABI_BYTES||
       c->tail_bytes<QD_DIRECT_TAIL_BYTES||call->pool1_writable_bytes!=c->active_vtcm_bytes||
       call->pools[1].bytes<c->active_vtcm_bytes||
       call->pools[1].bytes-c->active_vtcm_bytes<c->tail_bytes||
       (uintptr_t)c->tail!=(uintptr_t)call->pools[1].base+c->active_vtcm_bytes)return 0;
    if(ranges_overlap(c->tables,c->table_bytes,c->abi,c->abi_bytes)||
       ranges_overlap(c->tables,c->table_bytes,c->tail,c->tail_bytes)||
       ranges_overlap(c->abi,c->abi_bytes,c->tail,c->tail_bytes))return 0;
    for(uint32_t p=0;p<7;p++)if(p!=1&&
        (ranges_overlap(c->tables,c->table_bytes,call->pools[p].base,call->pools[p].bytes)||
         ranges_overlap(c->abi,c->abi_bytes,call->pools[p].base,call->pools[p].bytes)||
         ranges_overlap(c->tail,c->tail_bytes,call->pools[p].base,call->pools[p].bytes)))return 0;
    const Nprg2Program *p=call->program;const Nprg2Section *s[6]={&call->bound->geometry,&call->bound->tensor,&call->bound->command,&call->bound->refs,&call->bound->vectors,&call->bound->calls};
    for(uint32_t i=0;i<6;i++)if(ranges_overlap(c->tables,c->table_bytes,s[i]->data,s[i]->bytes)||ranges_overlap(c->abi,c->abi_bytes,s[i]->data,s[i]->bytes))return 0;
    if(ranges_overlap(c->tables,c->table_bytes,p->raw,p->raw_bytes)||ranges_overlap(c->abi,c->abi_bytes,p->raw,p->raw_bytes)||
       ranges_overlap(c->tables,c->table_bytes,p->tensors,(size_t)p->tensor_count*sizeof *p->tensors)||
       ranges_overlap(c->abi,c->abi_bytes,p->tensors,(size_t)p->tensor_count*sizeof *p->tensors)||
       ranges_overlap(c->tables,c->table_bytes,p->expanded_blocks,(size_t)p->expanded_block_count*sizeof *p->expanded_blocks)||
       ranges_overlap(c->abi,c->abi_bytes,p->expanded_blocks,(size_t)p->expanded_block_count*sizeof *p->expanded_blocks))return 0;
    return 1;
}

static int class_is(const uint8_t *g,const char *upper,const char *lower){
    return !memcmp(g+2,upper,strlen(upper))||!memcmp(g+2,lower,strlen(lower));
}
static int dims_from_geometry(const uint8_t *g,uint32_t *dims,uint32_t *max,uint32_t *pad){
    if(!(g[1]&G_SHAPE))return 0;
    for(uint32_t i=0;i<4;i++){
        int32_t d=(int16_t)((uint16_t)g[6+i*2]|(uint16_t)g[7+i*2]<<8);
        int32_t m=(int16_t)((uint16_t)g[14+i*2]|(uint16_t)g[15+i*2]<<8);
        int32_t p=(int16_t)((uint16_t)g[22+i*2]|(uint16_t)g[23+i*2]<<8);
        if(d<0||m<0||p<0)return 0;dims[i]=(uint32_t)d;max[i]=(uint32_t)m;pad[i]=(uint32_t)p;
    }
    return 1;
}
static uint32_t elem_bytes(uint8_t dtype){static const uint8_t size[8]={0,1,1,2,2,4,4,4};return dtype<8?size[dtype]:0;}
static int32_t rd_i32(const uint8_t *p){return (int32_t)((uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24);}
static int tensor_view(const Nprg2ExecCall *call,uint32_t input,uint32_t slot,
                       uint32_t *index,Nprg2TensorView *view){
    return (input?nprg2_exec_input(call,slot,index,view):nprg2_exec_output(call,slot,index,view))==NPRG2_EXEC_OK;
}
static int direct_extent(const Nprg2ExecCall *call,const Nprg2TensorView *view,uint32_t bytes,const uint8_t **ptr){
    if(view->storage_kind!=1||view->pool>=7||view->offset>call->pools[view->pool].bytes||
       bytes>call->pools[view->pool].bytes-view->offset||!call->pools[view->pool].base)return 0;
    *ptr=call->pools[view->pool].base+view->offset;return 1;
}
static uint32_t shape_byte_count(const uint8_t *g){
    uint64_t n=1;if(!(g[1]&G_SHAPE))return 0;
    for(uint32_t i=0;i<4;i++){int32_t d=(int16_t)((uint16_t)g[14+i*2]|(uint16_t)g[15+i*2]<<8);if(d<=0)return 0;n*=(uint32_t)d;if(n>UINT32_MAX)return 0;}
    uint32_t e=elem_bytes(g[0]);if(!e||n>UINT32_MAX/e)return 0;return (uint32_t)n*e;
}
static int build_native_request(const Nprg2ExecCall *call,Nprg2NativeConfig *cfg,NativeRequest *out){
    if(!call||!cfg||!out||call->command.opcode!=QDR_HMX_CONV||
       call->command.input_count<3||call->command.output_count!=1||!cfg->submit_and_drain||
       !cfg->active_vtcm_bytes||!native_config_ranges_ok(call,cfg))return 0;
    memset(out,0,sizeof *out);
    uint32_t xi,wi,fi,yi;Nprg2TensorView x,w,f,y;
    if(!tensor_view(call,1,0,&xi,&x)||!tensor_view(call,1,1,&wi,&w)||
       !tensor_view(call,1,2,&fi,&f)||!tensor_view(call,0,0,&yi,&y))return 0;
    (void)xi;(void)wi;(void)fi;(void)yi;
    const uint8_t *xg=x.geometry_record,*wg=w.geometry_record,*fg=f.geometry_record,*yg=y.geometry_record;
    if(!(xg[1]&G_CROUTON)||!(yg[1]&G_CROUTON)||!(xg[1]&G_SHAPE)||!(yg[1]&G_SHAPE)||
       (x.storage_kind!=2)||(y.storage_kind!=2)||!x.blocks||!y.blocks||!x.block_count||!y.block_count||
       !class_is(xg,"CB","cb")||!class_is(yg,"CB","cb")||
       w.storage_kind!=1||f.storage_kind!=1||w.pool!=1||f.pool!=1||
       !class_is(fg,"Fi","fi")||wg[0]!=2u||fg[0]!=5u||xg[0]!=1u||yg[0]!=1u||
       !(wg[1]&G_SHAPE)||!(fg[1]&G_SHAPE))return 0;
    uint32_t xd[4],xm[4],xp[4],yd[4],ym[4],yp[4],wd[4],wm[4],wp[4],fd[4],fm[4],fp[4];
    if(!dims_from_geometry(xg,xd,xm,xp)||!dims_from_geometry(yg,yd,ym,yp)||
       !dims_from_geometry(wg,wd,wm,wp)||!dims_from_geometry(fg,fd,fm,fp))return 0;
    if(!wm[2]||!wm[3]||!wm[0]||!wm[1])return 0;
    int special_co2=(wm[0]==3u&&wm[1]==3u&&wm[2]==32u&&wm[3]==32u&&
        wd[0]==wm[0]&&wd[1]==wm[1]&&wd[2]==wm[2]&&wd[3]==2u&&
        xd[1]==10u&&xd[2]==258u&&xm[2]==272u&&
        yd[1]==8u&&yd[2]==256u&&yd[3]==2u&&ym[1]==8u&&ym[2]==256u&&ym[3]==32u);
    for(uint32_t q=0;q<4;q++)if(((wd[q]!=wm[q])&&!(q==3&&special_co2))||wp[q]||fd[q]!=fm[q]||fp[q])return 0;
    if(fm[0]!=1u||fm[1]!=wm[3]/32u||fm[2]!=1u||fm[3]!=64u)return 0;
    uint32_t wbytes=shape_byte_count(wg),fbytes=shape_byte_count(fg);const uint8_t *wp0,*fp0;
    if(!wbytes||!fbytes||!direct_extent(call,&w,wbytes,&wp0)||!direct_extent(call,&f,fbytes,&fp0))return 0;
    if(wm[0]==1u&&wm[1]==1u){
        if((call->command.variant!=32&&call->command.variant!=33)||
           (call->command.variant==32&&!class_is(wg,"Xpb","xpb"))||
           (call->command.variant==33&&!class_is(wg,"Xwb","xwb")))return 0;
        out->family=NATIVE_FAMILY_1X1;QdN1x1Request *r=&out->one;
        for(uint32_t p=0;p<7;p++){out->pool_bytes[p]=call->program->pool_bytes[p];r->pool[p]=call->pools[p].base;}
        r->pool_count=7;r->pool_bytes=out->pool_bytes;
        uint32_t *d[4]={xd,yd},*m[4]={xm,ym},*pad[4]={xp,yp};
        QdN1x1Blocked *t[2]={&r->input,&r->output};
        for(uint32_t q=0;q<2;q++){memcpy(t[q]->dims,d[q],16);memcpy(t[q]->max_dims,m[q],16);memcpy(t[q]->padding,pad[q],16);}
        r->input.slice_offset=x.slice_origin;r->input.block_count=x.block_count;r->input.blocks=x.blocks;
        r->output.slice_offset=y.slice_origin;r->output.block_count=y.block_count;r->output.blocks=y.blocks;
        r->weights=wp0;r->weights_bytes=wbytes;r->weight_ci=wm[2];r->weight_co=wm[3];
        r->weight_format=call->command.variant==32?QD_N1X1_WEIGHT_XPB:QD_N1X1_WEIGHT_XWB;
#if defined(QDS_NATIVE_PADDED_1X1) && QDS_NATIVE_PADDED_1X1
        /* Only the two QDR Xpb profiles with the validated 32-channel native
         * expansion opt in. Keep weight_ci as the serialized CI used for
         * bounds; weight_native_ci selects the packed execution width. */
        if(r->weight_format==QD_N1X1_WEIGHT_XPB&&xm[3]==32u&&
           ((wd[2]==12u&&wm[2]==12u&&xd[3]==9u)||
            (wd[2]==20u&&wm[2]==20u&&xd[3]==18u)))
            r->weight_native_ci=32u;
#endif
        r->weight_kernel_h=1;r->weight_kernel_w=1;r->input_dtype=QD_N1X1_DTYPE_U8;r->output_dtype=QD_N1X1_DTYPE_U8;
        r->weight_dtype=QD_N1X1_DTYPE_I8;r->fi_dtype=QD_N1X1_DTYPE_I32;
        r->fi=(const int32_t*)fp0;r->fi_bytes=fbytes;
        return 1;
    }
    if(wm[0]==3u&&wm[1]==3u&&call->command.variant==33&&class_is(wg,"Xwb","xwb")){
        out->family=NATIVE_FAMILY_3X3;QdN3x3Request *r=&out->three;
        for(uint32_t p=0;p<7;p++){out->pool_bytes[p]=call->program->pool_bytes[p];r->pool[p]=call->pools[p].base;}
        r->pool_count=7;r->pool_bytes=out->pool_bytes;
        QdN3x3Blocked *t[2]={&r->input,&r->output};uint32_t *d[2]={xd,yd},*m[2]={xm,ym},*pad[2]={xp,yp};
        for(uint32_t q=0;q<2;q++){memcpy(t[q]->dims,d[q],16);memcpy(t[q]->max_dims,m[q],16);memcpy(t[q]->padding,pad[q],16);}
        r->input.slice_offset=x.slice_origin;r->input.block_count=x.block_count;r->input.blocks=x.blocks;
        r->output.slice_offset=y.slice_origin;r->output.block_count=y.block_count;r->output.blocks=y.blocks;
        r->weights=wp0;r->weights_bytes=wbytes;r->weight_ci=wm[2];r->weight_co=wm[3];r->weight_kernel_h=3;r->weight_kernel_w=3;
        r->input_dtype=QD_N3X3_DTYPE_U8;r->output_dtype=QD_N3X3_DTYPE_U8;r->weight_dtype=QD_N3X3_DTYPE_I8;
        r->fi_dtype=QD_N3X3_DTYPE_I32;r->weight_format=QD_N3X3_WEIGHT_XWB;r->fi=(const int32_t*)fp0;r->fi_bytes=fbytes;
        r->input_zero_point=(uint32_t)rd_i32(xg+30);r->output_zero_point=(uint32_t)rd_i32(yg+30);
        return 1;
    }
    return 0;
}
static int native_validate(const Nprg2ExecCall *call,void *ctx){
    Nprg2NativeConfig *c=(Nprg2NativeConfig*)ctx;NativeRequest n;QdDirectBinding b;
    if(!build_native_request(call,c,&n)||
       (c->family_mask&~(NPRG2_NATIVE_ALLOW_1X1|NPRG2_NATIVE_ALLOW_3X3))||
       (c->family_mask&&!(c->family_mask&(n.family==NATIVE_FAMILY_1X1?NPRG2_NATIVE_ALLOW_1X1:NPRG2_NATIVE_ALLOW_3X3))))return 0;
    if(n.family==NATIVE_FAMILY_3X3&&c->max_3x3_input_pages){
        uint64_t pages=(uint64_t)(n.three.input.max_dims[1]/8u)*
                       (uint64_t)(n.three.input.max_dims[2]/8u)*
                       (uint64_t)(n.three.input.max_dims[3]/32u);
        if(pages>c->max_3x3_input_pages)return 0;
    }
    memset(&b,0,sizeof b);
    void *vtcm=(void*)call->pools[1].base;void *tail=c->tail;
    if(n.family==NATIVE_FAMILY_1X1)return qd_direct_validate_1x1(&n.one,vtcm,c->active_vtcm_bytes,c->tables,c->table_bytes,c->abi,c->abi_bytes,tail,c->tail_bytes,&b)==QD_DIRECT_OK;
    return qd_direct_validate_3x3(&n.three,vtcm,c->active_vtcm_bytes,c->tables,c->table_bytes,c->abi,c->abi_bytes,tail,c->tail_bytes,&b)==QD_DIRECT_OK;
}
static int native_run(const Nprg2ExecCall *call,void *ctx){
    Nprg2NativeConfig *c=(Nprg2NativeConfig*)ctx;NativeRequest n;QdDirectBinding b;
    if(!native_validate(call,ctx)||!build_native_request(call,c,&n))return 0;memset(&b,0,sizeof b);
    void *vtcm=(void*)call->pools[1].base;int e;
    if(n.family==NATIVE_FAMILY_1X1)e=qd_direct_bind_1x1(&n.one,vtcm,c->active_vtcm_bytes,c->tables,c->table_bytes,c->abi,c->abi_bytes,c->tail,c->tail_bytes,&b);
    else e=qd_direct_bind_3x3(&n.three,vtcm,c->active_vtcm_bytes,c->tables,c->table_bytes,c->abi,c->abi_bytes,c->tail,c->tail_bytes,&b);
    if(e!=QD_DIRECT_OK)return 0;
    return c->submit_and_drain(c->submit_ctx,call,&b,n.family);
}
int nprg2_executor_register_native_conv(Nprg2Executor *e,Nprg2NativeConfig *c){
    if(!e||!c)return NPRG2_EXEC_ARGUMENT;
    int a=nprg2_executor_register(e,QDR_HMX_CONV,32,native_validate,native_run,c);if(a)return a;
    return nprg2_executor_register(e,QDR_HMX_CONV,33,native_validate,native_run,c);
}
