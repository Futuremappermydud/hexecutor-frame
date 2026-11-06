#ifndef QD_GRAPH_NATIVE_DIRECT_H
#define QD_GRAPH_NATIVE_DIRECT_H

/* Experimental bounded-range harness. Pool 1 moves once at each range
 * boundary; every command inside the range must use a direct native leaf.
 * This is deliberately separate from whole-model prepared-program execution. */
#include "../research/provider-runtime/qd_native_direct.h"
#if QDS_DIRECT_BIND_ONLY
static const volatile char graph_bind_only_feature[]="QD_BIND_ONLY_1";
#endif
#if QDS_DIRECT_COMPACT_INPUT
#if QDS_DIRECT_COMPACT_INPUT == 2
static const volatile char graph_compact_input_feature[]="QD_COMPACT_INPUT_2";
#else
static const volatile char graph_compact_input_feature[]="QD_COMPACT_INPUT_1";
#endif
#endif

typedef struct {
    uint32_t sizes[7];
    QdN1x1Request one;
    QdN3x3Request three;
    uint32_t family;
} QdGraphDirectRequest;

static int graph_direct_guards(const QdDirectBinding *b) {
    HVX_Vector difference=Q6_V_vzero(),a5=Q6_V_vsplat_R(0xa5a5a5a5u);
    const HVX_Vector *out=b->output_guard,*in=b->input_guard;
    for(uint32_t q=0;q<QD_DIRECT_PAGE_BYTES/128u;q++) {
        difference=Q6_V_vor_VV(difference,Q6_V_vxor_VV(out[q],a5));
        difference=Q6_V_vor_VV(difference,in[q]);
    }
    if(b->packed_weight_guard)difference=Q6_V_vor_VV(difference,Q6_V_vxor_VV(*(const HVX_Vector*)b->packed_weight_guard,a5));
    const HVX_Vector *fi=b->zero_fi;
    difference=Q6_V_vor_VV(difference,Q6_V_vxor_VV(fi[0],Q6_V_vsplat_R(0x404000u)));
    difference=Q6_V_vor_VV(difference,fi[1]);
    uint32_t any=0;
    for(uint32_t q=0;q<32;q++)any|=(uint32_t)Q6_R_vextract_VR(difference,q*4u);
    return any==0;
}

#if QDS_DIRECT_COMPACT_INPUT || QDS_PREPARED_COMPACT_3X3
#include "qd_graph_native_compact.h"
#endif

static int graph_direct_request(QdgGraph *g,const C *c,QdGraphDirectRequest *n) {
    if(c->op!=QDR_HMX_CONV || (c->variant!=32 && c->variant!=33) ||
       c->nin<3 || c->nout!=1)return 0;
    for(uint32_t k=0;k<3;k++)if(c->in[k]>=g->ntensors)return 0;
    if(c->out[0]>=g->ntensors)return 0;
    const T *x=dsp_tensor(g,c->in[0]),*w=dsp_tensor(g,c->in[1]);
    const T *f=dsp_tensor(g,c->in[2]),*y=dsp_tensor(g,c->out[0]);
    if(!(x->flags&QDR_BLOCKED) || !(y->flags&QDR_BLOCKED) ||
       !is_class(x,"CB","cb") || !is_class(y,"CB","cb") ||
       x->dtype!=DT_U8 || y->dtype!=DT_U8 || !x->crouton || !y->crouton ||
       !(w->flags&QDR_DIRECT) || !(f->flags&QDR_DIRECT) ||
       (w->flags&QDR_BLOCKED) || (f->flags&QDR_BLOCKED) ||
       w->pool!=1 || f->pool!=1 || w->dtype!=DT_I8 || f->dtype!=DT_I32 ||
       !is_class(f,"Fi","fi") || !x->blocks || !y->blocks)return 0;
    uint64_t wb=1,fb=4;
    for(uint32_t d=0;d<4;d++) {
        if(w->m[d]<=0 || w->d[d]!=w->m[d] || w->p[d] ||
           f->m[d]<=0 || f->d[d]!=f->m[d] || f->p[d])return 0;
        wb*=(uint32_t)w->m[d];fb*=(uint32_t)f->m[d];
        if(wb>UINT32_MAX || fb>UINT32_MAX)return 0;
    }
    if(!native_1x1_pool_span(g,w,(uint32_t)wb) ||
       !native_1x1_pool_span(g,f,(uint32_t)fb))return 0;
    memset(n,0,sizeof *n);
    for(uint32_t p=0;p<7;p++) {
        if(g->pool_bytes[p]>UINT32_MAX)return 0;
        n->sizes[p]=(uint32_t)g->pool_bytes[p];
        n->one.pool[p]=n->three.pool[p]=g->pool[p];
    }
    if(w->m[0]==1 && w->m[1]==1 &&
       ((c->variant==32 && is_class(w,"Xpb","xpb")) ||
        (c->variant==33 && is_class(w,"Xwb","xwb")))) {
        QdN1x1Request *r=&n->one;n->family=1;
        r->pool_count=7;r->pool_bytes=n->sizes;
        for(uint32_t d=0;d<4;d++) {
            r->input.dims[d]=x->d[d];r->input.max_dims[d]=x->m[d];r->input.padding[d]=x->p[d];
            r->output.dims[d]=y->d[d];r->output.max_dims[d]=y->m[d];r->output.padding[d]=y->p[d];
        }
        r->input.blocks=x->blocks;r->input.block_count=x->nblocks;r->input.slice_offset=x->slice;
        r->output.blocks=y->blocks;r->output.block_count=y->nblocks;r->output.slice_offset=y->slice;
        r->weights=g->pool[1]+w->offset;r->weights_bytes=(uint32_t)wb;
        r->fi=(const int32_t *)(g->pool[1]+f->offset);r->fi_bytes=(uint32_t)fb;
        r->weight_ci=w->m[2];r->weight_co=w->m[3];r->weight_kernel_h=r->weight_kernel_w=1;
        r->input_dtype=r->output_dtype=QD_N1X1_DTYPE_U8;
        r->weight_dtype=QD_N1X1_DTYPE_I8;r->fi_dtype=QD_N1X1_DTYPE_I32;
        r->weight_format=c->variant==32?QD_N1X1_WEIGHT_XPB:QD_N1X1_WEIGHT_XWB;
        return 1;
    }
    if(w->m[0]==3 && w->m[1]==3 && c->variant==33 && is_class(w,"Xwb","xwb")) {
        QdN3x3Request *r=&n->three;n->family=3;
        r->pool_count=7;r->pool_bytes=n->sizes;
        for(uint32_t d=0;d<4;d++) {
            r->input.dims[d]=x->d[d];r->input.max_dims[d]=x->m[d];r->input.padding[d]=x->p[d];
            r->output.dims[d]=y->d[d];r->output.max_dims[d]=y->m[d];r->output.padding[d]=y->p[d];
        }
        r->input.blocks=x->blocks;r->input.block_count=x->nblocks;r->input.slice_offset=x->slice;
        r->output.blocks=y->blocks;r->output.block_count=y->nblocks;r->output.slice_offset=y->slice;
        r->weights=g->pool[1]+w->offset;r->weights_bytes=(uint32_t)wb;
        r->fi=(const int32_t *)(g->pool[1]+f->offset);r->fi_bytes=(uint32_t)fb;
        r->weight_ci=w->m[2];r->weight_co=w->m[3];r->weight_kernel_h=r->weight_kernel_w=3;
        r->input_dtype=r->output_dtype=QD_N3X3_DTYPE_U8;
        r->weight_dtype=QD_N3X3_DTYPE_I8;r->fi_dtype=QD_N3X3_DTYPE_I32;
        r->weight_format=QD_N3X3_WEIGHT_XWB;r->input_zero_point=x->zp;r->output_zero_point=y->zp;
        return 1;
    }
    return 0;
}

static int graph_direct_bind(QdGraphDirectRequest *n,int commit,QdDirectBinding *b) {
    void *v=(void *)QD_DIRECT_VTCM_BASE,*tail=(uint8_t *)v+QD_DIRECT_TAIL_OFFSET;
    /* Legacy workers are quiescent throughout this range, so their lower
     * 512 KiB L2 scratch is available. Private stacks start at +512 KiB. */
    uint32_t *tables=(uint32_t *)0x30000000u,*abi=(uint32_t *)0x30000800u;
    if(n->family==1) {
        if(commit)return qd_direct_bind_1x1(&n->one,v,QD_DIRECT_ACTIVE_BYTES,tables,QD_DIRECT_TABLE_BYTES,abi,QD_DIRECT_ABI_BYTES,tail,QD_DIRECT_TAIL_BYTES,b);
        return qd_direct_validate_1x1(&n->one,v,QD_DIRECT_ACTIVE_BYTES,tables,QD_DIRECT_TABLE_BYTES,abi,QD_DIRECT_ABI_BYTES,tail,QD_DIRECT_TAIL_BYTES,b);
    }
    if(commit)return qd_direct_bind_3x3(&n->three,v,QD_DIRECT_ACTIVE_BYTES,tables,QD_DIRECT_TABLE_BYTES,abi,QD_DIRECT_ABI_BYTES,tail,QD_DIRECT_TAIL_BYTES,b);
    return qd_direct_validate_3x3(&n->three,v,QD_DIRECT_ACTIVE_BYTES,tables,QD_DIRECT_TABLE_BYTES,abi,QD_DIRECT_ABI_BYTES,tail,QD_DIRECT_TAIL_BYTES,b);
}

static int graph_direct_run(QdgGraph *g,uint32_t first,uint32_t last) {
    volatile uint32_t *trace=(volatile uint32_t *)(SERVICE+1024u);
    for(uint32_t q=0;q<16;q++)trace[q]=0;
    trace[0]=0x31524451u;trace[2]=((volatile uint32_t *)SERVICE)[2];
#if QDS_DIRECT_BIND_ONLY
    trace[15]=(uint32_t)graph_bind_only_feature[0];
#endif
    g->error_detail=0;
    if(first>=last || last>g->ncommands || g->pool_bytes[1]<QD_DIRECT_ACTIVE_BYTES ||
       !native_1x1_workers_idle() || !native_1x1_lease_valid() || !native_1x1_hmx_owned())
        return QDG_ERR_UNSUPPORTED;
    uint8_t *shadow=g->pool[1];g->pool[1]=(uint8_t *)QD_DIRECT_VTCM_BASE;
    int error=0;uint32_t ix=first;QdDirectBinding binding;
    /* No VTCM tensor, table or ABI writes before complete interval coverage. */
    for(ix=first;ix<last;ix++) {
        trace[3]=ix;
        C c;QdGraphDirectRequest n;dsp_command_begin(g,ix,&c);
        if(!graph_direct_request(g,&c,&n) || graph_direct_bind(&n,0,&binding)) {
            error=QDG_ERR_UNSUPPORTED;goto out;
        }
    }
#if QDS_DIRECT_COMPACT_INPUT
    /* Restrict this diagnostic to one proven 3x3 input geometry. Its input
     * descriptor needs exactly maxH/8 * maxW/8 * CI/32 pages. */
    if(last!=first+1u){error=QDG_ERR_UNSUPPORTED;goto out;}
    { C c;QdGraphDirectRequest n;dsp_command_begin(g,first,&c);
      if(!graph_direct_request(g,&c,&n) || n.family!=3 ||
         (uint64_t)(n.three.input.max_dims[1]/8u)*(n.three.input.max_dims[2]/8u)*
         (n.three.weight_ci/32u)>(QD_DIRECT_TAIL_BYTES-0x2000u-128u)/2048u) {
          error=QDG_ERR_UNSUPPORTED;goto out;
      }
    }
    trace[15]=(uint32_t)graph_compact_input_feature[0];
#endif
    trace[1]=1;
    uint64_t t=now_ns();
    dsp_clean_range(shadow,QD_DIRECT_ACTIVE_BYTES);
    dsp_packet_copy(g->pool[1],shadow,QD_DIRECT_ACTIVE_BYTES);
    g->dsp_read_cycles+=now_ns()-t;
    trace[1]=2;
    qdrc_invalidate_native();g->resident_key=0;
    for(ix=first;ix<last;ix++) {
        if(resident_cancelled()){error=12;break;}
        C c;QdGraphDirectRequest n;dsp_command_begin(g,ix,&c);t=now_ns();
        if(!graph_direct_request(g,&c,&n) || graph_direct_bind(&n,1,&binding)) {
            error=QDG_ERR_DSP;break;
        }
        trace[1]=3;trace[3]=ix;trace[4]=n.family;
        trace[6]=binding.in_desc[0];trace[7]=binding.out_desc[0];
        trace[8]=(uint32_t)(uintptr_t)binding.weights;trace[9]=(uint32_t)(uintptr_t)binding.fi;
        trace[10]=binding.cfg[12];trace[11]=binding.in_desc[2];
        trace[12]=binding.in_desc[3];trace[13]=binding.in_desc[4];
        /* The extracted generic body contains an original-address branch to
         * an unlinked aligned implementation. Never enter that branch. */
        if(n.family==3 && !(binding.cfg[12]&8u)){error=QDG_ERR_UNSUPPORTED;break;}
        uint64_t *diag=n.family==1?g->dsp_native_1x1_diag:g->dsp_native_3x3_diag;
#if QDS_DIRECT_BIND_ONLY
        volatile uint32_t *echo=(volatile uint32_t *)(SERVICE+0x5000u);
        echo[0]=0x31424451u;echo[1]=n.family;echo[2]=ix;
        echo[3]=binding.in_pages;echo[4]=binding.out_pages;
        echo[5]=n.family==1?n.one.weights_bytes:n.three.weights_bytes;
        echo[6]=n.family==1?n.one.fi_bytes:n.three.fi_bytes;
        if((uint64_t)echo[5]+echo[6]>0x18000u){error=QDG_ERR_UNSUPPORTED;break;}
        for(uint32_t q=0;q<512;q++)echo[16+q]=binding.in_table[q];
        for(uint32_t q=0;q<30;q++)echo[528+q]=binding.in_desc[q];
#endif
#if QDS_DIRECT_COMPACT_INPUT
        if(n.family!=3 || !graph_direct_compact_fits(&binding)){error=QDG_ERR_UNSUPPORTED;break;}
        graph_direct_compact_input(&binding);
        diag[6]+=(uint64_t)binding.in_pages*2048u;
#endif
        diag[6]+=binding.staged_weight_bytes;
        /* Scalar writes to the L2 tables/ABI must be published before the
         * native leaf's vector reads, as in qd_native_3x3_issue(). Only
         * metadata and tail control are cleaned; tensor data stays in VTCM. */
        dsp_clean_range(binding.in_table,QD_DIRECT_TABLE_BYTES);
        dsp_clean_range(binding.in_desc,QD_DIRECT_ABI_BYTES);
        dsp_clean_range((void *)(QD_DIRECT_VTCM_BASE+QD_DIRECT_TAIL_OFFSET),binding.tail_publish_bytes);
#if QDS_DIRECT_BIND_ONLY
        for(uint32_t q=0;q<512;q++)echo[560+q]=binding.in_table[q];
        for(uint32_t q=0;q<30;q++)echo[1072+q]=binding.in_desc[q];
        /* Diagnostic copies only: no HMX instruction consumes these operands. */
        dsp_packet_copy(SERVICE+0x8000u,binding.weights,echo[5]);
        dsp_packet_copy(SERVICE+0x8000u+echo[5],binding.fi,echo[6]);
        for(uint32_t q=0;q<binding.in_pages;q++) {
            uintptr_t p=echo[16+q];
            if(p<QD_DIRECT_VTCM_BASE || p-QD_DIRECT_VTCM_BASE>QD_DIRECT_ACTIVE_BYTES-2048u) {
                error=QDG_ERR_DSP;break;
            }
            dsp_packet_copy(SERVICE+0x20000u+q*2048u,(const void *)p,2048u);
        }
        if(error)break;
        trace[1]=6;
        continue;
#endif
        uint32_t saved_ssr;
        __asm__ volatile("%0=ssr":"=r"(saved_ssr));
        uint32_t hmx_ssr=saved_ssr|(1u<<26);
        __asm__ volatile("ssr=%0\n isync\n barrier"::"r"(hmx_ssr):"memory");
        trace[5]=saved_ssr;trace[1]=4;
        diag[0]++;diag[1]+=now_ns()-t;
        *binding.done=UINT32_MAX;
        __asm__ volatile("mxclracc\n barrier" ::: "memory");t=now_ns();
        if(n.family==1)qd_qnn_hmx_1x1(binding.out_desc,binding.in_desc,binding.weights,binding.fi,binding.cfg,binding.mode);
        else qd_qnn_hmx_stride1_generic(binding.out_desc,binding.in_desc,binding.weights,binding.fi,binding.cfg,binding.mode);
        trace[1]=5;
        diag[2]+=now_ns()-t;t=now_ns();
        qd_hmx_finish(binding.zero_fi,(void *)binding.done);
        uint64_t deadline=now_ns()+100000000ull;
        while(*binding.done && now_ns()<=deadline)__asm__ volatile("nop");
        if(*binding.done){diag[5]++;error=QDG_ERR_DSP;break;}
        __asm__ volatile("ssr=%0\n isync\n barrier"::"r"(saved_ssr):"memory");
        trace[1]=6;
        if(!graph_direct_guards(&binding))error=QDG_ERR_DSP;
#if QDS_DIRECT_COMPACT_INPUT
        if(!graph_direct_compact_guard(&binding))error=QDG_ERR_DSP;
#endif
        diag[3]+=now_ns()-t;
        if(error){diag[5]++;break;}
    }
    qdrc_invalidate_native();g->resident_key=0;
    if(!error) {
        t=now_ns();dsp_packet_copy(shadow,g->pool[1],QD_DIRECT_ACTIVE_BYTES);
        g->dsp_write_cycles+=now_ns()-t;
        trace[1]=7;
    }
out:
    g->pool[1]=shadow;
    if(error){g->stats.error_command=ix;g->error_detail=0x44495200u|(uint32_t)error;}
    trace[14]=(uint32_t)error;
    return error;
}
#endif
