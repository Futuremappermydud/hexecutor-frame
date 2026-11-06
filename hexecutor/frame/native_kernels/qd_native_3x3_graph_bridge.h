#ifndef QD_NATIVE_3X3_GRAPH_BRIDGE_H
#define QD_NATIVE_3X3_GRAPH_BRIDGE_H

/* Source-included bridge for the bounded V75 Xwb 3x3 stride-1 leaf.
 * Include after the graph metadata/cache and native 1x1 bridge helpers. */
#if !defined(QDG_DSP) || !defined(QDS_NATIVE_3X3) || !QDS_NATIVE_3X3
#error "qd_native_3x3_graph_bridge.h requires QDG_DSP and QDS_NATIVE_3X3=1"
#endif
#if !defined(QDS_NATIVE_1X1) || !QDS_NATIVE_1X1
#error "native 3x3 graph dispatch requires QDS_NATIVE_1X1=1 lease helpers"
#endif
#if !defined(QDS_PRESERVE_VTCM) || !QDS_PRESERVE_VTCM
#error "native 3x3 graph dispatch requires QDS_PRESERVE_VTCM=1"
#endif

#include "qd_native_3x3_adapter.h"
#include <string.h>

#define QD_N3X3_WORKSPACE_OFFSET 0x600000u
#define QD_N3X3_NATIVE_FAILURE  0x4e330001u

static int native_3x3_defer(QdgGraph *g) {
    if (g) g->dsp_native_3x3_diag[4]++;
    return QDG_DSP_DEFER;
}

static int native_3x3_pool_span(QdgGraph *g, const T *t, uint32_t bytes) {
    return t->pool < QDG_POOLS && g->pool[t->pool] &&
           t->offset <= g->pool_bytes[t->pool] &&
           (uint64_t)bytes <= g->pool_bytes[t->pool] - t->offset;
}

static int native_3x3_try(QdgGraph *g, const C *c, uint32_t index) {
    (void)index;
    if (!g || !c || c->op != QDR_HMX_CONV || c->nin < 3 || c->nout != 1 ||
        c->variant == 36)
        return native_3x3_defer(g);

    const uint32_t input_id=c->in[0], weight_id=c->in[1], fi_id=c->in[2];
    const uint32_t output_id=c->out[0];
    if (input_id>=g->ntensors || weight_id>=g->ntensors || fi_id>=g->ntensors ||
        output_id>=g->ntensors) return native_3x3_defer(g);

    const T *x=dsp_tensor(g,input_id), *y=dsp_tensor(g,output_id);
    const T *fi=dsp_tensor(g,fi_id);
    uint32_t weight_source=g->static_source[weight_id];
    if(weight_source==QDR_NONE)weight_source=weight_id;
    if(weight_source>=g->ntensors)return native_3x3_defer(g);
    const T *wt=dsp_tensor(g,weight_source);
    if(!x||!y||!fi||!wt)return native_3x3_defer(g);

    /* Narrowly match the simulator-proved CB/u8, Xwb, Fi geometry. In
     * particular W and Fi sizes use maximum dimensions: CO2 has a physical
     * CO32 write footprint and a full max-shape 3x3 weight tensor. */
    if (!(x->flags&QDR_BLOCKED)||(x->flags&QDR_DIRECT)||!x->crouton||
        !(y->flags&QDR_BLOCKED)||(y->flags&QDR_DIRECT)||!y->crouton||
        !is_class(x,"CB","cb")||!is_class(y,"CB","cb")||
        x->dtype!=DT_U8||y->dtype!=DT_U8||x->esize!=1||y->esize!=1||
        !(wt->flags&QDR_DIRECT)||(wt->flags&QDR_BLOCKED)||
        wt->dtype!=DT_I8||wt->esize!=1||!is_class(wt,"Xwb","xwb")||
        !(fi->flags&QDR_DIRECT)||(fi->flags&QDR_BLOCKED)||
        fi->dtype!=DT_I32||fi->esize!=4||!is_class(fi,"Fi","fi"))
        return native_3x3_defer(g);

    if(x->d[0]!=1||y->d[0]!=1||x->m[0]!=1||y->m[0]!=1||
       wt->d[0]!=3||wt->d[1]!=3||wt->m[0]!=3||wt->m[1]!=3||
       wt->d[2]<=0||wt->d[2]!=wt->m[2]||wt->d[3]<=0||
       wt->m[2]<=0||wt->m[3]<=0||wt->d[3]>wt->m[3]||
       wt->m[2]>256||wt->m[3]>128||
       (wt->m[2]&31)||(wt->m[3]&31)||
       x->d[3]!=wt->m[2]||x->m[3]!=wt->m[2]||x->p[3]!=0||
       y->m[3]!=wt->m[3]||y->d[3]<=0||y->d[3]>y->m[3]||y->p[3]!=0||
       (int64_t)x->d[1]!=(int64_t)y->d[1]+2||
       (int64_t)x->d[2]!=(int64_t)y->d[2]+2||x->zp!=0||
       x->d[1]!=10||y->d[1]!=8||x->m[1]!=24||y->m[1]!=8||
       x->p[1]!=7||x->p[2]!=7||x->p[0]||x->p[3]||
       y->p[0]||y->p[1]||y->p[2]||y->p[3]||
       x->d[1]<=0||x->d[2]<=0||y->d[1]<=0||y->d[2]<=0||
       (x->m[1]&7)||(x->m[2]&7)||(y->m[1]&7)||(y->m[2]&7)||
       x->m[2]>272||y->m[2]>256||(y->d[2]&7)||
       x->d[1]>x->m[1]||x->d[2]>x->m[2]||
       y->d[1]>y->m[1]||y->d[2]>y->m[2]||
       (int64_t)x->d[1]+x->p[1]+2>x->m[1]||
       (int64_t)x->d[2]+x->p[2]+2>x->m[2])
        return native_3x3_defer(g);

    if(y->d[3]!=wt->m[3] &&
       !(wt->m[2]==32&&wt->m[3]==32&&y->d[3]==2&&
         x->d[1]==10&&x->d[2]==258&&x->m[2]==272&&
         y->d[1]==8&&y->d[2]==256&&y->m[1]==8&&y->m[2]==256))
        return native_3x3_defer(g);

    /* Fi stores one 64-word row per output-channel block: 32 multipliers
     * followed by 32 biases. Shapes are [1, CO/32, 1, 64], not a flat
     * [1,1,1,2*CO] vector (both have the same byte count). */
    if(fi->m[0]!=1||fi->m[1]!=wt->m[3]/32||fi->m[2]!=1||fi->m[3]!=64||
       fi->d[0]!=fi->m[0]||fi->d[1]!=fi->m[1]||
       fi->d[2]!=fi->m[2]||fi->d[3]!=fi->m[3])
        return native_3x3_defer(g);
    uint64_t wbytes64=(uint64_t)9u*(uint32_t)wt->m[2]*(uint32_t)wt->m[3];
    uint64_t fibytes64=(uint64_t)(uint32_t)fi->m[1]*(uint32_t)fi->m[3]*fi->esize;
    if(wbytes64>UINT32_MAX||
       fibytes64!=(uint64_t)(uint32_t)wt->m[3]*8u||fibytes64>UINT32_MAX||
       (fi->offset&127u)||!native_3x3_pool_span(g,wt,(uint32_t)wbytes64)||
       !native_3x3_pool_span(g,fi,(uint32_t)fibytes64)||
       !x->blocks||!y->blocks||!x->nblocks||!y->nblocks)
        return native_3x3_defer(g);

    /* Once the command is inside the proven shape family, lease and global
     * HMX ownership failures are engine errors, not a reason to run fallback. */
    if(!g->use_hmx||!native_1x1_lease_valid()||!native_1x1_hmx_owned()||
       !native_1x1_workers_idle()) {
        g->error_detail=QD_N3X3_NATIVE_FAILURE|3u;
        g->dsp_native_3x3_diag[5]++;
        return QDG_ERR_DSP;
    }

    QdN3x3Request request;memset(&request,0,sizeof request);
    uint32_t pool_bytes[7];
    for(uint32_t p=0;p<7;p++) {
        if(p>=QDG_POOLS||g->pool_bytes[p]>UINT32_MAX)return native_3x3_defer(g);
        pool_bytes[p]=(uint32_t)g->pool_bytes[p];request.pool[p]=g->pool[p];
    }
    request.pool_count=7;request.pool_bytes=pool_bytes;
    for(uint32_t d=0;d<4;d++) {
        request.input.dims[d]=(uint32_t)x->d[d];
        request.input.max_dims[d]=(uint32_t)x->m[d];
        request.input.padding[d]=(uint32_t)x->p[d];
        request.output.dims[d]=(uint32_t)y->d[d];
        request.output.max_dims[d]=(uint32_t)y->m[d];
        request.output.padding[d]=(uint32_t)y->p[d];
    }
    request.input.slice_offset=x->slice;request.input.block_count=x->nblocks;
    request.input.blocks=x->blocks;
    request.output.slice_offset=y->slice;request.output.block_count=y->nblocks;
    request.output.blocks=y->blocks;
    request.weights=g->pool[wt->pool]+wt->offset;
    request.weights_bytes=(uint32_t)wbytes64;
    request.weight_ci=(uint32_t)wt->m[2];request.weight_co=(uint32_t)wt->m[3];
    request.weight_kernel_h=3;request.weight_kernel_w=3;
    request.input_dtype=QD_N3X3_DTYPE_U8;request.output_dtype=QD_N3X3_DTYPE_U8;
    request.weight_dtype=QD_N3X3_DTYPE_I8;request.fi_dtype=QD_N3X3_DTYPE_I32;
    request.weight_format=QD_N3X3_WEIGHT_XWB;
    request.fi=(const int32_t *)(g->pool[fi->pool]+fi->offset);
    request.fi_bytes=(uint32_t)fibytes64;request.input_zero_point=(uint32_t)x->zp;
    request.output_zero_point=(uint32_t)y->zp;
    request.workspace=(void *)(QDVT_BASE+QD_N3X3_WORKSPACE_OFFSET);
    request.workspace_bytes=QD_N3X3_WORKSPACE_MAX;

    if(((uintptr_t)request.weights&127u)||((uintptr_t)request.fi&127u))
        return native_3x3_defer(g);
    dsp_clean_range(request.weights,request.weights_bytes);
    dsp_clean_range(request.fi,request.fi_bytes);

    QdN3x3Prepared prepared;
    uint64_t started=now_ns();
    int status=qd_native_3x3_prepare(&request,&prepared);
    g->dsp_native_3x3_diag[1]+=now_ns()-started;
    if(status)return native_3x3_defer(g);
    g->dsp_native_3x3_diag[0]++;
    g->dsp_native_3x3_diag[6]+=prepared.workspace_used;

    if(!native_1x1_workers_idle()||!native_1x1_lease_valid()||
       !native_1x1_hmx_owned()) {
        g->error_detail=QD_N3X3_NATIVE_FAILURE;
        g->dsp_native_3x3_diag[5]++;
        return QDG_ERR_DSP;
    }

    g->resident_key=0;qdrc_invalidate_native();
    __asm__ volatile("barrier\n mxclracc":::"memory");
    started=now_ns();status=qd_native_3x3_issue(&prepared);
    g->dsp_native_3x3_diag[2]+=now_ns()-started;
    if(status) {
        g->resident_key=0;qdrc_invalidate_native();
        g->error_detail=QD_N3X3_NATIVE_FAILURE|((uint32_t)status<<8);
        g->dsp_native_3x3_diag[5]++;return QDG_ERR_DSP;
    }

    started=now_ns();status=qd_native_3x3_drain(&prepared);
    if(status) {
        g->resident_key=0;qdrc_invalidate_native();
        g->error_detail=QD_N3X3_NATIVE_FAILURE|((uint32_t)status<<8);
        g->dsp_native_3x3_diag[3]+=now_ns()-started;
        g->dsp_native_3x3_diag[5]++;return QDG_ERR_DSP;
    }
    if(!native_1x1_workers_idle()||!native_1x1_lease_valid()||
       !native_1x1_hmx_owned()) {
        g->resident_key=0;qdrc_invalidate_native();
        g->error_detail=QD_N3X3_NATIVE_FAILURE|2u;
        g->dsp_native_3x3_diag[3]+=now_ns()-started;
        g->dsp_native_3x3_diag[5]++;return QDG_ERR_DSP;
    }
    status=qd_native_3x3_finish(&prepared);
    g->dsp_native_3x3_diag[3]+=now_ns()-started;
    g->resident_key=0;qdrc_invalidate_native();
    if(status) {
        g->error_detail=QD_N3X3_NATIVE_FAILURE|((uint32_t)status<<8);
        g->dsp_native_3x3_diag[5]++;return QDG_ERR_DSP;
    }
    return 0;
}

#undef QD_N3X3_WORKSPACE_OFFSET
#undef QD_N3X3_NATIVE_FAILURE

#endif
