#ifndef QD_GRAPH_PREPARED_H
#define QD_GRAPH_PREPARED_H
#include "qd_prepared_config.h"
#include "native_program/prepared_runner.h"
#include "native_program/prepared_native.h"
#include "native_program/prepared_io.h"
#if QDS_PREPARED_PREPROCESS
#include "native_program/prepared_preprocess.h"
#endif
#if QDS_PREPARED_ELEMENTWISE
#include "native_program/prepared_elementwise.h"
#endif
#if QDS_PREPARED_REFERENCE
#include "native_program/prepared_reference.h"
#if QDS_PREPARED_COMPACT_3X3
#error QDS_PREPARED_REFERENCE registers no native conv and owns the VTCM tail after its control area
#endif
/* Exact full-model mode: no native HMX conv. Convolution runs the fast exact
 * kernel with its workspace in the VTCM tail after the 8 KiB control area and
 * its scalar-accessed state in the L2-TCM slot the elementwise handlers would
 * otherwise use; every other key without a specialized handler runs the
 * reference kernel. Reference scratch peak is 983,168 bytes over both models.
 * Profiles: per-variant cycles in g->stats, conv phases (staging, MAC,
 * requant, output write) in g->dsp_conv_phase[0..3]. */
#define QDS_REFERENCE_SCRATCH_BYTES 0x100000u
#define QDS_REFERENCE_NEAR 0x30078000u
#define QDS_REFERENCE_NEAR_BYTES 0x8000u
#define QDS_REFERENCE_CHAINS 64u
/* With QDS_PREPARED_ELEMENTWISE the HVX add/requant/linearclip handlers (13,
 * 70, 68) get executor scratch region NPRG2_EXEC_SCRATCH_NEAR, an L2-TCM slot
 * of their own, never the L1-cached heap scratch above (region 0) and never
 * the conv near slot at 0x30078000. 0x300f8000..0x30100000 is the top half
 * of the work slot: QdsWork uses 116 bytes at 0x300f0000, the hot program
 * ends at 0x300ef000 and the worker stacks at 0x300b0000. Demand is 30,748
 * bytes on Hexagon. They stay chained to the reference kernel, so a command
 * their validator rejects still runs the reference kernel in region 0. */
#define QDS_ELEMENTWISE_L2 0x300f8000u
#define QDS_ELEMENTWISE_L2_BYTES 0x8000u
static const volatile char prepared_reference_feature[]="QD_PREPARED_REF_1";
#endif

#if QDS_NATIVE_PADDED_1X1
static const volatile char prepared_padded_feature[]="QD_PREPARED_W1_1";
#endif
#if QDS_PREPARED_COMPACT_3X3
static const volatile char prepared_compact3_feature[]="QD_PREPARED_A3_1";
#endif

typedef struct {
    Nprg2PreparedRunner runner;
    Nprg2Executor executor;
    Nprg2NativeConfig native;
    Nprg2ExecBuffer inputs[4],outputs[2];
    Nprg2ExecStats stats;
#if QDS_PREPARED_REFERENCE
    Nprg2ReferenceConv conv;
    Nprg2ReferenceChain chains[QDS_REFERENCE_CHAINS];
    uint8_t *scratch;
#endif
} QdPreparedState;

#if QDS_PREPARED_REFERENCE
/* Chains the reference kernel under every specialized schema key, then
 * registers it for every key still free. */
static int prepared_register_reference(QdPreparedState *s) {
    Nprg2Executor *e=&s->executor;
    uint32_t chained=0;
    for(uint32_t i=0;i<NPRG2_EXEC_REGISTRY_CAPACITY;i++) {
        Nprg2ExecHandler *h=&e->handlers[i];
        if(!h->used || h->opcode==QDR_NOP || h->validate==nprg2_exec_reference_validate)continue;
        if(h->variant>QDR_VARIANTS || qdr_variant_ops[h->variant]!=h->opcode)continue;
        if(chained==QDS_REFERENCE_CHAINS ||
           nprg2_executor_chain_reference(e,h->opcode,h->variant,&s->chains[chained]))return 1;
        chained++;
    }
    return nprg2_executor_register_reference_missing(e,NULL)!=NPRG2_EXEC_OK;
}
#endif

/* Default prepared mode enables only direct 1x1. The separate compact-3x3
 * opt-in stages A within the bounded reserved tail; W/Fi/Y remain direct. */
static int prepared_submit(void *ctx,const Nprg2ExecCall *call,
                           const QdDirectBinding *b,uint8_t family) {
    QdgGraph *g=ctx;
    int family_ok=family==1;
#if QDS_PREPARED_COMPACT_3X3
    family_ok|=family==3 && graph_direct_compact_fits(b);
#endif
    if(!family_ok || !native_1x1_workers_idle() || !native_1x1_lease_valid() ||
       !native_1x1_hmx_owned())return 0;
    volatile uint32_t *trace=(volatile uint32_t *)(SERVICE+1024u);
    trace[1]=3;trace[3]=call->command_index;trace[4]=family;
    uint64_t *diag=family==1?g->dsp_native_1x1_diag:g->dsp_native_3x3_diag,t=now_ns();
#if QDS_PREPARED_COMPACT_3X3
    trace[15]=(uint32_t)prepared_compact3_feature[0];
    if(family==3){graph_direct_compact_input(b);diag[6]+=(uint64_t)b->in_pages*2048u;}
#endif
    diag[6]+=b->staged_weight_bytes;
#if QDS_NATIVE_PADDED_1X1
    if(b->staged_weight_bytes)trace[13]=(uint32_t)prepared_padded_feature[0];
#endif
    dsp_clean_range(b->in_table,QD_DIRECT_TABLE_BYTES);
    dsp_clean_range(b->in_desc,QD_DIRECT_ABI_BYTES);
    dsp_clean_range((void *)(QD_DIRECT_VTCM_BASE+QD_DIRECT_TAIL_OFFSET),b->tail_publish_bytes);
    uint32_t saved_ssr;
    __asm__ volatile("%0=ssr":"=r"(saved_ssr));
    uint32_t hmx_ssr=saved_ssr|(1u<<26);
    __asm__ volatile("ssr=%0\n isync\n barrier"::"r"(hmx_ssr):"memory");
    trace[5]=saved_ssr;trace[1]=4;diag[0]++;diag[1]+=now_ns()-t;
    *b->done=UINT32_MAX;
    __asm__ volatile("mxclracc\n barrier":::"memory");t=now_ns();
    if(family==1)qd_qnn_hmx_1x1(b->out_desc,b->in_desc,b->weights,b->fi,b->cfg,b->mode);
#if QDS_PREPARED_COMPACT_3X3
    else qd_qnn_hmx_stride1_generic(b->out_desc,b->in_desc,b->weights,b->fi,b->cfg,b->mode);
#endif
    trace[1]=5;diag[2]+=now_ns()-t;t=now_ns();
    qd_hmx_finish(b->zero_fi,(void *)b->done);
    uint64_t deadline=now_ns()+100000000ull;
    while(*b->done && now_ns()<=deadline)__asm__ volatile("nop");
    if(*b->done){diag[5]++;return 0;}
    __asm__ volatile("ssr=%0\n isync\n barrier"::"r"(saved_ssr):"memory");
    trace[1]=6;
    int okay=graph_direct_guards(b) && native_1x1_workers_idle() && native_1x1_lease_valid();
#if QDS_PREPARED_COMPACT_3X3
    if(family==3)okay &= graph_direct_compact_guard(b);
#endif
    diag[3]+=now_ns()-t;if(!okay)diag[5]++;
    return okay;
}

static int prepared_cancel(void *ctx,uint32_t index) {
    (void)ctx;(void)index;return resident_cancelled();
}

static QdPreparedState *prepared_open(QdgGraph *g,const QdsConfig *cfg,
                                     const void *const *inputs,const uint64_t *input_bytes,
                                     void *const *outputs,const uint64_t *output_bytes) {
    const QdsPreparedConfig *p=(const QdsPreparedConfig *)(SERVICE+QDS_CONFIG+sizeof *cfg);
    if(p->magic!=QDS_PREPARED_MAGIC || p->version!=1 || p->offset<0x100000u ||
       (p->offset&127u) || p->offset>=0x1000000u || p->bytes<256u ||
       p->bytes>0x1000000u-p->offset ||
       (p->offset<cfg->image+cfg->image_bytes && cfg->image<p->offset+p->bytes) ||
       (p->offset<cfg->pool[5]+cfg->pool_bytes[5] && cfg->pool[5]<p->offset+p->bytes) ||
       !native_1x1_lease_valid() || !native_1x1_hmx_owned() || !native_1x1_workers_idle())return NULL;
    QdPreparedState *s=calloc(1,sizeof *s);if(!s)return NULL;
    Nprg2Pool pools[7];
    for(uint32_t k=0;k<7;k++) {
        if(g->pool_bytes[k]>UINT32_MAX){free(s);return NULL;}
        pools[k]=(Nprg2Pool){g->pool[k],(uint32_t)g->pool_bytes[k]};
    }
    pools[1].base=(uint8_t *)QD_DIRECT_VTCM_BASE;
    uint8_t source_sha[32];
    nprg2_sha256(g->image,g->img.bytes,source_sha);
    Nprg2ExpectedHashes hashes={
        .context_sha256=(const uint8_t *)g->image+32,
        .constants_sha256=(const uint8_t *)g->image+64,
        .source_sha256=source_sha,.prepared_sha256=(const uint8_t *)p->sha256};
    int e=nprg2_runner_open(&s->runner,(const void *)(QDO_BASE+p->offset),p->bytes,
        &hashes,pools,(Nprg2Allocator){0},(void *)0x30002000u,0x76000u,
        (void *)0x300b0000u,0x3f000u,0x3f000u,1);
    if(e){free(s);return NULL;}
    nprg2_executor_init(&s->executor);
    s->native=(Nprg2NativeConfig){.active_vtcm_bytes=QD_DIRECT_ACTIVE_BYTES,
        .tables=(uint32_t *)0x30000000u,.table_bytes=QD_DIRECT_TABLE_BYTES,
        .abi=(uint32_t *)0x30000800u,.abi_bytes=QD_DIRECT_ABI_BYTES,
        .tail=(void *)(QD_DIRECT_VTCM_BASE+QD_DIRECT_TAIL_OFFSET),
        .tail_bytes=QD_DIRECT_TAIL_BYTES,.submit_and_drain=prepared_submit,.submit_ctx=g,.family_mask=1};
#if QDS_PREPARED_COMPACT_3X3
    s->native.family_mask=NPRG2_NATIVE_ALLOW_1X1|NPRG2_NATIVE_ALLOW_3X3;
    s->native.max_3x3_input_pages=(QD_DIRECT_TAIL_BYTES-0x2000u-128u)/2048u;
#endif
#if QDS_PREPARED_REFERENCE
    s->conv=(Nprg2ReferenceConv){(uint8_t *)(QD_DIRECT_VTCM_BASE+QD_DIRECT_TAIL_OFFSET+0x2000u),
                                 QD_DIRECT_TAIL_BYTES-0x2000u,
                                 (uint8_t *)QDS_REFERENCE_NEAR,QDS_REFERENCE_NEAR_BYTES,
                                 now_ns,g->dsp_conv_phase};
    s->scratch=malloc(QDS_REFERENCE_SCRATCH_BYTES);
    if(!s->scratch || nprg2_executor_register_reference_conv(&s->executor,&s->conv,NULL) ||
       nprg2_executor_register_weights(&s->executor) ||
       nprg2_executor_register_core_safe_ops(&s->executor)) {
        free(s->scratch);nprg2_runner_close(&s->runner);free(s);return NULL;
    }
#else
    if(nprg2_executor_register_weights(&s->executor) ||
       nprg2_executor_register_core_safe_ops(&s->executor) ||
       nprg2_executor_register_native_conv(&s->executor,&s->native)) {
        nprg2_runner_close(&s->runner);free(s);return NULL;
    }
#endif
#if QDS_PREPARED_ELEMENTWISE && QDS_PREPARED_REFERENCE
    if(NPRG2_ELEMENTWISE_SCRATCH_BYTES>QDS_ELEMENTWISE_L2_BYTES ||
       nprg2_runner_set_scratch_region(&s->runner,NPRG2_EXEC_SCRATCH_NEAR,
                                       (void *)QDS_ELEMENTWISE_L2,QDS_ELEMENTWISE_L2_BYTES) ||
       nprg2_executor_register_elementwise_region(&s->executor,NPRG2_EXEC_SCRATCH_NEAR)) {
        free(s->scratch);nprg2_runner_close(&s->runner);free(s);return NULL;
    }
#elif QDS_PREPARED_ELEMENTWISE
    if(NPRG2_ELEMENTWISE_SCRATCH_BYTES>0x8000u ||
       nprg2_executor_register_elementwise(&s->executor)) {
        nprg2_runner_close(&s->runner);free(s);return NULL;
    }
#endif
#if QDS_PREPARED_PREPROCESS
    if(nprg2_executor_register_preprocess(&s->executor)) {
        nprg2_runner_close(&s->runner);free(s);return NULL;
    }
#endif
#if QDS_PREPARED_REFERENCE
    if(prepared_register_reference(s)) {
        free(s->scratch);nprg2_runner_close(&s->runner);free(s);return NULL;
    }
#endif
    for(uint32_t k=0;k<4;k++)s->inputs[k]=(Nprg2ExecBuffer){(uint8_t *)inputs[k],(uint32_t)input_bytes[k]};
    for(uint32_t k=0;k<2;k++)s->outputs[k]=(Nprg2ExecBuffer){outputs[k],(uint32_t)output_bytes[k]};
    return s;
}

static int prepared_run(QdgGraph *g,QdPreparedState *s,uint32_t first,uint32_t last) {
    volatile uint32_t *trace=(volatile uint32_t *)(SERVICE+1024u);
    for(uint32_t q=0;q<16;q++)trace[q]=0;
    trace[0]=0x31525051u;trace[2]=((volatile uint32_t *)SERVICE)[2];g->error_detail=0;
    if(!s || first>=last || last>g->ncommands || !native_1x1_lease_valid() ||
       !native_1x1_hmx_owned() || !native_1x1_workers_idle())return QDG_ERR_UNSUPPORTED;
    trace[6]=s->runner.l2_hot_bytes;trace[7]=s->runner.l2_expanded_bytes;
    trace[10]=(uint32_t)(uintptr_t)s->runner.l2_hot;
    trace[11]=(uint32_t)(uintptr_t)s->runner.l2_expanded_blocks;
    void *scratch=NULL;uint32_t scratch_bytes=0;
#if QDS_PREPARED_ELEMENTWISE && !QDS_PREPARED_REFERENCE
    scratch=(void *)0x30078000u;scratch_bytes=0x8000u;
#endif
#if QDS_PREPARED_REFERENCE
    /* Region 0 only; the elementwise L2 region is set on the runner. */
    scratch=s->scratch;scratch_bytes=QDS_REFERENCE_SCRATCH_BYTES;
    trace[15]=(uint32_t)prepared_reference_feature[0];
#endif
    s->stats.clock=now_ns;s->stats.variant_cycles=g->stats.variant_ns;s->stats.variant_count=g->stats.variant_count;
    int e=nprg2_runner_preflight(&s->runner,&s->executor,first,last-first,QD_DIRECT_ACTIVE_BYTES,
        s->inputs,4,s->outputs,2,g,scratch,scratch_bytes,&s->stats);
    if(e)goto failed;
    trace[1]=1;
    uint64_t t=now_ns();
    dsp_clean_range(g->pool[1],QD_DIRECT_ACTIVE_BYTES);
    dsp_packet_copy((void *)QD_DIRECT_VTCM_BASE,g->pool[1],QD_DIRECT_ACTIVE_BYTES);
    g->dsp_read_cycles+=now_ns()-t;trace[1]=2;
    qdrc_invalidate_native();g->resident_key=0;
    e=nprg2_runner_execute(&s->runner,&s->executor,first,last-first,QD_DIRECT_ACTIVE_BYTES,
        s->inputs,4,s->outputs,2,g,scratch,scratch_bytes,prepared_cancel,NULL,&s->stats);
    qdrc_invalidate_native();g->resident_key=0;
    if(e)goto failed;
    t=now_ns();dsp_packet_copy(g->pool[1],(void *)QD_DIRECT_VTCM_BASE,QD_DIRECT_ACTIVE_BYTES);
    g->dsp_write_cycles+=now_ns()-t;trace[1]=7;
    trace[8]=s->stats.executed_commands;trace[9]=s->stats.validated_commands;return 0;
failed:
    trace[8]=s->stats.executed_commands;trace[9]=s->stats.validated_commands;
    g->stats.error_command=s->stats.failed_command;
    g->error_detail=0x4e505200u|(uint32_t)e;trace[14]=(uint32_t)e;
    return e==NPRG2_EXEC_UNSUPPORTED || e==NPRG2_EXEC_REJECTED?QDG_ERR_UNSUPPORTED:QDG_ERR_DSP;
}
#endif
