#ifndef QD_GRAPH_PREPARED_H
#define QD_GRAPH_PREPARED_H
#include "qd_prepared_config.h"
#include "native_program/prepared_runner.h"
#include "native_program/prepared_native.h"
#include "native_program/prepared_io.h"
#ifndef QDS_L2_CACHE
#define QDS_L2_CACHE 0
#endif
#if QDS_L2_CACHE
#include "qd_l2_cache.h"
static const volatile char qds_l2_engine_feature[] __attribute__((used))="QDL2ENGN03";
#endif
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
#if QDS_L2_CACHE==1
/* Lower L2 half stays TCM through the lease: conv near as without L2. */
#define QDS_REFERENCE_NEAR 0x30078000u
#define QDS_ELEMENTWISE_L2 QDS_L2_ELEMENTWISE
#elif QDS_L2_CACHE
/* Scalar-heavy conv state: C7 DDR hits L1/L2; VTCM scalar loads are slow. */
#define QDS_REFERENCE_NEAR QDS_L2_NEAR
#define QDS_ELEMENTWISE_L2 QDS_L2_ELEMENTWISE
#else
#define QDS_REFERENCE_NEAR 0x30078000u
#define QDS_ELEMENTWISE_L2 0x300f8000u
#endif
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
#define QDS_ELEMENTWISE_L2_BYTES 0x8000u
static const volatile char prepared_reference_feature[]="QD_PREPARED_REF_1";
#endif

/* QDS_PREPARED_THREADS=3 (default 1): the reference build's conv rows and the
 * large HVX elementwise / data-movement handlers run on the three HVX hardware
 * threads (the master plus the two followers parked in follower()). 1 keeps the
 * single-thread behaviour exactly (no hook, no helper memory is touched).
 *
 * Helper conv near slots, two of QDS_PAR_L2_BYTES each, in the work slot
 * 0x300f0000..0x300f8000 whose only occupant is QdsWork (bytes 0..0x200, see
 * the _Static_assert in qd_graph_dsp.h):
 *   0x300f0200..0x300f4000   part 1
 *   0x300f4000..0x300f7e00   part 2 (0x300f7e00..0x300f8000 unused)
 * Nothing else is placed there during a prepared call: the expanded block
 * table ends below 0x30078000, conv near is 0x30078000..0x30080000, the stacks
 * are 0x30080000..0x300b0000, the hot program 0x300b0000..0x300ef000 and the
 * elementwise region starts at 0x300f8000. The legacy dsp_* caches that used
 * 0x300b0000.. are not used in prepared mode. */
#ifndef QDS_PREPARED_THREADS
#define QDS_PREPARED_THREADS 1
#endif
#if QDS_L2_CACHE
/* Inside the lease no L2-TCM address is usable: page 7 (qd_l2_cache.h). */
#define QDS_PAR_L2_0 QDS_L2_PAR_0
#define QDS_PAR_L2_1 QDS_L2_PAR_1
#else
#define QDS_PAR_L2_0 0x300f0200u
#define QDS_PAR_L2_1 0x300f4000u
#endif
#define QDS_PAR_L2_BYTES 0x3e00u

#if QDS_NATIVE_PADDED_1X1
static const volatile char prepared_padded_feature[]="QD_PREPARED_W1_1";
#endif
#if QDS_PREPARED_COMPACT_3X3
static const volatile char prepared_compact3_feature[]="QD_PREPARED_A3_1";
#endif

/* Per-handler hardware profile (QDS_PREPARED_PROFILE, default on): the cost
 * is two pcycle reads per command in preflight and in run. */
#ifndef QDS_PREPARED_PROFILE
#define QDS_PREPARED_PROFILE 1
#endif

/* Conv epilogue fusion (QDS_PREPARED_FUSE, default off): a schedule pass at
 * open marks conv -> REQUANT_U8/ADD_U8 pairs, the fast conv then writes the
 * epilogue's output from registers (prepared_fuse.h, EXECUTOR.md). trace[12]
 * reports the pair count. */
#ifndef QDS_PREPARED_HMX
#define QDS_PREPARED_HMX 0
#endif
#if QDS_PREPARED_HMX
/* Exact HMX for eligible u8 convs (native_program/prepared_hmx_conv.inc).
 * HMX is powered and clocked for the whole prepared run (prepared_run
 * requires native_1x1_hmx_owned); the conv sets SSR.XE2 on worker 0 only
 * around its HMX work, as qd_graph_native_direct.h does, and waits for the
 * completion tile before leaving. */
void qd_hmx_wide(void *,void *,void *,void *,uint32_t);
void qd_hmx_finish(void *,void *);
void qd_hmx_wide_n(void *,void *,void *,void *,uint32_t,uint32_t);
void qd_hmx_native_n(void *,void *,void *,void *const *,uint32_t,uint32_t);
void qd_hmx_native16_n(void *,void *,void *,void *const *,uint32_t,uint32_t);
void qd_hmx_slide3_n(const uint32_t *,const void *,const void *,const uint32_t *,uint32_t,uint32_t);
static uint32_t prepared_hmx_enter(void) {
    uint32_t saved;__asm__ volatile("%0=ssr":"=r"(saved));
    uint32_t on=saved|(1u<<26);__asm__ volatile("ssr=%0\n isync\n barrier"::"r"(on):"memory");
    return saved;
}
static void prepared_hmx_leave(uint32_t saved) {__asm__ volatile("ssr=%0\n isync\n barrier"::"r"(saved):"memory");}
static uint64_t now_ns(void);
static uint64_t prepared_hmx_clock(void) {return now_ns();}
static const Nprg2HmxOps prepared_hmx_ops={prepared_hmx_enter,prepared_hmx_leave,qd_hmx_wide,qd_hmx_finish,prepared_hmx_clock,
    qd_hmx_wide_n,qd_hmx_native_n,qd_hmx_native16_n,
#if !defined(NO_SLIDE3)
    qd_hmx_slide3_n
#else
    NULL
#endif
};
#endif
#ifndef QDS_PREPARED_FUSE
#define QDS_PREPARED_FUSE 0
#endif

typedef struct {
    Nprg2PreparedRunner runner;
    Nprg2Executor executor;
    Nprg2NativeConfig native;
    Nprg2ExecBuffer inputs[4],outputs[2];
    Nprg2ExecStats stats;
    Nprg2Parallel par;
#if QDS_PREPARED_PROFILE
    Nprg2ExecProfile profile;
    uint64_t handler_run_cycles[NPRG2_EXEC_REGISTRY_CAPACITY],handler_run_count[NPRG2_EXEC_REGISTRY_CAPACITY],
             handler_validate_cycles[NPRG2_EXEC_REGISTRY_CAPACITY];
#endif
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

#if QDS_PREPARED_THREADS>1
/* Hook that runs parts on the parked followers (op 4 in follower()). A worker
 * that does not come back within the join deadline is not recoverable (it may
 * still be running our code): par_fault stops further jobs and cancels the
 * run, and the arena record shows started != finished for that worker. */
static volatile uint32_t prepared_par_fault;
/* The lease is validated once per run (prepared_run, below) and nothing in a
 * run changes it; re-reading it per job meant ~10 uncached arena loads and two
 * TLB reads for every conv and three per HMX batch. */
static uint32_t prepared_par_lease_ok;
static int prepared_par_ready(void *impl) {
    (void)impl;
    return !prepared_par_fault && prepared_par_lease_ok && native_1x1_workers_idle();
}
static int prepared_par_run(void *impl,uint32_t parts,Nprg2ParallelFn fn,void *ctx) {
    if(parts<2 || parts>3 || !prepared_par_ready(impl))return 0;
    work->op=4;work->job_fn=fn;work->job_ctx=ctx;work->job_parts=parts;work->error=0;
    uint32_t gen=work->generation+1;
    __asm__ volatile("barrier":::"memory");work->generation=gen;
    fn(ctx,0,parts);
    uint64_t end=now_ns()+400000000ull;
    while(work->done[1]!=gen || work->done[2]!=gen)
        if(now_ns()>end){prepared_par_fault=1;return 1;}
    __asm__ volatile("barrier":::"memory");
    return 1;
}
#endif

static int prepared_cancel(void *ctx,uint32_t index) {
    (void)ctx;
#if QDS_PREPARED_THREADS>1
    if(prepared_par_fault)return 1;
#endif
    /* The host's cancel words are two uncached arena loads: every 32nd command. */
    return !(index&31u) && resident_cancelled();
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
        &hashes,pools,(Nprg2Allocator){0},
#if QDS_L2_CACHE
        (void *)QDS_L2_BLOCKS,0x76000u,(void *)QDS_L2_HOT,QDS_L2_HOT_BYTES,
#else
        (void *)0x30002000u,0x76000u,(void *)0x300b0000u,0x3f000u,
#endif
        0x3f000u,1);
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
                                 now_ns,g->dsp_conv_phase,{NULL,NULL},0,NULL};
#if QDS_PREPARED_HMX
    s->conv.hmx=&prepared_hmx_ops;
#endif
#if QDS_PREPARED_THREADS>1
    s->conv.par_near[0]=(uint8_t *)QDS_PAR_L2_0;s->conv.par_near[1]=(uint8_t *)QDS_PAR_L2_1;
    s->conv.par_near_bytes=QDS_PAR_L2_BYTES;
    s->par=(Nprg2Parallel){.max_parts=3,.ready=prepared_par_ready,.run=prepared_par_run};
    nprg2_runner_set_parallel(&s->runner,&s->par);
#endif
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
#if QDS_PREPARED_FUSE
    /* A schedule the analysis cannot model simply runs unfused. */
    (void)nprg2_runner_build_fuse(&s->runner,(Nprg2Allocator){0});
#endif
    for(uint32_t k=0;k<4;k++)s->inputs[k]=(Nprg2ExecBuffer){(uint8_t *)inputs[k],(uint32_t)input_bytes[k]};
    for(uint32_t k=0;k<2;k++)s->outputs[k]=(Nprg2ExecBuffer){outputs[k],(uint32_t)output_bytes[k]};
    return s;
}

/* VTCM in, execute, VTCM out. In L2 mode this runs on the C7 DDR stack
 * with L2 as cache, so it must not touch any 0x30xxxxxx address. */
/* QDS_POOL1_NOCOPY: pool 1 holds only activations and per-run weight copies
 * (weights_to_vtcm); every byte the schedule reads from it is written earlier
 * in the same run, so its starting contents don't matter (host check: the
 * primary schedule matches with pool 1 pre-filled with 0, 0x37, 0xff and
 * random bytes). The run then works on VTCM as it finds it, with no DDR
 * copy in or out; the DDR pool 1 image is no longer a mirror. */
#ifndef QDS_POOL1_NOCOPY
#define QDS_POOL1_NOCOPY 0
#endif
static int prepared_active(QdgGraph *g,QdPreparedState *s,uint32_t first,uint32_t last,
                           void *scratch,uint32_t scratch_bytes) {
    volatile uint32_t *trace=(volatile uint32_t *)(SERVICE+1024u);
    uint64_t t=now_ns();
#if !QDS_POOL1_NOCOPY
    dsp_clean_range(g->pool[1],QD_DIRECT_ACTIVE_BYTES);
    dsp_packet_copy((void *)QD_DIRECT_VTCM_BASE,g->pool[1],QD_DIRECT_ACTIVE_BYTES);
#endif
    g->dsp_read_cycles+=now_ns()-t;trace[1]=2;
    qdrc_invalidate_native();g->resident_key=0;
    int e=nprg2_runner_execute(&s->runner,&s->executor,first,last-first,QD_DIRECT_ACTIVE_BYTES,
        s->inputs,4,s->outputs,2,g,scratch,scratch_bytes,prepared_cancel,NULL,&s->stats);
    qdrc_invalidate_native();g->resident_key=0;
    if(e)return e;
    t=now_ns();
#if !QDS_POOL1_NOCOPY
    dsp_packet_copy(g->pool[1],(void *)QD_DIRECT_VTCM_BASE,QD_DIRECT_ACTIVE_BYTES);
#endif
    g->dsp_write_cycles+=now_ns()-t;trace[1]=7;
    return 0;
}
#if QDS_L2_CACHE
typedef struct {QdgGraph *g;QdPreparedState *s;uint32_t first,last;void *scratch;uint32_t scratch_bytes;
                int e;uint32_t l2_error;} QdsLeaseCall;
static QdsLeaseCall qds_lease_call;
void qds_on_stack(void (*entry)(uint32_t),uint32_t arg,uint32_t stack_top);
static void __attribute__((noinline)) prepared_leased(uint32_t unused) {
    (void)unused;QdsLeaseCall *c=&qds_lease_call;QdsL2Cache l2;
    volatile uint32_t *trace=(volatile uint32_t *)(SERVICE+1024u);
    uint64_t t=now_ns();
    if(qds_l2_begin(&l2)){c->l2_error=1;if(qds_l2_end(&l2))c->l2_error|=4;return;}
    trace[4]=(uint32_t)(now_ns()-t);
#if QDS_L2_POOL_C7
    if(qds_l2_pool_cache(1)){c->l2_error=16;if(qds_l2_end(&l2))c->l2_error|=4;return;}
#endif
    qds_l2_lease_active=1;
    c->e=prepared_active(c->g,c->s,c->first,c->last,c->scratch,c->scratch_bytes);
    qds_l2_lease_active=0;
    t=now_ns();
#if QDS_L2_POOL_C7
    if(qds_l2_pool_cache(0))c->l2_error|=32;
#endif
    if(qds_l2_end(&l2))c->l2_error|=8;
    trace[5]=(uint32_t)(now_ns()-t);
}
#endif

static int prepared_run(QdgGraph *g,QdPreparedState *s,uint32_t first,uint32_t last) {
    volatile uint32_t *trace=(volatile uint32_t *)(SERVICE+1024u);
    for(uint32_t q=0;q<16;q++)trace[q]=0;
    trace[0]=0x31525051u;trace[2]=((volatile uint32_t *)SERVICE)[2];g->error_detail=0;
#if QDS_L2_CACHE
    trace[12]=(uint32_t)qds_l2_engine_feature[0];
#endif
    if(!s || first>=last || last>g->ncommands || !native_1x1_lease_valid() ||
       !native_1x1_hmx_owned() || !native_1x1_workers_idle())return QDG_ERR_UNSUPPORTED;
#if QDS_PREPARED_THREADS>1
    prepared_par_lease_ok=1;
#endif
    trace[6]=s->runner.l2_hot_bytes;trace[7]=s->runner.l2_expanded_bytes;
    trace[10]=(uint32_t)(uintptr_t)s->runner.l2_hot;
    trace[12]=s->runner.fuse?s->runner.fuse->count:0;
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
#if QDS_PREPARED_PROFILE
    /* Per run: cleared here, exported by the caller after the run. The
     * identical-request preflight is proved, not repeated, in execute. */
    memset(s->handler_run_cycles,0,sizeof s->handler_run_cycles);
    memset(s->handler_run_count,0,sizeof s->handler_run_count);
    memset(s->handler_validate_cycles,0,sizeof s->handler_validate_cycles);
    s->profile=(Nprg2ExecProfile){.clock=now_ns,.variant_cycles=g->stats.variant_ns,.variant_count=g->stats.variant_count,
        .handler_run_cycles=s->handler_run_cycles,.handler_run_count=s->handler_run_count,
        .handler_validate_cycles=s->handler_validate_cycles};
    nprg2_runner_set_profile(&s->runner,&s->profile,1);
#else
    nprg2_runner_set_profile(&s->runner,NULL,1);
#endif
#if QDS_PREPARED_THREADS>1
    memset(s->par.jobs_parallel,0,sizeof s->par.jobs_parallel);memset(s->par.jobs_serial,0,sizeof s->par.jobs_serial);
    memset(s->par.parts_run,0,sizeof s->par.parts_run);s->par.not_ready=s->par.declined=0;
#endif
    int e=nprg2_runner_preflight(&s->runner,&s->executor,first,last-first,QD_DIRECT_ACTIVE_BYTES,
        s->inputs,4,s->outputs,2,g,scratch,scratch_bytes,&s->stats);
    if(e)goto failed;
    trace[1]=1;
#if QDS_L2_CACHE
    /* Metadata was written by scalar code with L2 as TCM; publish it. */
    dsp_clean_range(s->runner.l2_expanded_blocks,s->runner.l2_expanded_bytes);
    dsp_clean_range(s->runner.l2_hot,s->runner.l2_hot_bytes);
    volatile uint32_t *guard=(volatile uint32_t *)QDS_L2_STACK_BASE;
    for(uint32_t i=0;i<32;i++)guard[i]=0x7154c9a3u;
    qds_lease_call=(QdsLeaseCall){g,s,first,last,scratch,scratch_bytes,0,0};
#if QDS_PREPARED_THREADS>1
    /* Followers leave their L2-TCM stacks for page-7 stacks (op 5) before the
     * lease and come back (op 6) after TCM is restored; in between they only
     * poll the uncached QdsWork and run op-4 jobs. */
    for(uint32_t t=1;t<3;t++)for(uint32_t i=0;i<32;i++)((volatile uint32_t *)QDS_L2_FSTACK(t))[i]=0x7154c9a3u;
    if(prepared_par_fault || send_work(5,0)){prepared_par_fault=1;trace[13]=64;e=NPRG2_EXEC_ARGUMENT;goto failed;}
#endif
    qds_on_stack(prepared_leased,0,QDS_L2_STACK_TOP);
    for(uint32_t i=0;i<32;i++)if(guard[i]!=0x7154c9a3u)qds_lease_call.l2_error|=2;
#if QDS_PREPARED_THREADS>1
    if(send_work(6,0)){prepared_par_fault=1;qds_lease_call.l2_error|=64;}
    for(uint32_t t=1;t<3;t++)for(uint32_t i=0;i<32;i++)
        if(((volatile uint32_t *)QDS_L2_FSTACK(t))[i]!=0x7154c9a3u)qds_lease_call.l2_error|=128;
#endif
    e=qds_lease_call.e;
    trace[13]=qds_lease_call.l2_error;
    if(qds_lease_call.l2_error){e=NPRG2_EXEC_ARGUMENT;goto failed;}
#else
    e=prepared_active(g,s,first,last,scratch,scratch_bytes);
#endif
#if QDS_PREPARED_THREADS>1
    prepared_par_lease_ok=0;
#endif
    if(e)goto failed;
    trace[8]=s->stats.executed_commands;trace[9]=s->stats.validated_commands;return 0;
failed:
#if QDS_PREPARED_THREADS>1
    prepared_par_lease_ok=0;
#endif
    trace[8]=s->stats.executed_commands;trace[9]=s->stats.validated_commands;
    g->stats.error_command=s->stats.failed_command;
    g->error_detail=0x4e505200u|(uint32_t)e;trace[14]=(uint32_t)e;
    return e==NPRG2_EXEC_UNSUPPORTED || e==NPRG2_EXEC_REJECTED?QDG_ERR_UNSUPPORTED:QDG_ERR_DSP;
}

#if QDS_PREPARED_PROFILE && QDS_PREPARED_THREADS>1
/* Thread record, 128 bytes at QDS_PROFILE2+4160 (the profile record's end):
 * u64 [0] magic 'QDPT' | 1<<32, [1] max_parts | par_fault<<32, [2] not_ready
 * (workers busy or lease invalid when a handler asked), [3] declined (hook run
 * refused), [4..6] parallel jobs conv/ew/dm, [7..9] jobs run serially in more
 * than one part (never expected on hardware), [10..12] parts executed,
 * [13] started[1] | started[2]<<32, [14] finished[1] | finished[2]<<32
 * (cumulative since boot; started != finished = a worker that never left its
 * part), [15] 0. Per-run counters are cleared by prepared_run. */
static void prepared_export_par(const QdPreparedState *s,uint8_t *dst) {
    uint64_t r[16]={0};
    r[0]=0x54505451ull|1ull<<32;r[1]=(uint64_t)s->par.max_parts|(uint64_t)prepared_par_fault<<32;
    r[2]=s->par.not_ready;r[3]=s->par.declined;
    for(int k=0;k<3;k++){r[4+k]=s->par.jobs_parallel[k];r[7+k]=s->par.jobs_serial[k];r[10+k]=s->par.parts_run[k];}
    r[13]=(uint64_t)work->job_started[1]|(uint64_t)work->job_started[2]<<32;
    r[14]=(uint64_t)work->job_finished[1]|(uint64_t)work->job_finished[2]<<32;
    memcpy(dst,r,sizeof r);
}
#endif
#if QDS_PREPARED_PROFILE
/* Writes the last run's executor profile to the service arena (4160 bytes at
 * QDS_PROFILE2): eight u64 [magic 'QDHP' | version 1<<32, preflight cycles,
 * validator cycles, dispatch cycles, handler run cycles, preflight calls |
 * reused<<32, executed | validated<<32, registry slots], then one 32-byte
 * record per registry slot: u32 key (opcode | variant<<8 | used<<16 |
 * no_effect<<24), u32 0, u64 run count, u64 run cycles, u64 validator cycles.
 * Cycles are pcycles. Dispatch is the run loop outside handler run callbacks
 * (decode, lookup, cancel, scratch selection); preflight includes validators. */
static void prepared_export_profile(const QdPreparedState *s,uint8_t *dst) {
    uint64_t head[8]={0,s->profile.preflight_cycles,s->profile.validate_cycles,
        s->profile.dispatch_cycles,s->profile.run_cycles,
        (uint64_t)s->profile.preflight_calls|(uint64_t)s->profile.preflights_reused<<32,
        (uint64_t)s->stats.executed_commands|(uint64_t)s->stats.validated_commands<<32,NPRG2_EXEC_REGISTRY_CAPACITY};
    head[0]=0x51444850ull|1ull<<32;
    memcpy(dst,head,sizeof head);
    for(uint32_t i=0;i<NPRG2_EXEC_REGISTRY_CAPACITY;i++) {
        const Nprg2ExecHandler *h=&s->executor.handlers[i];
        uint64_t rec[4]={0,s->handler_run_count[i],s->handler_run_cycles[i],s->handler_validate_cycles[i]};
        if(h->used)rec[0]=(uint64_t)h->opcode|(uint64_t)h->variant<<8|1ull<<16|(uint64_t)h->no_effect<<24;
        memcpy(dst+sizeof head+(size_t)i*32u,rec,32);
    }
}
#endif
#endif
