#ifndef QD_GRAPH_PREPARED_H
#define QD_GRAPH_PREPARED_H
#include "qd_prepared_config.h"
#include "qd_graph_dsp.h"
#include "native_program/prepared_runner.h"
#include "native_program/prepared_native.h"
#include "native_program/prepared_io.h"
#ifndef NPRG2_QNN_RUNLISTS
#define NPRG2_QNN_RUNLISTS 0
#endif
#ifndef QDS_QNN_CLAIM
#define QDS_QNN_CLAIM 0
#endif
#ifndef QDS_QNN_MAIN_PARTS
#define QDS_QNN_MAIN_PARTS 2u /* THREADS=4: 1 disables the HT5 conv helper */
#endif
#ifndef QDS_QNN_WORKER_PARTS
#define QDS_QNN_WORKER_PARTS 2u
#endif
#if QDS_PREPARED_DMA_ASYNC
#include "native_program/prepared_async_dma.h"
#if !QDS_PRESERVE_VTCM
#error QDS_PREPARED_DMA_ASYNC requires the preserved VTCM lease
#endif
#if !NPRG2_QNN_RUNLISTS
#error QDS_PREPARED_DMA_ASYNC requires the captured QNN checkpoint run lists
#endif
#endif
#if NPRG2_QNN_RUNLISTS
#include "native_program/qnn_runlist_primary.generated.h"
#endif
#ifndef QDS_L2_CACHE
#define QDS_L2_CACHE 0
#endif
#if QDS_L2_CACHE
#include "qd_l2_cache.h"
static const volatile char qds_l2_engine_feature[] __attribute__((used))="QDL2ENGN03";
#endif
#if QDS_PREPARED_THREADS==4
#if QDS_L2_CACHE
#define QDS_PAR_L2_2 QDS_L2_PAR_2
#else
/* Reserved private page-7 scratch. The physical htid5 stack uses 0xa70b0000
 * and this third helper near slot starts immediately above it. */
#define QDS_PAR_L2_2 0xa70c0000u
#endif
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
#define QDS_REFERENCE_SCRATCH_BYTES 0x193000u
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
#if NPRG2_QNN_RUNLISTS && (!QDS_PREPARED_REFERENCE || QDS_PREPARED_THREADS <= 1 || QDS_L2_CACHE != 1)
#error NPRG2_QNN_RUNLISTS requires reference kernels, two prepared threads, and QDS_L2_CACHE=1
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
#if QDS_PREPARED_THREADS==4
_Static_assert(QDS_PAR_L2_2+QDS_PAR_L2_BYTES<=QDS_HTID5_HVX_STATE,
               "HT5 context save overlaps helper near slot");
_Static_assert((QDS_HTID5_HVX_STATE&127u)==0 && QDS_HTID5_HVX_STATE_BYTES>=0x1200u,
               "HT5 HVX state area is misaligned or too small");
#endif

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
void qd_hmx_nativep_n(void *,void *,void *,void *const *,uint32_t,uint32_t);
void qd_hmx_slide3p_n(const uint32_t *,const void *,const void *,const uint32_t *,uint32_t,uint32_t);
static uint32_t prepared_hmx_enter(void) {
    uint32_t saved;__asm__ volatile("%0=ssr":"=r"(saved));
    uint32_t on=saved|(1u<<26);__asm__ volatile("ssr=%0\n isync\n barrier"::"r"(on):"memory");
    return saved;
}
static void prepared_hmx_leave(uint32_t saved) {__asm__ volatile("ssr=%0\n isync\n barrier"::"r"(saved):"memory");}
static uint64_t now_ns(void);
#ifndef QDS_PART_TIMING
#define QDS_PART_TIMING 0
#endif
#if QDS_PART_TIMING
/* Diagnostic: per hardware thread, cycles inside op-4 job parts (each tid writes only its own). */
static volatile uint64_t qd_part_cycles[4];
#endif
static uint64_t prepared_hmx_clock(void) {return now_ns();}
static const Nprg2HmxOps prepared_hmx_ops={prepared_hmx_enter,prepared_hmx_leave,qd_hmx_wide,qd_hmx_finish,prepared_hmx_clock,
    qd_hmx_wide_n,qd_hmx_native_n,qd_hmx_native16_n,
#if !defined(NO_SLIDE3)
    qd_hmx_slide3_n
#else
    NULL
#endif
    ,qd_hmx_nativep_n,qd_hmx_slide3p_n
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
#if QDS_PREPARED_DMA_ASYNC
    QdsPreparedDmaAsync dma_async;
#endif
#if QDS_PREPARED_PROFILE
    Nprg2ExecProfile profile;
    uint64_t handler_run_cycles[NPRG2_EXEC_REGISTRY_CAPACITY],handler_run_count[NPRG2_EXEC_REGISTRY_CAPACITY],
             handler_validate_cycles[NPRG2_EXEC_REGISTRY_CAPACITY];
#endif
#if QDS_PREPARED_REFERENCE
    Nprg2ReferenceConv conv;
    Nprg2ReferenceChain chains[QDS_REFERENCE_CHAINS];
    uint8_t *scratch;
#if NPRG2_QNN_RUNLISTS
    uint8_t *qnn_worker_mem;
    Nprg2Parallel qnn_worker_parallel;
#if QDS_PREPARED_THREADS==4
    Nprg2Parallel qnn_main_parallel;
#endif
#endif
#endif
} QdPreparedState;

/* A timed-out DMA retains its descriptor and mapping state through teardown. */
static void prepared_state_dispose(QdPreparedState *s){
    if(!s)return;
    nprg2_runner_close(&s->runner);
    if(s->runner.ready)return;
#if QDS_PREPARED_DMA_ASYNC
    if(s->dma_async.pending)return;
#endif
#if QDS_PREPARED_REFERENCE
    free(s->scratch);
#endif
#if NPRG2_QNN_RUNLISTS && QDS_PREPARED_REFERENCE
    free(s->qnn_worker_mem);
#endif
    free(s);
}

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
    if(parts<2 || parts>QDS_PREPARED_THREADS || !prepared_par_ready(impl))return 0;
    work->op=4;work->job_fn=fn;work->job_ctx=ctx;work->job_parts=parts;work->error=0;
    uint32_t gen=work->generation+1;
    __asm__ volatile("barrier":::"memory");work->generation=gen;
#if QDS_PART_TIMING
    uint64_t pt0=now_ns();fn(ctx,0,parts);if(nprg2_par_kind!=NPRG2_PAR_CONV)qd_part_cycles[0]+=now_ns()-pt0;
#else
    fn(ctx,0,parts);
#endif
    uint64_t end=now_ns()+400000000ull;
    for(uint32_t t=1;t<QDS_WORKER_SLOTS;t++)
        while(work->done[t]!=gen)
            if(now_ns()>end){prepared_par_fault=1;return 1;}
    __asm__ volatile("barrier":::"memory");
    return 1;
}
#endif

#if NPRG2_QNN_RUNLISTS && QDS_PREPARED_THREADS>1
/* The worker run list is owned by follower 1. Nested parts run on follower 1
 * and follower 2 through QdsWork's independent subjob mailbox. */
#if NPRG2_QNN_RUNLISTS && QDS_PREPARED_THREADS==4 && QDS_QNN_CLAIM
/* Claim-based parts across both QNN lists (QDS_QNN_CLAIM=1). Every logical
 * thread t has one private slot per owner lane o (slot[o][t]): only lane o
 * writes gen/cancel/fn/ctx/part, only thread t writes taken/done. A part is
 * run by exactly one party through a two-flag handshake:
 *   helper: taken=gen; barrier; if cancel==gen skip, else run; done=gen
 *   owner : cancel=gen; barrier; if taken!=gen run it itself (stolen)
 *           else wait for done (and run it if the helper skipped).
 * Owners post to their home helper (lane 0 -> HT5, lane 1 -> HT2), plus the
 * other helper and the other list thread when those advertise idle. Every
 * wait loop (checkpoint waits, part joins) serves incoming parts, so two
 * owners waiting on each other always progress. Lives in the lower L2-TCM
 * half (0x30074400, free between QdsWork and the checkpoint table). */
typedef struct {
    volatile uint32_t gen,cancel,taken,done,skipped;
    Nprg2ParallelFn fn;void *ctx;volatile uint32_t part,parts;uint32_t pad[7];
} QdsQnnSlot;
typedef struct { QdsQnnSlot slot[2][4]; volatile uint32_t idle[4],magic; } QdsQnnClaim;
#define QDS_QNN_CLAIM_MAGIC 0x514e434cu
#define qnn_claim ((QdsQnnClaim *)0x30074400u)
_Static_assert(0x30074400u+sizeof(QdsQnnClaim)<=0x30075000u,"QNN claim slots overlap the checkpoint table");
typedef struct { uint32_t lane,n,who[3]; Nprg2Parallel *par; uint64_t stolen,helped; } QdsQnnLane;
static QdsQnnLane qnn_lane[2];
static int qnn_claim_serve(uint32_t self) {
    /* Slots are live only between qnn_claim_reset and qnn_claim_close; any
     * slot still unretired at close was cancelled by its owner (skip only). */
    if(qnn_claim->magic!=QDS_QNN_CLAIM_MAGIC)return 0;
    __asm__ volatile("barrier":::"memory");
    for(uint32_t o=0;o<2;o++) {
        QdsQnnSlot *sl=&qnn_claim->slot[o][self];
        uint32_t g=sl->gen;
        if(g==sl->done)continue;
        __asm__ volatile("barrier":::"memory");
        sl->taken=g;__asm__ volatile("barrier":::"memory");
        if(sl->cancel==g){sl->skipped=g;__asm__ volatile("barrier":::"memory");sl->done=g;continue;}
        if(self>=2)qnn_claim->idle[self]=0;
        sl->fn(sl->ctx,sl->part,sl->parts);
        __asm__ volatile("barrier":::"memory");sl->done=g;
        if(self>=2)qnn_claim->idle[self]=1;
        return 1;
    }
    return 0;
}
static void qnn_claim_reset(void) {
    memset((void *)qnn_claim,0,sizeof(QdsQnnClaim));
    for(uint32_t l=0;l<2;l++){qnn_lane[l].stolen=qnn_lane[l].helped=0;}
    qnn_claim->idle[2]=qnn_claim->idle[3]=1;
    __asm__ volatile("barrier":::"memory");qnn_claim->magic=QDS_QNN_CLAIM_MAGIC;
    __asm__ volatile("barrier":::"memory");
}
static void qnn_claim_close(void) {
    __asm__ volatile("barrier":::"memory");qnn_claim->magic=0;__asm__ volatile("barrier":::"memory");
}
static int qnn_claim_ready(void *impl) {
    QdsQnnLane *L=impl;
    if(prepared_par_fault || !prepared_par_lease_ok)return 0;
    uint32_t home=L->lane?2u:3u,other=L->lane?3u:2u,peer=L->lane?0u:1u;
    uint32_t cand[3]={home,other,peer},n=0;
    for(uint32_t i=0;i<3;i++) {
        QdsQnnSlot *sl=&qnn_claim->slot[L->lane][cand[i]];
        if(sl->gen!=sl->done)continue;           /* previous part not retired yet */
        if(i && !qnn_claim->idle[cand[i]])continue;
        L->who[n++]=cand[i];
    }
    if(!n)return 0;
    L->n=n;L->par->max_parts=1u+n;return 1;
}
static int qnn_claim_run(void *impl,uint32_t parts,Nprg2ParallelFn fn,void *ctx) {
    QdsQnnLane *L=impl;
    if(parts<2 || parts>1u+L->n)return 0;
    uint32_t self=L->lane,np=parts-1u;QdsQnnSlot *sl[3];uint32_t g[3];
    for(uint32_t i=0;i<np;i++) {
        sl[i]=&qnn_claim->slot[L->lane][L->who[i]];
        sl[i]->fn=fn;sl[i]->ctx=ctx;sl[i]->part=i+1u;sl[i]->parts=parts;
        g[i]=sl[i]->done+1u;
        __asm__ volatile("barrier":::"memory");sl[i]->gen=g[i];
    }
    fn(ctx,0,parts);
    uint64_t end=now_ns()+400000000ull;
    for(uint32_t i=0;i<np;i++) {
        sl[i]->cancel=g[i];__asm__ volatile("barrier":::"memory");
        if(sl[i]->taken!=g[i]){fn(ctx,i+1u,parts);L->stolen++;continue;}
        while(sl[i]->done!=g[i]) {
            if(!qnn_claim_serve(self)) {
                if(now_ns()>end){prepared_par_fault=1;break;}
                __asm__ volatile("pause(#8)");
            }
        }
        __asm__ volatile("barrier":::"memory");
        if(sl[i]->skipped==g[i]){fn(ctx,i+1u,parts);L->stolen++;}
        else L->helped++;
    }
    __asm__ volatile("barrier":::"memory");
    return 1;
}
#endif
#if QDS_PREPARED_THREADS==4
/* Main (conv) list helper: logical worker 3 (physical HT5) runs part 1 of a
 * two-part job through QdsWork's msub mailbox while workers 1 and 2 own the
 * worker list. Conv part 1 uses helper slot par_near[0], which no other
 * run-list job touches. */
static int prepared_qnn_main_parallel_run(void *impl,uint32_t parts,Nprg2ParallelFn fn,void *ctx) {
    (void)impl;
    if(parts!=2 || prepared_par_fault || !prepared_par_lease_ok)return 0;
    work->msub_fn=fn;work->msub_ctx=ctx;work->msub_parts=parts;
    uint32_t gen=work->msub_generation+1;
    __asm__ volatile("barrier":::"memory");work->msub_generation=gen;
    fn(ctx,0,parts);
    uint64_t end=now_ns()+400000000ull;
    while(work->msub_done!=gen) {
        if(now_ns()>end)prepared_par_fault=1;
        __asm__ volatile("pause(#32)");
    }
    __asm__ volatile("barrier":::"memory");return 1;
}
#endif
static int prepared_qnn_worker_parallel_ready(void *impl) {
    (void)impl;return !prepared_par_fault && prepared_par_lease_ok;
}
static int prepared_qnn_worker_parallel_run(void *impl,uint32_t parts,Nprg2ParallelFn fn,void *ctx) {
    (void)impl;
    if(parts!=2 || !prepared_qnn_worker_parallel_ready(NULL))return 0;
    work->sub_fn=fn;work->sub_ctx=ctx;work->sub_parts=parts;
    uint32_t gen=work->sub_generation+1;
    __asm__ volatile("barrier":::"memory");work->sub_generation=gen;
    fn(ctx,0,parts);
    uint64_t end=now_ns()+400000000ull;
    while(work->sub_done!=gen) {
        if(now_ns()>end)prepared_par_fault=1;
        __asm__ volatile("pause(#32)");
    }
    __asm__ volatile("barrier":::"memory");return 1;
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
#if QDS_PREPARED_DMA_ASYNC && defined(__hexagon__)
    int dma_probe=qds_prepared_dma_async_probe(&s->dma_async,s->runner.pools[5].base,s->runner.pools[5].bytes);
    if(dma_probe<0){prepared_state_dispose(s);return NULL;}
    if(dma_probe>0 && (!qds_prepared_dma_ring_init(&s->dma_async) ||
       nprg2_runner_set_weights_dma_ring(&s->runner,&s->dma_async.ring)!=NPRG2_RUNNER_OK)){
        prepared_state_dispose(s);return NULL;
    }
#endif
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
#if QDS_PREPARED_THREADS==4
    s->conv.par_near[2]=(uint8_t *)QDS_PAR_L2_2;
#endif
    s->conv.par_near_bytes=QDS_PAR_L2_BYTES;
    s->par=(Nprg2Parallel){.max_parts=QDS_PREPARED_THREADS,.ready=prepared_par_ready,.run=prepared_par_run};
#if NPRG2_QNN_RUNLISTS && QDS_PREPARED_THREADS>1
    s->qnn_worker_parallel=(Nprg2Parallel){.max_parts=QDS_QNN_WORKER_PARTS,
        .ready=prepared_qnn_worker_parallel_ready,.run=prepared_qnn_worker_parallel_run};
#if QDS_PREPARED_THREADS==4
    s->qnn_main_parallel=(Nprg2Parallel){.max_parts=QDS_QNN_MAIN_PARTS,
        .ready=prepared_qnn_worker_parallel_ready,.run=prepared_qnn_main_parallel_run};
#endif
#if QDS_PREPARED_THREADS==4 && QDS_QNN_CLAIM
    qnn_lane[0]=(QdsQnnLane){.lane=0,.par=&s->qnn_main_parallel};
    qnn_lane[1]=(QdsQnnLane){.lane=1,.par=&s->qnn_worker_parallel};
    s->qnn_main_parallel=(Nprg2Parallel){.max_parts=2,.ready=qnn_claim_ready,.run=qnn_claim_run,.impl=&qnn_lane[0]};
    s->qnn_worker_parallel=(Nprg2Parallel){.max_parts=2,.ready=qnn_claim_ready,.run=qnn_claim_run,.impl=&qnn_lane[1]};
#endif
#endif
    nprg2_runner_set_parallel(&s->runner,&s->par);
#endif
    s->scratch=malloc(QDS_REFERENCE_SCRATCH_BYTES);
#if NPRG2_QNN_RUNLISTS
    s->qnn_worker_mem=malloc(QDS_REFERENCE_SCRATCH_BYTES+QDS_REFERENCE_NEAR_BYTES);
#endif
    if(!s->scratch
#if NPRG2_QNN_RUNLISTS
       || !s->qnn_worker_mem
#endif
       || nprg2_executor_register_reference_conv(&s->executor,&s->conv,NULL) ||
       nprg2_executor_register_weights(&s->executor) ||
       nprg2_executor_register_core_safe_ops(&s->executor)) {
        prepared_state_dispose(s);return NULL;
    }
#else
    if(nprg2_executor_register_weights(&s->executor) ||
       nprg2_executor_register_core_safe_ops(&s->executor) ||
       nprg2_executor_register_native_conv(&s->executor,&s->native)) {
        prepared_state_dispose(s);return NULL;
    }
#endif
#if QDS_PREPARED_ELEMENTWISE && QDS_PREPARED_REFERENCE
    if(NPRG2_ELEMENTWISE_SCRATCH_BYTES>QDS_ELEMENTWISE_L2_BYTES ||
       nprg2_runner_set_scratch_region(&s->runner,NPRG2_EXEC_SCRATCH_NEAR,
                                       (void *)QDS_ELEMENTWISE_L2,QDS_ELEMENTWISE_L2_BYTES) ||
       nprg2_executor_register_elementwise_region(&s->executor,NPRG2_EXEC_SCRATCH_NEAR)) {
        prepared_state_dispose(s);return NULL;
    }
#elif QDS_PREPARED_ELEMENTWISE
    if(NPRG2_ELEMENTWISE_SCRATCH_BYTES>0x8000u ||
       nprg2_executor_register_elementwise(&s->executor)) {
        prepared_state_dispose(s);return NULL;
    }
#endif
#if QDS_PREPARED_PREPROCESS
    if(nprg2_executor_register_preprocess(&s->executor)) {
        prepared_state_dispose(s);return NULL;
    }
#endif
#if QDS_PREPARED_REFERENCE
    if(prepared_register_reference(s)) {
        prepared_state_dispose(s);return NULL;
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

#if NPRG2_QNN_RUNLISTS
typedef struct {
    QdgGraph *graph;
    QdPreparedState *state;
    Nprg2ExecRequest requests[2];
    volatile uint32_t *checkpoints;
    volatile uint32_t error;
    volatile uint32_t completed[2];
    volatile uint32_t excl_flag[2],excl_victim;
#if QDS_PREPARED_PROFILE
    uint64_t wait[2],wall[2],pre[2];
#endif
} QdQnnListRun;
#if QDS_PREPARED_PROFILE
/* Per-list handler time: list 0 into the handler profile directly after the
 * join, list 1 accumulated here (no shared words between the two lists). */
static uint64_t qnn_prof_cycles[2][NPRG2_EXEC_REGISTRY_CAPACITY];
static uint32_t qnn_prof_count[2][NPRG2_EXEC_REGISTRY_CAPACITY];
#if QDS_REF_SPLIT_PROF
extern uint64_t nprg2_ref_split[2][48][80][3];
#endif
#ifndef QDS_QNN_WAITATTR
#define QDS_QNN_WAITATTR 0
#endif
#if QDS_QNN_WAITATTR
/* Diagnostic: a list's checkpoint-wait time charged to the handler of the
 * command that last released that checkpoint. Exported (instead of worker
 * busy) as handler validate_ms = main-list wait by worker producer. */
static volatile uint8_t qnn_cp_last_hi[QD_QNN_PRIMARY_CHECKPOINT_COUNT];
static uint64_t qnn_wait_attr[2][NPRG2_EXEC_REGISTRY_CAPACITY];
#endif
#endif
#define QDS_QNN_WORKER_NEAR 0xa7060000u
#define QDS_QNN_WORKER_SCRATCH 0xa7200000u
_Static_assert(QDS_QNN_WORKER_SCRATCH+QDS_REFERENCE_SCRATCH_BYTES<=0xa8000000u,"QNN worker scratch leaves private page 7");
static volatile uint32_t *const prepared_qnn_checkpoints=(volatile uint32_t *)0x30075000u;
static int prepared_qnn_matches(const QdPreparedState *s,uint32_t first,uint32_t last) {
    return s && first==0 && last==QD_QNN_PRIMARY_COMMAND_COUNT &&
        s->runner.program.command_count==QD_QNN_PRIMARY_COMMAND_COUNT &&
        !memcmp(s->runner.program.context_sha256,nprg2_qnn_primary_context_sha,32) &&
        !s->runner.fuse && s->qnn_worker_mem;
}
#ifndef QDS_QNN_EXCLUSIVE
#define QDS_QNN_EXCLUSIVE 0
#endif
static int prepared_qnn_command(QdQnnListRun *run,const Nprg2QnnRunEntry *entry,uint32_t list) {
    uint32_t wait=entry->wait_cp,done=entry->done_cp;
#if QDS_PREPARED_PROFILE
    uint64_t p0=now_ns();
#endif
    if(wait>=QD_QNN_PRIMARY_CHECKPOINT_COUNT || done>=QD_QNN_PRIMARY_CHECKPOINT_COUNT ||
       (done && nprg2_qnn_primary_cp_owner[done]!=list))return NPRG2_EXEC_REJECTED;
    uint64_t wait_deadline=now_ns()+4000000000ull;
    uint32_t spins=0;
#if QDS_PREPARED_THREADS==4 && QDS_QNN_CLAIM
    if(wait && run->checkpoints[wait]){qnn_claim->idle[list]=1;__asm__ volatile("barrier":::"memory");}
#endif
    while(wait && run->checkpoints[wait]) {
#if QDS_PREPARED_THREADS==4 && QDS_QNN_CLAIM
        if(qnn_claim_serve(list))continue;
#endif
        if((spins++&1023u)==0u) {
            __asm__ volatile("barrier":::"memory");
            if(run->error || resident_cancelled())return NPRG2_EXEC_CANCELLED;
            if(now_ns()>wait_deadline)return NPRG2_EXEC_REJECTED;
        }
        __asm__ volatile("pause(#32)");
    }
    __asm__ volatile("barrier":::"memory");
#if QDS_PREPARED_THREADS==4 && QDS_QNN_CLAIM
    qnn_claim->idle[list]=0;__asm__ volatile("barrier":::"memory");
    (void)qnn_claim_serve(list);   /* a part posted just before we left the wait */
#endif
    QdPreparedState *s=run->state;
#if QDS_PREPARED_PROFILE
    uint64_t p1=now_ns();run->wait[list]+=p1-p0;
#if QDS_QNN_WAITATTR
    if(wait)qnn_wait_attr[list][qnn_cp_last_hi[wait]]+=p1-p0;
#endif
#endif
#if QDS_PREPARED_DMA_ASYNC
    if(entry->dma_event_kind==2u && s->runner.weights_async_ready &&
       nprg2_exec_weights_dma_checkpoint_wait(&s->runner.weights_async,entry->dma_event_id)!=NPRG2_EXEC_OK)
        return NPRG2_EXEC_HANDLER_ERROR;
#endif
#if QDS_QNN_EXCLUSIVE
    /* Diagnostic: lists still follow their checkpoints but never run handlers
     * at the same time (Peterson lock, taken only after the wait). */
    run->excl_flag[list]=1;__asm__ volatile("barrier":::"memory");
    run->excl_victim=list;__asm__ volatile("barrier":::"memory");
    while(run->excl_flag[list^1u] && run->excl_victim==list)__asm__ volatile("pause(#8)");
    __asm__ volatile("barrier":::"memory");
#if QDS_PREPARED_PROFILE
    {uint64_t p1b=now_ns();run->wait[list]+=p1b-p1;p1=p1b;}
#endif
#endif
#if QDS_PREPARED_PROFILE
    {uint64_t pr=now_ns();nprg2_lane_prehandler[list]-=pr;run->pre[list]+=pr-p1;}
#endif
    int rc=nprg2_executor_run_preflighted_command_prepared(&s->executor,
        &run->requests[list],entry->command);
#if QDS_QNN_EXCLUSIVE
    __asm__ volatile("barrier":::"memory");run->excl_flag[list]=0;__asm__ volatile("barrier":::"memory");
#endif
    if(rc)return rc;
#if QDS_PREPARED_PROFILE
    {
        Nprg2Command cmd;uint64_t p2=now_ns();
        if(nprg2_command_at(run->requests[list].bound,entry->command,&cmd)==NPRG2_OK){
            int hi=nprg2_executor_handler_slot(&s->executor,cmd.opcode,cmd.variant);
            if(hi>=0){qnn_prof_cycles[list][hi]+=p2-p1;qnn_prof_count[list][hi]++;}
#if QDS_QNN_WAITATTR
            if(hi>=0 && done)qnn_cp_last_hi[done]=(uint8_t)hi;
#endif
        }
    }
#endif
#if QDS_PREPARED_DMA_ASYNC
    if(entry->dma_event_kind==1u && s->runner.weights_async_ready &&
       nprg2_exec_weights_dma_checkpoint_set(&s->runner.weights_async,entry->dma_event_id)!=NPRG2_EXEC_OK)
        return NPRG2_EXEC_HANDLER_ERROR;
#endif
    run->completed[list]++;
    if(done) {
        if(!run->checkpoints[done])return NPRG2_EXEC_REJECTED;
        __asm__ volatile("barrier":::"memory");
        run->checkpoints[done]--;
        __asm__ volatile("barrier":::"memory");
    }
    return NPRG2_EXEC_OK;
}
static void prepared_qnn_worker_list(void *opaque,uint32_t tid,uint32_t parts) {
    QdQnnListRun *run=opaque;
    if(tid!=1 || parts<2)return;
#if QDS_PREPARED_PROFILE
    uint64_t w0=now_ns();
#endif
    for(uint32_t i=0;i<QD_QNN_PRIMARY_WORKER_COUNT;i++) {
        int rc=prepared_qnn_command(run,&nprg2_qnn_primary_worker[i],1);
        if(rc){run->error=(uint32_t)rc;__asm__ volatile("barrier":::"memory");return;}
    }
#if QDS_PREPARED_PROFILE
    run->wall[1]=now_ns()-w0;
#endif
}
static int prepared_qnn_dma_finish(QdPreparedState *s,int status){
#if QDS_PREPARED_DMA_ASYNC
    if(s&&s->runner.weights_async_ready &&
       nprg2_exec_weights_dma_drain_all(&s->runner.weights_async)!=NPRG2_EXEC_OK)
        return NPRG2_EXEC_HANDLER_ERROR;
#else
    (void)s;
#endif
    return status;
}
static int prepared_qnn_runlists(QdgGraph *g,QdPreparedState *s,uint32_t first,uint32_t last) {
    if(!prepared_qnn_matches(s,first,last))return -1;
    QdQnnListRun run={.graph=g,.state=s,.checkpoints=prepared_qnn_checkpoints};
    if(nprg2_runner_prepare_preflighted_request(&s->runner,&s->executor,0,
        QD_QNN_PRIMARY_COMMAND_COUNT,QD_DIRECT_ACTIVE_BYTES,s->inputs,4,s->outputs,2,
        g,s->scratch,QDS_REFERENCE_SCRATCH_BYTES,&s->stats,0,
#if QDS_PREPARED_THREADS==4
        &s->qnn_main_parallel,
#else
        NULL,
#endif
        s->scratch,QDS_REFERENCE_SCRATCH_BYTES,s->runner.extra_scratch[0].data,
        s->runner.extra_scratch[0].bytes,&run.requests[0]) ||
       nprg2_runner_prepare_preflighted_request(&s->runner,&s->executor,0,
        QD_QNN_PRIMARY_COMMAND_COUNT,QD_DIRECT_ACTIVE_BYTES,s->inputs,4,s->outputs,2,
        g,s->scratch,QDS_REFERENCE_SCRATCH_BYTES,&s->stats,1,&s->qnn_worker_parallel,
        (void *)QDS_QNN_WORKER_SCRATCH,QDS_REFERENCE_SCRATCH_BYTES,
        (void *)QDS_QNN_WORKER_NEAR,QDS_REFERENCE_NEAR_BYTES,&run.requests[1]))
        return NPRG2_EXEC_ARGUMENT;
    for(uint32_t i=0;i<QD_QNN_PRIMARY_CHECKPOINT_COUNT;i++)run.checkpoints[i]=nprg2_qnn_primary_cp_initial[i];
#if QDS_PREPARED_THREADS==4 && QDS_QNN_CLAIM
    qnn_claim_reset();
#endif
    {Nprg2Parallel *wp=&s->qnn_worker_parallel;
     memset(wp->jobs_parallel,0,sizeof wp->jobs_parallel);memset(wp->jobs_serial,0,sizeof wp->jobs_serial);
     memset(wp->parts_run,0,sizeof wp->parts_run);wp->not_ready=wp->declined=0;}
#if QDS_PREPARED_THREADS==4
    {Nprg2Parallel *mp=&s->qnn_main_parallel;
     memset(mp->jobs_parallel,0,sizeof mp->jobs_parallel);memset(mp->jobs_serial,0,sizeof mp->jobs_serial);
     memset(mp->parts_run,0,sizeof mp->parts_run);mp->not_ready=mp->declined=0;}
#endif
#if QDS_PREPARED_PROFILE
    memset(qnn_prof_cycles,0,sizeof qnn_prof_cycles);memset(qnn_prof_count,0,sizeof qnn_prof_count);
#if QDS_REF_SPLIT_PROF
    memset(nprg2_ref_split,0,sizeof nprg2_ref_split);
#endif
#if QDS_QNN_WAITATTR
    memset(qnn_wait_attr,0,sizeof qnn_wait_attr);
#endif
    nprg2_lane_clock=now_ns;nprg2_lane_prehandler[0]=nprg2_lane_prehandler[1]=0;
    uint64_t m0=now_ns();
#endif
    __asm__ volatile("barrier":::"memory");
    work->job_fn=prepared_qnn_worker_list;work->job_ctx=&run;work->job_parts=2;
    uint32_t generation=send_work_start(8,0);
    uint32_t completed=0;
    for(uint32_t i=0;i<QD_QNN_PRIMARY_MAIN_COUNT;i++) {
        int rc=prepared_qnn_command(&run,&nprg2_qnn_primary_main[i],0);
        if(rc){run.error=(uint32_t)rc;break;}
        completed++;
    }
#if QDS_PREPARED_PROFILE
    run.wall[0]=now_ns()-m0;
#endif
    int join=send_work_wait(generation,4000000000ull);
#if QDS_PREPARED_THREADS==4 && QDS_QNN_CLAIM
    qnn_claim_close();
#endif
#if QDS_PREPARED_PROFILE
    if(!join){
        /* run_ms/dispatch_ms = main wall/wait, validate_ms = worker wait;
         * handler run_ms = main, validate_ms = worker; count = main | worker<<32. */
        s->profile.run_cycles=run.wall[0];s->profile.dispatch_cycles=run.wait[0];
        s->profile.validate_cycles=run.wait[1];
        for(uint32_t k=0;k<NPRG2_EXEC_REGISTRY_CAPACITY;k++){
            s->handler_run_cycles[k]=qnn_prof_cycles[0][k];s->handler_validate_cycles[k]=qnn_prof_cycles[1][k];
#if QDS_QNN_WAITATTR
            s->handler_validate_cycles[k]=qnn_wait_attr[0][k];
            s->handler_run_cycles[k]=qnn_wait_attr[1][k];
#endif
#if QDS_REF_SPLIT_PROF
            {const Nprg2ExecHandler *h=&s->executor.handlers[k];
             if(h->used && h->opcode<48u && h->variant<80u){
                 /* run_ms = setup_ex (both lanes), validate_ms = run_kernel; count low = entry overhead pcycles/16 */
                 s->handler_run_cycles[k]=nprg2_ref_split[0][h->opcode][h->variant][0]+nprg2_ref_split[1][h->opcode][h->variant][0];
                 s->handler_validate_cycles[k]=nprg2_ref_split[0][h->opcode][h->variant][1]+nprg2_ref_split[1][h->opcode][h->variant][1];
                 s->handler_run_count[k]=(s->handler_run_count[k]&~0xffffffffull)|(uint32_t)(qnn_prof_count[0][k]);
                 s->handler_run_count[k]|=0;
             }}
#endif
            s->handler_run_count[k]=(uint64_t)qnn_prof_count[0][k]|(uint64_t)qnn_prof_count[1][k]<<32;
        }
        ((volatile uint32_t *)(SERVICE+1024u))[3]=(uint32_t)run.wall[1];
        /* preflight_ms = worker-lane executor overhead before handler entry;
         * trace[10] = main-lane overhead (pcycles), trace[14] = worker DMA-wait+other pre (pcycles). */
        s->profile.preflight_cycles=nprg2_lane_prehandler[1];
        ((volatile uint32_t *)(SERVICE+1024u))[10]=(uint32_t)nprg2_lane_prehandler[0];
        ((volatile uint32_t *)(SERVICE+1024u))[14]=(uint32_t)run.pre[1];
    }
#endif
#if QDS_PREPARED_THREADS>1
    /* The main list runs unsplit; report the worker lane's split counters. */
    {Nprg2Parallel *wp=&s->qnn_worker_parallel;
     memcpy(s->par.jobs_parallel,wp->jobs_parallel,sizeof wp->jobs_parallel);
     memcpy(s->par.jobs_serial,wp->jobs_serial,sizeof wp->jobs_serial);
     memcpy(s->par.parts_run,wp->parts_run,sizeof wp->parts_run);
     s->par.not_ready=wp->not_ready;s->par.declined=wp->declined;}
#if QDS_PREPARED_THREADS==4
    /* Conv kind comes from the main lane (the worker lane runs no conv). */
    {Nprg2Parallel *mp=&s->qnn_main_parallel;
     s->par.jobs_parallel[NPRG2_PAR_CONV]=mp->jobs_parallel[NPRG2_PAR_CONV];
     s->par.jobs_serial[NPRG2_PAR_CONV]=mp->jobs_serial[NPRG2_PAR_CONV];
     s->par.parts_run[NPRG2_PAR_CONV]=mp->parts_run[NPRG2_PAR_CONV];
     s->par.not_ready+=mp->not_ready;s->par.declined+=mp->declined;}
#endif
#if QDS_PREPARED_THREADS==4 && QDS_QNN_CLAIM
    /* Claim mode: not_ready = parts the main lane ran itself (stolen back or
     * skipped), declined = the worker lane's; trace[12] = helped parts main | worker<<16. */
    s->par.not_ready=qnn_lane[0].stolen;s->par.declined=qnn_lane[1].stolen;
    ((volatile uint32_t *)(SERVICE+1024u))[12]=(uint32_t)(qnn_lane[0].helped|qnn_lane[1].helped<<16);
#endif
#endif
    if(join) {
        run.error=NPRG2_EXEC_CANCELLED;prepared_par_fault=1;
        __asm__ volatile("barrier":::"memory");
        /* Retain this stack/context, the lease and code until both workers
         * acknowledge. A permanently stalled worker requires host recovery. */
        for(uint32_t t=1;t<QDS_WORKER_SLOTS;t++)
            while(work->done[t]!=generation)__asm__ volatile("pause(#32)");
        return prepared_qnn_dma_finish(s,NPRG2_EXEC_CANCELLED);
    }
    if(run.error)return prepared_qnn_dma_finish(s,(int)run.error);
    if(completed!=QD_QNN_PRIMARY_MAIN_COUNT)return prepared_qnn_dma_finish(s,NPRG2_EXEC_REJECTED);
    for(uint32_t i=0;i<QD_QNN_PRIMARY_CHECKPOINT_COUNT;i++)
        if(run.checkpoints[i])return prepared_qnn_dma_finish(s,NPRG2_EXEC_REJECTED);
    s->stats.executed_commands=QD_QNN_PRIMARY_COMMAND_COUNT;
    volatile uint32_t *trace=(volatile uint32_t *)(SERVICE+1024u);
    trace[6]=run.completed[0];trace[7]=run.completed[1];trace[11]=1;
    return prepared_qnn_dma_finish(s,NPRG2_EXEC_OK);
}
#endif

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
    int e;
#if NPRG2_QNN_RUNLISTS
    e=prepared_qnn_runlists(g,s,first,last);
    if(e==-1){
#if QDS_PREPARED_DMA_ASYNC
        uint8_t dma_ready=s->runner.weights_async_ready;s->runner.weights_async_ready=0;
#endif
        e=nprg2_runner_execute(&s->runner,&s->executor,first,last-first,QD_DIRECT_ACTIVE_BYTES,
            s->inputs,4,s->outputs,2,g,scratch,scratch_bytes,prepared_cancel,NULL,&s->stats);
#if QDS_PREPARED_DMA_ASYNC
        s->runner.weights_async_ready=dma_ready;
#endif
    }
#else
    e=nprg2_runner_execute(&s->runner,&s->executor,first,last-first,QD_DIRECT_ACTIVE_BYTES,
        s->inputs,4,s->outputs,2,g,scratch,scratch_bytes,prepared_cancel,NULL,&s->stats);
#endif
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
    {
        /* With QDS_POOL1_NOCOPY the run never touches pool 1's DDR image, the first
         * 8 MiB of this page: only pools 3, 4, 6 and the I/O buffers can hold L1
         * lines. (Inputs too: their clean lines would be stale on the next run.) */
        uint32_t lo=0,hi=0;
#if QDS_POOL1_NOCOPY
        lo=0xffffffffu;
        static const uint8_t ps[3]={QDG_POOL_DDR3,QDG_POOL_DDR4,QDG_POOL_SPILL};
#define QDS_L1_SPAN(b,n) do { uint32_t b_=(uint32_t)(uintptr_t)(b),e_=b_+(uint32_t)(n); if((n) && b_<lo)lo=b_; if((n) && e_>hi)hi=e_; } while(0)
        for(uint32_t i=0;i<3;i++)QDS_L1_SPAN(c->g->pool[ps[i]],c->g->pool_bytes[ps[i]]);
        for(uint32_t i=0;i<4;i++)QDS_L1_SPAN(c->s->inputs[i].data,c->s->inputs[i].bytes);
        for(uint32_t i=0;i<2;i++)QDS_L1_SPAN(c->s->outputs[i].data,c->s->outputs[i].bytes);
#undef QDS_L1_SPAN
        if(lo<QDS_L2_POOL_VA || hi>QDS_L2_POOL_VA+0x1000000u || lo>=hi){lo=0;hi=0;} /* anything unexpected: whole page */
#endif
        if(qds_l2_pool_cache_range(0,lo,hi))c->l2_error|=32;
    }
#endif
    if(qds_l2_end(&l2))c->l2_error|=8;
    trace[5]=(uint32_t)(now_ns()-t);
}
#endif

static int prepared_run(QdgGraph *g,QdPreparedState *s,uint32_t first,uint32_t last) {
    volatile uint32_t *trace=(volatile uint32_t *)(SERVICE+1024u);
    for(uint32_t q=0;q<16;q++)trace[q]=0;
    trace[0]=0x31525051u;trace[2]=((volatile uint32_t *)SERVICE)[2];g->error_detail=0;
#if QDS_PART_TIMING
    qd_part_cycles[0]=qd_part_cycles[1]=qd_part_cycles[2]=0;
#endif
#if QDS_L2_CACHE
    trace[12]=(uint32_t)qds_l2_engine_feature[0];
#endif
    if(!s || first>=last || last>g->ncommands || !native_1x1_lease_valid() ||
       !native_1x1_hmx_owned() || !native_1x1_workers_idle())return QDG_ERR_UNSUPPORTED;
#if QDS_PREPARED_THREADS>1
    prepared_par_lease_ok=1;
#endif
#if QDS_PREPARED_DMA_ASYNC
    if(s->runner.weights_async_ready){
        s->dma_async.starts=s->dma_async.drains=0;
        s->dma_async.verified_bytes=s->dma_async.mismatch_count=0;
        if(!nprg2_dma_ring_reset(&s->dma_async.ring))return QDG_ERR_UNSUPPORTED;
    }
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
#if NPRG2_QNN_RUNLISTS
    if(prepared_qnn_matches(s,first,last) && !nprg2_exec_reference_clone_lane(0,1)) {
        e=NPRG2_EXEC_ARGUMENT;goto failed;
    }
#endif
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
    for(uint32_t t=1;t<QDS_WORKER_SLOTS;t++)for(uint32_t i=0;i<32;i++)((volatile uint32_t *)QDS_L2_FSTACK(t))[i]=0x7154c9a3u;
    if(prepared_par_fault || send_work(5,0)){prepared_par_fault=1;trace[13]=64;e=NPRG2_EXEC_ARGUMENT;goto failed;}
#endif
    qds_on_stack(prepared_leased,0,QDS_L2_STACK_TOP);
    for(uint32_t i=0;i<32;i++)if(guard[i]!=0x7154c9a3u)qds_lease_call.l2_error|=2;
#if QDS_PREPARED_THREADS>1
    if(send_work(6,0)){prepared_par_fault=1;qds_lease_call.l2_error|=64;}
    for(uint32_t t=1;t<QDS_WORKER_SLOTS;t++)for(uint32_t i=0;i<32;i++)
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
#if QDS_PART_TIMING
    trace[3]=(uint32_t)qd_part_cycles[0];trace[10]=(uint32_t)qd_part_cycles[1];trace[14]=(uint32_t)qd_part_cycles[2];
#endif
    trace[8]=s->stats.executed_commands;trace[9]=s->stats.validated_commands;
#if defined(NPRG2_APPROX_COST_VOLUME_STEP) && NPRG2_APPROX_COST_VOLUME_STEP > 1
    trace[6]=s->stats.approx_cost_volume_commands;
    trace[7]=s->stats.approx_cost_volume_conv_planes;
    trace[10]=s->stats.approx_cost_volume_interpolated_planes;
#endif
    return 0;
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
    uint64_t r[18]={0};
    r[0]=0x54505451ull|1ull<<32;r[1]=(uint64_t)s->par.max_parts|(uint64_t)prepared_par_fault<<32;
    r[2]=s->par.not_ready;r[3]=s->par.declined;
    for(int k=0;k<3;k++){r[4+k]=s->par.jobs_parallel[k];r[7+k]=s->par.jobs_serial[k];r[10+k]=s->par.parts_run[k];}
    r[13]=(uint64_t)work->job_started[1]|(uint64_t)work->job_started[2]<<32;
    r[14]=(uint64_t)work->job_finished[1]|(uint64_t)work->job_finished[2]<<32;
#if QDS_PREPARED_THREADS==4
    r[15]=work->job_started[3];r[16]=work->job_finished[3];
#endif
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
