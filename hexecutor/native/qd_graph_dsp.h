#ifndef QD_GRAPH_DSP_H
#define QD_GRAPH_DSP_H
#include "qd_conv.h"
#include "qd_owned_memory.h"
#define QDS_MAGIC 0x31534451u
#define QDS_CONFIG 4096u
#define QDS_WORK 8192u
#if QDS_L2_CACHE==1
/* Lower, still-TCM L2 half (qd_l2_cache.h layout): followers poll it during the lease. */
#define QDS_WORK_L2 0x30074000u
#elif QDS_L2_CACHE
#define QDS_WORK_L2 0xf1802000u
#else
#define QDS_WORK_L2 0x300f0000u
#endif
#define QDS_RESULT 2048u
#define QDS_PROFILE 16384u
/* Prepared per-handler profile, see prepared_export_profile (qd_graph_prepared.h). */
#define QDS_PROFILE2 20480u
#ifndef QDS_PREPARED_THREADS
#define QDS_PREPARED_THREADS 1
#endif
/* The existing callback wakes physical hthreads 0, 1, and 2. In the opt-in
 * four-worker experiment htid5 is explicitly resumed by the master; htid3
 * runs the stock eye-tracking loop (it polls DMA 0x1522000 independently of
 * our wake and uses VTCM 0x31700000..~0x317d4600) and htid4 owns CPU/IPCC
 * servicing. Keep engine work
 * indices dense even though the fourth physical id is sparse. */
#if QDS_PREPARED_THREADS == 4
#define QDS_WORKER_SLOTS 4u
static inline uint32_t qds_logical_tid(uint32_t tid) { return tid==5u?3u:tid; }
static inline uint32_t qds_physical_tid(uint32_t logical) { return logical==3u?5u:logical; }
#else
#define QDS_WORKER_SLOTS 3u
static inline uint32_t qds_logical_tid(uint32_t tid) { return tid; }
static inline uint32_t qds_physical_tid(uint32_t logical) { return logical; }
#endif
#ifdef HEXECUTOR_TENSOR_BOOT
/* A DDR stack makes this worker's C code about ten times slower, and every
 * operator joins on its slowest part. */
#define QDS_HTID5_STACK_BOTTOM 0x300e0000u
#define QDS_HTID5_STACK_TOP    0x300f0000u
#else
#define QDS_HTID5_STACK_BOTTOM 0xa70a0000u
#define QDS_HTID5_STACK_TOP    0xa70b0000u
#endif
#define QDS_HTID5_HVX_STATE    0xa70d0000u
#define QDS_HTID5_HVX_STATE_BYTES 0x1200u
#ifndef QDS_PREPARED_DMA_ASYNC
#define QDS_PREPARED_DMA_ASYNC 0
#endif
#ifndef QDS_PREPARED_DMA_VERIFY
#define QDS_PREPARED_DMA_VERIFY 0
#endif
#ifndef QDS_PREPARED_DMA_MAX_TRANSFERS
#define QDS_PREPARED_DMA_MAX_TRANSFERS 0u
#endif
#ifndef QDS_PREPARED_DMA_MODE
#define QDS_PREPARED_DMA_MODE 0u
#endif
#define QDS_DMA_VERIFY_DIAG_OFFSET 1152u
#define QDS_DMA_DESC_BASE 0xf1808000u
#define QDS_DMA_STATE_OFFSET 0x3f00u
#define QDS_DMA_STATE_IDLE 0u
#define QDS_DMA_STATE_INFLIGHT 1u
#define QDS_DMA_STATE_PENDING_ERROR 101u
#define QDS_DMA_PENDING_ERROR 101u
#define QDS_OP_RUN 3u
typedef struct {
    uint32_t magic,version,image,image_bytes,pool[7],pool_bytes[7];
    uint32_t input[4],input_bytes[4],output[2],output_bytes[2];
    uint32_t heap,heap_bytes,code_bytes;
} QdsConfig;
/* Check the complete allocation contract before initializing the heap or
 * touching any graph-owned bytes. Scratch and worker staging have fixed slots. */
static uint32_t qds_config_valid(const QdsConfig *c) {
    uint32_t offsets[15],bytes[15],n=0;
    if(c->magic!=QDS_MAGIC || c->version!=1 || !c->code_bytes || c->code_bytes>0x100000u ||
       c->heap!=0x2000000u || c->heap_bytes!=0x2000000u-4096u)return 0;
    offsets[n]=c->image;bytes[n++]=c->image_bytes;
    for(uint32_t i=0;i<7;i++)if(c->pool_bytes[i]) {
        offsets[n]=c->pool[i];bytes[n++]=c->pool_bytes[i];
        uint32_t lo=i==5?0x100000u:0x1000000u,hi=i==5?0x1000000u:0x2000000u;
        if(c->pool[i]<lo || c->pool[i]>hi || c->pool_bytes[i]>hi-c->pool[i])return 0;
    }
    for(uint32_t i=0;i<4;i++){offsets[n]=c->input[i];bytes[n++]=c->input_bytes[i];}
    for(uint32_t i=0;i<2;i++){offsets[n]=c->output[i];bytes[n++]=c->output_bytes[i];}
    if(c->image<0x100000u || c->image>=0x1000000u || c->image_bytes<256u ||
       c->image_bytes>0x1000000u-c->image)return 0;
    /* The first seven descriptors are image/pools; I/O occupies the last six. */
    for(uint32_t i=n-6;i<n;i++)if(offsets[i]<0x1000000u)return 0;
    for(uint32_t i=0;i<n;i++) {
        if((offsets[i]&127u) || !bytes[i] || offsets[i]>=0x2000000u || bytes[i]>0x2000000u-offsets[i])return 0;
        for(uint32_t j=0;j<i;j++)if(offsets[i]<offsets[j]+bytes[j] && offsets[j]<offsets[i]+bytes[i])return 0;
    }
    return 1;
}
typedef void (*QdsJobFn)(void *ctx,uint32_t part,uint32_t parts);
#ifndef NPRG2_QNN_RUNLISTS
#define NPRG2_QNN_RUNLISTS 0
#endif
#ifndef NPRG2_OVERLAP
#define NPRG2_OVERLAP 0
#endif
typedef struct {
    volatile uint32_t generation,op,sequence,error,done[QDS_WORKER_SLOTS];
    uint32_t arena,prepared_a;
    QdConv conv;
    /* op 4 (prepared threads): followers with tid < job_parts run
     * job_fn(job_ctx,tid,job_parts). job_started/finished count per follower
     * (index tid) and let the host see a worker that never came back. */
    QdsJobFn job_fn;void *job_ctx;volatile uint32_t job_parts,job_started[QDS_WORKER_SLOTS],job_finished[QDS_WORKER_SLOTS];
    /* Dedicated follower-2 mailbox for a nested background list kernel
     * (used with NPRG2_OVERLAP / NPRG2_QNN_RUNLISTS; idle otherwise). */
    volatile uint32_t sub_generation,sub_done;
    QdsJobFn sub_fn;void *sub_ctx;volatile uint32_t sub_parts;
    /* Second mailbox, served by logical worker 3 (physical HT5): nested
     * parts of the QNN main list (QDS_PREPARED_THREADS=4). */
    volatile uint32_t msub_generation,msub_done;
    QdsJobFn msub_fn;void *msub_ctx;volatile uint32_t msub_parts;
} QdsWork;
/* QdsWork owns 0x300f0000..0x300f0200; 0x300f0200.. is the prepared helper
 * slots' (qd_graph_prepared.h). */
_Static_assert(sizeof(QdsWork)<=0x200,"QdsWork overlaps the prepared helper near slots");
uint32_t qdrc_execute(const QdConv *c,uint8_t *arena,uint32_t tid,uint32_t prepared_a);
uint32_t qdrc_power(uint32_t on);
#if QDS_NATIVE_1X1
void qdrc_invalidate_native(void);
#endif
#endif
