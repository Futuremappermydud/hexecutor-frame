/* The unchanged loader enters this small trampoline. Extended code, pools and
 * heap all live in private cdsp_mem, never outside owned mappings. Every worker
 * returns here before the JTLB is restored and EXIT is published. */
#include "qd_graph_dsp.h"
#include "qd_protocol.h"
#ifndef QDS_L2_CACHE
#define QDS_L2_CACHE 0
#endif
#if QDS_L2_CACHE
/* Boot is unchanged in L2 mode; the engine owns the lease (qd_l2_cache.h). */
static const char qds_l2_boot_feature[] __attribute__((used))="QDL2BOOT02";
#endif
#ifndef QDS_PRESERVE_VTCM
#define QDS_PRESERVE_VTCM 0
#endif
#if QDS_PRESERVE_VTCM
#include "qd_vtcm_lease.h"
#define QDS_LEASE_STATE 3328u
#define QDS_LEASE_TIMING 3456u
#define QDS_LEASE_STATUS 3472u
/* Read from the actual ELF by the host, rather than inferred from a filename. */
static const uint8_t qds_boot_features[12] __attribute__((used)) = {'Q','D','B','O','O','T','0','1',1,0,0,0};
#endif
#define ARENA 0xf1800000u
#ifndef QDS_ENTRY_GUARD
#define QDS_ENTRY_GUARD 0
#endif
#if QDS_ENTRY_GUARD
#include "qd_graph_entry.h"
static const uint8_t qds_entry_features[12] __attribute__((used)) =
    {'Q','D','E','N','T','R','Y','1',1,0,0,0};
#endif
static uint64_t cycles(void){uint64_t c;__asm__ volatile("%0=upcycle":"=r"(c));return c;}
#if QDS_PRESERVE_VTCM
#include "qd_axim.h"
#include <stddef.h>
_Static_assert(offsetof(QdVtcmLease,checksum_xor)==44,"host fingerprint ABI changed");
#endif
extern void qds_on_stack(void (*entry)(uint32_t),uint32_t tid,uint32_t stack);
#if QDS_PRESERVE_VTCM
static void qds_save_vtcm(uint32_t unused) {
    (void)unused;
    *(volatile uint32_t *)(ARENA+QDS_LEASE_STATUS)=qdvt_save((QdVtcmLease *)(ARENA+QDS_LEASE_STATE),
        (volatile uint8_t *)QDVT_BASE,QDVT_BASE,QDVT_BYTES,QDVT_BACKUP_BASE,(uint8_t *)QDVT_BACKUP_BASE);
}
static void qds_restore_vtcm(uint32_t unused) {
    (void)unused;
    QdVtcmLease *lease=(QdVtcmLease *)(ARENA+QDS_LEASE_STATE);
    uint32_t e=qdvt_restore_verify(lease,(volatile uint8_t *)QDVT_BASE);
    if(!e)e=qdvt_release_verified(lease);
    *(volatile uint32_t *)(ARENA+QDS_LEASE_STATUS)=e;
}
static uint32_t qds_lease_on_stack(void (*entry)(uint32_t)) {
    /* HVX spills and scalar fingerprint reductions must share coherent TCM,
     * rather than the firmware's scalar-cached DDR stack. */
    volatile uint32_t *guard=(volatile uint32_t *)0x30080000u;
    for(uint32_t i=0;i<32;i++)guard[i]=0x7154c9a3u;
    qds_on_stack(entry,0,0x30090000u);
    for(uint32_t i=0;i<32;i++)if(guard[i]!=0x7154c9a3u)return 9;
    return *(volatile uint32_t *)(ARENA+QDS_LEASE_STATUS);
}
#endif
__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;__asm__ volatile("%0=htid":"=r"(tid));if(tid>=3)return;
    volatile uint32_t *a=(volatile uint32_t *)ARENA,*sync=(volatile uint32_t *)(ARENA+3584);
    QdoMap *map=(QdoMap *)(ARENA+QDO_STATE);
    const QdsConfig *cfg=(const QdsConfig *)(ARENA+QDS_CONFIG);
#if QDS_ENTRY_GUARD
    /* A callback can be invoked while the host is still preparing its arena.
     * Reject unpublished/foreign commands before any memory ownership change.
     * A valid nonzero sequence also prevents followers passing the initial
     * zero-valued sync word before the master publishes initialization. */
    if(!qds_initial_command_valid(a,QDS_OP_RUN) || !qds_config_valid(cfg)) {
        if(!tid) {
            volatile uint32_t *result=(volatile uint32_t *)(ARENA+QDS_RESULT);
            result[0]=QDS_MAGIC;result[1]=1;
        }
        volatile QdStatus *st=(volatile QdStatus *)(ARENA+QD_STATUS_OFFSET)+tid;
        st->error=1;st->rows=0;st->reserved=QDC_EXITED;
        __asm__ volatile("barrier":::"memory");st->done_sequence=a[2];
        return;
    }
    const uint32_t entry_sequence=a[2];
    /* Do not borrow memory for a callback that did not launch all three
     * workers. Arrival slots are generation-tagged and written only by their
     * own worker; the master never clears a follower's published arrival. */
    sync[8+tid]=entry_sequence;
    __asm__ volatile("barrier":::"memory");
#endif
#if QDS_PRESERVE_VTCM
    QdVtcmLease *lease=(QdVtcmLease *)(ARENA+QDS_LEASE_STATE);
    volatile uint64_t *timing=(volatile uint64_t *)(ARENA+QDS_LEASE_TIMING);
    volatile uint32_t *lease_clocks=(volatile uint32_t *)(ARENA+QDC_CLOCK_OFFSET);
    _Static_assert(sizeof(QdVtcmLease)<=128,"VTCM lease metadata overlaps timings");
#endif
    if(tid==0) {
        sync[0]=0;sync[1]=sync[2]=sync[3]=0;
        uint32_t err=cfg->magic!=QDS_MAGIC || cfg->version!=1 || !cfg->code_bytes || cfg->code_bytes>0x100000;
#if QDS_ENTRY_GUARD
        uint64_t arrival_deadline=cycles()+2000000000u;
        while(!err && !qds_entry_all_arrived(sync+8,entry_sequence))
            if(cycles()>arrival_deadline)err=10;
#endif
        if(!err)err=qdo_map(map);
#if QDS_PRESERVE_VTCM
        for(uint32_t i=0;i<sizeof *lease/4;i++)((uint32_t *)lease)[i]=0;
        timing[0]=timing[1]=0;
        if(!err) {
            err=clock_raise(lease_clocks);
            uint64_t started=cycles();
            if(!err)err=qds_lease_on_stack(qds_save_vtcm);
            timing[0]=cycles()-started;
            uint32_t ce=clock_restore(lease_clocks);if(!err)err=ce;
            if(err)err+=32;
        }
#endif
        if(!err) {
            volatile uint32_t *w=(volatile uint32_t *)QDS_WORK_L2;
            for(uint32_t i=0;i<sizeof(QdsWork)/4;i++)w[i]=0;
            for(uint32_t p=QDO_BASE;p<QDO_BASE+cfg->code_bytes;p+=32) {
                __asm__ volatile("dcinva(%0)\n icinva(%0)"::"r"(p):"memory");
            }
            __asm__ volatile("isync":::"memory");
        }
        sync[4]=err;
        __asm__ volatile("barrier":::"memory");sync[0]=a[2];
    } else {
        uint64_t end=cycles()+2000000000u;
        while(sync[0]!=a[2])if(cycles()>end)return;
    }
    __asm__ volatile("barrier":::"memory");
    if(!sync[4]) {
        /* L2 mode too: the engine switches worker 0 to a DDR stack only
         * inside its cache lease (qd_l2_cache.h). */
        volatile uint32_t *guard=(volatile uint32_t *)(0x30080000u+tid*0x10000u);
        for(uint32_t i=0;i<32;i++)guard[i]=0x7154c9a3u;
        qds_on_stack((void (*)(uint32_t))QDO_BASE,tid,0x30090000u+tid*0x10000u);
        for(uint32_t i=0;i<32;i++)if(guard[i]!=0x7154c9a3u)sync[4]=9;
    }
    __asm__ volatile("barrier":::"memory");sync[1+tid]=1;
    if(tid==0) {
        uint64_t end=cycles()+2000000000u;
        while(!sync[2] || !sync[3])if(cycles()>end){sync[4]=8;return;}
        uint32_t err=0;
#if QDS_PRESERVE_VTCM
        if(lease->phase==1) {
            err=clock_raise(lease_clocks);
            uint64_t started=cycles();
            if(!err)err=qds_lease_on_stack(qds_restore_vtcm);
            timing[1]=cycles()-started;
            uint32_t ce=clock_restore(lease_clocks);if(!err)err=ce;
            if(err){sync[4]=64+err;}
        } else if(lease->magic==QDVT_MAGIC && lease->phase!=2) {
            err=QDVT_BAD_STATE;sync[4]=64+err;
        }
#endif
        /* A failed memory restore retains the private mapping and snapshot
         * for host-directed recovery. All workers still quiesce normally. */
        if(!err){err=qdo_unmap(map);if(err)sync[4]=err;}
        __asm__ volatile("barrier":::"memory");sync[0]=0xffffffffu;
    } else while(sync[0]!=0xffffffffu) {}
    volatile QdStatus *st=(volatile QdStatus *)(ARENA+QD_STATUS_OFFSET)+tid;
    st->error=sync[4];st->rows=0;st->reserved=QDC_EXITED;
    __asm__ volatile("barrier":::"memory");st->done_sequence=a[2];
}
