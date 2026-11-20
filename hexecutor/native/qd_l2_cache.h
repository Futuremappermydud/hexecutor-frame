/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_L2_CACHE_H
#define QD_L2_CACHE_H
/* Frame V75, L2CFG=5: the whole 1 MiB L2 array becomes cache and no L2-TCM
 * address is usable while the lease is active (0x300c6d00 SMMU-faulted).
 * QDS_L2_CACHE=1 (L2CFG=1): only the upper 512 KiB (0x30080000..) become
 * cache (L2_HANDOFF.md section 5: fields 1-4 hold 512 KiB); the lower half
 * stays TCM and live through the lease (expanded block tables, conv near).
 * Its contents are checked across both mode switches.
 *
 * The lease is per prepared run on worker 0 only. Everything outside the
 * lease (bootstrap, preflight, followers) keeps the baseline TCM stacks.
 * Inside the lease worker 0 runs on a C7 (WB, L2-cacheable) private DDR
 * stack. The stack is entered with L2CFG=0, so the code that runs on it
 * before begin/after end must be scalar-only on the stack; HVX here touches
 * only TCM and the uncached backup. With L2 as cache, L2 is the coherence
 * point for scalar L1 and HVX accesses to C7 memory, as in normal QuRT.
 *
 * Attributes (qurt_types.h): 0 WB non-L2, 5 WT L2, 6 uncached, 7 WB L2. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#define QDS_L2_ELEMENTWISE 0xa7050000u
#define QDS_L2_NEAR        0xa7060000u
#define QDS_L2_BLOCKS      0xa7100000u
#if QDS_L2_CACHE==1
/* Lower, still-TCM half during the lease:
 *   0x30000000..0x30002000  native tables/ABI (unchanged)
 *   0x30002000..0x3003c000  hot bound program (235,208 bytes used)
 *   0x3003c000..0x3006c000  lease stacks: worker 0, followers 1 and 2 (64 KiB each)
 *   0x3006c000..0x30073e00  helper conv near slots (2 x 0x3e00)
 *   0x30074000..0x30074200  QdsWork (QDS_WORK_L2), polled by the followers
 *   0x30078000..0x30080000  conv near (as without L2)
 * The expanded block tables go to page 7 (QDS_L2_BLOCKS). The stacks are
 * left out of the lower-half checks across the mode switches. */
#define QDS_L2_HOT         0x30002000u
#define QDS_L2_HOT_BYTES   0x3a000u
#define QDS_L2_STACK_BASE  0x3003c000u
#define QDS_L2_STACK_TOP   0x3004c000u
#if QDS_PREPARED_THREADS==4
#define QDS_L2_FSTACK(tid) ((tid)==3u?0xa70b0000u:(0x3004c000u+((tid)-1u)*0x10000u))
#else
#define QDS_L2_FSTACK(tid) (0x3004c000u+((tid)-1u)*0x10000u)
#endif
#define QDS_L2_PAR_0       0x3006c000u
#define QDS_L2_PAR_1       0x30070000u
#define QDS_L2_PAR_2       0xa70c0000u
#define QDS_L2_LIVE_SKIP   0x3003c000u
#define QDS_L2_LIVE_SKIP_END 0x3006c000u
#define QDS_L2_WORK_SKIP   0x30074000u
#define QDS_L2_WORK_SKIP_END 0x30078000u
#else
#define QDS_L2_HOT         0xa7180000u
#define QDS_L2_HOT_BYTES   0x3f000u
#define QDS_L2_STACK_BASE  0xa7030000u
#define QDS_L2_STACK_TOP   0xa7040000u
/* QDS_PREPARED_THREADS=3: the helper conv near slots and the two followers'
 * lease stacks (64 KiB each, bottom 128 B guarded by the engine) on page 7. */
#define QDS_L2_PAR_0       0xa7070000u
#define QDS_L2_PAR_1       0xa7074000u
#if QDS_PREPARED_THREADS==4
#define QDS_L2_FSTACK(tid) ((tid)==3u?0xa70b0000u:(0xa7080000u+((tid)-1u)*0x10000u))
#else
#define QDS_L2_FSTACK(tid) (0xa7080000u+((tid)-1u)*0x10000u)
#endif
#define QDS_L2_PAR_2       0xa70c0000u
#endif
#define QDS_L2_BACKUP      0xf1c00000u
#define QDS_L2_TCM         0x30000000u
#define QDS_L2_TCM_BYTES   0x100000u
#define QDS_L2_FIELD       ((uint32_t)QDS_L2_CACHE)
#if QDS_L2_CACHE!=1 && QDS_L2_CACHE!=5
#error QDS_L2_CACHE must be 1 or 5
#endif
/* The part of TCM that becomes cache: saved before, restored after. */
#define QDS_L2_SAVE        (QDS_L2_CACHE==1?0x30080000u:QDS_L2_TCM)
#define QDS_L2_SAVE_BYTES  (QDS_L2_CACHE==1?0x80000u:QDS_L2_TCM_BYTES)
/* CCR bits OR-ed in for the lease only (17 HFD, 19 HFDL2: hardware data
 * prefetch to L1/L2). Firmware runs with 0x150049 (instruction prefetch only). */
#ifndef QDS_L2_CCR_SET
#define QDS_L2_CCR_SET     0u /* (1u<<17)|(1u<<19) measured: no change */
#endif
static uint32_t qds_l2_syscfg(void){uint32_t v;__asm__ volatile("%0=syscfg":"=r"(v));return v;}
static uint32_t qds_l2_ccr(void){uint32_t v;__asm__ volatile("%0=ccr":"=r"(v));return v;}
static void qds_l2_setcfg(uint32_t v){
    __asm__ volatile("isync\n\tsyncht\n\tsyscfg=%0\n\tisync"::"r"(v):"memory");
}
/* Publish dirty scalar L1 lines of TCM (worker stacks are attr-7 TCM), so the
 * vector save sees them and no TCM line is evicted while L2 is cache. */
static void qds_l2_l1_range(uint32_t base,uint32_t bytes,int clean){
    for(uint32_t a=base;a<base+bytes;a+=32) {
        if(clean)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
        else __asm__ volatile("dcinva(%0)"::"r"(a):"memory");
    }
    __asm__ volatile("barrier":::"memory");
}
static void qds_l2_l1_tcm(int clean){qds_l2_l1_range(QDS_L2_TCM,QDS_L2_TCM_BYTES,clean);}
/* The backup mirrors TCM: backup offset == TCM offset. */
static void qds_l2_vcopy_n(uint32_t dst,uint32_t src,uint32_t bytes){
    HVX_Vector *d=(HVX_Vector *)dst;const HVX_Vector *s=(const HVX_Vector *)src;
    for(uint32_t i=0;i<bytes/128u;i+=4){
        HVX_Vector a=s[i],b=s[i+1],c=s[i+2],e=s[i+3];
        d[i]=a;d[i+1]=b;d[i+2]=c;d[i+3]=e;
    }
    __asm__ volatile("barrier":::"memory");
}
static void qds_l2_vcopy(uint32_t dst,uint32_t src){qds_l2_vcopy_n(dst,src,QDS_L2_TCM_BYTES);}
static uint32_t qds_l2_vdiff_n(uint32_t off,uint32_t bytes){
    const HVX_Vector *a=(const HVX_Vector *)(QDS_L2_TCM+off),*b=(const HVX_Vector *)(QDS_L2_BACKUP+off);
    HVX_Vector acc=Q6_V_vzero();
    for(uint32_t i=0;i<bytes/128u;i++)acc=Q6_V_vor_VV(acc,Q6_V_vxor_VV(a[i],b[i]));
    uint32_t bad=0;
    for(uint32_t i=0;i<32u;i++)bad|=(uint32_t)Q6_R_vextract_VR(acc,i*4u);
    return bad;
}
static uint32_t qds_l2_vdiff(void){
#if QDS_L2_CACHE==1
    return qds_l2_vdiff_n(0,QDS_L2_LIVE_SKIP-QDS_L2_TCM) |
           qds_l2_vdiff_n(QDS_L2_LIVE_SKIP_END-QDS_L2_TCM,QDS_L2_WORK_SKIP-QDS_L2_LIVE_SKIP_END) |
           qds_l2_vdiff_n(QDS_L2_WORK_SKIP_END-QDS_L2_TCM,QDS_L2_TCM+QDS_L2_TCM_BYTES-QDS_L2_WORK_SKIP_END);
#else
    return qds_l2_vdiff_n(0,QDS_L2_TCM_BYTES);
#endif
}
/* QDS_L2_SIG: check the mode switches with a signature of live TCM instead of comparing
 * against the uncached backup. Reading TCM needs no DDR traffic, so the lower half never
 * goes to the backup. Word sum and XOR over all vectors, folded to 64 bits; no HVX on the
 * stack (it may be entered before L2 is cache). */
#ifdef QDS_L2_SIG
static uint64_t qds_l2_sig_range(uint64_t h,uint32_t base,uint32_t bytes){
    const HVX_Vector *a=(const HVX_Vector *)base;
    HVX_Vector x=Q6_V_vzero(),t=Q6_V_vzero();
    for(uint32_t i=0;i<bytes/128u;i++){
        HVX_Vector v=a[i];
        x=Q6_V_vxor_VV(x,v);t=Q6_Vw_vadd_VwVw(Q6_Vw_vsub_VwVw(t,x),v);
    }
    for(uint32_t i=0;i<32u;i++){
        h=(h^(uint32_t)Q6_R_vextract_VR(x,i*4u))*0x100000001b3ull;
        h=(h^(uint32_t)Q6_R_vextract_VR(t,i*4u))*0x100000001b3ull;
    }
    return h;
}
#if QDS_L2_CACHE==1
static uint64_t qds_l2_sig_lower(void){
    uint64_t h=0xcbf29ce484222325ull;
    h=qds_l2_sig_range(h,QDS_L2_TCM,QDS_L2_LIVE_SKIP-QDS_L2_TCM);
    h=qds_l2_sig_range(h,QDS_L2_LIVE_SKIP_END,QDS_L2_WORK_SKIP-QDS_L2_LIVE_SKIP_END);
    return qds_l2_sig_range(h,QDS_L2_WORK_SKIP_END,QDS_L2_SAVE-QDS_L2_WORK_SKIP_END);
}
#endif
#endif
#if QDS_L2_CACHE==1
static uint32_t qds_l2_lower_diff(void){
    return qds_l2_vdiff_n(0,QDS_L2_LIVE_SKIP-QDS_L2_TCM) |
           qds_l2_vdiff_n(QDS_L2_LIVE_SKIP_END-QDS_L2_TCM,QDS_L2_WORK_SKIP-QDS_L2_LIVE_SKIP_END) |
           qds_l2_vdiff_n(QDS_L2_WORK_SKIP_END-QDS_L2_TCM,QDS_L2_SAVE-QDS_L2_WORK_SKIP_END);
}
#endif
/* Optional: remap private DDR page 1 (graph pools, spill, I/O; uncached
 * C6 outside the lease) as C7 while L2 is cache. Entering from C6 leaves no
 * L1 lines; leaving cleans L1 for the page before C6 is restored, then the
 * caller's qds_l2_end flushes L2 to DDR for the host. */
#ifndef QDS_L2_POOL_C7
#define QDS_L2_POOL_C7 1
#endif
#define QDS_L2_POOL_TLB  65u
#define QDS_L2_POOL_VA   0xa1000000u
static uint64_t qds_l2_pool_entry(uint32_t attr){
    uint64_t e;__asm__ volatile("%0=tlbr(%1)":"=r"(e):"r"(QDS_L2_POOL_TLB):"memory");
    if(((uint32_t)(e>>32)&0xfffffu)!=(QDS_L2_POOL_VA>>12))return 0;
    return (e&~(0xfull<<24))|((uint64_t)attr<<24);
}
/* Leaving: every L1 line of the page must go. flush_first..flush_end limits that
 * to what the run can touch (0, 0: the whole page, 524288 cache operations). */
static uint32_t qds_l2_pool_cache_range(int on,uint32_t flush_first,uint32_t flush_end){
    uint64_t e=qds_l2_pool_entry(on?7u:6u);
    if(!e)return 1;
    if(!on){
        if(!flush_end){flush_first=QDS_L2_POOL_VA;flush_end=QDS_L2_POOL_VA+0x1000000u;}
        for(uint32_t a=flush_first&~31u;a<flush_end;a+=32)
            __asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
        __asm__ volatile("barrier\n\tsyncht":::"memory");
    }
    __asm__ volatile("tlbw(%0,%1)\n\tisync"::"r"(e),"r"(QDS_L2_POOL_TLB):"memory");
    return 0;
}
static uint32_t qds_l2_pool_cache(int on){return qds_l2_pool_cache_range(on,0,0);}
typedef struct {uint32_t syscfg,ccr,active;
#ifdef QDS_L2_SIG
    uint64_t sig_upper,sig_lower;
#endif
} QdsL2Cache;
static uint32_t qds_l2_begin(QdsL2Cache *s){
    s->active=0;s->syscfg=qds_l2_syscfg();s->ccr=qds_l2_ccr();
    if(((s->syscfg>>16)&7u)!=0u || s->syscfg!=0x0a80a057u || s->ccr!=0x150049u)return 1;
#if QDS_L2_CACHE==1 && defined(QDS_L2_SIG)
    qds_l2_l1_tcm(1);
    s->sig_upper=qds_l2_sig_range(0xcbf29ce484222325ull,QDS_L2_SAVE,QDS_L2_SAVE_BYTES);
    s->sig_lower=qds_l2_sig_lower();
    qds_l2_vcopy_n(QDS_L2_BACKUP+(QDS_L2_SAVE-QDS_L2_TCM),QDS_L2_SAVE,QDS_L2_SAVE_BYTES);
#elif QDS_L2_CACHE==1 && defined(QDS_L2_NO_VERIFY)
    /* Opt-in: only the half that becomes cache is saved, nothing is compared. */
    qds_l2_l1_range(QDS_L2_SAVE,QDS_L2_SAVE_BYTES,1);
    qds_l2_vcopy_n(QDS_L2_BACKUP+(QDS_L2_SAVE-QDS_L2_TCM),QDS_L2_SAVE,QDS_L2_SAVE_BYTES);
#else
    qds_l2_l1_tcm(1);
    qds_l2_vcopy(QDS_L2_BACKUP,QDS_L2_TCM);
#endif
    uint32_t cleared=s->ccr&~(15u<<16);
    __asm__ volatile("ccr=%0\n\tisync"::"r"(cleared):"memory");
    qds_l2_setcfg(s->syscfg&~(7u<<16));
    __asm__ volatile("l2kill\n\tsyncht":::"memory");
    qds_l2_setcfg(s->syscfg|(QDS_L2_FIELD<<16));
    s->active=1;
    uint32_t bad=qds_l2_syscfg()!=(s->syscfg|(QDS_L2_FIELD<<16));
#if QDS_L2_CACHE==1 && defined(QDS_L2_SIG)
    bad|=qds_l2_sig_lower()!=s->sig_lower;
#elif QDS_L2_CACHE==1 && !defined(QDS_L2_NO_VERIFY)
    /* The live lower half must have come through l2kill and the switch. */
    bad|=qds_l2_lower_diff()!=0;
#endif
    __asm__ volatile("ccr=%0\n\tisync"::"r"(s->ccr|QDS_L2_CCR_SET):"memory");
    return bad;
}
static uint32_t qds_l2_end(QdsL2Cache *s){
    if(!s->active)return 0;
    uint32_t cleared=s->ccr&~(15u<<16);
    __asm__ volatile("ccr=%0\n\tisync"::"r"(cleared):"memory");
#if QDS_L2_CACHE==1 && defined(QDS_L2_SIG)
    qds_l2_l1_range(QDS_L2_TCM,QDS_L2_SAVE-QDS_L2_TCM,1);
    s->sig_lower=qds_l2_sig_lower();
#elif QDS_L2_CACHE==1 && !defined(QDS_L2_NO_VERIFY)
    /* Snapshot the live lower half (scalar lines published first) so the
     * switch back can be checked like the restored upper half. */
    qds_l2_l1_range(QDS_L2_TCM,QDS_L2_SAVE-QDS_L2_TCM,1);
    qds_l2_vcopy_n(QDS_L2_BACKUP,QDS_L2_TCM,QDS_L2_SAVE-QDS_L2_TCM);
#endif
    /* One statement: no stack access between the flush and TCM mode. */
    __asm__ volatile("syncht\n\tl2gcleaninv\n\tsyncht\n\tisync\n\tsyncht\n\tsyscfg=%0\n\tisync"
                     ::"r"(s->syscfg):"memory");
    qds_l2_vcopy_n(QDS_L2_SAVE,QDS_L2_BACKUP+(QDS_L2_SAVE-QDS_L2_TCM),QDS_L2_SAVE_BYTES);
#if QDS_L2_CACHE==1 && defined(QDS_L2_SIG)
    uint32_t bad=qds_l2_sig_lower()!=s->sig_lower ||
                 qds_l2_sig_range(0xcbf29ce484222325ull,QDS_L2_SAVE,QDS_L2_SAVE_BYTES)!=s->sig_upper;
#elif defined(QDS_L2_NO_VERIFY)
    uint32_t bad=0;
#else
    uint32_t bad=qds_l2_vdiff();
#endif
    qds_l2_l1_range(QDS_L2_SAVE,QDS_L2_SAVE_BYTES,0);
    __asm__ volatile("ccr=%0\n\tisync"::"r"(s->ccr):"memory");
    s->active=0;
    return bad || qds_l2_syscfg()!=s->syscfg || qds_l2_ccr()!=s->ccr;
}
#endif
