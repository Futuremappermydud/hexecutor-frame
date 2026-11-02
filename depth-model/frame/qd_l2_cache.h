#ifndef QD_L2_CACHE_H
#define QD_L2_CACHE_H
/* Frame V75, L2CFG=5: the whole 1 MiB L2 array becomes cache and no L2-TCM
 * address is usable while the lease is active (0x300c6d00 SMMU-faulted).
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
#define QDS_L2_STACK_BASE  0xa7030000u
#define QDS_L2_STACK_TOP   0xa7040000u
#define QDS_L2_ELEMENTWISE 0xa7050000u
#define QDS_L2_NEAR        0xa7060000u
/* QDS_PREPARED_THREADS=3: the helper conv near slots and the two followers'
 * lease stacks (64 KiB each, bottom 128 B guarded by the engine) on page 7. */
#define QDS_L2_PAR_0       0xa7070000u
#define QDS_L2_PAR_1       0xa7074000u
#define QDS_L2_FSTACK(tid) (0xa7080000u+((tid)-1u)*0x10000u)
#define QDS_L2_BACKUP      0xf1c00000u
#define QDS_L2_TCM         0x30000000u
#define QDS_L2_TCM_BYTES   0x100000u
#define QDS_L2_FIELD       5u
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
static void qds_l2_l1_tcm(int clean){
    for(uint32_t a=QDS_L2_TCM;a<QDS_L2_TCM+QDS_L2_TCM_BYTES;a+=32) {
        if(clean)__asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
        else __asm__ volatile("dcinva(%0)"::"r"(a):"memory");
    }
    __asm__ volatile("barrier":::"memory");
}
static void qds_l2_vcopy(uint32_t dst,uint32_t src){
    HVX_Vector *d=(HVX_Vector *)dst;const HVX_Vector *s=(const HVX_Vector *)src;
    for(uint32_t i=0;i<QDS_L2_TCM_BYTES/128u;i+=4){
        HVX_Vector a=s[i],b=s[i+1],c=s[i+2],e=s[i+3];
        d[i]=a;d[i+1]=b;d[i+2]=c;d[i+3]=e;
    }
    __asm__ volatile("barrier":::"memory");
}
static uint32_t qds_l2_vdiff(void){
    const HVX_Vector *a=(const HVX_Vector *)QDS_L2_TCM,*b=(const HVX_Vector *)QDS_L2_BACKUP;
    HVX_Vector acc=Q6_V_vzero();
    for(uint32_t i=0;i<QDS_L2_TCM_BYTES/128u;i++)acc=Q6_V_vor_VV(acc,Q6_V_vxor_VV(a[i],b[i]));
    uint32_t bad=0;
    for(uint32_t i=0;i<32u;i++)bad|=(uint32_t)Q6_R_vextract_VR(acc,i*4u);
    return bad;
}
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
static uint32_t qds_l2_pool_cache(int on){
    uint64_t e=qds_l2_pool_entry(on?7u:6u);
    if(!e)return 1;
    if(!on){
        for(uint32_t a=QDS_L2_POOL_VA;a<QDS_L2_POOL_VA+0x1000000u;a+=32)
            __asm__ volatile("dccleaninva(%0)"::"r"(a):"memory");
        __asm__ volatile("barrier\n\tsyncht":::"memory");
    }
    __asm__ volatile("tlbw(%0,%1)\n\tisync"::"r"(e),"r"(QDS_L2_POOL_TLB):"memory");
    return 0;
}
typedef struct {uint32_t syscfg,ccr,active;} QdsL2Cache;
static uint32_t qds_l2_begin(QdsL2Cache *s){
    s->active=0;s->syscfg=qds_l2_syscfg();s->ccr=qds_l2_ccr();
    if(((s->syscfg>>16)&7u)!=0u || s->syscfg!=0x0a80a057u || s->ccr!=0x150049u)return 1;
    qds_l2_l1_tcm(1);
    qds_l2_vcopy(QDS_L2_BACKUP,QDS_L2_TCM);
    uint32_t cleared=s->ccr&~(15u<<16);
    __asm__ volatile("ccr=%0\n\tisync"::"r"(cleared):"memory");
    qds_l2_setcfg(s->syscfg&~(7u<<16));
    __asm__ volatile("l2kill\n\tsyncht":::"memory");
    qds_l2_setcfg(s->syscfg|(QDS_L2_FIELD<<16));
    s->active=1;
    uint32_t bad=qds_l2_syscfg()!=(s->syscfg|(QDS_L2_FIELD<<16));
    __asm__ volatile("ccr=%0\n\tisync"::"r"(s->ccr|QDS_L2_CCR_SET):"memory");
    return bad;
}
static uint32_t qds_l2_end(QdsL2Cache *s){
    if(!s->active)return 0;
    uint32_t cleared=s->ccr&~(15u<<16);
    __asm__ volatile("ccr=%0\n\tisync"::"r"(cleared):"memory");
    /* One statement: no stack access between the flush and TCM mode. */
    __asm__ volatile("syncht\n\tl2gcleaninv\n\tsyncht\n\tisync\n\tsyncht\n\tsyscfg=%0\n\tisync"
                     ::"r"(s->syscfg):"memory");
    qds_l2_vcopy(QDS_L2_TCM,QDS_L2_BACKUP);
    uint32_t bad=qds_l2_vdiff();
    qds_l2_l1_tcm(0);
    __asm__ volatile("ccr=%0\n\tisync"::"r"(s->ccr):"memory");
    s->active=0;
    return bad || qds_l2_syscfg()!=s->syscfg || qds_l2_ccr()!=s->ccr;
}
#endif
