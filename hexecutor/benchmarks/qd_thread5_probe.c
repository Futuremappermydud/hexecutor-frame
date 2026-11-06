/* Bounded callback launch proof. No firmware code changes. Optional vector
 * mode temporarily leases the firmware HT3 vector configuration on parked HT5.
 * The stock FLOPS dispatch starts 0..2. Only after all three reach this callback
 * do we select the default callback-only work opcode and resume parked HT5.
 * MODECTL[21] must be set after HT5's done marker before publishing master done.
 */
#include <stdint.h>
#include "../native/qd_protocol.h"
#include "../native/qd_conv.h"
#define ARENA 0xf1800000u
#define WAIT5 (1u<<21)
static uint32_t mode(void){uint32_t v;__asm__ volatile("%0=modectl":"=r"(v));return v;}
static uint64_t cycles(void){uint64_t v;__asm__ volatile("%0=upcycle":"=r"(v));return v;}
static void fence(void){__asm__ volatile("barrier":::"memory");}
#ifdef QDS_TEST_FULL_HVX_CONTEXT
extern uint32_t qds_hvx_context_enter(uint32_t save_area);
extern void qds_hvx_context_leave(uint32_t save_area,uint32_t saved_ssr);
#endif
/* Same enable/context bits as firmware's physical HT3 startup value. Preserve
 * the three vector registers used, and restore exact SSR before returning.
 * The master checks that HT3 is in firmware Wait before requesting this mode. */
__attribute__((noinline)) static void vector_probe(uint32_t saved,volatile uint32_t *out) {
#ifdef QDS_TEST_FULL_HVX_CONTEXT
    saved=qds_hvx_context_enter(ARENA+8192);
#endif
    uint32_t on=(saved&~0xb8000000u)|0x98000000u;
    uint32_t addr=ARENA+4096;
    __asm__ volatile("ssr=%0\n isync\n barrier"::"r"(on):"memory");
    __asm__ volatile(
        "vmem(%0+#0)=v0\n vmem(%0+#1)=v1\n vmem(%0+#2)=v2\n barrier\n"
        "v0=vmem(%0+#3)\n v1=vmem(%0+#4)\n v2.b=vadd(v0.b,v1.b)\n"
        "vmem(%0+#5)=v2\n barrier\n"
        "v0=vmem(%0+#0)\n v1=vmem(%0+#1)\n v2=vmem(%0+#2)\n barrier\n"
        ::"r"(addr):"memory","v0","v1","v2");
#ifdef QDS_TEST_FULL_HVX_CONTEXT
    qds_hvx_context_leave(ARENA+8192,saved);
    uint32_t again=qds_hvx_context_enter(ARENA+16384);
    qds_hvx_context_leave(ARENA+16384,again);
    uint32_t bad=0;
    for(uint32_t i=0;i<4608;i++)
        bad|=((volatile uint8_t *)ARENA)[8192+i]^((volatile uint8_t *)ARENA)[16384+i];
    out[43]=bad?2:1;
#else
    __asm__ volatile("ssr=%0\n isync\n barrier"::"r"(saved):"memory");
#endif
    uint32_t restored;__asm__ volatile("%0=ssr":"=r"(restored));
    out[40]=on;out[41]=restored;fence();out[42]=1;
}
__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid,sp,ssr;
    __asm__ volatile("%0=htid":"=r"(tid));
    if(tid>2 && tid!=5)return;
    volatile uint32_t *a=(volatile uint32_t *)ARENA;
    volatile uint32_t *out=(volatile uint32_t *)(ARENA+2048);
    volatile QdStatus *st=(volatile QdStatus *)(ARENA+QD_STATUS_OFFSET)+tid;
    uint32_t seq=a[2],err=0;
    __asm__ volatile("%0=r29":"=r"(sp));
    __asm__ volatile("%0=ssr":"=r"(ssr));
    out[16+tid*4]=tid;out[17+tid*4]=sp;out[18+tid*4]=ssr;
    out[19+tid*4]=*(volatile uint32_t *)0xf1647084u;
    fence();out[8+tid]=seq;fence();
    if(tid==5 && a[4])vector_probe(ssr,out);
    if(tid==0) {
        uint64_t deadline=cycles()+50000000ull;
        out[0]=0x35544851u;out[1]=mode();out[2]=0;out[3]=0;
        while(out[9]!=seq || out[10]!=seq) {
            if(cycles()>deadline){err=1;break;}
        }
        if(!err && !(mode()&WAIT5))err=2;
        if(!err && *(volatile uint32_t *)0xf1647084u==2)err=3;
        if(!err && a[4] && !(mode()&(1u<<19)))err=5;
        if(!err) {
            volatile uint32_t *work=(volatile uint32_t *)0xf0b12000u;
            uint32_t saved=work[0];out[4]=saved;
            work[0]=0;fence();
            uint32_t mask=1u<<5;
            __asm__ volatile("resume(%0)"::"r"(mask):"memory");
            out[2]=1;
            while(st[5].done_sequence!=seq || !(mode()&WAIT5)) {
                if(cycles()>deadline){err=4;break;}
            }
            /* Preserve the callback-only opcode if HT5 may still enter late.
             * The host retains all session memory on this error. */
            if(!err){work[0]=saved;fence();out[3]=mode();}
        }
        out[5]=err;
    }
    st->error=err;st->rows=tid;st->reserved=QDC_EXITED;
    fence();st->done_sequence=seq;
}
