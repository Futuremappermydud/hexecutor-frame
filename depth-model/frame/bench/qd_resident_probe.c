/* Decoder-only one-shot probe. Uses the established arena and loader. It does
 * not map the graph pools, touch VTCM/HMX, or execute a graph kernel. */
#include "../qd_protocol.h"
#include "../qd_conv.h"
#include "../qd_resident.h"

#define ARENA 0xf1800000u
#define PROBE_MAGIC 0x31505251u /* QRP1 */
static uint64_t cycles(void) {
    uint64_t value;__asm__ volatile("%0 = upcycle" : "=r"(value));return value;
}
__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    if(tid>=3)return;
    const volatile uint32_t *cmd=(const volatile uint32_t *)ARENA;
    uint32_t sequence=cmd[2];
    volatile uint32_t *out=(volatile uint32_t *)(ARENA+2048+64*tid);
    for(uint32_t i=0;i<16;i++)out[i]=0;
    out[0]=PROBE_MAGIC;out[10]=tid;
    if(tid==0) {
        QdrImage view;
        QdrStatus status={QDR_BAD_HEADER,QDR_NONE,QDR_NONE,0};
        uint32_t off=cmd[3],bytes=cmd[4],mode=cmd[5];
        uint64_t start=cycles();
        if(cmd[0]==PROBE_MAGIC && cmd[1]==1 && sequence && cmd[7]==0 && mode<=1 &&
           off>=0x10000 && !(off&127) && off<=QD_ARENA_BYTES && bytes<=QD_ARENA_BYTES-off) {
            if(mode==0) {
                if(!qdr_open((const void *)(ARENA+off),bytes,&view,&status)) {
                    out[7]=view.count[QDR_TENSORS];out[8]=view.count[QDR_COMMANDS];
                }
            } else {
                qdr_dispatch((const void *)(ARENA+off),bytes,(const QdrKernel *)0,0,
                             cmd[6],(QdrCancel)0,(void *)0,&status);
            }
        }
        uint64_t elapsed=cycles()-start;
        out[1]=status.code;out[2]=status.command;out[3]=status.tensor;out[4]=status.detail;
        out[5]=(uint32_t)elapsed;out[6]=(uint32_t)(elapsed>>32);out[9]=bytes;
    }
    volatile QdStatus *s=(volatile QdStatus *)(ARENA+QD_STATUS_OFFSET+tid*sizeof(QdStatus));
    s->error=0;s->rows=0;s->reserved=QDC_EXITED;
    __asm__ volatile("barrier" ::: "memory");
    s->done_sequence=sequence;
    __asm__ volatile("barrier" ::: "memory");
}
