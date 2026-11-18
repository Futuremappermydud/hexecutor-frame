#ifndef QD_GRAPH_ENTRY_H
#define QD_GRAPH_ENTRY_H
#include <stdint.h>
#include "qd_conv.h"
#ifdef __hexagon__
#define QDS_ENTRY_BARRIER() __asm__ volatile("barrier" ::: "memory")
#else
#define QDS_ENTRY_BARRIER() __asm__ volatile("" ::: "memory")
#endif

/* Validate the published host command before mapping DDR, borrowing VTCM, or
 * starting followers. Sequence zero is the unpublished setup state. */
static inline int qds_initial_command_valid_flags(const volatile uint32_t *command,
                                                 uint32_t run_op, uint32_t allowed_flags) {
    if(!command)return 0;
    uint32_t sequence=command[2];
    QDS_ENTRY_BARRIER();
    int valid=sequence && command[0]==QDC_MAGIC && command[1]==QDC_VERSION &&
              command[3]==run_op && command[4]<=command[5] && !(command[6]&~allowed_flags);
    QDS_ENTRY_BARRIER();
    return valid && command[2]==sequence;
}

static inline int qds_initial_command_valid(const volatile uint32_t *command, uint32_t run_op) {
    return qds_initial_command_valid_flags(command, run_op, 1u);
}

static inline int qds_entry_all_arrived(const volatile uint32_t *arrivals,
                                      uint32_t sequence, uint32_t workers) {
    QDS_ENTRY_BARRIER();
    if(!sequence || workers<1 || workers>4)return 0;
    for(uint32_t i=0;i<workers;i++)if(arrivals[i]!=sequence)return 0;
    return 1;
}
#undef QDS_ENTRY_BARRIER
#endif
