#ifndef NPRG2_PREPARED_FUSE_H
#define NPRG2_PREPARED_FUSE_H

#include "nprg2_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Schedule-level conv -> elementwise epilogue fusion plan (2026-11-01).
 *
 * A pair (conv, epilogue) says: the u8 blocked output of HMX conv command
 * `conv` is the (only) conv-produced input of the later REQUANT_U8/70 or
 * ADD_U8/13 command `epi`, both tensors have identical geometry and block
 * tables, and moving the epilogue's reads and writes from its place in the
 * schedule up to the conv is invisible to every command in between and to
 * the other pairs. If `dead`, nothing reads the conv's raw output afterwards
 * (before it is fully overwritten), so a fused conv need not store it.
 * The plan only encodes schedule facts; whether a handler can execute a pair
 * exactly is decided per preflight (Nprg2ExecHandler.fuse) and recorded in
 * `active`. A pair is effective only for requests whose interval contains
 * both commands. */
typedef struct { uint32_t conv, epi; uint8_t operand, dead, active, pad; } Nprg2FusePair;
typedef struct {
    Nprg2FusePair *pairs;
    uint32_t count, commands;
    /* Per command: 0, or (pair+1) for the conv, 0x8000|(pair+1) for the epilogue. */
    uint16_t *role;
    Nprg2Allocator allocator;
} Nprg2FusePlan;

enum { NPRG2_FUSE_OK=0, NPRG2_FUSE_ARGUMENT=1, NPRG2_FUSE_MEMORY=2, NPRG2_FUSE_UNSUPPORTED=3 };
#define NPRG2_FUSE_EPI 0x8000u

/* Analyses the bound schedule (pool 1 block liveness, no pool access) and
 * fills the plan; zero pairs is a valid result. UNSUPPORTED (plan left empty)
 * if the schedule uses a pool-1 access the analysis cannot model. */
int nprg2_fuse_plan_build(Nprg2FusePlan *plan, const Nprg2Program *program, const Nprg2Bound *bound,
                          Nprg2Allocator allocator);
void nprg2_fuse_plan_free(Nprg2FusePlan *plan);

#ifdef __cplusplus
}
#endif
#endif
