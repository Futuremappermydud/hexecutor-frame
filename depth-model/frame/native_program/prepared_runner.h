#ifndef NPRG2_PREPARED_RUNNER_H
#define NPRG2_PREPARED_RUNNER_H

#include "prepared_executor.h"

#define NPRG2_RUNNER_POOL1_ACTIVE_BYTES 0x00780000u
#define NPRG2_RUNNER_POOL1_TOTAL_BYTES  0x00800000u

enum { NPRG2_RUNNER_OK=0,NPRG2_RUNNER_ARGUMENT=1,NPRG2_RUNNER_MEMORY=2,
       NPRG2_RUNNER_ALIAS=3,NPRG2_RUNNER_READER=4,NPRG2_RUNNER_BIND=5,
       NPRG2_RUNNER_NOT_READY=6 };

typedef struct {
    Nprg2Program program;
    Nprg2Bound bound;
    Nprg2Pool pools[7];
    uint32_t *setup_expanded_blocks;
    uint32_t *l2_expanded_blocks;
    uint32_t l2_expanded_bytes, l2_hot_bytes;
    uint8_t *l2_hot;
    /* Executor scratch regions 1.. passed with every request (zero = absent).
     * Set with nprg2_runner_set_scratch_region after open. */
    Nprg2ExecBuffer extra_scratch[NPRG2_EXEC_SCRATCH_REGIONS-1u];
    /* Optional profile and preflight reuse for every request (set after open). */
    Nprg2ExecProfile *profile;
    Nprg2Parallel *parallel;
    uint8_t reuse_preflight;
    /* Conv epilogue fusion plan (NULL = off), see nprg2_runner_build_fuse. */
    Nprg2FusePlan fuse_plan;
    Nprg2FusePlan *fuse;
    uint8_t ready;
} Nprg2PreparedRunner;

/* Opens/validates NPRG in setup memory, copies the expanded block table once
 * into the caller's L2 arena, and creates bound hot tables. It performs no
 * graph-pool writes. The caller certifies pool1's VTCM lease and supplies the
 * real memory pointers/intervals. */
int nprg2_runner_open(Nprg2PreparedRunner *runner,
                      const void *prepared_raw,uint32_t prepared_bytes,
                      const Nprg2ExpectedHashes *expected,
                      const Nprg2Pool pools[7],Nprg2Allocator allocator,
                      void *l2_expanded_blocks,uint32_t l2_expanded_capacity,
                      void *l2_hot,uint32_t l2_hot_capacity,
                      uint32_t l2_hot_available_bytes,
                      uint8_t pool1_is_verified_vtcm);

/* Profile for every later preflight/execute (NULL = none). With reuse set,
 * execute skips its repeated preflight when it can prove from the stats of the
 * preceding nprg2_runner_preflight (same stats object, same request) that the
 * interval was just validated; otherwise it preflights as usual. The caller
 * must not change graph metadata between the two calls (pool bytes may change). */
int nprg2_runner_set_profile(Nprg2PreparedRunner *runner,Nprg2ExecProfile *profile,int reuse_preflight);

/* Builds the conv epilogue fusion plan from the bound schedule (analysis only,
 * no graph-pool access) and enables it for every later request. Pairs the
 * conv handler's fuse hook rejects run unfused. Returns OK with zero pairs
 * when nothing qualifies and NOT_READY-free UNSUPPORTED codes from
 * nprg2_fuse_plan_build otherwise (fusion then stays off). */
int nprg2_runner_build_fuse(Nprg2PreparedRunner *runner,Nprg2Allocator allocator);
/* Turns a built plan on or off (A/B switch). */
int nprg2_runner_enable_fuse(Nprg2PreparedRunner *runner,int enable);

/* Executes exactly [command_first, command_first+command_count). The executor
 * preflights the complete requested interval before the first graph-pool
 * write; unsupported operations fail closed without a legacy route. */
int nprg2_runner_execute(Nprg2PreparedRunner *runner,
                         const Nprg2Executor *executor,
                         uint32_t command_first,uint32_t command_count,
                         uint32_t pool1_writable_bytes,
                         const Nprg2ExecBuffer *inputs,uint32_t input_count,
                         const Nprg2ExecBuffer *outputs,uint32_t output_count,
                         void *user,void *scratch,uint32_t scratch_bytes,
                         Nprg2ExecCancel cancelled,void *cancel_ctx,
                         Nprg2ExecStats *stats);

/* Preflights the same runner request without any graph-pool writes. Use this
 * before leasing/copying pool1 into VTCM; execute intentionally preflights
 * again after the boundary operation. */
int nprg2_runner_preflight(Nprg2PreparedRunner *runner,
                           const Nprg2Executor *executor,
                           uint32_t command_first,uint32_t command_count,
                           uint32_t pool1_writable_bytes,
                           const Nprg2ExecBuffer *inputs,uint32_t input_count,
                           const Nprg2ExecBuffer *outputs,uint32_t output_count,
                           void *user,void *scratch,uint32_t scratch_bytes,
                           Nprg2ExecStats *stats);

/* Sets executor scratch region `region` (1..NPRG2_EXEC_SCRATCH_REGIONS-1) for
 * later preflight/execute calls; bytes=0 removes it. The executor checks it
 * against graph memory and the other regions on every request. */
int nprg2_runner_set_scratch_region(Nprg2PreparedRunner *runner,uint8_t region,
                                    void *scratch,uint32_t scratch_bytes);

/* Parallel hook for every later request (NULL = serial). */
int nprg2_runner_set_parallel(Nprg2PreparedRunner *runner,Nprg2Parallel *parallel);

void nprg2_runner_close(Nprg2PreparedRunner *runner);

#endif
