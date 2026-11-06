#ifndef NPRG2_PREPARED_EXECUTOR_H
#define NPRG2_PREPARED_EXECUTOR_H

#include "nprg2_runtime.h"
#include "prepared_fuse.h"
#include "prepared_dma_ring.h"

#ifndef NPRG2_PAR_MAX_PARTS
#define NPRG2_PAR_MAX_PARTS 3u
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NPRG2_EXEC_REGISTRY_CAPACITY 128u

enum {
    NPRG2_EXEC_OK=0, NPRG2_EXEC_ARGUMENT=1, NPRG2_EXEC_REGISTRY_FULL=2,
    NPRG2_EXEC_DUPLICATE=3, NPRG2_EXEC_UNSUPPORTED=4,
    NPRG2_EXEC_REJECTED=5, NPRG2_EXEC_CANCELLED=6, NPRG2_EXEC_HANDLER_ERROR=7
};

typedef struct Nprg2Executor Nprg2Executor;

/* Optional data-parallel hook (QDS_PREPARED_THREADS). A handler that can split
 * its work into independent parts asks nprg2_parallel_parts() how many parts
 * it may use (1 when no hook, a one-part hook, or the workers are not ready
 * right now) and then calls nprg2_parallel_for(), which runs fn(ctx, part,
 * parts) for every part in [0,parts) exactly once and returns when all have
 * finished. The parts must write disjoint bytes and may only read memory that
 * no part writes; then any execution order, including the serial fallback
 * below, gives identical results. Part 0 always runs on the calling thread.
 *
 *   max_parts   parts the hook can run concurrently including the caller
 *               (0/1 = serial). Handlers never ask for more.
 *   ready       optional; called by nprg2_parallel_parts, nonzero = workers
 *               idle and leases valid. When it returns 0 the handler runs
 *               serially (one part).
 *   run         optional; runs the parts concurrently and returns 1, or
 *               returns 0 without having run anything (not idle etc.), in
 *               which case nprg2_parallel_for runs the parts on the caller.
 *   order       serial execution order of the parts when there is no run hook
 *               or it declined: 0 ascending, 1 descending, 2 pseudo-random
 *               (seeded from seed). Host tests use 1 and 2 with max_parts 3 to
 *               prove the parts are order independent.
 * The counters accumulate until the owner clears them: per kind, parallel
 * jobs, serial jobs of more than one part, and parts executed. */
typedef void (*Nprg2ParallelFn)(void *ctx, uint32_t part, uint32_t parts);
enum { NPRG2_PAR_CONV=0, NPRG2_PAR_EW=1, NPRG2_PAR_DM=2, NPRG2_PAR_KINDS=3 };
typedef struct Nprg2Parallel {
    uint32_t max_parts;
    int (*ready)(void *impl);
    int (*run)(void *impl, uint32_t parts, Nprg2ParallelFn fn, void *ctx);
    void *impl;
    uint32_t order;
    uint64_t seed;
    uint64_t jobs_parallel[NPRG2_PAR_KINDS], jobs_serial[NPRG2_PAR_KINDS], parts_run[NPRG2_PAR_KINDS];
    uint64_t not_ready, declined;
} Nprg2Parallel;
/* Parts the caller may use for a job of `want` natural parts (>= 1). */
extern volatile uint32_t nprg2_par_kind; /* kind of the job being dispatched (diagnostics) */
uint32_t nprg2_parallel_parts(Nprg2Parallel *p, uint32_t want);
void nprg2_parallel_for(Nprg2Parallel *p, uint32_t kind, uint32_t parts, Nprg2ParallelFn fn, void *ctx);

typedef struct { uint8_t *data; uint32_t bytes; } Nprg2ExecBuffer;
enum { NPRG2_DMA_DRAIN_HAZARD=1, NPRG2_DMA_DRAIN_CANCEL=2,
       NPRG2_DMA_DRAIN_ERROR=3, NPRG2_DMA_DRAIN_END=4, NPRG2_DMA_DRAIN_NEXT_COPY=5 };
typedef int (*Nprg2WeightsDmaStart)(void *ctx,uint32_t command_index,void *dst,
                                   const void *src,size_t bytes,uint64_t timeout_cycles);
typedef int (*Nprg2WeightsDmaDrain)(void *ctx,uint32_t reason,uint32_t command_index,
                                   uint64_t timeout_cycles);
typedef struct {
    const uint32_t *first_hazard;
    uint32_t command_count;
    Nprg2DmaRing *ring;
    Nprg2WeightsDmaStart start;
    Nprg2WeightsDmaDrain drain;
    void *ctx;
    uint64_t timeout_cycles;
    uint8_t pending,ring_mode;
    uint32_t pending_command,pending_hazard;
    uint64_t starts,drains,hazard_drains,cancel_drains,error_drains,end_drains,
             next_copy_drains,cpu_fallbacks,failures;
} Nprg2WeightsAsync;
/* Scratch regions. Each handler names the region it receives as
 * call->scratch. Region 0 is the request's scratch/scratch_bytes; region k>0
 * is request->extra_scratch[k-1]. A handler whose region is too small is
 * rejected in preflight. Every region must stay disjoint from graph memory,
 * the control arrays and the other regions. */
#define NPRG2_EXEC_SCRATCH_REGIONS 2u
enum { NPRG2_EXEC_SCRATCH_DEFAULT=0, NPRG2_EXEC_SCRATCH_NEAR=1 };
typedef struct Nprg2ExecCall Nprg2ExecCall;
struct Nprg2ExecCall {
    uint32_t command_index;
    Nprg2Command command;
    const Nprg2Program *program;
    const Nprg2Bound *bound;
    const Nprg2Pool *pools;
    uint32_t pool1_writable_bytes;
    const Nprg2ExecBuffer *inputs;
    uint32_t input_count;
    const Nprg2ExecBuffer *outputs;
    uint32_t output_count;
    void *user;
    Nprg2WeightsAsync *weights_async;
    /* NULL = serial. Shared by every call of a request. */
    Nprg2Parallel *parallel;
    /* The region the executor selected for this handler. */
    void *scratch;
    uint32_t scratch_bytes;
    /* Every request region, index 0 = default, so a wrapping handler (the
     * reference chain) can hand the wrapped handler its own region. */
    Nprg2ExecBuffer scratch_region[NPRG2_EXEC_SCRATCH_REGIONS];
    /* Set only on run calls: the executor has just validated this command's
     * exact request (in this run or an identical preflight), so a handler
     * may skip a purely repeated dry-run of its own validator. Always 0 in
     * validation calls. */
    uint8_t preflighted;
    /* Execution owner for handlers with per-thread mutable scratch/memos. */
    uint8_t execution_lane;
    /* Conv epilogue fusion (see prepared_fuse.h): set on the conv's call when
     * the executor activated a pair for this request; the epilogue command's
     * call, and whether the conv's raw output is dead (need not be stored).
     * NULL/0 otherwise, in validation as well as run calls. */
    const Nprg2ExecCall *epilogue;
    uint8_t epilogue_dead;
};

typedef int (*Nprg2ExecValidate)(const Nprg2ExecCall *call, void *handler_ctx);
typedef int (*Nprg2ExecRun)(const Nprg2ExecCall *call, void *handler_ctx);
/* Optional handler hook: can this conv handler execute `epilogue` fused into
 * its own output step, exactly, for this request? Called in preflight for
 * every plan pair whose two commands lie in the interval; nonzero activates
 * the pair (the conv is then validated and run with call->epilogue set and
 * the epilogue command is not run). It must validate the epilogue's views. */
typedef int (*Nprg2ExecFuse)(const Nprg2ExecCall *conv, const Nprg2ExecCall *epilogue, int dead, void *handler_ctx);
typedef int (*Nprg2ExecCancel)(void *ctx, uint32_t next_command);
/* Optional hardware profile, passed in the request (never in the stats, which
 * preflight clears). All arrays are caller-owned and accumulate until the
 * caller clears them. With no clock nothing is timed. Cycle values are in
 * clock() units (the DSP's pcycle counter on Frame).
 *   variant_*          run time and count per variant (< NPRG2_EXEC_PROFILE_VARIANTS)
 *   handler_*          per registry slot (NPRG2_EXEC_REGISTRY_CAPACITY entries):
 *                      run cycles, run count (also counts effect-free
 *                      commands, which are not timed) and validator cycles
 *   preflight_cycles   whole preflight calls, including the validators
 *   validate_cycles    the validator callbacks alone
 *   dispatch_cycles    run loop time outside handler run callbacks: command
 *                      decode, registry lookup, scratch selection, cancel
 *   run_cycles         the handler run callbacks alone */
#define NPRG2_EXEC_PROFILE_VARIANTS 80u
typedef struct {
    uint64_t (*clock)(void);
    uint64_t *variant_cycles, *variant_count;
    uint64_t *handler_run_cycles, *handler_run_count, *handler_validate_cycles;
    uint64_t preflight_cycles, validate_cycles, dispatch_cycles, run_cycles;
    uint32_t preflight_calls, preflights_reused;
} Nprg2ExecProfile;
/* Commands whose run is skipped without decoding (no_effect handlers, i.e. the
 * NOPs) are recorded by a full preflight in a bitmap over command indices below
 * this limit; larger indices are always decoded and dispatched. */
#define NPRG2_EXEC_STRIP_COMMANDS 8192u
/* One run-list record: command index, registry slot, role (fl) and, for a
 * conv whose epilogue pair is active, the epilogue's command index. */
enum { NPRG2_RUN_SKIP=1, NPRG2_RUN_FUSED_CONV=2, NPRG2_RUN_DEAD=4 };
typedef struct { uint16_t ix, epi; uint8_t hi, fl; } Nprg2ExecRunRec;
typedef struct {
    uint32_t validated_commands, executed_commands, failed_command;
    uint8_t failed_opcode, failed_variant;
    /* Signature of the request the last successful preflight covered, and the
     * digest of the program and bound tables' contents at that time; see
     * Nprg2ExecRequest.reuse_preflight. */
    uint64_t preflight_signature, preflight_digest;
    /* Set by a preflight (full or reused) that has just checked the digest;
     * the run that follows consumes it instead of hashing the tables again. */
    uint8_t digest_fresh;
    /* Valid after a full preflight of an interval inside the bitmap range:
     * bit set = the command has an effect and is dispatched; clear = a
     * no_effect command, counted but not decoded. stripped_by_slot counts them
     * per registry slot (for the handler profile). */
    uint8_t strip_valid;
    /* Step-2 cost-volume approximation activation counters; reset per run. */
    uint32_t approx_cost_volume_commands;
    uint16_t approx_cost_volume_conv_planes, approx_cost_volume_interpolated_planes;
    uint32_t effect_bits[NPRG2_EXEC_STRIP_COMMANDS/32u];
    uint32_t stripped_by_slot[NPRG2_EXEC_REGISTRY_CAPACITY];
    /* Compact run list, built by the same full preflight that sets strip_valid
     * (runs_valid): one record per dispatched command (effect-bit set), in
     * schedule order, with the registry slot and the fusion role the run loop
     * would otherwise re-derive. Everything from `runs` on is not cleared by
     * a preflight; only the first run_count records are meaningful. */
    uint8_t runs_valid;
    uint32_t run_count;
    Nprg2ExecRunRec runs[NPRG2_EXEC_STRIP_COMMANDS];
} Nprg2ExecStats;

typedef struct {
    const Nprg2Program *program;
    const Nprg2Bound *bound;
    Nprg2Pool pools[7];
    /* Caller-selected pool1 graph-write ceiling; excludes preserved VTCM tail. */
    uint32_t pool1_writable_bytes;
    const Nprg2ExecBuffer *inputs;
    uint32_t input_count;
    const Nprg2ExecBuffer *outputs;
    uint32_t output_count;
    void *user;
    Nprg2WeightsAsync *weights_async;
    void *scratch;
    uint32_t scratch_bytes;
    /* Regions 1.. (zero bytes = absent). */
    Nprg2ExecBuffer extra_scratch[NPRG2_EXEC_SCRATCH_REGIONS-1u];
    uint32_t command_first, command_count;
    Nprg2ExecCancel cancelled;
    void *cancel_ctx;
    Nprg2ExecStats *stats;
    Nprg2ExecProfile *profile;
    /* Optional epilogue fusion plan (NULL = off). Pairs are active only when
     * their handler's fuse hook accepted them in the preflight of this
     * interval; the plan's `active` flags are written by preflight. */
    Nprg2FusePlan *fuse;
    Nprg2Parallel *parallel;
    /* Run only. When set, stats is non-NULL and the same stats object holds
     * the signature of a successful preflight of this exact request (program,
     * bound tables, pools, ceiling, I/O and scratch buffers, interval,
     * registry, and a digest of the program/bound table contents), preflight
     * and run skip the repeated preflight, across inferences too. Any
     * difference, or no earlier preflight, runs the full preflight as usual.
     * Valid only for handlers whose validators read nothing but the request
     * and immutable data (pool 5): a validator that depends on other pool
     * bytes must re-derive them in its run callback (the elementwise handlers
     * and the reference chain do). */
    uint8_t reuse_preflight;
    uint8_t execution_lane;
} Nprg2ExecRequest;

typedef struct {
    uint8_t opcode, variant, used, scratch_region;
    /* Run does nothing beyond what validation proved (a NOP): the run loop
     * counts the command without calling run. */
    uint8_t no_effect;
    Nprg2ExecValidate validate;
    Nprg2ExecRun run;
    Nprg2ExecFuse fuse;
    void *ctx;
    uint32_t scratch_bytes, concurrent_worker_scratch_bytes;
} Nprg2ExecHandler;

struct Nprg2Executor { Nprg2ExecHandler handlers[NPRG2_EXEC_REGISTRY_CAPACITY]; uint32_t count; };

void nprg2_executor_init(Nprg2Executor *executor);
int nprg2_executor_register(Nprg2Executor *executor, uint8_t opcode, uint8_t variant,
                            Nprg2ExecValidate validate, Nprg2ExecRun run, void *ctx);
int nprg2_executor_register_scratch(Nprg2Executor *executor, uint8_t opcode, uint8_t variant,
                                    Nprg2ExecValidate validate, Nprg2ExecRun run, void *ctx,
                                    uint32_t scratch_bytes,
                                    uint32_t concurrent_worker_scratch_bytes);
/* As register_scratch, with the handler's scratch taken from region
 * scratch_region (< NPRG2_EXEC_SCRATCH_REGIONS). */
int nprg2_executor_register_scratch_region(Nprg2Executor *executor, uint8_t opcode, uint8_t variant,
                                           Nprg2ExecValidate validate, Nprg2ExecRun run, void *ctx,
                                           uint32_t scratch_bytes,
                                           uint32_t concurrent_worker_scratch_bytes,
                                           uint8_t scratch_region);
/* Declares a registered handler's run callback effect-free given a passed
 * validation, so run may skip calling it. UNSUPPORTED if not registered. */
/* Installs the fuse hook of a registered conv handler. */
int nprg2_executor_set_fuse(Nprg2Executor *executor, uint8_t opcode, uint8_t variant, Nprg2ExecFuse fuse);
int nprg2_executor_mark_no_effect(Nprg2Executor *executor, uint8_t opcode, uint8_t variant);
int nprg2_exec_input(const Nprg2ExecCall *call, uint32_t slot,
                     uint32_t *tensor_index, Nprg2TensorView *view);
int nprg2_exec_output(const Nprg2ExecCall *call, uint32_t slot,
                      uint32_t *tensor_index, Nprg2TensorView *view);
/* Read-only validation of the complete requested interval. It enforces the
 * same mapping, alias, scratch, coverage, and feature checks as execute. */
int nprg2_executor_preflight(const Nprg2Executor *executor,
                             const Nprg2ExecRequest *request);
/* Entire requested interval is validated before the first handler writes. No
 * legacy fallback route exists. Handler scratch must be caller-owned. */
int nprg2_executor_run(const Nprg2Executor *executor, const Nprg2ExecRequest *request);
/* Dispatch one command from a previously successful full-range preflight.
 * The proof is read-only and remains valid while commands from independent
 * QNN run lists execute concurrently. Caller supplies private scratch/near
 * regions for each list; `request->fuse` must be NULL. */
int nprg2_executor_run_preflighted_command(const Nprg2Executor *executor,
                                            const Nprg2ExecRequest *full_request,
                                            uint32_t command_index,
                                            uint8_t execution_lane,
                                            Nprg2Parallel *parallel_override,
                                            void *scratch,uint32_t scratch_bytes,
                                            void *near_scratch,uint32_t near_scratch_bytes);
/* Prepare the immutable per-list request once after full preflight; subsequent
 * commands dispatch directly without reconstructing or revalidating it. */
int nprg2_executor_prepare_preflighted_request(const Nprg2Executor *executor,
                                               const Nprg2ExecRequest *full_request,
                                               uint8_t execution_lane,
                                               Nprg2Parallel *parallel_override,
                                               void *scratch,uint32_t scratch_bytes,
                                               void *near_scratch,uint32_t near_scratch_bytes,
                                               Nprg2ExecRequest *prepared_request);
/* Diagnostic: per lane, sum of clock() at handler entry (caller subtracts entry times). */
extern uint64_t (*nprg2_lane_clock)(void);
extern uint64_t nprg2_lane_prehandler[3];
/* CMD_MEMO: one proven plan per command (see prepared_executor.c). */
void nprg2_exec_cmd_memo_begin(const void *program,const uint8_t *pool1);
void *nprg2_exec_cmd_memo(const Nprg2ExecCall *c);
void *nprg2_exec_cmd_memo_new(const Nprg2ExecCall *c,uint32_t bytes);
void nprg2_exec_cmd_memo_commit(const Nprg2ExecCall *c,void *p);
int nprg2_executor_handler_slot(const Nprg2Executor *executor,uint8_t op,uint8_t variant);
int nprg2_executor_run_preflighted_command_prepared(const Nprg2Executor *executor,
                                                    const Nprg2ExecRequest *prepared_request,
                                                    uint32_t command_index);

/* Implements the existing weights_to_vtcm CPU semantics for validated direct
 * tensors: memmove(min(max-shape source bytes, max-shape destination bytes)). */
int nprg2_exec_weights_validate(const Nprg2ExecCall *call, void *ctx);
int nprg2_exec_weights_run(const Nprg2ExecCall *call, void *ctx);
int nprg2_exec_weights_async_try(const Nprg2ExecCall *call,void *dst,const void *src,size_t bytes);
int nprg2_exec_weights_dma_checkpoint_set(Nprg2WeightsAsync *async,uint32_t event_id);
int nprg2_exec_weights_dma_checkpoint_wait(Nprg2WeightsAsync *async,uint32_t event_id);
int nprg2_exec_weights_dma_drain_all(Nprg2WeightsAsync *async);
int nprg2_executor_register_weights(Nprg2Executor *executor);

#ifdef __cplusplus
}
#endif
#endif
