#ifndef NPRG2_PREPARED_EXECUTOR_H
#define NPRG2_PREPARED_EXECUTOR_H

#include "nprg2_runtime.h"

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
typedef struct { uint8_t *data; uint32_t bytes; } Nprg2ExecBuffer;
/* Scratch regions. Each handler names the region it receives as
 * call->scratch. Region 0 is the request's scratch/scratch_bytes; region k>0
 * is request->extra_scratch[k-1]. A handler whose region is too small is
 * rejected in preflight. Every region must stay disjoint from graph memory,
 * the control arrays and the other regions. */
#define NPRG2_EXEC_SCRATCH_REGIONS 2u
enum { NPRG2_EXEC_SCRATCH_DEFAULT=0, NPRG2_EXEC_SCRATCH_NEAR=1 };
typedef struct {
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
} Nprg2ExecCall;

typedef int (*Nprg2ExecValidate)(const Nprg2ExecCall *call, void *handler_ctx);
typedef int (*Nprg2ExecRun)(const Nprg2ExecCall *call, void *handler_ctx);
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
typedef struct {
    uint32_t validated_commands, executed_commands, failed_command;
    uint8_t failed_opcode, failed_variant;
    /* Signature of the request the last successful preflight covered; see
     * Nprg2ExecRequest.reuse_preflight. */
    uint64_t preflight_signature;
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
    void *scratch;
    uint32_t scratch_bytes;
    /* Regions 1.. (zero bytes = absent). */
    Nprg2ExecBuffer extra_scratch[NPRG2_EXEC_SCRATCH_REGIONS-1u];
    uint32_t command_first, command_count;
    Nprg2ExecCancel cancelled;
    void *cancel_ctx;
    Nprg2ExecStats *stats;
    Nprg2ExecProfile *profile;
    /* Run only. When set, stats is non-NULL and the same stats object holds
     * the signature of a successful preflight of this exact request (program,
     * bound tables, pools, ceiling, I/O and scratch buffers, interval,
     * registry), run skips its own repeated preflight. Any difference, or no
     * earlier preflight, runs the full preflight as usual. */
    uint8_t reuse_preflight;
} Nprg2ExecRequest;

typedef struct {
    uint8_t opcode, variant, used, scratch_region;
    /* Run does nothing beyond what validation proved (a NOP): the run loop
     * counts the command without calling run. */
    uint8_t no_effect;
    Nprg2ExecValidate validate;
    Nprg2ExecRun run;
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

/* Implements the existing weights_to_vtcm CPU semantics for validated direct
 * tensors: memmove(min(max-shape source bytes, max-shape destination bytes)). */
int nprg2_exec_weights_validate(const Nprg2ExecCall *call, void *ctx);
int nprg2_exec_weights_run(const Nprg2ExecCall *call, void *ctx);
int nprg2_executor_register_weights(Nprg2Executor *executor);

#ifdef __cplusplus
}
#endif
#endif
