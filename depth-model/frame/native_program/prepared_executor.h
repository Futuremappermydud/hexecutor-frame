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
    void *scratch;
    uint32_t scratch_bytes;
} Nprg2ExecCall;

typedef int (*Nprg2ExecValidate)(const Nprg2ExecCall *call, void *handler_ctx);
typedef int (*Nprg2ExecRun)(const Nprg2ExecCall *call, void *handler_ctx);
typedef int (*Nprg2ExecCancel)(void *ctx, uint32_t next_command);
/* With clock and both arrays set, run adds each executed command's clock
 * delta and count at its variant (variants below NPRG2_EXEC_PROFILE_VARIANTS). */
#define NPRG2_EXEC_PROFILE_VARIANTS 80u
typedef struct {
    uint32_t validated_commands, executed_commands, failed_command;
    uint8_t failed_opcode, failed_variant;
    uint64_t (*clock)(void);
    uint64_t *variant_cycles, *variant_count;
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
    uint32_t command_first, command_count;
    Nprg2ExecCancel cancelled;
    void *cancel_ctx;
    Nprg2ExecStats *stats;
} Nprg2ExecRequest;

typedef struct {
    uint8_t opcode, variant, used, reserved;
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
