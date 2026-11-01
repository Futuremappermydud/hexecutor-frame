#ifndef NPRG2_PREPARED_REFERENCE_H
#define NPRG2_PREPARED_REFERENCE_H

#include "prepared_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Exact ports of qd_graph.c's CPU-backend kernels over bound NPRG v2 tensor
 * views. Every kernel builds dense temporaries in the caller's scratch arena,
 * then writes its destination with qd_graph's Memory.write semantics.
 *
 * Validation runs the same kernel without moving data: it checks metadata,
 * call arguments, every touched block/byte range against the pool bounds and
 * the pool-1 write ceiling, and the exact scratch peak. Gather axes, transpose
 * permutations and depth-to-space factors held as direct constant-pool (pool 5)
 * tensors are read during validation so results are sized exactly. Other
 * conditions that depend on live tensor values (gather/resize indices, runtime
 * shifts) are checked by run before that command's first graph write. Convolution
 * reads W through the current tensor location, never a static-source alias. */
int nprg2_exec_reference_validate(const Nprg2ExecCall *call, void *ctx);
int nprg2_exec_reference_run(const Nprg2ExecCall *call, void *ctx);
/* Scratch bytes the command needs, or 0 with *bytes untouched on rejection. */
int nprg2_exec_reference_scratch(const Nprg2ExecCall *call, uint32_t *bytes);
/* Source line of the most recent kernel rejection (diagnostics only). */
uint32_t nprg2_exec_reference_error_line(void);

int nprg2_executor_register_reference(Nprg2Executor *executor, uint8_t opcode, uint8_t variant);
/* Registers the reference kernel for every schema (opcode, variant) whose key
 * is still free; already-registered keys keep their handler. */
int nprg2_executor_register_reference_missing(Nprg2Executor *executor, uint32_t *added);

/* Exact fast convolution for every free HMX conv key (opcode 6). Raw weight
 * blocks are read in place ([O%32][I%4] 128-byte tiles), taps accumulate in
 * int32 (u16 activations as two u8 halves) and requantization is the
 * reference expression. All temporaries live in the workspace, not the call
 * scratch. The path is chosen from shapes alone, so validation and run agree;
 * a command whose int32 bound or workspace need does not fit takes the
 * reference convolution. A pool range that overlaps the workspace is clipped
 * below it for these commands, so no conv reads or writes the workspace
 * through a pool. On Hexagon with HVX the tiles feed vrmpy directly; the
 * workspace should then be VTCM. Storage must outlive the executor.
 *
 * near (optional) is memory with fast scalar access, L2-TCM on the Frame,
 * where scalar VTCM loads cost ~45 cycles. When a command's per-chunk state
 * fits, the input staged in the workspace is copied by vector in column
 * chunks into near, and tile offsets, accumulators, requant parameters and
 * output rows live there too; otherwise the command keeps the workspace-only
 * path. near must not overlap any pool, the workspace or the call scratch.
 * clock (optional) with phase[4] accumulates, per fast command: input staging,
 * multiply-accumulate, requantization, output write. */
typedef struct {
    uint8_t *workspace; uint32_t workspace_bytes;
    uint8_t *near; uint32_t near_bytes;
    uint64_t (*clock)(void); uint64_t *phase;
} Nprg2ReferenceConv;
int nprg2_executor_register_reference_conv(Nprg2Executor *executor, Nprg2ReferenceConv *conv, uint32_t *added);
/* nprg2_exec_reference_scratch as the handler registered with conv sees the
 * command: call-scratch bytes, and workspace bytes (0 off the fast path). */
int nprg2_exec_reference_conv_measure(const Nprg2ExecCall *call, const Nprg2ReferenceConv *conv,
                                      uint32_t *scratch_bytes, uint32_t *workspace_bytes);
/* Path the most recently measured or run convolution took: 0 reference,
 * 1 fast with the workspace only, 2 fast with near (diagnostics only). */
uint32_t nprg2_exec_reference_conv_path(void);

/* Rewrites an existing registration so commands its validator rejects use the
 * reference kernel instead. Storage must outlive the executor. The chained
 * entry takes the default scratch region (the reference kernel's); the
 * wrapped handler is still called with its own registered region and byte
 * demand, and falls through to the reference kernel when that region is
 * absent or too small. */
typedef struct {
    Nprg2ExecValidate validate; Nprg2ExecRun run; void *ctx;
    uint32_t scratch_bytes; uint8_t scratch_region;
} Nprg2ReferenceChain;
int nprg2_executor_chain_reference(Nprg2Executor *executor, uint8_t opcode, uint8_t variant,
                                   Nprg2ReferenceChain *storage);

#ifdef __cplusplus
}
#endif
#endif
