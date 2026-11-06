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
 * clock (optional) with phase[8] accumulates, per fast command: input staging,
 * multiply-accumulate, requantization, output write; u8 convs in phase[0..3],
 * u16 in phase[4..7], summed over every thread part (CPU time, not wall). */
typedef struct {
    uint8_t *workspace; uint32_t workspace_bytes;
    uint8_t *near; uint32_t near_bytes;
    uint64_t (*clock)(void); uint64_t *phase;
    /* Optional: two more near-style regions of par_near_bytes (128-byte
     * aligned, disjoint from everything else) for helper threads. A command
     * whose output rows can be split uses them with call->parallel (see
     * nprg2_parallel_parts); without them, or when a slot is not usable or
     * too small for the command, it runs on the calling thread. */
    uint8_t *par_near[3]; uint32_t par_near_bytes;
    /* Optional: exact HMX for eligible u8 convs (prepared_hmx_conv.inc).
     * The workspace must then be VTCM, and so must the weights' pool. */
    const struct Nprg2HmxOps *hmx;
} Nprg2ReferenceConv;
/* HMX entry points, all called on the executor's own thread (the HMX
 * owner): enter sets up the thread for HMX instructions (SSR.XE2 on the
 * Frame) and returns what leave restores; wide/finish are qd_hmx_wide and
 * qd_hmx_finish (qd_hmx_ops*.S). Off Hexagon only the pointer's presence
 * matters: the conv path then runs an exact scalar model of the same tiles. */
typedef struct Nprg2HmxOps {
    uint32_t (*enter)(void);
    void (*leave)(uint32_t saved);
    void (*wide)(void *a, void *b, void *fi, void *c, uint32_t kb);
    void (*finish)(void *zero, void *completion);
    uint64_t (*clock)(void);
    /* Optional qd_hmx_wide_n: one N block over mt M tiles with one bias load
     * and clear (kbnb = KB | NB<<16); used instead of wide when present. */
    void (*wide_n)(void *a, void *b, void *fi, void *c, uint32_t kbnb, uint32_t mt);
    /* Optional qd_hmx_native_n (NPRG2_HW_ROUNDING builds): native int8, one N
     * block over mt 64-pixel crouton tiles, cvt.ub stored :cm at out[t]. */
    void (*native_n)(void *a, void *b, void *fi, void *const *out, uint32_t kb, uint32_t mt);
    /* Optional qd_hmx_native16_n (NPRG2_HW_ROUNDING builds): u16 activations
     * (32-pixel u16 crouton tiles, :2x1), cvt.uh stored at out[t]. */
    void (*native16_n)(void *a, void *b, void *fi, void *const *out, uint32_t kb, uint32_t mt);
    /* Optional qd_hmx_slide3_n (NPRG2_HW_ROUNDING builds): native int8 3x3
     * stride 1 straight from the input crouton blocks (no A tiles), one N
     * block over one 8-row strip: rec = {rsA, rtA, rsC, rtC} per (column, cb),
     * st = {rsO, rtO} store words per column (qd_hmx_ops.S). */
    void (*slide3_n)(const uint32_t *rec, const void *b, const void *fi, const uint32_t *st, uint32_t cbs, uint32_t ncol);
    /* NPRG2_PLANES builds: as native_n and slide3_n, two conversions per tile
     * (qd_hmx_ops.S). native: out[t] = low plane, high plane 2048 bytes after it. */
    void (*nativep_n)(void *a, void *b, void *fi, void *const *out, uint32_t kb, uint32_t mt);
    void (*slide3p_n)(const uint32_t *rec, const void *b, const void *fi, const uint32_t *st, uint32_t cbs, uint32_t ncol);
} Nprg2HmxOps;
/* Fuse hook registered with the conv keys: the conv handler can run the
 * REQUANT_U8/70 or ADD_U8/13 command `epilogue` fused into its output step
 * (Nprg2ExecFuse; see prepared_fuse.h). */
int nprg2_exec_reference_fuse(const Nprg2ExecCall *conv, const Nprg2ExecCall *epilogue, int dead, void *ctx);
int nprg2_executor_register_reference_conv(Nprg2Executor *executor, Nprg2ReferenceConv *conv, uint32_t *added);
/* nprg2_exec_reference_scratch as the handler registered with conv sees the
 * command: call-scratch bytes, and workspace bytes (0 off the fast path). */
int nprg2_exec_reference_conv_measure(const Nprg2ExecCall *call, const Nprg2ReferenceConv *conv,
                                      uint32_t *scratch_bytes, uint32_t *workspace_bytes);
/* Duplicate the main preflight memo into an independent lane before threaded
 * run lists start. Only the owner lane mutates each per-command memo afterward. */
int nprg2_exec_reference_clone_lane(uint8_t source_lane,uint8_t destination_lane);
/* NPRG2_PLANES on Hexagon: repeat 3x3 u8 convs from a per-command plan captured by
 * their first full run (prepared_hmx_conv.inc, ConvPlan). begin before a run list,
 * with the list's request and the registered conv; run returns 1 ran, 0 use the
 * normal path, -1 error. */
int nprg2_exec_reference_plan_begin(const Nprg2ExecRequest *request, const Nprg2ReferenceConv *conv);
int nprg2_exec_reference_plan_run(uint32_t command, Nprg2Parallel *parallel, int32_t *err);
/* Path the most recently measured or run convolution took: 0 reference,
 * 1 fast with the workspace only, 2 fast with near, 3 HMX (diagnostics only). */
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
