#ifndef NPRG2_PLAN_CACHE_H
#define NPRG2_PLAN_CACHE_H

#include "prepared_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NPRG2_PLAN_MAGIC 0x504c414eu /* 'NPLN' little-endian */
#define NPRG2_PLAN_VERSION 1u
#define NPRG2_PLAN_HEADER_BYTES 128u
#define NPRG2_PLAN_TAG_UNCOVERED 0xffu
#define NPRG2_PLAN_MAX_COMMANDS 5120u
#define NPRG2_PLAN_TOTAL_BYTES (NPRG2_PLAN_HEADER_BYTES + NPRG2_PLAN_MAX_COMMANDS) /* 5248 */
#define NPRG2_PLAN_ALIGNMENT 128u

enum {
    NPRG2_PLAN_OK = 0,
    NPRG2_PLAN_ARGUMENT = 1,
    NPRG2_PLAN_CAPACITY = 2,
    NPRG2_PLAN_UNSUPPORTED = 3,
    NPRG2_PLAN_REJECTED = 4,
    NPRG2_PLAN_SCRATCH = 5,
    NPRG2_PLAN_MISMATCH = 6
};

/* Exactly 128 bytes serialized header */
typedef struct {
    uint32_t magic;                 /* 0x00: NPRG2_PLAN_MAGIC */
    uint32_t version;               /* 0x04: NPRG2_PLAN_VERSION */
    uint32_t header_bytes;          /* 0x08: 128 */
    uint32_t total_bytes;           /* 0x0c: 5248 */
    uint32_t command_count;         /* 0x10: 5068 (primary) / 5040 (alternate) */
    uint32_t command_capacity;      /* 0x14: 5120 (128-byte aligned) */
    uint32_t registry_count;        /* 0x18: executor handlers count at build time */
    uint32_t pool1_writable_bytes;  /* 0x1c: active VTCM write ceiling */
    uint32_t max_scratch_bytes;     /* 0x20: peak command scratch required */
    uint16_t dynamic_command_count; /* 0x24: count of commands needing per-run check */
    uint16_t reserved16;            /* 0x26: padding */
    uint32_t flags;                 /* 0x28: status flags */
    uint8_t  fingerprint[32];       /* 0x2c..0x4b: session/bound metadata SHA-256 */
    uint32_t pool_bases[7];         /* 0x4c..0x67: 28 bytes pool base identities */
    uint8_t  pad[24];               /* 0x68..0x7f: pad to exactly 128 bytes */
} Nprg2PlanHeader;

/* In-memory plan cache descriptor */
typedef struct {
    const Nprg2PlanHeader *header;
    const uint8_t *tags;            /* Pointer to 5120-byte bytecode tag table */
    uint32_t command_count;
    uint32_t pool1_writable_bytes;
    Nprg2Pool pools[7];
    void *arena;
    size_t arena_capacity;
    uint8_t ready;
} Nprg2PlanCache;

/* Build parameters bundle */
typedef struct {
    const Nprg2Program *program;
    const Nprg2Executor *executor;
    const Nprg2Pool *pools;             /* 7 pool descriptors */
    uint32_t pool1_writable_bytes;
    void *scratch;                      /* Setup-time alias proof scratch */
    uint32_t scratch_bytes;
    const uint8_t *session_fingerprint; /* Optional 32-byte hash */
    const Nprg2ExecBuffer *inputs;
    uint32_t input_count;
    const Nprg2ExecBuffer *outputs;
    uint32_t output_count;
} Nprg2PlanBuildConfig;

/* Dynamic per-run validation function pointer */
typedef int (*Nprg2ExecValidateDynamic)(const Nprg2ExecCall *call, void *handler_ctx);

/* Computes required bytes rounded up to 128-byte boundary */
size_t nprg2_plan_required_bytes(uint32_t command_count);

/* One-time static compilation of the plan cache */
int nprg2_plan_build(const Nprg2Bound *bound,
                     const Nprg2PlanBuildConfig *config,
                     void *cache_arena,
                     size_t capacity,
                     Nprg2PlanCache *out);

/* Convenience builder directly from an Nprg2ExecRequest */
int nprg2_plan_build_from_request(const Nprg2Executor *executor,
                                  const Nprg2ExecRequest *request,
                                  void *cache_arena,
                                  size_t capacity,
                                  Nprg2PlanCache *out);

/* Fast per-run preflight using the precomputed plan cache (~0.05 ms) */
int nprg2_executor_preflight_cached(const Nprg2Executor *executor,
                                    const Nprg2PlanCache *cache,
                                    const Nprg2ExecRequest *request);

/* Fast run using cached tag table for direct O(1) handler dispatch */
int nprg2_executor_run_cached(const Nprg2Executor *executor,
                              const Nprg2PlanCache *cache,
                              const Nprg2ExecRequest *request);

#ifdef __cplusplus
}
#endif

#endif /* NPRG2_PLAN_CACHE_H */
