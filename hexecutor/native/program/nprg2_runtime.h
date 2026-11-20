/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef NPRG2_RUNTIME_H
#define NPRG2_RUNTIME_H

#include <stddef.h>
#include <stdint.h>
#include "../qd_resident_schema.h"

#define NPRG2_MAGIC 0x4752504eu /* "NPRG" little-endian */
#define NPRG2_VERSION 2u
#define NPRG2_HEADER_BYTES 256u
#define NPRG2_TENSOR_BYTES 12u
#define NPRG2_GEOMETRY_BYTES 44u
#define NPRG2_COMMAND_BYTES 20u
#define NPRG2_MAX_IMAGE_BYTES (8u*1024u*1024u)

enum {
    NPRG2_OK=0, NPRG2_BAD_ARGUMENT=1, NPRG2_BAD_HEADER=2, NPRG2_BAD_CRC=3,
    NPRG2_BAD_HASH=4, NPRG2_BAD_SECTION=5, NPRG2_BAD_TENSOR=6,
    NPRG2_BAD_BLOCK=7, NPRG2_BAD_COMMAND=8, NPRG2_BAD_POOL=9,
    NPRG2_NO_MEMORY=10, NPRG2_UNSUPPORTED=11, NPRG2_SCRATCH=12,
    NPRG2_BAD_BIND=13
};

typedef void *(*Nprg2Alloc)(void *ctx, size_t bytes);
typedef void (*Nprg2Free)(void *ctx, void *ptr);
typedef struct { Nprg2Alloc alloc; Nprg2Free free; void *ctx; } Nprg2Allocator;
typedef struct { uint8_t *base; uint32_t bytes; } Nprg2Pool;
typedef struct { uint32_t offset, bytes; const uint8_t *data; } Nprg2Section;

typedef struct {
    uint16_t geometry;
    uint8_t storage_kind, pool;
    uint32_t offset, block_first, block_count, slice_origin;
} Nprg2Tensor;

typedef struct {
    uint16_t sequence;
    uint8_t opcode, variant;
    uint16_t io_index, input_first, output_first, vector_first, call_first;
    uint8_t input_count, output_count, vector_count, call_count, duplicate_pad;
} Nprg2Command;

typedef struct {
    const uint8_t *raw;
    uint32_t raw_bytes, tensor_count, geometry_count, block_stream_bytes;
    uint32_t command_count, ref_count, vector_count, call_count;
    uint32_t pool_bytes[7];
    uint16_t graph_input_first, graph_input_count, graph_output_first, graph_output_count;
    uint8_t context_sha256[32], constants_sha256[32], source_sha256[32];
    Nprg2Section geometry, tensor, block_stream, command, refs, vectors, calls;
    Nprg2Pool session_pools[7];
    Nprg2Tensor *tensors;
    uint32_t *expanded_blocks, expanded_block_count;
    Nprg2Allocator allocator;
} Nprg2Program;

typedef struct {
    const uint8_t *context_sha256, *constants_sha256, *source_sha256, *prepared_sha256;
} Nprg2ExpectedHashes;

typedef struct {
    uint32_t scratch_bytes, concurrent_worker_scratch_bytes;
    uint8_t supported, may_fallback;
} Nprg2Capability;
typedef int (*Nprg2CoverageResolver)(void *ctx, const Nprg2Program *program,
                                     uint32_t command_index, const Nprg2Command *command,
                                     Nprg2Capability *capability);

typedef struct {
    uint8_t *hot_memory;
    uint32_t hot_capacity, scratch_budget, l2_available_bytes;
    uint8_t pool1_is_verified_vtcm;
    Nprg2CoverageResolver resolve;
    void *resolver_ctx;
} Nprg2BindConfig;

typedef struct {
    Nprg2Section geometry, tensor, command, refs, vectors, calls;
    uint32_t hot_bytes, maximum_command_scratch;
    uint8_t ready;
} Nprg2Bound;

typedef struct {
    uint16_t geometry;
    uint8_t storage_kind, pool;
    uint32_t offset;
    const uint32_t *blocks;
    uint32_t block_count, slice_origin;
    const uint8_t *geometry_record;
} Nprg2TensorView;

/* Hashes must come from a trusted package/manifest. open is read-only with
 * respect to all supplied graph pools. A NULL allocator uses malloc/free. */
int nprg2_open(Nprg2Program *out, const void *raw, uint32_t raw_bytes,
               const Nprg2ExpectedHashes *expected, const Nprg2Pool pools[7],
               Nprg2Allocator allocator);
void nprg2_close(Nprg2Program *program);

/* Resolver coverage is checked for every command before hot_memory is written.
 * pool1 must be declared verified VTCM; this API does not map or verify hardware. */
int nprg2_bind_mock(const Nprg2Program *program, const Nprg2Pool pools[7],
                    const Nprg2BindConfig *config, Nprg2Bound *bound);
/* Copies/rewrites hot metadata without claiming command coverage. This is a
 * placement-only step: a real executor must preflight its requested interval
 * before any graph-pool writes. */
int nprg2_bind_metadata(const Nprg2Program *program, const Nprg2Pool pools[7],
                        const Nprg2BindConfig *config, Nprg2Bound *bound);
int nprg2_command_at(const Nprg2Bound *bound, uint32_t index, Nprg2Command *out);
int nprg2_tensor_at(const Nprg2Program *program, uint32_t index,
                    const Nprg2Tensor **tensor, const uint8_t **geometry_record);
/* Execution accessor: descriptor/geometry pointers reference the bound hot
 * memory; expanded block arrays point to the once-decoded private heap table. */
int nprg2_bound_tensor_at(const Nprg2Program *program, const Nprg2Bound *bound,
                          uint32_t index, Nprg2TensorView *out);

uint32_t nprg2_crc32(const void *data, size_t bytes, size_t zero_offset, size_t zero_bytes);
void nprg2_sha256(const void *data, size_t bytes, uint8_t out[32]);

#endif
