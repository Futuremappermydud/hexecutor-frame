#ifndef NPRG2_PREPARED_ELEMENTWISE_H
#define NPRG2_PREPARED_ELEMENTWISE_H

#include "prepared_executor.h"

/* Exact blocked add-u8, requant-u8, and u8->u16 LinearClip profiles. Alias proofs use
 * caller scratch: 3 * 2560 uintptr_t entries (61,440 bytes on 64-bit hosts).
 * No fallback or temporary pool is used. */
#define NPRG2_ELEMENTWISE_SCRATCH_BYTES (3u * 2560u * (uint32_t)sizeof(uintptr_t) + 7u * (uint32_t)sizeof(uint8_t *))

int nprg2_executor_register_elementwise(Nprg2Executor *executor);
/* As above, with the alias-proof scratch taken from executor scratch region
 * scratch_region instead of the default one. The Frame reference build uses
 * NPRG2_EXEC_SCRATCH_NEAR (an L2-TCM slot) so the HVX handlers never get the
 * reference kernel's L1-cached heap scratch. */
int nprg2_executor_register_elementwise_region(Nprg2Executor *executor, uint8_t scratch_region);

#endif
