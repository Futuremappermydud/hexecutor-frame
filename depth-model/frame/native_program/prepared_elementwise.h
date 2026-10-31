#ifndef NPRG2_PREPARED_ELEMENTWISE_H
#define NPRG2_PREPARED_ELEMENTWISE_H

#include "prepared_executor.h"

/* Exact blocked add-u8 and u8->u16 LinearClip profiles. Alias proofs use
 * caller scratch: 3 * 2560 uintptr_t entries (61,440 bytes on 64-bit hosts).
 * No fallback or temporary pool is used. */
#define NPRG2_ELEMENTWISE_SCRATCH_BYTES (3u * 2560u * (uint32_t)sizeof(uintptr_t) + 7u * (uint32_t)sizeof(uint8_t *))

int nprg2_executor_register_elementwise(Nprg2Executor *executor);

#endif
