#include "prepared_spill_fill.h"

#include <limits.h>
#include <string.h>

enum { SPILL_POOL = 6, SPILL_OPCODE = 26, FILL_VARIANT = 8, SPILL_VARIANT = 9 };

typedef struct {
    uint32_t count;
    uint32_t spill_offset;
    uint32_t pair_first;
    uint8_t spill_to_pool;
} SpillFillPlan;

static uint32_t rd32sf(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static int prepare_spill_fill(const Nprg2ExecCall *c, SpillFillPlan *plan) {
    if (!c || !plan || c->command.opcode != SPILL_OPCODE ||
        (c->command.variant != FILL_VARIANT && c->command.variant != SPILL_VARIANT) ||
        c->command.input_count || c->command.output_count || c->command.call_count ||
        c->command.duplicate_pad ||
        c->command.vector_count < 4u || !c->bound || !c->bound->vectors.data ||
        c->command.vector_first > c->bound->vectors.bytes / 4u ||
        c->command.vector_count > c->bound->vectors.bytes / 4u - c->command.vector_first ||
        !c->pools || !c->pools[SPILL_POOL].base ||
        (uintptr_t)c->pools[SPILL_POOL].base > UINTPTR_MAX - c->pools[SPILL_POOL].bytes) {
        return 0;
    }

    const uint8_t *vectors = c->bound->vectors.data + (size_t)c->command.vector_first * 4u;
    uint32_t count = c->command.vector_count;
    if (((count - 4u) & 1u) != 0u) return 0;

    uint64_t spill_offset = rd32sf(vectors + 12u);
    uint64_t spill_size = c->pools[SPILL_POOL].bytes;
    for (uint32_t i = 4u; i < count; i += 2u) {
        uint32_t encoded = rd32sf(vectors + (size_t)i * 4u);
        uint32_t bytes = rd32sf(vectors + (size_t)(i + 1u) * 4u);
        uint32_t pool = encoded & 0x7ffu;
        uint32_t address = encoded & ~0x7ffu;
        if (pool >= 7u || pool == SPILL_POOL || !c->pools[pool].base ||
            (uintptr_t)c->pools[pool].base > UINTPTR_MAX - c->pools[pool].bytes ||
            (uint64_t)address + bytes > c->pools[pool].bytes ||
            spill_offset + bytes > spill_size || spill_offset + bytes > UINT32_MAX) {
            return 0;
        }
        if (c->command.variant == FILL_VARIANT && pool == 1u &&
            (address > c->pool1_writable_bytes ||
             bytes > c->pool1_writable_bytes - address)) {
            return 0;
        }
        spill_offset += bytes;
    }

    plan->count = count;
    plan->spill_offset = rd32sf(vectors + 12u);
    plan->pair_first = c->command.vector_first + 4u;
    plan->spill_to_pool = c->command.variant == FILL_VARIANT;
    return 1;
}

static int validate_spill_fill(const Nprg2ExecCall *c, void *ctx) {
    (void)ctx;
    SpillFillPlan plan;
    return prepare_spill_fill(c, &plan);
}

static int run_spill_fill(const Nprg2ExecCall *c, void *ctx) {
    (void)ctx;
    SpillFillPlan plan;
    if (!prepare_spill_fill(c, &plan)) return 0;
    const uint8_t *vectors = c->bound->vectors.data + (size_t)c->command.vector_first * 4u;
    uint64_t spill_offset = plan.spill_offset;
    for (uint32_t i = 4u; i < plan.count; i += 2u) {
        uint32_t encoded = rd32sf(vectors + (size_t)i * 4u);
        uint32_t bytes = rd32sf(vectors + (size_t)(i + 1u) * 4u);
        uint32_t pool = encoded & 0x7ffu;
        uint32_t address = encoded & ~0x7ffu;
        uint8_t *pool_ptr = c->pools[pool].base + address;
        uint8_t *spill_ptr = c->pools[SPILL_POOL].base + (uint32_t)spill_offset;
        if (plan.spill_to_pool) memmove(pool_ptr, spill_ptr, bytes);
        else memmove(spill_ptr, pool_ptr, bytes);
        spill_offset += bytes;
    }
    return 1;
}

int nprg2_executor_register_spill_fill(Nprg2Executor *e) {
    if (!e) return NPRG2_EXEC_ARGUMENT;
    int r = nprg2_executor_register(e, SPILL_OPCODE, FILL_VARIANT,
                                    validate_spill_fill, run_spill_fill, NULL);
    if (r != NPRG2_EXEC_OK) return r;
    return nprg2_executor_register(e, SPILL_OPCODE, SPILL_VARIANT,
                                   validate_spill_fill, run_spill_fill, NULL);
}
