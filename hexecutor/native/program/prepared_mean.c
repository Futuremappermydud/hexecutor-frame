#include "prepared_mean.h"

#include <stdint.h>
#include <string.h>

enum {
    MEAN_DTYPE_U16 = 3,
    MEAN_STORAGE_DIRECT = 1,
    MEAN_GEOMETRY_SHAPE = 1,
    MEAN_ELEMENT_COUNT = 32
};

static uint16_t rd16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static int32_t rdi16(const uint8_t *p) {
    return (int16_t)rd16(p);
}

static int32_t rd32(const uint8_t *p) {
    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

int nprg2_exec_mean_validate(const Nprg2ExecCall *c, void *ctx) {
    (void)ctx;
    if (!c || c->command.opcode != QDR_MEAN || c->command.variant != 47) return 0;
    if (c->command.input_count != 3 || c->command.output_count != 1) return 0;

    uint32_t in_idx, out_idx;
    Nprg2TensorView in, out;
    if (nprg2_exec_input(c, 0, &in_idx, &in) != NPRG2_EXEC_OK ||
        nprg2_exec_output(c, 0, &out_idx, &out) != NPRG2_EXEC_OK) return 0;

    const uint8_t *ig = in.geometry_record, *og = out.geometry_record;
    if (!ig || !og) return 0;

    /* Dtype check: both u16 */
    if (ig[0] != MEAN_DTYPE_U16 || og[0] != MEAN_DTYPE_U16) return 0;

    /* Storage kind: both direct */
    if (in.storage_kind != MEAN_STORAGE_DIRECT || out.storage_kind != MEAN_STORAGE_DIRECT) return 0;
    if (in.pool >= 7 || out.pool >= 7 || !c->pools[in.pool].base || !c->pools[out.pool].base) return 0;

    /* Class name check: both "FH" */
    if (memcmp(ig + 2, "FH", 2) || memcmp(og + 2, "FH", 2)) return 0;

    /* Input shape check: exactly 32 elements (1x2x4x4) */
    uint32_t in_count = (uint32_t)rdi16(ig + 6) * (uint32_t)rdi16(ig + 8) *
                        (uint32_t)rdi16(ig + 10) * (uint32_t)rdi16(ig + 12);
    if (in_count != MEAN_ELEMENT_COUNT) return 0;

    /* Output shape check: exactly 1 element (1x1x1x1) */
    uint32_t out_count = (uint32_t)rdi16(og + 6) * (uint32_t)rdi16(og + 8) *
                         (uint32_t)rdi16(og + 10) * (uint32_t)rdi16(og + 12);
    if (out_count != 1) return 0;

    /* Pool boundary checks */
    uint32_t in_bytes = in_count * 2u;
    uint32_t out_bytes = out_count * 2u;
    if (in.offset + in_bytes > c->pools[in.pool].bytes ||
        out.offset + out_bytes > c->pools[out.pool].bytes) return 0;
    if (out.pool == 1 && out.offset + out_bytes > c->pool1_writable_bytes) return 0;

    /* Disjointness: input and output buffers must not alias if in same pool */
    if (in.pool == out.pool && in.offset < out.offset + out_bytes && out.offset < in.offset + in_bytes) {
        return 0;
    }

    /* Quantization metadata check: matching scale and zp */
    if (rd32(ig + 30) != rd32(og + 30)) return 0;
    if (memcmp(ig + 34, og + 34, sizeof(double))) return 0;

    return 1;
}

int nprg2_exec_mean_run(const Nprg2ExecCall *c, void *ctx) {
    if (!nprg2_exec_mean_validate(c, ctx)) return 0;

    uint32_t in_idx, out_idx;
    Nprg2TensorView in, out;
    nprg2_exec_input(c, 0, &in_idx, &in);
    nprg2_exec_output(c, 0, &out_idx, &out);

    const uint16_t *src = (const uint16_t *)(c->pools[in.pool].base + in.offset);
    uint16_t *dst = (uint16_t *)(c->pools[out.pool].base + out.offset);

    int64_t sum = 0;
    for (uint32_t j = 0; j < MEAN_ELEMENT_COUNT; j++) {
        sum += (int64_t)src[j];
    }

    int64_t val = (sum + 33) >> 5;
    if (val < 0) val = 0;
    if (val > 65535) val = 65535;

    dst[0] = (uint16_t)val;
    return 1;
}

int nprg2_executor_register_mean(Nprg2Executor *e) {
    if (!e) return NPRG2_EXEC_ARGUMENT;
    return nprg2_executor_register(e, QDR_MEAN, 47, nprg2_exec_mean_validate, nprg2_exec_mean_run, NULL);
}
