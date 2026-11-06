#include "prepared_reshape_cb.h"

#include <stdint.h>
#include <string.h>
#if defined(CMD_MEMO) && defined(__hexagon__) && defined(__HVX__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

enum { VARIANT_RESHAPE_HW_CB = 49, DTYPE_U8 = 1, STORAGE_BLOCKED = 2,
       SHAPE = 1, QUANT = 32, CROUTON = 64, PAGE_BYTES = 2048,
       SOURCE_PAGES = 32, DEST_PAGES = 4 };

typedef struct {
    const uint8_t *source[SOURCE_PAGES];
    uint8_t *dest[DEST_PAGES];
    uint32_t source_row;
} ReshapePlan;

static uint16_t rd16(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8);
}

static int shape(const Nprg2TensorView *v, const uint16_t dims[4],
                 const uint16_t maximum[4], int source) {
    if (!v || !v->geometry_record || v->storage_kind != STORAGE_BLOCKED ||
        !v->blocks || !v->block_count) return 0;
    const uint8_t *g = v->geometry_record;
    if (g[0] != DTYPE_U8 || (g[1] & (SHAPE | QUANT | CROUTON)) !=
            (SHAPE | QUANT | CROUTON) || memcmp(g + 2, "CB", 2)) return 0;
    for (unsigned i = 0; i < 4; i++) {
        uint16_t pad = rd16(g + 22 + 2 * i);
        if (rd16(g + 6 + 2 * i) != dims[i] ||
            rd16(g + 14 + 2 * i) != maximum[i] ||
            (source && i == 1 ? pad > 7u : pad != 0u)) return 0;
    }
    return 1;
}

static int page(const Nprg2ExecCall *c, const Nprg2TensorView *v,
                uint32_t logical_index, int output, uint8_t **ptr) {
    if (v->slice_origin > v->block_count ||
        logical_index >= v->block_count - v->slice_origin) return 0;
    uint32_t encoded = v->blocks[v->slice_origin + logical_index];
    uint32_t pool = encoded & 127u, offset = encoded & ~127u;
    if (pool != 1 || !c->pools[1].base || offset % PAGE_BYTES ||
        offset > c->pools[1].bytes || PAGE_BYTES > c->pools[1].bytes - offset ||
        (output && (offset > c->pool1_writable_bytes ||
                    PAGE_BYTES > c->pool1_writable_bytes - offset))) return 0;
    *ptr = c->pools[1].base + offset;
    return 1;
}

static int make_plan(const Nprg2ExecCall *c, ReshapePlan *plan) {
    if (!c || !plan || !c->pools || c->command.opcode != QDR_COPY_VALUES ||
        c->command.variant != VARIANT_RESHAPE_HW_CB ||
        c->command.input_count != 1 || c->command.output_count != 1 ||
        c->command.vector_count || c->command.call_count ||
        c->command.duplicate_pad || c->command.io_index != UINT16_MAX) return 0;
    uint32_t in_id, out_id;
    Nprg2TensorView in, out;
    if (nprg2_exec_input(c, 0, &in_id, &in) != NPRG2_EXEC_OK ||
        nprg2_exec_output(c, 0, &out_id, &out) != NPRG2_EXEC_OK) return 0;
    (void)in_id; (void)out_id;
    static const uint16_t in_d[4] = {1, 1, 64, 128};
    static const uint16_t in_m[4] = {1, 8, 64, 128};
    static const uint16_t out_d[4] = {1, 8, 8, 128};
    if (!shape(&in, in_d, in_m, 1) || !shape(&out, out_d, out_d, 0) ||
        memcmp(in.geometry_record + 30, out.geometry_record + 30, 12)) return 0;
    plan->source_row = rd16(in.geometry_record + 22 + 2u);

    /* The active source comprises 8 width tiles by 4 depth tiles. The
     * destination comprises 4 depth tiles. Require full-page disjointness
     * before writing, including duplicated page descriptors. */
    for (uint32_t i = 0; i < SOURCE_PAGES; i++) {
        uint8_t *p;
        if (!page(c, &in, i, 0, &p)) return 0;
        plan->source[i] = p;
    }
    for (uint32_t i = 0; i < DEST_PAGES; i++) {
        if (!page(c, &out, i, 1, &plan->dest[i])) return 0;
    }
    for (uint32_t i = 0; i < SOURCE_PAGES; i++) {
        uintptr_t a = (uintptr_t)plan->source[i];
        for (uint32_t j = 0; j < i; j++)
            if (a == (uintptr_t)plan->source[j]) return 0;
        for (uint32_t j = 0; j < DEST_PAGES; j++)
            if (a == (uintptr_t)plan->dest[j]) return 0;
    }
    for (uint32_t i = 0; i < DEST_PAGES; i++)
        for (uint32_t j = 0; j < i; j++)
            if (plan->dest[i] == plan->dest[j]) return 0;
    return 1;
}

int nprg2_exec_reshape_cb_validate(const Nprg2ExecCall *c, void *ctx) {
    (void)ctx;
    ReshapePlan plan;
    return make_plan(c, &plan);
}

#ifdef CMD_MEMO
/* Every row is 8 pixels of 32 bytes, contiguous and 256-byte aligned on both sides. */
static void reshape_rows(const ReshapePlan *p) {
    for (uint32_t h = 0; h < 8; h++)
        for (uint32_t d = 0; d < 4; d++) {
            const uint8_t *src = p->source[h * 4 + d] + p->source_row * 256;
            uint8_t *dst = p->dest[d] + h * 256;
#if defined(__hexagon__) && defined(__HVX__)
            const HVX_Vector a = ((const HVX_Vector *)src)[0], b = ((const HVX_Vector *)src)[1];
            ((HVX_Vector *)dst)[0] = a; ((HVX_Vector *)dst)[1] = b;
#else
            memcpy(dst, src, 256);
#endif
        }
}
#endif
int nprg2_exec_reshape_cb_run(const Nprg2ExecCall *c, void *ctx) {
    (void)ctx;
#ifdef CMD_MEMO
    {
        const ReshapePlan *m = nprg2_exec_cmd_memo(c);
        if (m) { reshape_rows(m); return 1; }
    }
#endif
    ReshapePlan plan;
    if (!make_plan(c, &plan)) return 0;
#ifdef CMD_MEMO
    if (plan.source_row < 8u) {
        ReshapePlan *n = nprg2_exec_cmd_memo_new(c, sizeof plan);
        if (n) { *n = plan; nprg2_exec_cmd_memo_commit(c, n); reshape_rows(n); return 1; }
    }
#endif
    for (uint32_t h = 0; h < 8; h++)
        for (uint32_t w = 0; w < 8; w++)
            for (uint32_t d = 0; d < 4; d++) {
                /* Logical input width is 8*h+w. CB has 8x8x32 bytes/page;
                 * the single live input row sits at the source H padding
                 * (0..7) within each of its 32 source pages. */
                const uint8_t *src = plan.source[h * 4 + d] +
                    plan.source_row * 256 + w * 32;
                uint8_t *dst = plan.dest[d] + h * 256 + w * 32;
                memcpy(dst, src, 32);
            }
    return 1;
}

int nprg2_executor_register_reshape_cb(Nprg2Executor *e) {
    if (!e) return NPRG2_EXEC_ARGUMENT;
    return nprg2_executor_register(e, QDR_COPY_VALUES, VARIANT_RESHAPE_HW_CB,
                                   nprg2_exec_reshape_cb_validate,
                                   nprg2_exec_reshape_cb_run, NULL);
}
