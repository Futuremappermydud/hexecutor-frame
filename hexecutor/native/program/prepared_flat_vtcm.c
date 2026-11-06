#include "prepared_flat_vtcm.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

enum { FLAT_VTCM_OPCODE = QDR_COPY_VALUES, FLAT_VTCM_VARIANT = 65,
       FLAT_VTCM_F32 = 6, FLAT_VTCM_DIRECT = 1, FLAT_VTCM_SHAPE = 1 };

static uint16_t rd16fv(const uint8_t *p) { return (uint16_t)p[0] | (uint16_t)p[1] << 8; }
static int32_t rdi16fv(const uint8_t *p) { return (int16_t)rd16fv(p); }

typedef struct { Nprg2TensorView in, out; uint32_t bytes; } FlatVtcmPlan;

static int make_plan(const Nprg2ExecCall *c, FlatVtcmPlan *p) {
    if (!c || !p || c->command.opcode != FLAT_VTCM_OPCODE ||
        c->command.variant != FLAT_VTCM_VARIANT || c->command.input_count != 1 ||
        c->command.output_count != 1 || c->command.vector_count || c->command.call_count ||
        c->command.duplicate_pad || c->command.io_index != UINT16_MAX) return 0;

    uint32_t ii, oi;
    if (nprg2_exec_input(c, 0, &ii, &p->in) || nprg2_exec_output(c, 0, &oi, &p->out)) return 0;
    const Nprg2TensorView *in = &p->in, *out = &p->out;
    const uint8_t *ig = in->geometry_record, *og = out->geometry_record;
    if (!ig || !og || in->storage_kind != FLAT_VTCM_DIRECT ||
        out->storage_kind != FLAT_VTCM_DIRECT || in->pool != 3 || out->pool != 1 ||
        !c->pools[1].base || !c->pools[3].base || ig[0] != FLAT_VTCM_F32 ||
        og[0] != FLAT_VTCM_F32 || memcmp(ig + 2, "ff", 2) || memcmp(og + 2, "Ff", 2) ||
        !(ig[1] & FLAT_VTCM_SHAPE) || !(og[1] & FLAT_VTCM_SHAPE) ||
        (ig[1] & ~FLAT_VTCM_SHAPE) || (og[1] & ~FLAT_VTCM_SHAPE)) return 0;

    uint64_t elements = 1;
    for (unsigned d = 0; d < 4; ++d) {
        int32_t id = rdi16fv(ig + 6 + 2*d), im = rdi16fv(ig + 14 + 2*d);
        int32_t ip = rdi16fv(ig + 22 + 2*d), od = rdi16fv(og + 6 + 2*d);
        int32_t om = rdi16fv(og + 14 + 2*d), op = rdi16fv(og + 22 + 2*d);
        if (id <= 0 || id != im || id != od || id != om || ip || op) return 0;
        elements *= (uint32_t)id;
        if (elements > UINT32_MAX / sizeof(float)) return 0;
    }
    p->bytes = (uint32_t)elements * sizeof(float);
    if (in->offset > c->pools[3].bytes || p->bytes > c->pools[3].bytes - in->offset ||
        out->offset > c->pool1_writable_bytes || p->bytes > c->pool1_writable_bytes - out->offset ||
        out->offset > c->pools[1].bytes || p->bytes > c->pools[1].bytes - out->offset) return 0;

    uintptr_t src_base = (uintptr_t)c->pools[3].base;
    uintptr_t dst_base = (uintptr_t)c->pools[1].base;
    if (src_base > UINTPTR_MAX - in->offset || dst_base > UINTPTR_MAX - out->offset) return 0;
    uintptr_t src = src_base + in->offset;
    uintptr_t dst = dst_base + out->offset;
    if (src > UINTPTR_MAX - p->bytes || dst > UINTPTR_MAX - p->bytes ||
        (src < dst + p->bytes && dst < src + p->bytes)) return 0;
    return 1;
}

int nprg2_exec_flat_vtcm_validate(const Nprg2ExecCall *c, void *ctx) {
    (void)ctx;
    FlatVtcmPlan p;
    return make_plan(c, &p);
}

int nprg2_exec_flat_vtcm_run(const Nprg2ExecCall *c, void *ctx) {
    (void)ctx;
#ifdef CMD_MEMO
    typedef struct { uint8_t *dst; const uint8_t *src; uint32_t n; } FlatMemo;
    {
        const FlatMemo *m = nprg2_exec_cmd_memo(c);
        if (m) { memcpy(m->dst, m->src, m->n); return 1; }
    }
#endif
    FlatVtcmPlan p;
    if (!make_plan(c, &p)) return 0;
#ifdef CMD_MEMO
    {
        FlatMemo *n = nprg2_exec_cmd_memo_new(c, sizeof *n);
        if (n) {
            n->dst = c->pools[1].base + p.out.offset; n->src = c->pools[3].base + p.in.offset; n->n = p.bytes;
            nprg2_exec_cmd_memo_commit(c, n);
        }
    }
#endif
    memcpy(c->pools[1].base + p.out.offset, c->pools[3].base + p.in.offset, p.bytes);
    return 1;
}

int nprg2_executor_register_flat_vtcm(Nprg2Executor *e) {
    return nprg2_executor_register(e, FLAT_VTCM_OPCODE, FLAT_VTCM_VARIANT,
                                   nprg2_exec_flat_vtcm_validate,
                                   nprg2_exec_flat_vtcm_run, NULL);
}
