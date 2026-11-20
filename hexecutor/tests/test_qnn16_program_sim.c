/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Hexagon simulator run of a whole Q16P program with the HMX 1x1 leaf.
 * Usage: hexagon-sim -mv75 test_qnn16_program.elf -- [--compare] PROGRAM.q16p CONSTANTS.q16c OUT_DIR IN0.f32 [IN1.f32 ...]
 * Writes OUT_DIR/out<N>.f32 per graph output and prints pcycles per
 * operation kind. --compare runs every depthwise and space-to-depth op twice,
 * float path then HVX tile path, and reports their output differences.
 * Inputs, programs and outputs are ignored local files. */
#include "program.h"

#include <hexagon_sim_timer.h>
#include <hexagon_standalone.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *load(const char *path, uint32_t *bytes) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = memalign(128, (size_t)n + 128);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        return NULL;
    }
    fclose(f);
    *bytes = (uint32_t)n;
    return p;
}

static uint64_t pcycles(void) { return hexagon_sim_read_pcycles(); }

static Qnn16Tensor view_of(const Q16Program *p, uint8_t *vtcm, uint32_t id) {
    const Q16Tensor *t = &p->tensors[id];
    Qnn16Tensor v;
    v.base = vtcm;
    v.block_offsets = t->direct ? 0 : p->words + t->block_first;
    v.block_count = t->block_count;
    v.base_bytes = Q16P_COMPLETION_OFFSET;
    v.direct_offset = t->offset;
    v.dims = t->dims;
    v.full = t->full;
    v.padding = t->padding;
    v.direct = t->direct;
    return v;
}
static float h2f(uint16_t h) {
    _Float16 v;
    memcpy(&v, &h, 2);
    return (float)v;
}
static int aliases(const Q16Program *p, uint32_t a, uint32_t b) {
    const Q16Tensor *x = &p->tensors[a], *y = &p->tensors[b];
    for (uint32_t i = 0; i < x->block_count; i++)
        for (uint32_t j = 0; j < y->block_count; j++)
            if (p->words[x->block_first + i] == p->words[y->block_first + j] &&
                p->words[x->block_first + i] != Q16P_NONE)
                return 1;
    return 0;
}
static uint32_t compare_failures, compared;
/* ---- Fuzz mode: random operands for one kind, generic path against tile path,
 * with the work split in four parts as on the device. ---- */
static void four_parts(Q16Task task, void *ctx) {
    for (uint32_t part = 0; part < 4u; part++)
        task(ctx, part, 4u);
}
static void four_parts_rev(Q16Task task, void *ctx) {
    for (uint32_t part = 4u; part-- > 0;)
        task(ctx, part, 4u);
}
static uint32_t rng_state = 0x1234567u;
static uint32_t rng(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}
static float rnd(void) { /* [-4, 4), exact zero now and then */
    uint32_t r = rng();
    if (!(r & 31u))
        return 0.0f;
    return ((float)(r & 0xffffu) / 65536.0f - 0.5f) * 8.0f;
}
static void fuzz_tensor(const Q16Program *p, uint8_t *vtcm, uint32_t id, int nonneg_tail) {
    const Q16Tensor *t = &p->tensors[id];
    uint32_t n = q16_tensor_elements(t);
    if (t->direct == 2u)
        return;
    if (t->dtype == Q16_F32) {
        float *f = (float *)(vtcm + t->offset);
        for (uint32_t i = 0; i < n; i++) {
            float v = rnd();
            f[i] = nonneg_tail && i >= n / 2u && v < 0 ? -v : v;
        }
        return;
    }
    uint16_t *h = memalign(128, n * 2u + 128);
    for (uint32_t i = 0; i < n; i++) {
        _Float16 v = (_Float16)rnd();
        memcpy(&h[i], &v, 2);
    }
    Qnn16Tensor v = view_of(p, vtcm, id);
    qnn16_scatter_f16(&v, h);
    free(h);
}
static void fuzz_inputs(const Q16Program *p, uint8_t *vtcm, const Q16Op *o) {
    if (o->kind == Q16_WELFORD_MERGE) { /* partials: [mean c][m2 c], m2 >= 0 */
        for (uint32_t i = 0; i < o->p[1]; i++)
            fuzz_tensor(p, vtcm, p->words[o->p[0] + i], 1);
        return;
    }
    if (o->x != Q16P_NONE)
        fuzz_tensor(p, vtcm, o->x, 0);
    if (o->kind == Q16_MUL)
        fuzz_tensor(p, vtcm, o->y, 0);
    if (o->kind == Q16_CONCAT)
        for (uint32_t i = 0; i < o->p[2]; i++)
            fuzz_tensor(p, vtcm, p->words[o->p[1] + i], 0);
}
/* Shape signature, so fuzz mode runs each distinct operand layout once. */
static uint32_t signature(const Q16Program *p, const Q16Op *o) {
    uint32_t h = o->kind * 2654435761u;
    const uint32_t ids[3] = {o->x, o->out, o->kind == Q16_MUL ? o->y : Q16P_NONE};
    for (uint32_t k = 0; k < 3u; k++) {
        if (ids[k] == Q16P_NONE)
            continue;
        const Q16Tensor *t = &p->tensors[ids[k]];
        const uint32_t f[14] = {t->direct, t->dtype, t->dims.b, t->dims.h, t->dims.w, t->dims.c, t->full.b,
                                t->full.h, t->full.w, t->full.c, t->padding.b, t->padding.h, t->padding.w,
                                t->padding.c};
        for (uint32_t i = 0; i < 14u; i++)
            h = (h ^ f[i]) * 16777619u;
    }
    const int words = o->kind == Q16_WELFORD_MERGE || o->kind == Q16_CONCAT;
    for (uint32_t i = 0; i < 5u; i++)
        if (i != 3u && !(words && i == 0u))
            h = (h ^ o->p[i]) * 16777619u;
    return h;
}
/* Lanes outside the logical tensor must keep their contents: snapshot the
 * output's storage before the tile run, then count changed bytes outside it. */
typedef struct {
    uint8_t *copy;
    uint32_t bytes;
} Guard;
static Guard guard_take(const Q16Program *p, uint8_t *vtcm, uint32_t id) {
    const Q16Tensor *t = &p->tensors[id];
    Guard g = {0, 0};
    if (t->direct == 2u)
        return g;
    if (t->direct) {
        g.bytes = t->bytes + 256u;
        g.copy = memalign(128, g.bytes);
        memcpy(g.copy, vtcm + t->offset - 128u, g.bytes);
        return g;
    }
    g.bytes = t->block_count * 2048u;
    g.copy = memalign(128, g.bytes);
    for (uint32_t i = 0; i < t->block_count; i++)
        if (p->words[t->block_first + i] != Q16P_NONE)
            memcpy(g.copy + i * 2048u, vtcm + p->words[t->block_first + i], 2048);
    return g;
}
static uint32_t guard_check(const Q16Program *p, uint8_t *vtcm, uint32_t id, Guard g) {
    const Q16Tensor *t = &p->tensors[id];
    uint32_t bad = 0;
    if (!g.copy)
        return 0;
    if (t->direct) {
        const uint8_t *now = vtcm + t->offset - 128u;
        for (uint32_t i = 0; i < 128u; i++)
            bad += now[i] != g.copy[i];
        for (uint32_t i = 128u + t->bytes; i < g.bytes; i++)
            bad += now[i] != g.copy[i];
        free(g.copy);
        return bad;
    }
    const uint32_t nh = (t->full.h + 7u) / 8u, nw = (t->full.w + 3u) / 4u, nc = (t->full.c + 31u) / 32u;
    for (uint32_t i = 0; i < t->block_count; i++) {
        uint32_t off = p->words[t->block_first + i];
        if (off == Q16P_NONE)
            continue;
        const uint32_t g0 = i % nc, tx = (i / nc) % nw, ty = (i / (nc * nw)) % nh, b = i / (nc * nw * nh);
        for (uint32_t k = 0; k < 2048u; k += 2u) {
            const uint32_t x = tx * 4u + ((k >> 7) & 1u) * 2u + ((k >> 1) & 1u), y = ty * 8u + ((k >> 8) & 7u);
            const uint32_t c = g0 * 32u + ((k >> 2) & 31u);
            const int logical = b >= t->padding.b && b < t->padding.b + t->dims.b && y >= t->padding.h &&
                                y < t->padding.h + t->dims.h && x >= t->padding.w && x < t->padding.w + t->dims.w &&
                                c >= t->padding.c && c < t->padding.c + t->dims.c;
            if (!logical && memcmp(vtcm + off + k, g.copy + i * 2048u + k, 2))
                bad++;
        }
    }
    free(g.copy);
    return bad;
}
static int guarding;
/* Float path, then tile path; prints the largest differences. */
static uint32_t compare_op(Q16Exec *e, uint32_t i) {
    const Q16Program *p = e->program;
    const Q16Op *o = &p->ops[i];
    /* In-place operations: keep the input and restore it between the two runs. */
    uint16_t *saved = 0;
    Qnn16Tensor xv;
    if (o->x != Q16P_NONE && aliases(p, o->x, o->out)) {
        if (p->tensors[o->x].dtype != Q16_F16 || p->tensors[o->out].dtype != Q16_F16) {
            printf("compare op %u kind %u: input and output alias, skipped\n", (unsigned)i, (unsigned)o->kind);
            e->flags = 0;
            return q16_exec_op(e, i);
        }
        xv = view_of(p, e->vtcm, o->x);
        saved = memalign(128, q16_tensor_elements(&p->tensors[o->x]) * 2u + 128);
        qnn16_gather_f16(saved, &xv);
    }
    const Q16Tensor *ot = &p->tensors[o->out];
    if (ot->dtype == Q16_F32) {
        uint32_t n = q16_tensor_elements(ot);
        float *ref = memalign(128, n * 4u + 128), *got = memalign(128, n * 4u + 128);
        e->flags = Q16_EXEC_NO_TILE;
        uint32_t err = q16_exec_op(e, i);
        memcpy(ref, e->vtcm + ot->offset, n * 4u);
        e->flags = 0;
        Guard gd = {0, 0};
        if (guarding)
            gd = guard_take(p, e->vtcm, o->out);
        err |= q16_exec_op(e, i);
        uint32_t spilled = guarding ? guard_check(p, e->vtcm, o->out, gd) : 0;
        if (spilled)
            printf("  op %u wrote %u bytes outside its float32 output\n", (unsigned)i, (unsigned)spilled);
        memcpy(got, e->vtcm + ot->offset, n * 4u);
        uint32_t diff = 0, shown = 0;
        float big = 0;
        for (uint32_t k = 0; k < n; k++)
            if ((ref[k] < 0 ? -ref[k] : ref[k]) > big)
                big = ref[k] < 0 ? -ref[k] : ref[k];
        for (uint32_t k = 0; k < n; k++)
            if (memcmp(&ref[k], &got[k], 4)) {
                uint32_t a, b;
                memcpy(&a, &ref[k], 4);
                memcpy(&b, &got[k], 4);
                /* -0 versus +0 is numerically equal and allowed. */
                if (((a | b) & 0x7fffffffu) == 0)
                    continue;
                /* Moments: the closed form differs from the recurrence by float32 rounding. */
                float d = ref[k] - got[k], m = ref[k] < 0 ? -ref[k] : ref[k];
                if (d < 0)
                    d = -d;
                if ((o->kind == Q16_WELFORD_PARTIAL || o->kind == Q16_WELFORD_MERGE) && d <= m * 1e-4f + big * 1e-5f)
                    continue;
                diff++;
                if (shown++ < 3)
                    printf("  f32 elem %u: ref %08x got %08x\n", (unsigned)k, (unsigned)a, (unsigned)b);
            }
        compared++;
        if (diff || spilled)
            compare_failures++;
        printf("compare op %u kind %u: %u f32 elems, %u differ\n", (unsigned)i, (unsigned)o->kind, (unsigned)n,
               (unsigned)diff);
        free(ref);
        free(got);
        return err;
    }
    uint32_t n = q16_tensor_elements(&p->tensors[o->out]);
    uint16_t *ref = memalign(128, n * 2u + 128), *got = memalign(128, n * 2u + 128);
    Qnn16Tensor v = view_of(p, e->vtcm, o->out);
    e->flags = Q16_EXEC_NO_TILE;
    uint32_t err = q16_exec_op(e, i);
    qnn16_gather_f16(ref, &v);
    if (saved) {
        qnn16_scatter_f16(&xv, saved);
        free(saved);
    }
    e->flags = 0;
    Guard gd = {0, 0};
    if (guarding)
        gd = guard_take(p, e->vtcm, o->out);
    uint64_t t = pcycles();
    err |= q16_exec_op(e, i);
    t = pcycles() - t;
    uint32_t spilled = guarding ? guard_check(p, e->vtcm, o->out, gd) : 0;
    if (spilled)
        printf("  op %u changed %u halves outside its logical output\n", (unsigned)i, (unsigned)spilled);
    qnn16_gather_f16(got, &v);
    float worst = 0, worst_rel = 0;
    uint32_t bad = 0, diff = 0;
    for (uint32_t k = 0; k < n; k++) {
        float a = h2f(ref[k]), b = h2f(got[k]), d = a > b ? a - b : b - a;
        float m = a < 0 ? -a : a;
        if (ref[k] != got[k])
            diff++;
        if (d > worst)
            worst = d;
        if (m > 0.0625f && d / m > worst_rel)
            worst_rel = d / m;
        /* Loose screen: a few FP16 ULPs; o->kind S2D must be exact. */
        if (o->kind == Q16_S2D || o->kind == Q16_MINMAX || o->kind == Q16_X2S || o->kind == Q16_TRANSPOSE ||
                    o->kind == Q16_FORMAT || o->kind == Q16_CONCAT || o->kind == Q16_IM2COL ||
                    o->kind == Q16_SHUFFLE || o->kind == Q16_RESHAPE || o->kind == Q16_DECIMATE ||
                    o->kind == Q16_RESIZE_NEAREST || o->kind == Q16_CAST_F16
                ? ref[k] != got[k] : (d > m * (1.0f / 256.0f) + (1.0f / 4096.0f) || d != d))
            bad++;
    }
    compared++;
    if (bad || spilled)
        compare_failures++;
    printf("compare op %u kind %u: %u elems, %u differ, %u over, max abs %.6g, max rel %.3g, tile %llu pcycles\n",
           (unsigned)i, (unsigned)o->kind, (unsigned)n, (unsigned)diff, (unsigned)bad, (double)worst,
           (double)worst_rel, (unsigned long long)t);
    free(ref);
    free(got);
    return err;
}

int main(int argc, char **argv) {
    /* --parts4 / --parts4-rev anywhere: run the job parts one after another, forward or
     * reversed, instead of serially (exposes cross-part hazards of batched operations). */
    Q16ParallelFn emulate = 0;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--parts4") || !strcmp(argv[a], "--parts4-rev")) {
            emulate = argv[a][8] ? four_parts_rev : four_parts;
            for (int b = a; b < argc - 1; b++)
                argv[b] = argv[b + 1];
            argc--;
            a--;
        }
    }
    int compare = argc > 1 && !strcmp(argv[1], "--compare");
    if (compare) {
        argv++;
        argc--;
    }
    uint32_t stop = 0xffffffffu;
    if (argc > 2 && !strcmp(argv[1], "--stop")) { /* run only the first N operations */
        stop = (uint32_t)strtoul(argv[2], 0, 10);
        argv += 2;
        argc -= 2;
    }
    /* --fuzz KIND [--all]: only ops of that kind (one per operand layout unless --all), random inputs. */
    uint32_t fuzz = 0, fuzz_all = 0;
    if (argc > 2 && !strcmp(argv[1], "--fuzz")) {
        fuzz = (uint32_t)strtoul(argv[2], 0, 10);
        argv += 2;
        argc -= 2;
        if (argc > 1 && !strcmp(argv[1], "--all")) {
            fuzz_all = 1;
            argv++;
            argc--;
        }
    }
    if (argc < 5) {
        printf("usage: PROGRAM CONSTANTS OUT_DIR INPUT...\n");
        return 2;
    }
    uint32_t base = __rdcfg(__vtcm_base) << 16;
    for (uint32_t i = 0; i < Q16P_VTCM_BYTES; i += 4096)
        add_translation((void *)(base + i), (void *)(base + i), 6);
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" ::"r"(ssr) : "memory");

    uint32_t image_bytes = 0, constant_bytes = 0;
    void *image = load(argv[1], &image_bytes);
    uint8_t *constants = load(argv[2], &constant_bytes);
    if (!image || !constants) {
        printf("FAIL cannot read program\n");
        return 1;
    }
    Q16Program program;
    uint32_t bad;
    uint32_t err = q16_program_open(&program, image, image_bytes, constant_bytes, &bad);
    if (err) {
        printf("FAIL program rejected: error %u at op %u\n", (unsigned)err, (unsigned)bad);
        return 1;
    }
    const Q16Header *h = program.header;
    if ((uint32_t)(argc - 4) != h->inputs) {
        printf("FAIL program has %u inputs\n", (unsigned)h->inputs);
        return 1;
    }
    uint8_t *exchange = memalign(128, h->exchange_bytes);
    uint8_t *scratch = memalign(128, h->scratch_bytes);
    memset(exchange, 0, h->exchange_bytes);
    for (uint32_t i = 0; i < h->inputs; i++) {
        uint32_t n = 0;
        uint8_t *in = load(argv[4 + i], &n);
        if (!in || n != program.io[i].bytes) {
            printf("FAIL input %u: %u bytes, want %u\n", (unsigned)i, (unsigned)n, (unsigned)program.io[i].bytes);
            return 1;
        }
        memcpy(exchange + program.io[i].offset, in, n);
    }
    Q16Exec e = {&program, (uint8_t *)base, exchange, scratch, constants, qnn16_conv1x1, qnn16_hmx_finish, emulate, 0, 0};
    static uint64_t kind_cycles[Q16_KIND_END];
    static uint32_t kind_count[Q16_KIND_END];
    qnn16_hmx_reset();
    if (fuzz) {
        static uint32_t seen[4096];
        uint32_t nseen = 0;
        guarding = 1;
        e.parallel = four_parts;
        for (uint32_t i = 0; i < h->ops; i++) {
            const Q16Op *o = &program.ops[i];
            if (o->kind == Q16_COPY || o->kind == Q16_ZERO) {
                e.flags = 0;
                if (q16_exec_op(&e, i)) {
                    printf("FAIL op %u\n", (unsigned)i);
                    return 1;
                }
                continue;
            }
            if (o->kind != fuzz)
                continue;
            uint32_t sig = signature(&program, o), k = 0;
            while (k < nseen && seen[k] != sig)
                k++;
            if (k < nseen && !fuzz_all)
                continue;
            if (k == nseen && nseen < 4096u)
                seen[nseen++] = sig;
            fuzz_inputs(&program, e.vtcm, o);
            const Q16Tensor *xt = &program.tensors[o->x == Q16P_NONE ? o->out : o->x], *yt = &program.tensors[o->out];
            printf("fuzz op %u: x %s [%u %u %u %u] -> out %s [%u %u %u %u]\n", (unsigned)i,
                   xt->direct ? "direct" : "tiles", (unsigned)xt->dims.b, (unsigned)xt->dims.h, (unsigned)xt->dims.w,
                   (unsigned)xt->dims.c, yt->direct ? "direct" : "tiles", (unsigned)yt->dims.b, (unsigned)yt->dims.h,
                   (unsigned)yt->dims.w, (unsigned)yt->dims.c);
            if (compare_op(&e, i)) {
                printf("FAIL op %u kind %u\n", (unsigned)i, (unsigned)o->kind);
                return 1;
            }
        }
        printf("fuzz kind %u: %u layouts, %u failed\n", (unsigned)fuzz, (unsigned)compared, (unsigned)compare_failures);
        printf(compare_failures || !compared ? "FAIL qnn16 fuzz\n" : "PASS qnn16 fuzz\n");
        return compare_failures || !compared ? 1 : 0;
    }
    uint64_t start = pcycles();
    for (uint32_t i = 0; i < h->ops && i < stop;) {
        uint64_t t = pcycles();
        uint32_t k = program.ops[i].kind, ran = 1;
        if (compare && (k == Q16_DW3X3 || k == Q16_DW3X3_S2 || k == Q16_DW1X1 || k == Q16_S2D || k == Q16_CONV3X3 || k == Q16_PW1X1 || k == Q16_CONV3X3_S2 || k == Q16_SWISH || k == Q16_ADD_SWISH ||
                        k == Q16_INSTANCE_NORM || k == Q16_WELFORD_PARTIAL ||
                        k == Q16_MUL || k == Q16_MINMAX || k == Q16_X2S || k == Q16_TRANSPOSE ||
                        k == Q16_FORMAT || k == Q16_CAST_F32 || k == Q16_AVGPOOL || k == Q16_CONCAT || k == Q16_IM2COL ||
                        k == Q16_SHUFFLE || k == Q16_RESHAPE || k == Q16_RESIZE_BILINEAR || k == Q16_RESIZE_NEAREST ||
                        k == Q16_DECIMATE || k == Q16_SOFTMAX || k == Q16_REDUCE_SUM || k == Q16_REDUCE_MEAN || k == Q16_L2NORM || k == Q16_CAST_F16 ||
                        k == Q16_SLICE_F32 || k == Q16_WELFORD_MERGE))
            err = compare_op(&e, i);
        else if (stop == 0xffffffffu)
            err = q16_exec_run(&e, i, &ran);
        else
            err = q16_exec_op(&e, i);
        if (err) {
            printf("FAIL op %u kind %u: error %u\n", (unsigned)i, (unsigned)program.ops[i].kind, (unsigned)err);
            return 1;
        }
        kind_cycles[k] += pcycles() - t;
        kind_count[k] += ran;
#ifdef Q16_SIM_PER_OP /* QNN16_SIM_CFLAGS=-DQ16_SIM_PER_OP: one line per operation (or batched run) */
        printf("op %u kind %u: %llu pcycles\n", (unsigned)i, (unsigned)k, (unsigned long long)(pcycles() - t));
#endif
        i += ran;
    }
    uint64_t total = pcycles() - start;
    for (uint32_t k = 1; k < Q16_KIND_END; k++)
        if (kind_count[k])
            printf("kind %2u: %4u ops %12llu pcycles\n", (unsigned)k, (unsigned)kind_count[k],
                   (unsigned long long)kind_cycles[k]);
    for (uint32_t i = 0; i < h->outputs; i++) {
        char path[512];
        snprintf(path, sizeof path, "%s/out%u.f32", argv[3], (unsigned)i);
        FILE *f = fopen(path, "wb");
        const Q16Io *io = &program.io[h->inputs + i];
        if (!f || fwrite(exchange + io->offset, 1, io->bytes, f) != io->bytes) {
            printf("FAIL cannot write %s\n", path);
            return 1;
        }
        fclose(f);
    }
    if (compare)
        printf("compare: %u ops, %u over the screen\n", (unsigned)compared, (unsigned)compare_failures);
    printf("PASS qnn16 program sim: %u ops, %u matrix ops, %llu pcycles\n", (unsigned)h->ops,
           (unsigned)e.matrix_ops, (unsigned long long)total);
    return 0;
}
