/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Whole-graph FP16 check on the instruction simulator. Runs one model serially
 * and compares output bits with a saved reference (normally a Frame report),
 * so a backend change that should not alter numerics can be checked off-device.
 * usage: test_model IMAGE CONSTANTS INPUT_F32 REFERENCE_PREFIX */
#include "../native/tensor/fp16.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hexagon_standalone.h>
#include <hexagon_sim_timer.h>
static void *slurp(const char *path, uint32_t *bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("FAIL open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    *bytes = (uint32_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = aligned_alloc(128, (*bytes + 255u) & ~127u);
    if (!p || fread(p, 1, *bytes, f) != *bytes)
        exit(1);
    fclose(f);
    return p;
}
int main(int argc, char **argv) {
    if (argc != 5)
        return 2;
    uint32_t base = __rdcfg(__vtcm_base) << 16;
    for (uint32_t i = 0; i < 0x800000u; i += 4096)
        add_translation((void *)(base + i), (void *)(base + i), 6);
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" ::"r"(ssr) : "memory");
    uint32_t image_bytes, constant_bytes, input_bytes, bad = 0;
    void *image = slurp(argv[1], &image_bytes), *constants = slurp(argv[2], &constant_bytes);
    void *input = slurp(argv[3], &input_bytes);
    const HxHeader *h = image;
    uint8_t *exchange = aligned_alloc(128, h->arena_bytes);
    static HxGraph g;
    uint32_t error = hx_graph_open(&g, image, image_bytes, constants, constant_bytes, exchange,
                                   h->arena_bytes, &bad);
    if (error) {
        printf("FAIL open error %u op %u\n", (unsigned)error, (unsigned)bad);
        return 1;
    }
    uint8_t *arena = (uint8_t *)base;
    uint32_t *tables = malloc(hx_fp16_table_words(&g) * 4u);
    static struct HxFp16Plan plan;
    g.arena = arena;
    if (!hx_fp16_plan(&g, &plan, tables, arena + 0x800000)) {
        puts("FAIL plan");
        return 1;
    }
    g.fp16 = &plan;
    uint32_t resident = 0, copied = 0;
    for (uint32_t i = 0; i < h->ops; i++)
        if (g.ops[i].kind == HX_CONV) {
            uint32_t n = g.ops[i].p[10] * (g.tensors[g.ops[i].out].stride / 32u) * 2048u;
            if (plan.weights[i])
                resident += n;
            else
                copied += n;
        }
    printf("plan peak %u resident weights %u copied weights %u workspace %u\n", (unsigned)plan.peak,
           (unsigned)resident, (unsigned)copied, (unsigned)plan.workspace_bytes);
    hx_fp16_stage(&g, NULL);
    /* Simulated cycles per operator kind: three per packet for one thread, so
     * this tracks issue cost. The simulator does not model HMX or memory time. */
    static const char *const names[] = {"",    "CONV", "DW",   "ADD",  "PRELU", "MAX",
                                        "PAD", "RESZ", "RSHP", "CAT",  "MEAN",  "LOGI"};
    uint64_t kinds[12] = {0}, begin, end, total = 0;
    begin = hexagon_sim_read_pcycles();
    hx_fp16_import(&g, g.io[0], input, NULL);
    end = hexagon_sim_read_pcycles();
    printf("cycles IMPORT=%llu", (unsigned long long)(end - begin));
    total += end - begin;
    for (uint32_t i = 0; i < h->ops; i++) {
        begin = hexagon_sim_read_pcycles();
        if (g.ops[i].kind == HX_CONV)
            hx_fp16_conv(&g, i, NULL);
        else
            hx_fp16_op(&g, i, 0, 1);
        hx_fp16_complete(&g, i);
        end = hexagon_sim_read_pcycles();
        kinds[g.ops[i].kind] += end - begin;
        total += end - begin;
    }
    for (uint32_t k = 1; k < 12; k++)
        if (kinds[k])
            printf(" %s=%llu", names[k], (unsigned long long)kinds[k]);
    printf(" total=%llu\n", (unsigned long long)total);
    int failed = 0;
    for (uint32_t i = 0; i < h->outputs; i++) {
        const uint32_t id = g.io[h->inputs + i];
        const HxTensor *t = &g.tensors[id];
        float *got = aligned_alloc(128, hx_tensor_bytes(t));
        hx_fp16_export(&g, id, got, NULL);
        char path[512];
        snprintf(path, sizeof path, "%s%u.f32", argv[4], (unsigned)i);
        uint32_t bytes;
        float *want = slurp(path, &bytes);
        /* Reference files hold logical channels only. */
        uint32_t differ = 0, n = t->h * t->w * t->c, first = n;
        float worst = 0;
        if (bytes != n * 4u) {
            printf("FAIL reference size %u for %u\n", (unsigned)bytes, (unsigned)(n * 4u));
            return 1;
        }
        for (uint32_t j = 0; j < n; j++) {
            float a = got[(j / t->c) * t->stride + j % t->c], b = want[j];
            if (memcmp(&a, &b, 4)) {
                if (first == n)
                    first = j;
                differ++;
                float d = a > b ? a - b : b - a;
                if (d > worst)
                    worst = d;
            }
        }
        printf("output %u: %u/%u differ, first %u, worst %g\n", (unsigned)i, (unsigned)differ,
               (unsigned)n, (unsigned)first, (double)worst);
        failed |= differ != 0;
    }
    puts(failed ? "DIFFERENT" : "PASS identical to reference");
    return failed;
}
