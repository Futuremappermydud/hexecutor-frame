/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Several FP16 graphs in one VTCM arena on the instruction simulator.
 * Runs an alternating sequence twice: with standalone plans, re-staging every
 * time the graph changes (the engine's single-resident behaviour), and with
 * hx_fp16_share's disjoint records, staging each graph once. Outputs must be
 * bit-identical; cycles show the staging that sharing removes.
 * usage: test_share ROUNDS IMAGE CONSTANTS INPUT_F32 [IMAGE CONSTANTS INPUT_F32 ...] */
#include "../native/tensor/fp16.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hexagon_standalone.h>
#include <hexagon_sim_timer.h>
#define MAX_GRAPHS 8
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
static HxGraph graphs[MAX_GRAPHS];
static struct HxFp16Plan plans[MAX_GRAPHS];
static float *inputs[MAX_GRAPHS];
static uint32_t output_words(const HxGraph *g) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < g->header->outputs; i++) {
        const HxTensor *t = &g->tensors[g->io[g->header->inputs + i]];
        n += t->h * t->w * t->c;
    }
    return n;
}
static void run(HxGraph *g, uint32_t index, float *out) {
    hx_fp16_import(g, g->io[0], inputs[index], NULL);
    for (uint32_t i = 0; i < g->header->ops; i++) {
        if (g->ops[i].kind == HX_CONV)
            hx_fp16_conv(g, i, NULL);
        else
            hx_fp16_op(g, i, 0, 1);
        hx_fp16_complete(g, i);
    }
    for (uint32_t i = 0; i < g->header->outputs; i++) {
        uint32_t id = g->io[g->header->inputs + i];
        const HxTensor *t = &g->tensors[id];
        /* Compare logical channels only; padding lanes are not defined. */
        float *full = aligned_alloc(128, (hx_tensor_bytes(t) + 127u) & ~127u);
        if (!full) {
            puts("FAIL out of memory");
            exit(1);
        }
        hx_fp16_export(g, id, full, NULL);
        for (uint32_t p = 0; p < t->h * t->w; p++)
            memcpy(out + p * t->c, full + p * t->stride, t->c * 4u);
        free(full);
        out += t->h * t->w * t->c;
    }
}
/* One pass over the sequence; returns total cycles and fills outputs. */
static uint64_t per_graph[2][MAX_GRAPHS];
static uint64_t pass(uint32_t count, uint32_t rounds, int shared, float **outputs,
                     uint64_t *stage_cycles) {
    uint32_t staged = 0;
    uint64_t total = 0;
    *stage_cycles = 0;
    for (uint32_t r = 0; r < rounds; r++)
        for (uint32_t k = 0; k < count; k++) {
            uint64_t begin = hexagon_sim_read_pcycles();
            if (getenv("SHARE_TRACE"))
                printf("pass %d round %u graph %u staged %x\n", shared, (unsigned)r, (unsigned)k,
                       (unsigned)staged);
            if (!(staged & (1u << k))) {
                hx_fp16_stage(&graphs[k], NULL);
                staged = shared ? staged | (1u << k) : 1u << k;
            }
            uint64_t mid = hexagon_sim_read_pcycles();
            run(&graphs[k], k, outputs[k] + (size_t)r * output_words(&graphs[k]));
            uint64_t end = hexagon_sim_read_pcycles();
            *stage_cycles += mid - begin;
            per_graph[shared][k] += end - mid;
            total += end - begin;
        }
    return total;
}
int main(int argc, char **argv) {
    if (argc < 5 || (argc - 2) % 3 || (argc - 2) / 3 > MAX_GRAPHS)
        return 2;
    const uint32_t rounds = (uint32_t)atoi(argv[1]), count = (uint32_t)(argc - 2) / 3;
    uint32_t base = __rdcfg(__vtcm_base) << 16;
    for (uint32_t i = 0; i < 0x800000u; i += 4096)
        add_translation((void *)(base + i), (void *)(base + i), 6);
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" ::"r"(ssr) : "memory");
    uint8_t *arena = (uint8_t *)base;
    float *standalone[MAX_GRAPHS], *shared[MAX_GRAPHS];
    HxGraph *list[MAX_GRAPHS];
    for (uint32_t k = 0; k < count; k++) {
        uint32_t image_bytes, constant_bytes, input_bytes, bad = 0;
        void *image = slurp(argv[2 + 3 * k], &image_bytes);
        void *constants = slurp(argv[3 + 3 * k], &constant_bytes);
        inputs[k] = slurp(argv[4 + 3 * k], &input_bytes);
        const HxHeader *h = image;
        uint8_t *exchange = aligned_alloc(128, h->arena_bytes);
        uint32_t error = hx_graph_open(&graphs[k], image, image_bytes, constants, constant_bytes,
                                       exchange, h->arena_bytes, &bad);
        if (error) {
            printf("FAIL open graph %u error %u op %u\n", (unsigned)k, (unsigned)error,
                   (unsigned)bad);
            return 1;
        }
        graphs[k].arena = arena;
        uint32_t *tables = malloc(hx_fp16_table_words(&graphs[k]) * 4u);
        if (!exchange || !tables) {
            puts("FAIL out of memory");
            return 1;
        }
        if (!hx_fp16_plan(&graphs[k], &plans[k], tables, arena + 0x800000)) {
            printf("FAIL plan graph %u\n", (unsigned)k);
            return 1;
        }
        graphs[k].fp16 = &plans[k];
        list[k] = &graphs[k];
        size_t words = (size_t)rounds * output_words(&graphs[k]);
        standalone[k] = calloc(words, 4);
        shared[k] = calloc(words, 4);
        if (!standalone[k] || !shared[k]) {
            puts("FAIL out of memory");
            return 1;
        }
    }
    uint64_t stage_a, stage_b;
    uint64_t a = pass(count, rounds, 0, standalone, &stage_a);
    if (!hx_fp16_share(list, count, arena + 0x800000)) {
        puts("FAIL share");
        return 1;
    }
    for (uint32_t k = 0; k < count; k++) {
        uint32_t resident = 0, copied = 0;
        for (uint32_t i = 0; i < graphs[k].header->ops; i++)
            if (graphs[k].ops[i].kind == HX_CONV) {
                uint32_t n = graphs[k].ops[i].p[10] *
                             (graphs[k].tensors[graphs[k].ops[i].out].stride / 32u) * 2048u;
                if (plans[k].weights[i])
                    resident += n;
                else
                    copied += n;
            }
        printf("graph %u: peak %u shared resident weights %u copied %u workspace %u\n",
               (unsigned)k, (unsigned)plans[k].peak, (unsigned)resident, (unsigned)copied,
               (unsigned)plans[k].workspace_bytes);
    }
    uint64_t b = pass(count, rounds, 1, shared, &stage_b);
    int failed = 0;
    for (uint32_t k = 0; k < count; k++) {
        size_t words = (size_t)rounds * output_words(&graphs[k]);
        if (memcmp(standalone[k], shared[k], words * 4)) {
            printf("graph %u outputs DIFFER\n", (unsigned)k);
            failed = 1;
        }
    }
    printf("standalone total=%llu stage=%llu\nshared     total=%llu stage=%llu\n",
           (unsigned long long)a, (unsigned long long)stage_a, (unsigned long long)b,
           (unsigned long long)stage_b);
    for (uint32_t k = 0; k < count; k++)
        printf("graph %u run cycles (excluding staging): standalone %llu shared %llu\n", (unsigned)k,
               (unsigned long long)per_graph[0][k], (unsigned long long)per_graph[1][k]);
    puts(failed ? "FAIL outputs differ" : "PASS identical outputs");
    return failed;
}
