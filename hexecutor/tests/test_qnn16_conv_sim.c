/* Hexagon simulator check of QNN FP16 run-list handlers against fixtures
 * exported by hexecutor/offline/qnn16_oracle.py from QAIRT's x86 HTP simulator.
 * Usage: hexagon-sim -mv75 test_qnn16_conv.elf -- FIXTURE...
 * The x86 simulator evaluates HMX in exact FP32 before rounding to FP16, while
 * the ISS models the hardware datapath (as QNN on silicon runs it), so each
 * output half must be within the fixture's per-element tolerance:
 * 2^-10 * sum |a * w * scale| plus one FP16 ULP plus 2^-14 (subnormals); padding
 * lanes are don't-care.
 * Fixtures are ignored local artifacts. */
#include "qnn16.h"

#include <hexagon_standalone.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VTCM_MAP (8u << 20)
#define BLOCK 2048u

static uint8_t *vtcm;
static const char *dump_dir; /* --dump=DIR: write mismatching outputs as opN.got */
/* cvt.hf = acc() parameters by table count: 2 converts; a second table with 6
 * applies its clamp (found by sweep against QNN, see the qnn16 docs). --acc2=A,B
 * overrides for experiments. */
static uint32_t acc_params[2][2] = {{2, 0}, {2, 6}};
static uint32_t vtcm_used;

static void *vtcm_take(uint32_t bytes) {
    uint32_t at = (vtcm_used + BLOCK - 1) & ~(BLOCK - 1);
    if (at + bytes > VTCM_MAP - BLOCK) return NULL; /* last tile: completion */
    vtcm_used = at + bytes;
    return vtcm + at;
}

/* Distance in representable FP16 values; +0 and -0 are one value. */
static uint32_t ulp_distance(uint16_t a, uint16_t b) {
    int32_t x = (a & 0x8000u) ? -(int32_t)(a & 0x7fffu) : (int32_t)a;
    int32_t y = (b & 0x8000u) ? -(int32_t)(b & 0x7fffu) : (int32_t)b;
    return (uint32_t)(x > y ? x - y : y - x);
}

static uint32_t worst_ulp, total_diff, total_halves;
static float worst_ratio;

static float half_to_float(uint16_t h) {
    union {
        uint16_t u;
        __fp16 f;
    } v = {h};
    return (float)v.f;
}

static int read_exact(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

static int run_conv1x1(FILE *f, uint32_t count, const char *name) {
    int failures = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t hdr[5];
        if (!read_exact(f, hdr, sizeof hdr)) return -1;
        const uint32_t seq = hdr[0], m = hdr[1], k = hdr[2], n = hdr[3], tables = hdr[4];
        if (tables < 1 || tables > 2) return -1;
        vtcm_used = 0;
        uint8_t *a = vtcm_take(m * k * BLOCK), *w = vtcm_take(n * k * BLOCK), *fi = vtcm_take(n * tables * 256u),
                *c = vtcm_take(m * n * BLOCK);
        uint8_t *expect = malloc(m * n * BLOCK);
        float *tol = malloc(m * n * (BLOCK / 2) * sizeof(float));
        const uint8_t **ap = malloc(m * k * sizeof *ap);
        uint8_t **cp = malloc(m * n * sizeof *cp);
        if (!a || !w || !fi || !c || !expect || !tol || !ap || !cp) {
            printf("SKIP %s op %u: m=%u k=%u n=%u does not fit\n", name, (unsigned)seq, (unsigned)m, (unsigned)k,
                   (unsigned)n);
            fseek(f, (long)(m * k * BLOCK + n * k * BLOCK + n * tables * 256u + m * n * BLOCK * 3u), SEEK_CUR);
            free(expect), free(tol), free(ap), free(cp);
            continue;
        }
        /* Reversed input and output block placement exercises the block tables. */
        for (uint32_t j = 0; j < m * k; j++) {
            ap[j] = a + (m * k - 1 - j) * BLOCK;
            if (!read_exact(f, (void *)ap[j], BLOCK)) return -1;
        }
        if (!read_exact(f, w, n * k * BLOCK) || !read_exact(f, fi, n * tables * 256u) || !read_exact(f, expect, m * n * BLOCK) ||
            !read_exact(f, tol, m * n * (BLOCK / 2) * sizeof(float)))
            return -1;
        for (uint32_t j = 0; j < m * n; j++) cp[j] = c + (m * n - 1 - j) * BLOCK;
        memset(c, 0xa5, m * n * BLOCK);
        Qnn16Conv1x1 p = {ap, w, fi, (uint8_t *const *)cp, m, k, n, tables, acc_params[tables - 1]};
        qnn16_hmx_reset();
        qnn16_conv1x1(&p);
        qnn16_hmx_finish(vtcm + VTCM_MAP - BLOCK);
        uint32_t bad = 0, first = 0, far = 0;
        for (uint32_t j = 0; j < m * n; j++)
            for (uint32_t e = 0; e < BLOCK / 2; e++) {
                const uint32_t at = j * (BLOCK / 2) + e;
                uint16_t got = ((const uint16_t *)cp[j])[e], want = ((const uint16_t *)expect)[at];
                if (tol[at] > 3.0e38f) continue; /* padding lane */
                float g = half_to_float(got), x = half_to_float(want), diff = g > x ? g - x : x - g;
                uint32_t d = ulp_distance(got, want);
                if (d) total_diff++;
                if (d > worst_ulp) worst_ulp = d;
                if (!(diff <= tol[at])) {
                    if (!bad++) first = at;
                    far++;
                }
                if (diff / tol[at] > worst_ratio) worst_ratio = diff / tol[at];
                total_halves++;
            }
        if (far && dump_dir) {
            char path[256];
            snprintf(path, sizeof path, "%s/op%u.got", dump_dir, (unsigned)seq);
            FILE *d = fopen(path, "wb");
            if (d) {
                for (uint32_t j = 0; j < m * n; j++) fwrite(cp[j], 1, BLOCK, d);
                fclose(d);
            }
        }
        if (far) {
            uint16_t got = ((const uint16_t *)cp[first / 1024])[first % 1024];
            uint16_t want = ((const uint16_t *)expect)[first];
            printf("FAIL %s op %u m=%u k=%u n=%u: %u halves outside tolerance, first %u got %04x want %04x\n", name, (unsigned)seq,
                   (unsigned)m, (unsigned)k, (unsigned)n, (unsigned)bad, (unsigned)first, got, want);
            failures++;
        }
        free(expect), free(tol), free(ap), free(cp);
    }
    return failures;
}

int main(int argc, char **argv) {
    uint32_t base = __rdcfg(__vtcm_base) << 16;
    for (uint32_t i = 0; i < VTCM_MAP; i += 4096) add_translation((void *)(base + i), (void *)(base + i), 6);
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" ::"r"(ssr) : "memory");
    vtcm = (uint8_t *)base;
    int failures = 0, cases = 0;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--acc2=", 7)) {
            sscanf(argv[i] + 7, "%lx,%lx", (unsigned long *)&acc_params[1][0], (unsigned long *)&acc_params[1][1]);
            continue;
        }
        if (!strncmp(argv[i], "--dump=", 7)) {
            dump_dir = argv[i] + 7;
            continue;
        }
        FILE *f = fopen(argv[i], "rb");
        char magic[4];
        uint32_t h[3];
        if (!f || !read_exact(f, magic, 4) || memcmp(magic, "Q16F", 4) || !read_exact(f, h, sizeof h) || h[0] != 3) {
            printf("FAIL cannot read fixture %s\n", argv[i]);
            return 1;
        }
        int r = h[1] == 1 ? run_conv1x1(f, h[2], argv[i]) : -2;
        fclose(f);
        if (r < 0) {
            printf("FAIL fixture %s: %s\n", argv[i], r == -2 ? "unknown kind" : "truncated");
            return 1;
        }
        failures += r;
        cases += (int)h[2];
    }
    if (failures) {
        printf("FAIL %d of %d QNN FP16 cases\n", failures, cases);
        return 1;
    }
    printf("PASS qnn16 conv1x1: %d cases within tolerance; %u of %u valid halves differ from the x86 oracle "
           "(worst %u ULP, worst error/tolerance %.3f)\n",
           cases, (unsigned)total_diff, (unsigned)total_halves, (unsigned)worst_ulp, (double)worst_ratio);
    return 0;
}
