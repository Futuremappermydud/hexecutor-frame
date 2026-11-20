/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_reshape_cb.h"

#if defined(__hexagon__)
#include <hexagon_standalone.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

static uint8_t *read_file(const char *path, uint32_t *n) {
    FILE *f = fopen(path, "rb");
    CHECK(f);
    CHECK(!fseek(f, 0, SEEK_END));
    long z = ftell(f);
    CHECK(z > 0 && (uint64_t)z <= 32u * 1024u * 1024u);
    CHECK(!fseek(f, 0, SEEK_SET));
    uint8_t *p = malloc((size_t)z);
    CHECK(p && fread(p, 1, (size_t)z, f) == (size_t)z);
    fclose(f);
    *n = (uint32_t)z;
    return p;
}

static void pool_alloc(const uint8_t *raw, Nprg2Pool p[7]) {
    for (uint32_t i = 0; i < 7; i++) {
        p[i].bytes = rd32(raw + 144 + i * 4);
        p[i].base = !p[i].bytes ? NULL : (i == 1 ? aligned_alloc(128, p[i].bytes) : malloc(p[i].bytes));
        CHECK(!p[i].bytes || p[i].base);
        if (p[i].bytes) memset(p[i].base, 0, p[i].bytes);
    }
}

static void pool_free(Nprg2Pool p[7]) {
    for (int i = 0; i < 7; i++) free(p[i].base);
}

static int resolve_all(void *ctx, const Nprg2Program *p, uint32_t i, const Nprg2Command *c, Nprg2Capability *cap) {
    (void)ctx; (void)p; (void)i; (void)c;
    cap->supported = 1; cap->may_fallback = 0; cap->scratch_bytes = 0; cap->concurrent_worker_scratch_bytes = 0;
    return 1;
}

static void run_reshape_case(const char *qpath, const char *npath, const char *fxpath, uint32_t target_cmd, const char *model) {
    uint32_t qn, nn;
    uint8_t *q = read_file(qpath, &qn), *raw = read_file(npath, &nn), qsha[32], nsha[32];
    nprg2_sha256(q, qn, qsha);
    nprg2_sha256(raw, nn, nsha);
    Nprg2ExpectedHashes exp = {q + 32, q + 64, qsha, nsha};

    Nprg2Pool pools[7] = {0};
    pool_alloc(raw, pools);
    Nprg2Program p;
    CHECK(nprg2_open(&p, raw, nn, &exp, pools, (Nprg2Allocator){0}) == NPRG2_OK);

    uint8_t *hot = calloc(1, 300000);
    CHECK(hot);
    Nprg2BindConfig bc = {
        .hot_memory = hot,
        .hot_capacity = 300000,
        .l2_available_bytes = 786432,
        .pool1_is_verified_vtcm = 1,
        .resolve = resolve_all
    };
    Nprg2Bound b;
    CHECK(nprg2_bind_mock(&p, pools, &bc, &b) == NPRG2_OK);

    Nprg2Executor e;
    nprg2_executor_init(&e);
    CHECK(nprg2_executor_register_reshape_cb(&e) == NPRG2_EXEC_OK);

    Nprg2ExecRequest census = {.program = &p, .bound = &b,
        .pool1_writable_bytes = 0x780000u, .command_count = 1};
    memcpy(census.pools, pools, sizeof pools);
    uint32_t accepted = 0;
    for (uint32_t ix = 0; ix < p.command_count; ix++) {
        Nprg2Command candidate;
        CHECK(nprg2_command_at(&b, ix, &candidate) == NPRG2_OK);
        if (candidate.opcode == QDR_COPY_VALUES && candidate.variant == 49u) {
            census.command_first = ix;
            int rc = nprg2_executor_preflight(&e, &census);
            if (rc != NPRG2_EXEC_OK)
                fprintf(stderr, "Reshape census rejected model=%s command=%lu seq=%lu rc=%d\n",
                        model, (unsigned long)ix, (unsigned long)candidate.sequence, rc);
            CHECK(rc == NPRG2_EXEC_OK);
            accepted++;
        }
    }
    CHECK(accepted == 80u);

    uint32_t fx_bytes;
    uint8_t *fixture = read_file(fxpath, &fx_bytes);
    CHECK(fx_bytes >= 68u && !memcmp(fixture, "EPCP", 4) && rd32(fixture + 4) == 1u);

    Nprg2Command c;
    CHECK(nprg2_command_at(&b, target_cmd, &c) == NPRG2_OK);
    CHECK(rd32(fixture + 8) == c.opcode && rd32(fixture + 12) == c.variant &&
          rd32(fixture + 16) == c.sequence && rd32(fixture + 20) == target_cmd);

    uint32_t page_count = rd32(fixture + 36);
    CHECK(page_count == 4);

    uint8_t *want_pool[7] = {0};
    const uint8_t *cur = fixture + 68;
    for (uint32_t pi = 0; pi < 7; pi++) {
        uint32_t size = pools[pi].bytes;
        if (size) {
            memcpy(pools[pi].base, cur, size);
            if (pi == 1) {
                want_pool[pi] = malloc(size);
                CHECK(want_pool[pi]);
                memcpy(want_pool[pi], cur, size);
            } else {
                want_pool[pi] = (uint8_t *)cur;
            }
        }
        cur += size;
    }

    uint32_t pool1_bytes = pools[1].bytes;
    uint8_t *want = want_pool[1];
    for (uint32_t page = 0; page < page_count; page++) {
        uint32_t off = rd32(cur);
        cur += 4;
        CHECK(off <= pool1_bytes && 2048u <= pool1_bytes - off);
        memcpy(want + off, cur, 2048u);
        cur += 2048u;
    }

    Nprg2ExecRequest r = {
        .program = &p,
        .bound = &b,
        .pool1_writable_bytes = 0x780000u,
        .command_first = target_cmd,
        .command_count = 1
    };
    memcpy(r.pools, pools, sizeof pools);

    /* 1. Preflight check */
    CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_OK);

    /* 2. Run execution */
    CHECK(nprg2_executor_run(&e, &r) == NPRG2_EXEC_OK);

    /* 3. Physical pool verification */
    for (uint32_t pi = 0; pi < 7; pi++) {
        if (pools[pi].bytes && memcmp(pools[pi].base, want_pool[pi], pools[pi].bytes)) {
            uint32_t diff = 0;
            while (diff < pools[pi].bytes && pools[pi].base[diff] == want_pool[pi][diff]) diff++;
            fprintf(stderr, "Pool mismatch model=%s pool=%lu off=%lu got=%u cpu=%u\n",
                    model, (unsigned long)pi, (unsigned long)diff,
                    (unsigned int)pools[pi].base[diff], (unsigned int)want_pool[pi][diff]);
            exit(1);
        }
    }

    /* 4. Fail-closed negative checks */
    uint8_t *cmd_raw = (uint8_t *)b.command.data + (size_t)target_cmd * NPRG2_COMMAND_BYTES;
    uint8_t old_var = cmd_raw[3];
    cmd_raw[3] = 13u; /* Valid variant, unregistered for this executor. */
    CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_UNSUPPORTED);
    cmd_raw[3] = old_var;
    CHECK(!memcmp(pools[1].base, want_pool[1], pools[1].bytes));

    /* Out of bounds pool-1 page, duplicate destination, and input alias. */
    Nprg2ExecCall call = {.command_index = target_cmd, .command = c, .program = &p, .bound = &b, .pools = pools};
    uint32_t oi, ii; Nprg2TensorView ov, iv;
    CHECK(nprg2_exec_input(&call, 0, &ii, &iv) == NPRG2_EXEC_OK);
    CHECK(nprg2_exec_output(&call, 0, &oi, &ov) == NPRG2_EXEC_OK);
    (void)ii; (void)oi;
    uint32_t *output_blocks = (uint32_t *)ov.blocks;
    uint32_t original = output_blocks[ov.slice_origin];
    output_blocks[ov.slice_origin] = 0x780000u | 1u;
    CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_REJECTED);
    output_blocks[ov.slice_origin] = original;
    CHECK(!memcmp(pools[1].base, want_pool[1], pools[1].bytes));
    output_blocks[ov.slice_origin] = output_blocks[ov.slice_origin + 1u];
    CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_REJECTED);
    output_blocks[ov.slice_origin] = original;
    CHECK(!memcmp(pools[1].base, want_pool[1], pools[1].bytes));
    output_blocks[ov.slice_origin] = iv.blocks[iv.slice_origin];
    CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_REJECTED);
    output_blocks[ov.slice_origin] = original;
    CHECK(!memcmp(pools[1].base, want_pool[1], pools[1].bytes));

    printf("PASS Reshape CB CPU-prefix %s: 80/80 metadata accepted, op=16 var=49 cmd=%lu seq=%lu full pools exact and bad pages reject\n",
           model, (unsigned long)target_cmd, (unsigned long)c.sequence);

    free(want_pool[1]);
    free(fixture);
    free(hot);
    nprg2_close(&p);
    pool_free(pools);
    free(q);
    free(raw);
}

static uint32_t fixture_command(const char *path) {
    FILE *f = fopen(path, "rb");
    CHECK(f);
    uint8_t h[24];
    CHECK(fread(h, 1, sizeof h, f) == sizeof h && !memcmp(h, "EPCP", 4));
    fclose(f);
    return rd32(h + 20);
}

/* Source H padding selects the active row inside each CB source page; every
 * value 0..7 occurs 10 times per model. The simulator build covers the two
 * boundary rows and one interior row to bound Rosetta runtime. */
#if defined(RESHAPE_SIM_SUBSET)
static const unsigned kPads[] = {0, 3, 7};
#else
static const unsigned kPads[] = {0, 1, 2, 3, 4, 5, 6, 7};
#endif

int main(void) {
    static const char *const models[2] = {"primary", "alternate"};
    for (unsigned m = 0; m < 2; m++) {
        for (unsigned i = 0; i < sizeof kPads / sizeof kPads[0]; i++) {
            char qpath[96], npath[96], fxpath[128];
            snprintf(qpath, sizeof qpath, "hexecutor/native/build/resident-%s.qdrx", models[m]);
            snprintf(npath, sizeof npath, "samples/depth-model/deployment/programs/%s-v2.nprg", models[m]);
            snprintf(fxpath, sizeof fxpath,
                     "hexecutor/native/program/fixtures/%s-smoke-reshape-cb-pad%u.epcpu",
                     models[m], kPads[i]);
            printf("Reshape CB %s source pad %u:\n", models[m], kPads[i]);
            run_reshape_case(qpath, npath, fxpath, fixture_command(fxpath), models[m]);
        }
    }
    return 0;
}
