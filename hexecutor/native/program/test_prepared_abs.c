/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_abs.h"

#ifdef __hexagon__
#include <hexagon_standalone.h>
#define INIT_SIM() do { SIM_ACQUIRE_HVX; SIM_SET_HVX_DOUBLE_MODE; } while (0)
#define TEARDOWN_SIM() SIM_RELEASE_HVX
#else
#define INIT_SIM() do {} while (0)
#define TEARDOWN_SIM() do {} while (0)
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static uint16_t rd16(const uint8_t *p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static void wr32(uint8_t *p, uint32_t x) { p[0] = (uint8_t)x; p[1] = (uint8_t)(x >> 8); p[2] = (uint8_t)(x >> 16); p[3] = (uint8_t)(x >> 24); }

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

static void check_pools_untouched(const Nprg2Pool pools[7], uint8_t * const want_pool[7]) {
    for (uint32_t pi = 0; pi < 7; pi++) {
        if (pools[pi].bytes) {
            CHECK(memcmp(pools[pi].base, want_pool[pi], pools[pi].bytes) == 0);
        }
    }
}

static void run_abs_case(Nprg2Program *p, Nprg2Bound *b, Nprg2Executor *e,
                         Nprg2Pool pools[7], uint32_t pool1_writable,
                         uint32_t cmd_idx, const char *fx_path,
                         const char *model_name) {
    Nprg2Command c;
    CHECK(nprg2_command_at(b, cmd_idx, &c) == NPRG2_OK);
    CHECK(c.opcode == QDR_ABS);

    uint32_t fixture_bytes;
    uint8_t *fixture = read_file(fx_path, &fixture_bytes);
    CHECK(fixture_bytes >= 68u && !memcmp(fixture, "EPCP", 4) && rd32(fixture + 4) == 1u);
    CHECK(rd32(fixture + 8) == c.opcode && rd32(fixture + 12) == c.variant &&
          rd32(fixture + 16) == c.sequence && rd32(fixture + 20) == cmd_idx);

    uint32_t page_count = rd32(fixture + 36);
    uint8_t *want_pool[7] = {0};
    uint64_t need = 68ull + (uint64_t)page_count * 2052u;
    for (uint32_t pi = 0; pi < 7; pi++) {
        uint32_t size = rd32(fixture + 40 + 4u * pi);
        CHECK(size == pools[pi].bytes);
        need += size;
    }
    CHECK(need == fixture_bytes);

    const uint8_t *cursor = fixture + 68;
    for (uint32_t pi = 0; pi < 7; pi++) {
        uint32_t size = pools[pi].bytes;
        if (size) {
            memcpy(pools[pi].base, cursor, size);
            if (pi == 1) {
                want_pool[pi] = malloc(size);
                CHECK(want_pool[pi]);
                memcpy(want_pool[pi], cursor, size);
            } else {
                want_pool[pi] = (uint8_t *)cursor;
            }
        }
        cursor += size;
    }

    uint32_t pool1_bytes = pools[1].bytes;
    uint8_t *want = want_pool[1];
    CHECK(want);
    for (uint32_t page = 0; page < page_count; page++) {
        uint32_t off = rd32(cursor);
        cursor += 4;
        CHECK(off <= pool1_bytes && 2048u <= pool1_bytes - off);
        memcpy(want + off, cursor, 2048u);
        cursor += 2048u;
    }

    Nprg2ExecRequest r = {
        .program = p,
        .bound = b,
        .pool1_writable_bytes = pool1_writable,
        .command_first = cmd_idx,
        .command_count = 1
    };
    memcpy(r.pools, pools, sizeof(r.pools));

    /* Preflight and execute */
    CHECK(nprg2_executor_preflight(e, &r) == NPRG2_EXEC_OK);
    CHECK(nprg2_executor_run(e, &r) == NPRG2_EXEC_OK);

    /* Assert exact physical pool match across all pools */
    for (uint32_t pi = 0; pi < 7; pi++) {
        if (pools[pi].bytes && memcmp(pools[pi].base, want_pool[pi], pools[pi].bytes)) {
            uint32_t i = 0;
            while (i < pools[pi].bytes && pools[pi].base[i] == want_pool[pi][i]) i++;
            fprintf(stderr, "CPU-prefix physical pool mismatch model=%s pool=%lu cmd=%lu seq=%lu off=%lu got=%u cpu=%u\n",
                    model_name, (unsigned long)pi, (unsigned long)cmd_idx, (unsigned long)c.sequence,
                    (unsigned long)i, pools[pi].base[i], want_pool[pi][i]);
            exit(1);
        }
    }

    /* Negative tests: proving atomic fail-closed rejection without writing to any pool */
    uint8_t *cmd_raw = (uint8_t *)b->command.data + (size_t)cmd_idx * NPRG2_COMMAND_BYTES;
    Nprg2ExecCall call = {
        .command_index = cmd_idx,
        .command = c,
        .program = p,
        .bound = b,
        .pools = pools,
        .pool1_writable_bytes = pool1_writable
    };
    uint32_t in_idx, out_idx;
    Nprg2TensorView in_view, out_view;
    CHECK(nprg2_exec_input(&call, 0, &in_idx, &in_view) == NPRG2_EXEC_OK);
    CHECK(nprg2_exec_output(&call, 0, &out_idx, &out_view) == NPRG2_EXEC_OK);

    /* 1. Unsupported variant (valid variant <= 75 but unregistered for ABS) */
    uint8_t old_variant = cmd_raw[3];
    cmd_raw[3] = 13u;
    int rc = nprg2_executor_preflight(e, &r);
    if (rc != NPRG2_EXEC_UNSUPPORTED) {
        fprintf(stderr, "cmd %lu preflight returned %d (expected %d)\n",
                (unsigned long)cmd_idx, rc, NPRG2_EXEC_UNSUPPORTED);
    }
    CHECK(rc == NPRG2_EXEC_UNSUPPORTED);
    check_pools_untouched(pools, want_pool);
    cmd_raw[3] = old_variant;

    /* 2. Corrupt input geometry: wrong dtype */
    uint8_t *in_geom = (uint8_t *)in_view.geometry_record;
    uint8_t old_in_dtype = in_geom[0];
    in_geom[0] = 2u; /* DTYPE_U8 instead of DTYPE_U16 */
    CHECK(nprg2_executor_preflight(e, &r) == NPRG2_EXEC_REJECTED);
    check_pools_untouched(pools, want_pool);
    in_geom[0] = old_in_dtype;

    /* 3. Corrupt output geometry: dimension mismatch */
    uint8_t *out_geom = (uint8_t *)out_view.geometry_record;
    uint16_t old_dim = rd16(out_geom + 12);
    out_geom[12] ^= 0x01;
    CHECK(nprg2_executor_preflight(e, &r) == NPRG2_EXEC_REJECTED);
    check_pools_untouched(pools, want_pool);
    out_geom[12] = (uint8_t)(old_dim & 0xff);
    out_geom[13] = (uint8_t)(old_dim >> 8);

    if (c.variant == 12) {
        /* Flat FH tests: out-of-bounds offset and alias overlap */
        uint8_t *out_ten = (uint8_t *)b->tensor.data + (size_t)out_idx * NPRG2_TENSOR_BYTES;
        uint32_t old_off = rd32(out_ten + 4);

        /* Out-of-bounds offset exceeding pool1_writable_bytes */
        wr32(out_ten + 4, pool1_writable);
        CHECK(nprg2_executor_preflight(e, &r) == NPRG2_EXEC_REJECTED);
        check_pools_untouched(pools, want_pool);
        wr32(out_ten + 4, old_off);

        /* Alias overlap: in.pool == out.pool and overlapping offsets */
        wr32(out_ten + 4, in_view.offset);
        CHECK(nprg2_executor_preflight(e, &r) == NPRG2_EXEC_REJECTED);
        check_pools_untouched(pools, want_pool);
        wr32(out_ten + 4, old_off);
    } else if (c.variant == 11) {
        /* Crouton CH tests: class corruption, storage kind corruption, block offset corruption */
        /* Change class name from "CH" to "CB" */
        uint8_t old_c1 = out_geom[3];
        out_geom[3] = 'B';
        CHECK(nprg2_executor_preflight(e, &r) == NPRG2_EXEC_REJECTED);
        check_pools_untouched(pools, want_pool);
        out_geom[3] = old_c1;

        /* Change storage kind from 2 (blocked) to 1 (direct) */
        uint8_t *out_ten = (uint8_t *)b->tensor.data + (size_t)out_idx * NPRG2_TENSOR_BYTES;
        uint8_t old_kind = out_ten[2];
        out_ten[2] = 1u;
        CHECK(nprg2_executor_preflight(e, &r) == NPRG2_EXEC_REJECTED);
        check_pools_untouched(pools, want_pool);
        out_ten[2] = old_kind;

        /* Corrupt block entry: set first accessed block to invalid pool entry (pool >= 7) */
        uint32_t *out_blocks = (uint32_t *)out_view.blocks;
        uint32_t oi0 = out_view.slice_origin;
        uint32_t old_block0 = out_blocks[oi0];
        out_blocks[oi0] = 7u; /* pool 7 is invalid (pool >= 7) */
        int run_rc = nprg2_executor_run(e, &r);
        if (run_rc != NPRG2_EXEC_HANDLER_ERROR) {
            fprintf(stderr, "Corrupted block nprg2_executor_run returned %d (expected %d)\n", run_rc, NPRG2_EXEC_HANDLER_ERROR);
        }
        CHECK(run_rc == NPRG2_EXEC_HANDLER_ERROR || run_rc == NPRG2_EXEC_REJECTED);
        check_pools_untouched(pools, want_pool);
        out_blocks[oi0] = old_block0;
    }

    printf("PASS %s cmd %lu (seq %lu, var %u): exact pool match & fail-closed negative tests OK\n",
           model_name, (unsigned long)cmd_idx, (unsigned long)c.sequence, c.variant);

    free(want_pool[1]);
    free(fixture);
}

static void run_primary(void) {
    const char *qpath = "hexecutor/native/build/resident-primary.qdrx";
    const char *npath = "samples/depth-model/deployment/programs/primary-v2.nprg";
    uint32_t qn, nn;
    uint8_t *q = read_file(qpath, &qn), *raw = read_file(npath, &nn), qsha[32], nsha[32];
    nprg2_sha256(q, qn, qsha);
    nprg2_sha256(raw, nn, nsha);
    Nprg2ExpectedHashes x = {q + 32, q + 64, qsha, nsha};
    Nprg2Pool pools[7] = {0};
    pool_alloc(raw, pools);
    Nprg2Program p;
    CHECK(nprg2_open(&p, raw, nn, &x, pools, (Nprg2Allocator){0}) == NPRG2_OK);
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
    CHECK(nprg2_executor_register_abs(&e) == NPRG2_EXEC_OK);

    run_abs_case(&p, &b, &e, pools, 0x780000u, 74u,
                 "hexecutor/native/program/fixtures/primary-smoke-abs-fh.epcpu", "primary");
    run_abs_case(&p, &b, &e, pools, 0x780000u, 4741u,
                 "hexecutor/native/program/fixtures/primary-smoke-abs-ch.epcpu", "primary");

    free(hot);
    nprg2_close(&p);
    pool_free(pools);
    free(q);
    free(raw);
}

static void run_alternate(void) {
    const char *qpath = "hexecutor/native/build/resident-alternate.qdrx";
    const char *npath = "samples/depth-model/deployment/programs/alternate-v2.nprg";
    uint32_t qn, nn;
    uint8_t *q = read_file(qpath, &qn), *raw = read_file(npath, &nn), qsha[32], nsha[32];
    nprg2_sha256(q, qn, qsha);
    nprg2_sha256(raw, nn, nsha);
    Nprg2ExpectedHashes x = {q + 32, q + 64, qsha, nsha};
    Nprg2Pool pools[7] = {0};
    pool_alloc(raw, pools);
    Nprg2Program p;
    CHECK(nprg2_open(&p, raw, nn, &x, pools, (Nprg2Allocator){0}) == NPRG2_OK);
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
    CHECK(nprg2_executor_register_abs(&e) == NPRG2_EXEC_OK);

    run_abs_case(&p, &b, &e, pools, 0x780000u, 72u,
                 "hexecutor/native/program/fixtures/alternate-smoke-abs-fh.epcpu", "alternate");

    free(hot);
    nprg2_close(&p);
    pool_free(pools);
    free(q);
    free(raw);
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    INIT_SIM();

    run_primary();
    run_alternate();

    printf("ALL PREPARED ABS VERIFICATION TESTS PASSED\n");
    TEARDOWN_SIM();
    return 0;
}
