/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_conv_u16.h"

#if defined(__hexagon__)
#include <hexagon_standalone.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); exit(1); } } while (0)

static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void wr32(uint8_t *p, uint32_t x) {
    p[0] = (uint8_t)x;
    p[1] = (uint8_t)(x >> 8);
    p[2] = (uint8_t)(x >> 16);
    p[3] = (uint8_t)(x >> 24);
}

static uint8_t *read_file(const char *path, uint32_t *n) {
    FILE *f = fopen(path, "rb");
    CHECK(f);
    CHECK(!fseek(f, 0, SEEK_END));
    long z = ftell(f);
    CHECK(z > 0 && (uint64_t)z <= 32u * 1024u * 1024u);
    CHECK(!fseek(f, 0, SEEK_SET));
    uint8_t *p = (uint8_t *)malloc((size_t)z);
    CHECK(p && fread(p, 1, (size_t)z, f) == (size_t)z);
    fclose(f);
    *n = (uint32_t)z;
    return p;
}

static void pool_alloc(const uint8_t *raw, Nprg2Pool p[7]) {
    for (uint32_t i = 0; i < 7; i++) {
        p[i].bytes = rd32(raw + 144 + i * 4);
        if (!p[i].bytes) {
            p[i].base = NULL;
        } else {
#if defined(__hexagon__)
            p[i].base = (i == 1) ? (uint8_t *)aligned_alloc(128, p[i].bytes) : (uint8_t *)malloc(p[i].bytes);
#else
            if (i == 1) {
                void *ptr = NULL;
                CHECK(posix_memalign(&ptr, 128, p[i].bytes) == 0);
                p[i].base = (uint8_t *)ptr;
            } else {
                p[i].base = (uint8_t *)malloc(p[i].bytes);
            }
#endif
            CHECK(p[i].base);
            memset(p[i].base, 0, p[i].bytes);
        }
    }
}

static void pool_free(Nprg2Pool p[7]) {
    for (int i = 0; i < 7; i++) free(p[i].base);
}

static int resolve_all(void *ctx, const Nprg2Program *p, uint32_t i, const Nprg2Command *c, Nprg2Capability *cap) {
    (void)ctx; (void)p; (void)i; (void)c;
    cap->supported = 1;
    cap->may_fallback = 0;
    cap->scratch_bytes = 0;
    cap->concurrent_worker_scratch_bytes = 0;
    return 1;
}

static int find_command(const Nprg2Program *p, const Nprg2Bound *b, uint8_t op, uint8_t variant) {
    for (uint32_t i = 0; i < p->command_count; i++) {
        Nprg2Command c;
        CHECK(nprg2_command_at(b, i, &c) == NPRG2_OK);
        if (c.opcode == op && c.variant == variant) return (int)i;
    }
    return -1;
}

static void run_model(const char *qpath, const char *npath, const char *c3x3_fixture, const char *c1x1_fixture) {
    uint32_t qn, nn;
    uint8_t *q = read_file(qpath, &qn);
    uint8_t *raw = read_file(npath, &nn);
    uint8_t qsha[32], nsha[32];
    nprg2_sha256(q, qn, qsha);
    nprg2_sha256(raw, nn, nsha);
    Nprg2ExpectedHashes expected_hashes = {q + 32, q + 64, qsha, nsha};

    Nprg2Pool pools[7] = {0};
    pool_alloc(raw, pools);

    Nprg2Program p;
    CHECK(nprg2_open(&p, raw, nn, &expected_hashes, pools, (Nprg2Allocator){0}) == NPRG2_OK);

    uint8_t *hot = (uint8_t *)calloc(1, 300000);
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
    CHECK(nprg2_executor_register_conv_u16(&e) == NPRG2_EXEC_OK);

    Nprg2ExecRequest r = {
        .program = &p,
        .bound = &b,
        .pool1_writable_bytes = 0x780000u,
        .scratch = NULL,
        .scratch_bytes = 0
    };
    memcpy(r.pools, pools, sizeof pools);
    Nprg2ExecStats stats;
    r.stats = &stats;

    int cmds[2] = {
        find_command(&p, &b, QDR_HMX_CONV, 35),
        find_command(&p, &b, QDR_HMX_CONV, 34)
    };
    CHECK(cmds[0] >= 0 && cmds[1] >= 0);

    for (int which = 0; which < 2; which++) {
        uint32_t first = (uint32_t)cmds[which];
        Nprg2Command c;
        CHECK(nprg2_command_at(&b, first, &c) == NPRG2_OK);
        r.command_first = first;
        r.command_count = 1;

        const char *fx_path = (which == 0 ? c3x3_fixture : c1x1_fixture);
        uint32_t fixture_bytes;
        uint8_t *fixture = read_file(fx_path, &fixture_bytes);
        CHECK(fixture_bytes >= 68u && !memcmp(fixture, "EPCP", 4) && rd32(fixture + 4) == 1u);
        CHECK(rd32(fixture + 8) == c.opcode && rd32(fixture + 12) == c.variant &&
              rd32(fixture + 16) == c.sequence && rd32(fixture + 20) == first);

        uint32_t fi_pool = rd32(fixture + 24), fi_offset = rd32(fixture + 28),
                 fi_bytes = rd32(fixture + 32), page_count = rd32(fixture + 36);

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
                    want_pool[pi] = (uint8_t *)malloc(size);
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

        if (fi_pool != UINT32_MAX) {
            CHECK(fi_pool < 7 && pools[fi_pool].base && fi_bytes == 256u &&
                  fi_offset <= pools[fi_pool].bytes && fi_bytes <= pools[fi_pool].bytes - fi_offset);
        } else {
            CHECK(!fi_bytes);
        }

        for (uint32_t page = 0; page < page_count; page++) {
            uint32_t off = rd32(cursor);
            cursor += 4;
            CHECK(off <= pool1_bytes && 2048u <= pool1_bytes - off);
            memcpy(want + off, cursor, 2048u);
            cursor += 2048u;
        }

        /* 1. Preflight validation */
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_OK);

        /* 2. Execution */
        Nprg2ExecCall call = {
            .command_index = first,
            .command = c,
            .program = &p,
            .bound = &b,
            .pools = pools,
            .pool1_writable_bytes = r.pool1_writable_bytes,
            .scratch = NULL,
            .scratch_bytes = 0
        };
        (void)call;

        CHECK(nprg2_executor_run(&e, &r) == NPRG2_EXEC_OK);

        /* 3. 100% bit-exact physical pool match across all pools including all output pages */
        for (uint32_t pi = 0; pi < 7; pi++) {
            if (pools[pi].bytes && memcmp(pools[pi].base, want_pool[pi], pools[pi].bytes)) {
                uint32_t diff_idx = 0;
                while (diff_idx < pools[pi].bytes && pools[pi].base[diff_idx] == want_pool[pi][diff_idx]) diff_idx++;
                fprintf(stderr, "CPU-prefix physical pool mismatch model=%s which=%d pool=%u cmd=%u seq=%u off=%u got=%u cpu=%u\n",
                        qpath, which, (unsigned)pi, (unsigned)first, (unsigned)c.sequence,
                        (unsigned)diff_idx, (unsigned)pools[pi].base[diff_idx], (unsigned)want_pool[pi][diff_idx]);
                exit(1);
            }
        }

        /* 4. Fail-closed check: Invalid variant */
        uint8_t *cmd_raw = (uint8_t *)b.command.data + (size_t)first * NPRG2_COMMAND_BYTES;
        uint8_t orig_variant = cmd_raw[3];
        cmd_raw[3] = 36u;
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_UNSUPPORTED);
        for (uint32_t pi = 0; pi < 7; pi++) {
            if (pools[pi].bytes) CHECK(!memcmp(pools[pi].base, want_pool[pi], pools[pi].bytes));
        }
        cmd_raw[3] = orig_variant;
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_OK);

        /* 5. Fail-closed check: Bad block table */
        uint32_t yi;
        Nprg2TensorView yv;
        CHECK(nprg2_exec_output(&call, 0, &yi, &yv) == NPRG2_EXEC_OK);
        CHECK(yv.block_count > 0 && yv.blocks);
        uint32_t *out_blocks = (uint32_t *)yv.blocks;
        uint32_t orig_block = out_blocks[0];

        /* Corrupt to unaligned block offset (not 2048-aligned) */
        out_blocks[0] = (orig_block & ~127u) + 128u + 1u;
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_REJECTED);
        for (uint32_t pi = 0; pi < 7; pi++) {
            if (pools[pi].bytes) CHECK(!memcmp(pools[pi].base, want_pool[pi], pools[pi].bytes));
        }

        /* Corrupt to out-of-bounds writable VTCM */
        out_blocks[0] = 0x780000u | 1u;
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_REJECTED);
        for (uint32_t pi = 0; pi < 7; pi++) {
            if (pools[pi].bytes) CHECK(!memcmp(pools[pi].base, want_pool[pi], pools[pi].bytes));
        }

        /* Corrupt to invalid pool */
        out_blocks[0] = (orig_block & ~127u) | 2u;
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_REJECTED);
        for (uint32_t pi = 0; pi < 7; pi++) {
            if (pools[pi].bytes) CHECK(!memcmp(pools[pi].base, want_pool[pi], pools[pi].bytes));
        }
        out_blocks[0] = orig_block;
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_OK);

        /* 6. Fail-closed check: Unaligned direct offsets (weights & FI) */
        uint32_t wi, fi;
        Nprg2TensorView wv, fv;
        CHECK(nprg2_exec_input(&call, 1, &wi, &wv) == NPRG2_EXEC_OK);
        CHECK(nprg2_exec_input(&call, 2, &fi, &fv) == NPRG2_EXEC_OK);

        uint8_t *w_rec = (uint8_t *)b.tensor.data + (size_t)wi * NPRG2_TENSOR_BYTES;
        uint8_t *f_rec = (uint8_t *)b.tensor.data + (size_t)fi * NPRG2_TENSOR_BYTES;
        uint32_t orig_w_off = rd32(w_rec + 4);
        uint32_t orig_f_off = rd32(f_rec + 4);

        /* Unaligned w offset */
        wr32(w_rec + 4, orig_w_off + 1);
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_REJECTED);
        for (uint32_t pi = 0; pi < 7; pi++) {
            if (pools[pi].bytes) CHECK(!memcmp(pools[pi].base, want_pool[pi], pools[pi].bytes));
        }
        wr32(w_rec + 4, orig_w_off);
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_OK);

        /* Unaligned fi offset */
        wr32(f_rec + 4, orig_f_off + 1);
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_REJECTED);
        for (uint32_t pi = 0; pi < 7; pi++) {
            if (pools[pi].bytes) CHECK(!memcmp(pools[pi].base, want_pool[pi], pools[pi].bytes));
        }
        wr32(f_rec + 4, orig_f_off);
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_OK);

        free(want_pool[1]);
        free(fixture);
    }

    printf("PASS prepared CH/u16 conv captured CPU prefix: %s\n", qpath);
    free(hot);
    nprg2_close(&p);
    pool_free(pools);
    free(q);
    free(raw);
}

int main(int argc, char **argv) {
#if defined(__hexagon__)
    SIM_ACQUIRE_HVX;
    SIM_SET_HVX_DOUBLE_MODE;
#endif

    if (argc == 1) {
        run_model("hexecutor/native/build/resident-primary.qdrx",
                  "depth-model/deployment/programs/primary-v2.nprg",
                  "hexecutor/native/program/fixtures/primary-smoke-conv-u16-3x3.epcpu",
                  "hexecutor/native/program/fixtures/primary-smoke-conv-u16-1x1.epcpu");
#if defined(__hexagon__)
        SIM_RELEASE_HVX;
#endif
        return 0;
    }

    CHECK(argc == 5);
    run_model(argv[1], argv[2], argv[3], argv[4]);

#if defined(__hexagon__)
    SIM_RELEASE_HVX;
#endif
    return 0;
}
