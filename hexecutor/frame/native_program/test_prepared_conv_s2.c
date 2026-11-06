#include "prepared_conv_s2.h"

#include <hexagon_standalone.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { \
    if (!(x)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
        exit(1); \
    } \
} while (0)

static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
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
        p[i].base = !p[i].bytes ? NULL : (i == 1 ? (uint8_t *)aligned_alloc(128, p[i].bytes) : (uint8_t *)malloc(p[i].bytes));
        CHECK(!p[i].bytes || p[i].base);
        if (p[i].bytes) memset(p[i].base, 0, p[i].bytes);
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

static int find_command_by_seq(const Nprg2Program *p, const Nprg2Bound *b, uint32_t seq) {
    for (uint32_t i = 0; i < p->command_count; i++) {
        Nprg2Command c;
        CHECK(nprg2_command_at(b, i, &c) == NPRG2_OK);
        if (c.sequence == seq) return (int)i;
    }
    return -1;
}

static const struct {
    uint32_t seq;
    const char *fixture_path;
    uint32_t expected_cin;
    uint32_t expected_cout;
    uint32_t expected_pages;
} TEST_CASES[] = {
    { 224,  "frame/native_program/fixtures/primary-smoke-conv-s2-224.epcpu",  128, 64,  32 },
    { 363,  "frame/native_program/fixtures/primary-smoke-conv-s2-363.epcpu",  256, 128, 32 },
    { 3407, "frame/native_program/fixtures/primary-smoke-conv-s2-3407.epcpu", 512, 128, 16 },
};

int main(void) {
    SIM_ACQUIRE_HVX;
    SIM_SET_HVX_DOUBLE_MODE;

    const char *qpath = "frame/build/resident-primary.qdrx";
    const char *npath = "frame/native_program/examples/primary-v2.nprg";

    uint32_t qn, nn;
    uint8_t *q = read_file(qpath, &qn);
    uint8_t *raw = read_file(npath, &nn);
    uint8_t qsha[32], nsha[32];
    nprg2_sha256(q, qn, qsha);
    nprg2_sha256(raw, nn, nsha);
    Nprg2ExpectedHashes x = { q + 32, q + 64, qsha, nsha };

    Nprg2Pool pools[7] = {0};
    pool_alloc(raw, pools);
    Nprg2Program p;
    CHECK(nprg2_open(&p, raw, nn, &x, pools, (Nprg2Allocator){0}) == NPRG2_OK);

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
    CHECK(nprg2_executor_register_conv_s2(&e) == NPRG2_EXEC_OK);

    Nprg2ExecRequest r = {
        .program = &p,
        .bound = &b,
        .pool1_writable_bytes = 0x780000u,
        .scratch = NULL,
        .scratch_bytes = 0
    };
    memcpy(r.pools, pools, sizeof(pools));

    for (size_t tc = 0; tc < sizeof(TEST_CASES) / sizeof(TEST_CASES[0]); tc++) {
        uint32_t seq = TEST_CASES[tc].seq;
        const char *fx_path = TEST_CASES[tc].fixture_path;
        int cmd_idx = find_command_by_seq(&p, &b, seq);
        CHECK(cmd_idx >= 0);

        uint32_t first = (uint32_t)cmd_idx;
        Nprg2Command c;
        CHECK(nprg2_command_at(&b, first, &c) == NPRG2_OK);
        CHECK(c.opcode == 6 && c.variant == 36 && c.sequence == seq);

        r.command_first = first;
        r.command_count = 1;

        uint32_t fixture_bytes;
        uint8_t *fixture = read_file(fx_path, &fixture_bytes);
        CHECK(fixture_bytes >= 68u && !memcmp(fixture, "EPCP", 4) && rd32(fixture + 4) == 1u);
        CHECK(rd32(fixture + 8) == c.opcode && rd32(fixture + 12) == c.variant);
        CHECK(rd32(fixture + 16) == c.sequence && rd32(fixture + 20) == first);

        uint32_t fi_pool = rd32(fixture + 24);
        uint32_t fi_offset = rd32(fixture + 28);
        uint32_t fi_bytes = rd32(fixture + 32);
        uint32_t page_count = rd32(fixture + 36);
        CHECK(page_count == TEST_CASES[tc].expected_pages);

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
        CHECK(fi_pool == 1 && fi_bytes == TEST_CASES[tc].expected_cout * 8u);
        CHECK(fi_offset <= pool1_bytes && fi_bytes <= pool1_bytes - fi_offset);

        for (uint32_t page = 0; page < page_count; page++) {
            uint32_t off = rd32(cursor);
            cursor += 4;
            CHECK(off <= pool1_bytes && 2048u <= pool1_bytes - off);
            memcpy(want + off, cursor, 2048u);
            cursor += 2048u;
        }

        /* 1. Test preflight validation */
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_OK);

        /* 2. Test descriptor setup */
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
        Nprg2ConvS2Descriptors desc;
        CHECK(nprg2_conv_s2_setup_descriptors(&call, &desc) == 1);
        CHECK(desc.input_shape.dims[3] == TEST_CASES[tc].expected_cin);
        CHECK(desc.output_shape.dims[3] == TEST_CASES[tc].expected_cout);
        CHECK(desc.weight_shape.dims[2] == TEST_CASES[tc].expected_cin / 4u);
        CHECK(desc.weight_shape.dims[3] == TEST_CASES[tc].expected_cout);
        CHECK(desc.input_obj.data != 0 && desc.output_obj.data != 0);
        CHECK(desc.weight_obj.data != 0 && desc.fi_obj.data != 0);

        /* 3. Run execution */
        CHECK(nprg2_executor_run(&e, &r) == NPRG2_EXEC_OK);

        /* 4. Verify bit-exact output match across all pools */
        for (uint32_t pi = 0; pi < 7; pi++) {
            if (pools[pi].bytes && memcmp(pools[pi].base, want_pool[pi], pools[pi].bytes)) {
                uint32_t i = 0;
                while (i < pools[pi].bytes && pools[pi].base[i] == want_pool[pi][i]) i++;
                fprintf(stderr, "CPU-prefix mismatch seq=%lu pool=%lu off=%lu got=%u cpu=%u\n",
                        (unsigned long)seq, (unsigned long)pi, (unsigned long)i,
                        (unsigned int)pools[pi].base[i], (unsigned int)want_pool[pi][i]);
                exit(1);
            }
        }

        /* 5. Rejection proof: corrupt output block table to exceed writable bytes */
        uint32_t oi, ii;
        Nprg2TensorView ov, iv;
        CHECK(!nprg2_exec_output(&call, 0, &oi, &ov));
        CHECK(!nprg2_exec_input(&call, 0, &ii, &iv));
        CHECK(ov.block_count && iv.block_count);

        uint32_t oi0 = ov.slice_origin;
        uint32_t ii0 = iv.slice_origin;
        uint32_t *out_blocks = (uint32_t *)ov.blocks;
        uint32_t orig_block = out_blocks[oi0];

        /* Corrupt with offset exceeding pool1_writable_bytes */
        out_blocks[oi0] = 0x780001u;
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_REJECTED);
        for (uint32_t pi = 0; pi < 7; pi++) {
            if (pools[pi].bytes) CHECK(!memcmp(pools[pi].base, want_pool[pi], pools[pi].bytes));
        }

        /* Corrupt to alias input block */
        out_blocks[oi0] = iv.blocks[ii0];
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_REJECTED);
        for (uint32_t pi = 0; pi < 7; pi++) {
            if (pools[pi].bytes) CHECK(!memcmp(pools[pi].base, want_pool[pi], pools[pi].bytes));
        }

        /* Restore original block */
        out_blocks[oi0] = orig_block;
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_OK);

        printf("PASS prepared conv_s2 seq %lu (CI=%lu CO=%lu pages=%lu) bit-exact\n",
               (unsigned long)seq, (unsigned long)TEST_CASES[tc].expected_cin,
               (unsigned long)TEST_CASES[tc].expected_cout, (unsigned long)page_count);

        free(want_pool[1]);
        free(fixture);
    }

    /* 6. Verify rejection of unsupported variant */
    int unsupported = find_command_by_seq(&p, &b, 247); /* Variant 33 */
    if (unsupported >= 0) {
        r.command_first = (uint32_t)unsupported;
        r.command_count = 1;
        CHECK(nprg2_executor_preflight(&e, &r) == NPRG2_EXEC_UNSUPPORTED);
    }

    printf("ALL TESTS PASSED: prepared conv_s2 bit-exact proof verified under Hexagon simulator\n");

    free(hot);
    nprg2_close(&p);
    pool_free(pools);
    free(q);
    free(raw);
    SIM_RELEASE_HVX;
    return 0;
}
