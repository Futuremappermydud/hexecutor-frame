#include "prepared_block_zap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static void wr32(uint8_t *p, uint32_t x) {
    for (unsigned i = 0; i < 4; i++) p[i] = (uint8_t)(x >> (8u * i));
}
static uint8_t *read_file(const char *path, uint32_t *bytes) {
    FILE *f = fopen(path, "rb"); CHECK(f);
    CHECK(!fseek(f, 0, SEEK_END)); long n = ftell(f);
    CHECK(n > 0 && n <= 32L * 1024L * 1024L && !fseek(f, 0, SEEK_SET));
    uint8_t *data = malloc((size_t)n); CHECK(data);
    CHECK(fread(data, 1, (size_t)n, f) == (size_t)n); fclose(f);
    *bytes = (uint32_t)n; return data;
}
static int resolve_all(void *ctx, const Nprg2Program *p, uint32_t i,
                       const Nprg2Command *cmd, Nprg2Capability *cap) {
    (void)ctx; (void)p; (void)i; (void)cmd;
    cap->supported = 1; cap->may_fallback = 0;
    cap->scratch_bytes = cap->concurrent_worker_scratch_bytes = 0;
    return 1;
}
static void oracle(uint8_t *pool, const uint8_t *v) {
    uint32_t header = rd32(v + 4), count = header & 0x7fffu;
    uint32_t nzero = ((header >> 16) & 7u) * 256u;
    uint8_t fill = (uint8_t)rd32(v), tile[2048];
    for (uint32_t i = 0; i < count; i++) {
        uint32_t src = rd32(v + (2u + 2u * i) * 4u);
        uint32_t dst = rd32(v + (3u + 2u * i) * 4u);
        memcpy(tile, pool + src, sizeof tile);
        memset(tile + sizeof tile - nzero, fill, nzero);
        memcpy(pool + dst, tile, sizeof tile);
    }
}
static void run_model(const char *qpath, const char *npath) {
    uint32_t qbytes, nbytes;
    uint8_t *q = read_file(qpath, &qbytes), *raw = read_file(npath, &nbytes);
    uint8_t qsha[32], nsha[32]; CHECK(qbytes >= 96);
    nprg2_sha256(q, qbytes, qsha); nprg2_sha256(raw, nbytes, nsha);
    Nprg2ExpectedHashes hashes = {q + 32, q + 64, qsha, nsha};
    Nprg2Pool pools[7] = {0};
    for (unsigned i = 0; i < 7; i++) {
        pools[i].bytes = rd32(raw + 144u + 4u * i);
        pools[i].base = pools[i].bytes ? malloc(pools[i].bytes) : NULL;
        CHECK(!pools[i].bytes || pools[i].base);
        if (pools[i].bytes) memset(pools[i].base, (int)(i + 29u), pools[i].bytes);
    }
    Nprg2Program program;
    CHECK(nprg2_open(&program, raw, nbytes, &hashes, pools, (Nprg2Allocator){0}) == NPRG2_OK);
    uint8_t *hot = calloc(1, 300000u); CHECK(hot);
    Nprg2BindConfig config = {.hot_memory=hot, .hot_capacity=300000u,
        .l2_available_bytes=786432u, .pool1_is_verified_vtcm=1, .resolve=resolve_all};
    Nprg2Bound bound;
    CHECK(nprg2_bind_mock(&program, pools, &config, &bound) == NPRG2_OK);
    Nprg2Executor e; nprg2_executor_init(&e);
    CHECK(nprg2_executor_register_block_zap(&e) == NPRG2_EXEC_OK);
    Nprg2ExecRequest req = {.program=&program, .bound=&bound,
        .pool1_writable_bytes=0x780000u, .command_count=1};
    memcpy(req.pools, pools, sizeof pools);
    uint8_t *expected = malloc(pools[1].bytes); CHECK(expected);
    for (uint32_t i = 0; i < pools[1].bytes; i++)
        pools[1].base[i] = (uint8_t)(i * 37u + i / 2048u);
    memcpy(expected, pools[1].base, pools[1].bytes);

    uint32_t covered = 0;
    for (uint32_t i = 0; i < program.command_count; i++) {
        Nprg2Command cmd; CHECK(nprg2_command_at(&bound, i, &cmd) == NPRG2_OK);
        if (cmd.opcode != QDR_BLOCK_ZAP) continue;
        CHECK(cmd.variant == 5 && cmd.vector_count >= 5);
        req.command_first = i;
        uint8_t *v = (uint8_t *)bound.vectors.data + (size_t)cmd.vector_first * 4u;
        if (!covered) {
            uint32_t old = rd32(v + 8u);
            wr32(v + 8u, 0x780000u);
            CHECK(nprg2_executor_preflight(&e, &req) == NPRG2_EXEC_REJECTED);
            CHECK(!memcmp(pools[1].base, expected, pools[1].bytes));
            wr32(v + 8u, old);
            old = rd32(v + 20u); wr32(v + 20u, 0x780000u);
            CHECK(nprg2_executor_preflight(&e, &req) == NPRG2_EXEC_REJECTED);
            CHECK(!memcmp(pools[1].base, expected, pools[1].bytes));
            wr32(v + 20u, old);
            old = rd32(v + 4u); wr32(v + 4u, 0u);
            CHECK(nprg2_executor_preflight(&e, &req) == NPRG2_EXEC_REJECTED);
            CHECK(!memcmp(pools[1].base, expected, pools[1].bytes));
            wr32(v + 4u, old);
        }
        CHECK(nprg2_executor_preflight(&e, &req) == NPRG2_EXEC_OK);
        CHECK(!memcmp(pools[1].base, expected, pools[1].bytes));
        oracle(expected, v);
        CHECK(nprg2_executor_run(&e, &req) == NPRG2_EXEC_OK);
        CHECK(!memcmp(pools[1].base, expected, pools[1].bytes));
        covered++;
    }
    CHECK(covered == 33u);
    printf("PASS prepared BlockZap %s: %u commands, full pool1 exact\n", qpath, covered);
    free(expected); free(hot); nprg2_close(&program);
    for (unsigned i = 0; i < 7; i++) free(pools[i].base);
    free(q); free(raw);
}
int main(void) {
    run_model("frame/build/resident-primary.qdrx", "frame/native_program/examples/primary-v2.nprg");
    run_model("frame/build/resident-alternate.qdrx", "frame/native_program/examples/alternate-v2.nprg");
    return 0;
}
