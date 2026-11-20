/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "prepared_slice_pad.h"

#ifdef __hexagon__
#include <hexagon_standalone.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);exit(1);}}while(0)
#define HEADER_BYTES 64u
#define ACTIVE_BYTES 0x780000u

static uint32_t rd32(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static void wr32(uint8_t *p, uint32_t x){p[0]=(uint8_t)x;p[1]=(uint8_t)(x>>8);p[2]=(uint8_t)(x>>16);p[3]=(uint8_t)(x>>24);}

static uint8_t *read_file(const char *path, uint32_t *n){
    FILE *f = fopen(path, "rb");
    CHECK(f);
    CHECK(!fseek(f, 0, SEEK_END));
    long z = ftell(f);
    CHECK(z > 0 && (uint64_t)z <= 64ULL * 1024ULL * 1024ULL);
    CHECK(!fseek(f, 0, SEEK_SET));
    uint8_t *p = malloc((size_t)z);
    CHECK(p && fread(p, 1, (size_t)z, f) == (size_t)z);
    fclose(f);
    *n = (uint32_t)z;
    return p;
}

static int resolve_all(void *ctx, const Nprg2Program *p, uint32_t i, const Nprg2Command *c, Nprg2Capability *cap){
    (void)ctx; (void)p; (void)i; (void)c;
    cap->supported = 1;
    cap->may_fallback = 0;
    cap->scratch_bytes = 0;
    cap->concurrent_worker_scratch_bytes = 0;
    return 1;
}

static void pool_alloc(const uint8_t *r, Nprg2Pool p[7]){
    for (unsigned i = 0; i < 7; i++){
        p[i].bytes = rd32(r + 144 + 4 * i);
        p[i].base = p[i].bytes ? malloc(p[i].bytes) : NULL;
        CHECK(!p[i].bytes || p[i].base);
        if (p[i].bytes) memset(p[i].base, 0, p[i].bytes);
    }
}

static void pool_free(Nprg2Pool p[7]){
    for (unsigned i = 0; i < 7; i++) free(p[i].base);
}

static int compare_pools(Nprg2Pool p[7], const uint8_t *before[7]){
    for (unsigned i = 0; i < 7; i++) {
        if (p[i].bytes && memcmp(p[i].base, before[i], p[i].bytes)) return 0;
    }
    return 1;
}

static int compare_after(Nprg2Pool pools[7], const uint8_t *before[7], uint32_t off, const uint8_t *patch, uint32_t bytes){
    for (unsigned p = 0; p < 7; p++) {
        if (p != 1 && pools[p].bytes && memcmp(pools[p].base, before[p], pools[p].bytes)) {
            uint32_t i = 0;
            while (i < pools[p].bytes && pools[p].base[i] == before[p][i]) i++;
            fprintf(stderr, "CPU-prefix pool mismatch pool=%u byte=%lu\n", p, (unsigned long)i);
            return 0;
        }
    }
    if (memcmp(pools[1].base, before[1], off)) {
        fprintf(stderr, "pool1 mismatch before target block\n");
        return 0;
    }
    if (memcmp(pools[1].base + off, patch, bytes)) {
        fprintf(stderr, "pool1 target block mismatch\n");
        return 0;
    }
    uint32_t end = off + bytes;
    if (memcmp(pools[1].base + end, before[1] + end, pools[1].bytes - end)) {
        fprintf(stderr, "pool1 mismatch after target block\n");
        return 0;
    }
    return 1;
}

static void run_case(const char *qpath, const char *npath, const char *fx68_path, const char *fx69_path){
    uint32_t qn, nn;
    uint8_t *q = read_file(qpath, &qn);
    uint8_t *raw = read_file(npath, &nn);
    uint8_t qsha[32], nsha[32];
    nprg2_sha256(q, qn, qsha);
    nprg2_sha256(raw, nn, nsha);
    Nprg2ExpectedHashes ex = {.context_sha256 = q + 32, .constants_sha256 = q + 64, .source_sha256 = qsha, .prepared_sha256 = nsha};

    Nprg2Pool pools[7] = {0};
    pool_alloc(raw, pools);
    Nprg2Program program;
    CHECK(nprg2_open(&program, raw, nn, &ex, pools, (Nprg2Allocator){0}) == NPRG2_OK);

    uint8_t *hot = calloc(1, 300000);
    CHECK(hot);
    Nprg2BindConfig bc = {.hot_memory = hot, .hot_capacity = 300000, .l2_available_bytes = 786432, .pool1_is_verified_vtcm = 1, .resolve = resolve_all};
    Nprg2Bound bound;
    CHECK(nprg2_bind_mock(&program, pools, &bc, &bound) == NPRG2_OK);

    Nprg2Executor executor;
    nprg2_executor_init(&executor);
    CHECK(nprg2_executor_register_slice_pad(&executor) == NPRG2_EXEC_OK);

    const char *fx_paths[2] = {fx68_path, fx69_path};
    for (unsigned which = 0; which < 2; which++) {
        uint32_t fn;
        uint8_t *fx = read_file(fx_paths[which], &fn);
        CHECK(fn >= HEADER_BYTES);
        CHECK(!memcmp(fx, "PPFX", 4));
        CHECK(rd32(fx + 4) == 1u);
        CHECK(rd32(fx + 8) == QDR_INPUT_SLICE_PAD);
        CHECK(rd32(fx + 12) == 2u);

        uint32_t seq = rd32(fx + 16);
        uint32_t cmd = rd32(fx + 20);
        uint32_t io = rd32(fx + 24);
        uint32_t io_bytes = rd32(fx + 28);
        uint32_t patches = rd32(fx + 32);
        CHECK(io < 2 && io_bytes > 0 && patches == 1);

        Nprg2Command command;
        CHECK(nprg2_command_at(&bound, cmd, &command) == NPRG2_OK);
        CHECK(command.opcode == QDR_INPUT_SLICE_PAD && command.variant == 2u && command.sequence == seq);

        const uint8_t *cur = fx + HEADER_BYTES;
        const uint8_t *before[7] = {0};
        for (unsigned p = 0; p < 7; p++) {
            uint32_t z = rd32(fx + 36 + 4 * p);
            CHECK(z == pools[p].bytes);
            if (z) {
                before[p] = cur;
                memcpy(pools[p].base, cur, z);
                cur += z;
            }
        }

        Nprg2ExecBuffer inputs[2] = {{0}};
        inputs[io].data = (uint8_t *)cur;
        inputs[io].bytes = io_bytes;
        cur += io_bytes;

        uint32_t off = rd32(cur);
        uint32_t n = rd32(cur + 4);
        cur += 8;
        CHECK(n == 2580u && off <= pools[1].bytes && n <= pools[1].bytes - off);
        const uint8_t *patch = cur;
        cur += n;
        CHECK((uint32_t)(cur - fx) == fn);

        Nprg2ExecRequest request = {
            .program = &program,
            .bound = &bound,
            .pool1_writable_bytes = ACTIVE_BYTES,
            .inputs = inputs,
            .input_count = 2,
            .command_first = cmd,
            .command_count = 1
        };
        memcpy(request.pools, pools, sizeof pools);

        /* 1. Read-only preflight: must return OK and leave memory untouched */
        CHECK(nprg2_executor_preflight(&executor, &request) == NPRG2_EXEC_OK);
        CHECK(compare_pools(pools, before));

        /* 2. Fail-closed: unsupported variant must reject before writes */
        uint8_t *cmd_bytes = (uint8_t *)bound.command.data + (size_t)cmd * NPRG2_COMMAND_BYTES;
        uint8_t old_variant = cmd_bytes[3];
        cmd_bytes[3] = 1u;
        CHECK(nprg2_executor_preflight(&executor, &request) == NPRG2_EXEC_UNSUPPORTED);
        CHECK(compare_pools(pools, before));
        cmd_bytes[3] = old_variant;

        /* 3. Fail-closed: destination offset exceeding writable pool1 ceiling */
        Nprg2ExecCall call = {
            .command_index = cmd,
            .command = command,
            .program = &program,
            .bound = &bound,
            .pools = pools,
            .pool1_writable_bytes = ACTIVE_BYTES,
            .inputs = inputs,
            .input_count = 2
        };
        uint32_t out_id;
        Nprg2TensorView out_view;
        CHECK(nprg2_exec_output(&call, 0, &out_id, &out_view) == NPRG2_EXEC_OK);
        uint8_t *tensor_row = (uint8_t *)bound.tensor.data + (size_t)out_id * NPRG2_TENSOR_BYTES;
        uint32_t old_offset = rd32(tensor_row + 4);
        wr32(tensor_row + 4, ACTIVE_BYTES - n + 1u);
        CHECK(nprg2_executor_preflight(&executor, &request) == NPRG2_EXEC_REJECTED);
        CHECK(compare_pools(pools, before));
        wr32(tensor_row + 4, old_offset);

        /* 4. Fail-closed: wrong input capacity */
        inputs[io].bytes = io_bytes - 1u;
        CHECK(nprg2_executor_preflight(&executor, &request) == NPRG2_EXEC_REJECTED);
        CHECK(compare_pools(pools, before));
        inputs[io].bytes = io_bytes;

        /* 5. Fail-closed: input buffer aliasing output in pool1 */
        inputs[io].data = pools[1].base + off;
        CHECK(nprg2_executor_preflight(&executor, &request) == NPRG2_EXEC_ARGUMENT);
        CHECK(compare_pools(pools, before));
        inputs[io].data = (uint8_t *)(cur - n - 8 - io_bytes);

        /* 6. Execution: must return OK */
        CHECK(nprg2_executor_run(&executor, &request) == NPRG2_EXEC_OK);

        /* 7. Exact bit-for-bit output and untouched background */
        CHECK(compare_after(pools, before, off, patch, n));

        printf("PASS InputSlicePad CPU-prefix %s: op=%u var=%u cmd=%lu seq=%lu dst_off=%lu bytes=%lu (100%% exact)\n",
               qpath, (unsigned)command.opcode, (unsigned)command.variant,
               (unsigned long)cmd, (unsigned long)seq, (unsigned long)off, (unsigned long)n);

        free(fx);
    }

    free(hot);
    nprg2_close(&program);
    pool_free(pools);
    free(q);
    free(raw);
}

int main(int argc, char **argv){
#ifdef __hexagon__
    SIM_ACQUIRE_HVX;
    SIM_SET_HVX_DOUBLE_MODE;
#endif

    if (argc == 1) {
        run_case("hexecutor/native/build/resident-alternate.qdrx",
                 "depth-model/deployment/programs/alternate-v2.nprg",
                 "hexecutor/native/program/fixtures/alternate-smoke-slice-pad-cmd68.ppcpu",
                 "hexecutor/native/program/fixtures/alternate-smoke-slice-pad-cmd69.ppcpu");
    } else if (argc == 5) {
        run_case(argv[1], argv[2], argv[3], argv[4]);
    } else {
        fprintf(stderr, "Usage: %s [qdrx nprg fx68 fx69]\n", argv[0]);
        return 1;
    }

#ifdef __hexagon__
    SIM_RELEASE_HVX;
#endif
    return 0;
}
