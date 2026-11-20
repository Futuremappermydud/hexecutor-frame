/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Reference-kernel schedule test for the host and the V75 simulator.
 *
 *   test_prepared_reference_sim [primary|alternate|both] [prefix|full]
 *   test_prepared_reference_sim <primary|alternate> split <segments> <dir>
 *   test_prepared_reference_sim <primary|alternate> tail <first> <dir> [window]
 *   test_prepared_reference_sim segment <file.rseg> [log every N commands]
 *   test_prepared_reference_sim [primary|alternate|both] profile   (host)
 *
 * Any mode also accepts "fast" (fast exact convolution in a VTCM-tail-sized
 * workspace), "ew" (Hexagon only: the HVX elementwise handlers, chained over the
 * reference kernel as on the Frame) and "cycles" (per-command cost table; pcycles in the simulator).
 *
 * prefix starts from a captured NativeGraph CPU prefix (.mpfx from
 * capture_multi_prefix_cpu.py) and requires all seven pools to equal the
 * capture afterwards. full starts from zeroed pools plus the constant pool,
 * runs the complete schedule on the native-smoke inputs, and requires both
 * graph outputs to match the emulator smoke sha256.
 *
 * split (host) runs the full schedule, cuts it into segments of roughly equal
 * host run time, checks the final outputs against the emulator sha256, and
 * writes each segment's start and end state (all pools plus both graph
 * outputs) as <dir>/<model>-seg<k>.rseg. segment runs one such file and
 * requires the end state bit for bit, so the segments can run in parallel
 * simulator processes and together cover the whole schedule.
 *
 * Every key is registered to prepared_reference. The census sizes scratch from
 * nprg2_exec_reference_scratch, one preflight covers the range without writing
 * any pool, and one run executes it. Run from the depth-model directory. */
#include "prepared_reference.h"
#include "prepared_fuse.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __hexagon__
#include <hexagon_sim_timer.h>
#include <hexagon_standalone.h>
#include "prepared_elementwise.h"
#endif

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

enum { MPFX_HEADER = 64, RSEG_HEADER = 72, POOL1_CEILING = 0x780000, MAX_SEGMENTS = 64, CONV_WORKSPACE = 0x7e000,
       CONV_NEAR = 0x8000 };

/* "fast" anywhere on the command line registers the fast exact convolution
 * with a VTCM-tail-sized workspace and the Frame's 32 KiB near slot; "cycles"
 * times every command and prints the cost per (opcode, variant) and the fast
 * convolution's phases after each run. */
static int use_fast, use_cycles;
/* "fuse": conv epilogue fusion plan (prepared_fuse.h) for every request. A
 * dead conv output is never stored, so pool 1 may differ from the unfused
 * host end state in those blocks (env_pools_match masks exactly them). */
static int use_fuse;
/* "ew" (Hexagon builds only): register the HVX elementwise handlers for scratch
 * region 1, chained in front of the reference kernel as the Frame build does. */
static int use_ew;
#ifdef __hexagon__
static Nprg2ReferenceChain ew_chains[32];
static uint8_t ew_scratch[NPRG2_ELEMENTWISE_SCRATCH_BYTES + 8] __attribute__((aligned(128)));
#endif
static Nprg2ReferenceConv conv;
static uint64_t conv_phase[8];
/* "hmx" (Hexagon only, implies fast): pool 1 and the conv workspace live in
 * VTCM, as on the Frame, and eligible u8 convs run on the simulated HMX
 * (prepared_hmx_conv.inc) with the hybrid's qd_hmx_wide/qd_hmx_finish. */
static uint8_t *hmx_vtcm;
#ifdef __hexagon__
extern void qd_hmx_wide(void *, void *, void *, void *, uint32_t);
extern void qd_hmx_finish(void *, void *);
extern void qd_hmx_wide_n(void *, void *, void *, void *, uint32_t, uint32_t);
extern void qd_hmx_native_n(void *, void *, void *, void *const *, uint32_t, uint32_t);
extern void qd_hmx_native16_n(void *, void *, void *, void *const *, uint32_t, uint32_t);
extern void qd_hmx_slide3_n(const uint32_t *, const void *, const void *, const uint32_t *, uint32_t, uint32_t);
/* SIM_HMX_N=0 keeps the per-tile qd_hmx_wide (the v6 call). */
#ifndef SIM_HMX_N
#define SIM_HMX_N 1
#endif
static const Nprg2HmxOps sim_hmx = {.wide = qd_hmx_wide, .finish = qd_hmx_finish, .wide_n = SIM_HMX_N ? qd_hmx_wide_n : NULL,
                                    .native_n = qd_hmx_native_n, .native16_n = qd_hmx_native16_n,
#ifndef NO_SLIDE3
                                    .slide3_n = qd_hmx_slide3_n
#endif
};
static void hmx_map_vtcm(void) {
    uint32_t base = __rdcfg(__vtcm_base) << 16, pn = base >> 12, index = 120;
    uint64_t entry = ((uint64_t)(0xc0000000u | pn) << 32) | (0xf6000000u | (pn << 1) | 0x40u);
    __asm__ volatile("tlbw(%0,%1)\n isync" ::"r"(entry), "r"(index) : "memory");
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" ::"r"(ssr) : "memory");
    hmx_vtcm = (uint8_t *)base;
}
#endif
/* "par" / "parrev" / "parshuf" / "thr": QDS_PREPARED_THREADS-style parallel hook
 * with 3 parts and the Frame's two 0x3e00-byte helper conv slots. par runs the
 * parts serially ascending, parrev descending, parshuf shuffled (any order
 * must give the same pools); thr (Hexagon only) runs parts 1 and 2 on real
 * hardware threads 1 and 2 of the simulator, the Frame's follower scheme. */
#ifndef SJ_MODE
#define SJ_MODE 0
#endif
static int use_par;
static Nprg2Parallel par_hook;
enum { PAR_SLOT = 0x3e00 };
static uint8_t par_slot[2][PAR_SLOT] __attribute__((aligned(128)));
#ifdef __hexagon__
#include <hexagon_standalone.h>
static volatile struct { uint32_t gen, op, parts; Nprg2ParallelFn fn; void *ctx; uint32_t done[3]; } sj;
static uint8_t sj_stack[3][65536] __attribute__((aligned(16)));
static volatile uint32_t sj_ran[3];
static void sj_worker(void *p) {
    uint32_t tid = (uint32_t)(uintptr_t)p, seen = 0;
    /* The simulator shares HVX contexts between threads unless each thread acquires one. */
    acquire_vector_unit(HEXAGON_VECTOR_WAIT);
    for (;;) {
        while (sj.gen == seen) __asm__ volatile("pause(#32)");
        __asm__ volatile("barrier" ::: "memory");
        seen = sj.gen;
        if (sj.op == 3) { sj.done[tid] = seen; return; }
#if SJ_MODE==2
        if (tid==2) while (sj.done[1] != seen) __asm__ volatile("pause(#32)");
#endif
        if (tid < sj.parts) { sj.fn(sj.ctx, tid, sj.parts); sj_ran[tid]++; }
        __asm__ volatile("barrier" ::: "memory");
        sj.done[tid] = seen;
    }
}
static int sj_run(void *impl, uint32_t parts, Nprg2ParallelFn fn, void *ctx) {
    (void)impl;
    if (parts < 2 || parts > 3) return 0;
    sj.op = 4; sj.fn = fn; sj.ctx = ctx; sj.parts = parts;
    uint32_t gen = sj.gen + 1;
#if SJ_MODE>=1
    fn(ctx, 0, parts);
#endif
    __asm__ volatile("barrier" ::: "memory");
    sj.gen = gen;
#if SJ_MODE==0
    fn(ctx, 0, parts);
#endif
    while (sj.done[1] != gen || sj.done[2] != gen) __asm__ volatile("pause(#32)");
    __asm__ volatile("barrier" ::: "memory");
    return 1;
}
static void sj_stop(void);
static void sj_start(void) {
    memset(sj_stack, 0xa5, sizeof sj_stack);
    acquire_vector_unit(HEXAGON_VECTOR_WAIT); /* every thread, the main one too */
    for (int t = 1; t < 3; t++) thread_create(sj_worker, sj_stack[t] + sizeof sj_stack[t] - 64, t, (void *)(uintptr_t)t);
    par_hook.run = sj_run;
    atexit(sj_stop);
}
static void sj_stop(void) {
    if (!par_hook.run) return;
    uint32_t gen = sj.gen + 1;
    sj.op = 3; __asm__ volatile("barrier" ::: "memory"); sj.gen = gen;
    thread_join(0x6);
    for (int t = 1; t < 3; t++) {
        uint32_t i = 0;
        while (i < sizeof sj_stack[t] && sj_stack[t][i] == 0xa5) i++;
        printf("worker %d stack high-water %lu of %lu bytes\n", t, (unsigned long)(sizeof sj_stack[t] - i), (unsigned long)sizeof sj_stack[t]);
    }
    printf("worker parts run: %lu %lu\n", (unsigned long)sj_ran[1], (unsigned long)sj_ran[2]);
    printf("parallel jobs conv/ew/dm %llu/%llu/%llu parts %llu/%llu/%llu declined %llu\n",
           (unsigned long long)par_hook.jobs_parallel[0], (unsigned long long)par_hook.jobs_parallel[1], (unsigned long long)par_hook.jobs_parallel[2],
           (unsigned long long)par_hook.parts_run[0], (unsigned long long)par_hook.parts_run[1], (unsigned long long)par_hook.parts_run[2],
           (unsigned long long)par_hook.declined);
}
#endif

/* Simulator pcycles on Hexagon (link -lhexagon), clock() ticks on the host. */
static uint64_t ticks(void) {
#ifdef __hexagon__
    return hexagon_sim_read_pcycles();
#else
    return (uint64_t)clock();
#endif
}
static const char *tick_unit(void) {
#ifdef __hexagon__
    return "pcycles";
#else
    return "clock ticks";
#endif
}

static const uint32_t kOutputBytes[2] = {163840, 655360};
#ifndef NPRG2_HW_ROUNDING
#define NPRG2_HW_ROUNDING 0
#endif
#if NPRG2_HW_ROUNDING
/* Silicon-rounding oracle (recovered/<model>/hw_smoke.json, tools/make_hw_oracle.py). */
static const char *const kExpect[2][2] = {
    {"1e8ea71c160e22ab8f9ee06110598b471cddc02d3e8ae9b9c6276759c73ef178",
     "97b0d8066a3a89ef14f1e45376b9b70ab6000b645287d1b7e5617097914bb16c"},
    {"1e8ea71c160e22ab8f9ee06110598b471cddc02d3e8ae9b9c6276759c73ef178",
     "655381a40e758f03e6f12ad79480114acc147b56a2140c6fddc25a88a6eaa916"}};
#else
static const char *const kExpect[2][2] = {
    {"dfacee192b7a534ef79634993d03a1c870e1550e5ba1c485af5ab17466616ad3",
     "a978ee8291ea05c0b482b72cdc54f09c1ad8294a366f3ba215bb32653cb3f32b"},
    {"dfacee192b7a534ef79634993d03a1c870e1550e5ba1c485af5ab17466616ad3",
     "98d28b66c3489d1424569aad3e1c2b9bfa12fac3aba5b11e67b0fde10d8bc36a"}};
#endif

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static void wr32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }

static uint8_t *read_file(const char *path, uint32_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    CHECK(!fseek(f, 0, SEEK_END));
    long z = ftell(f);
    CHECK(z > 0 && (uint64_t)z <= 64u * 1024u * 1024u);
    CHECK(!fseek(f, 0, SEEK_SET));
    uint8_t *p = malloc((size_t)z);
    CHECK(p && fread(p, 1, (size_t)z, f) == (size_t)z);
    fclose(f);
    *n = (uint32_t)z;
    return p;
}

static int resolve_all(void *ctx, const Nprg2Program *p, uint32_t i, const Nprg2Command *c, Nprg2Capability *cap) {
    (void)ctx; (void)p; (void)i; (void)c;
    cap->supported = 1; cap->may_fallback = 0; cap->scratch_bytes = 0; cap->concurrent_worker_scratch_bytes = 0;
    return 1;
}

static int model_code(const char *model) { return strcmp(model, "primary") ? 1 : 0; }

static int outputs_match_emulator(const char *model, uint8_t *const *out) {
    for (int j = 0; j < 2; j++) {
        uint8_t sha[32];
        char hex[65];
        nprg2_sha256(out[j], kOutputBytes[j], sha);
        for (int b = 0; b < 32; b++) snprintf(hex + 2 * b, 3, "%02x", sha[b]);
        if (strcmp(hex, kExpect[model_code(model)][j])) {
            fprintf(stderr, "%s output %d sha256 %s, emulator %s\n", model, j, hex, kExpect[model_code(model)][j]);
            return 0;
        }
    }
    return 1;
}

/* An opened, bound program with the reference kernel on every key. */
typedef struct {
    const char *model;
    uint8_t *q, *raw, *hot, *scratch;
    uint32_t scratch_bytes;
    uint32_t progress; /* nonzero: run in chunks of this many commands and log each */
    Nprg2Pool pools[7];
    Nprg2Program p;
    Nprg2Bound b;
    Nprg2Executor e;
    uint8_t *out[2];
    Nprg2ExecBuffer in[4], ob[2];
    Nprg2FusePlan plan;
    uint8_t mask[4096];
} Env;

static void env_open(Env *v, const char *model, uint8_t *const *inputs, const uint32_t *input_bytes) {
    char path[256];
    uint32_t qn, nn;
    memset(v, 0, sizeof *v);
    v->model = model;
    snprintf(path, sizeof path, "hexecutor/native/build/resident-%s.qdrx", model);
    v->q = read_file(path, &qn);
    snprintf(path, sizeof path, "depth-model/deployment/programs/%s-v2.nprg", model);
    v->raw = read_file(path, &nn);
    uint8_t qsha[32], nsha[32];
    nprg2_sha256(v->q, qn, qsha);
    nprg2_sha256(v->raw, nn, nsha);
    Nprg2ExpectedHashes exp = {v->q + 32, v->q + 64, qsha, nsha};
    for (uint32_t i = 0; i < 7; i++) {
        v->pools[i].bytes = rd32(v->raw + 144 + i * 4);
        if (!v->pools[i].bytes) continue;
        v->pools[i].base = i == 1 && hmx_vtcm ? hmx_vtcm : aligned_alloc(2048, (v->pools[i].bytes + 2047u) & ~2047u);
        CHECK(i != 1 || !hmx_vtcm || v->pools[1].bytes <= 0x800000u);
        CHECK(v->pools[i].base);
        memset(v->pools[i].base, 0, v->pools[i].bytes);
    }
    CHECK(nprg2_open(&v->p, v->raw, nn, &exp, v->pools, (Nprg2Allocator){0}) == NPRG2_OK);
    v->hot = calloc(1, 300000);
    CHECK(v->hot);
    Nprg2BindConfig bc = {.hot_memory = v->hot, .hot_capacity = 300000, .l2_available_bytes = 786432,
                          .pool1_is_verified_vtcm = 1, .resolve = resolve_all};
    CHECK(nprg2_bind_mock(&v->p, v->pools, &bc, &v->b) == NPRG2_OK);
    if (use_fuse) {
        uint64_t t0 = ticks();
        int frc = nprg2_fuse_plan_build(&v->plan, &v->p, &v->b, (Nprg2Allocator){0});
        CHECK(frc == NPRG2_FUSE_OK);
        printf("%s: fusion plan %lu pairs, build cost %llu %s\n", model, (unsigned long)v->plan.count,
               (unsigned long long)(ticks() - t0), tick_unit());
    }
    nprg2_executor_init(&v->e);
    uint32_t added = 0;
    if (use_fast) {
        if (!conv.workspace) {
            /* The Frame's layout: workspace at tail+0x2000, above pool 1's writable ceiling. */
            conv.workspace = hmx_vtcm ? hmx_vtcm + 0x782000u : aligned_alloc(2048, CONV_WORKSPACE);
#ifdef __hexagon__
            if (hmx_vtcm) conv.hmx = &sim_hmx;
#endif
            conv.workspace_bytes = CONV_WORKSPACE;
            conv.near = aligned_alloc(128, CONV_NEAR);
            conv.near_bytes = CONV_NEAR;
            CHECK(conv.workspace && conv.near);
            if (use_cycles) { conv.clock = ticks; conv.phase = conv_phase; }
            if (use_par) { conv.par_near[0] = par_slot[0]; conv.par_near[1] = par_slot[1]; conv.par_near_bytes = PAR_SLOT; }
        }
        CHECK(nprg2_executor_register_reference_conv(&v->e, &conv, &added) == NPRG2_EXEC_OK);
    }
#ifdef __hexagon__
    if (use_ew) {
        CHECK(nprg2_executor_register_elementwise_region(&v->e, NPRG2_EXEC_SCRATCH_NEAR) == NPRG2_EXEC_OK);
        uint32_t chained = 0;
        for (uint32_t i = 0; i < NPRG2_EXEC_REGISTRY_CAPACITY; i++) {
            Nprg2ExecHandler *h = &v->e.handlers[i];
            if (!h->used || h->opcode == QDR_NOP || h->validate == nprg2_exec_reference_validate) continue;
            CHECK(chained < sizeof ew_chains / sizeof ew_chains[0]);
            CHECK(nprg2_executor_chain_reference(&v->e, h->opcode, h->variant, &ew_chains[chained]) == NPRG2_EXEC_OK);
            chained++;
        }
        printf("%s: %lu HVX elementwise keys chained over the reference kernel\n", model, (unsigned long)chained);
    }
#endif
    CHECK(nprg2_executor_register_reference_missing(&v->e, &added) == NPRG2_EXEC_OK);
    for (int j = 0; j < 2; j++) {
        v->out[j] = calloc(1, kOutputBytes[j]);
        CHECK(v->out[j]);
        v->ob[j] = (Nprg2ExecBuffer){v->out[j], kOutputBytes[j]};
    }
    for (int j = 0; j < 4; j++) v->in[j] = (Nprg2ExecBuffer){inputs[j], input_bytes[j]};
}

static void env_close(Env *v) {
    nprg2_fuse_plan_free(&v->plan);
    free(v->scratch);
    free(v->hot);
    nprg2_close(&v->p);
    for (int i = 0; i < 7; i++) if (i != 1 || !hmx_vtcm) free(v->pools[i].base);
    for (int j = 0; j < 2; j++) free(v->out[j]);
    free(v->raw);
    free(v->q);
}

/* Sizes scratch for [first, first+count) and prints the range's census. */
static void env_census(Env *v, uint32_t first, uint32_t count) {
    uint32_t peak = 0, peak_cmd = first, histogram[QDR_OPCODES + 1] = {0}, ws_peak = 0, near = 0, hmx = 0;
    for (uint32_t i = first; i < first + count; i++) {
        Nprg2ExecCall call = {.command_index = i, .program = &v->p, .bound = &v->b, .pools = v->pools,
                              .pool1_writable_bytes = POOL1_CEILING,
                              .inputs = v->in, .input_count = 4, .outputs = v->ob, .output_count = 2};
        CHECK(nprg2_command_at(&v->b, i, &call.command) == NPRG2_OK);
        uint32_t need = 0, ws = 0;
        if (!nprg2_exec_reference_conv_measure(&call, use_fast ? &conv : NULL, &need, &ws)) {
            fprintf(stderr, "%s census reject cmd=%lu op=%u var=%u line=%lu\n", v->model, (unsigned long)i,
                    call.command.opcode, call.command.variant, (unsigned long)nprg2_exec_reference_error_line());
            exit(1);
        }
        if (call.command.opcode <= QDR_OPCODES) histogram[call.command.opcode]++;
        if (need > peak) { peak = need; peak_cmd = i; }
        if (ws > ws_peak) ws_peak = ws;
        if (call.command.opcode == QDR_HMX_CONV && nprg2_exec_reference_conv_path() == 2) near++;
        if (call.command.opcode == QDR_HMX_CONV && nprg2_exec_reference_conv_path() >= 3) hmx++;  /* 4: native int8 */
    }
    if (use_fast)
        printf("%s: fast convolution workspace peak %lu bytes, %lu convolutions with near memory, %lu on HMX\n", v->model,
               (unsigned long)ws_peak, (unsigned long)near, (unsigned long)hmx);
    free(v->scratch);
    v->scratch_bytes = (peak + 4095u) & ~4095u;
    v->scratch = v->scratch_bytes ? aligned_alloc(128, v->scratch_bytes) : NULL;
    CHECK(!v->scratch_bytes || v->scratch);
    printf("%s: commands %lu..%lu, scratch peak %lu bytes at cmd %lu; opcodes:", v->model, (unsigned long)first,
           (unsigned long)(first + count - 1u), (unsigned long)peak, (unsigned long)peak_cmd);
    for (uint32_t op = 0; op <= QDR_OPCODES; op++)
        if (histogram[op]) printf(" %lu:%lu", (unsigned long)op, (unsigned long)histogram[op]);
    printf("\n");
    fflush(stdout);
}

/* Marks the pool-1 blocks of command cmd's output (a u8 CH tensor) as deliberately stale. */
static void mask_conv_out(Env *v, uint32_t cmd) {
    Nprg2Command c;
    CHECK(nprg2_command_at(&v->b, cmd, &c) == NPRG2_OK && c.output_count == 1);
    uint32_t ix = v->b.refs.data[c.output_first * 2u] | (uint32_t)v->b.refs.data[c.output_first * 2u + 1u] << 8;
    Nprg2TensorView tv;
    CHECK(nprg2_bound_tensor_at(&v->p, &v->b, ix, &tv) == NPRG2_OK && tv.storage_kind == 2);
    const uint8_t *g = tv.geometry_record;
    int m1 = (int16_t)(g[16] | g[17] << 8), m2 = (int16_t)(g[18] | g[19] << 8), m3 = (int16_t)(g[20] | g[21] << 8);
    uint64_t n = (uint64_t)(m1 / 8) * (m2 / 8) * (m3 / 32);
    for (uint64_t i = 0; i < n; i++) {
        uint32_t e = tv.blocks[tv.slice_origin + i];
        if ((e & 127u) == 1u) v->mask[(e & ~127u) / 2048u] = 1;
    }
}

/* One preflight, then one run. With start (seven pools, then two outputs; a
 * NULL entry means all zero), the preflight must leave every pool and output
 * equal to it. The simulator heap has no room for a separate copy. */
static void env_run(Env *v, uint32_t first, uint32_t count, const uint8_t *const *start) {
    Nprg2ExecStats st;
    Nprg2ExecRequest r = {.program = &v->p, .bound = &v->b, .pool1_writable_bytes = POOL1_CEILING,
                          .inputs = v->in, .input_count = 4, .outputs = v->ob, .output_count = 2,
                          .scratch = v->scratch, .scratch_bytes = v->scratch_bytes,
                          .command_first = first, .command_count = count, .stats = &st, .fuse = use_fuse ? &v->plan : NULL};
    memcpy(r.pools, v->pools, sizeof v->pools);
    if (use_par) r.parallel = &par_hook;
#ifdef __hexagon__
    if (use_ew) r.extra_scratch[0] = (Nprg2ExecBuffer){ew_scratch, NPRG2_ELEMENTWISE_SCRATCH_BYTES};
#endif
    uint64_t preflight_started = ticks();
    int rc = nprg2_executor_preflight(&v->e, &r);
    if (use_cycles)
        printf("%s: preflight of commands %lu..%lu cost %llu %s\n", v->model, (unsigned long)first,
               (unsigned long)(first + count - 1u), (unsigned long long)(ticks() - preflight_started), tick_unit());
    if (rc != NPRG2_EXEC_OK) {
        fprintf(stderr, "%s preflight rc=%d cmd=%lu op=%u var=%u line=%lu\n", v->model, rc,
                (unsigned long)st.failed_command, st.failed_opcode, st.failed_variant,
                (unsigned long)nprg2_exec_reference_error_line());
        exit(1);
    }
    for (int i = 0; start && i < 9; i++) {
        const uint8_t *now = i < 7 ? v->pools[i].base : v->out[i - 7];
        uint32_t n = i < 7 ? v->pools[i].bytes : kOutputBytes[i - 7];
        if (start[i]) CHECK(!n || !memcmp(now, start[i], n));
        else for (uint32_t j = 0; j < n; j++) CHECK(!now[j]);
    }
    static uint64_t cost[QDR_OPCODES + 1][QDR_VARIANTS + 1];
    static uint32_t calls[QDR_OPCODES + 1][QDR_VARIANTS + 1];
    memset(cost, 0, sizeof cost);
    memset(calls, 0, sizeof calls);
    memset(conv_phase, 0, sizeof conv_phase);
    uint64_t started = ticks(), total = 0;
    uint32_t fused_spans = 0;
    uint32_t step = v->progress ? v->progress : count, logged = 0;
    for (uint32_t done = 0; done < count; done += r.command_count) {
        r.command_first = first + done;
        r.command_count = use_cycles ? 1u : count - done < step ? count - done : step;
        if (use_fuse && use_cycles) {
            /* One step spans the overlapping fusion pairs, so fused conv and epilogue run in one request. */
            uint32_t end = r.command_first;
            for (uint32_t k = r.command_first; k <= end; k++) {
                uint16_t role = v->plan.role[k];
                if (role && !(role & NPRG2_FUSE_EPI) && v->plan.pairs[role - 1u].epi < first + count && v->plan.pairs[role - 1u].epi > end)
                    end = v->plan.pairs[role - 1u].epi;
            }
            r.command_count = end + 1u - r.command_first;
        }
        uint64_t t0 = ticks();
        rc = nprg2_executor_run(&v->e, &r);
        uint64_t spent = ticks() - t0;
        if (rc != NPRG2_EXEC_OK) {
            fprintf(stderr, "%s run rc=%d cmd=%lu op=%u var=%u line=%lu\n", v->model, rc, (unsigned long)st.failed_command,
                    st.failed_opcode, st.failed_variant, (unsigned long)nprg2_exec_reference_error_line());
            exit(1);
        }
        CHECK(st.executed_commands == r.command_count && st.validated_commands == r.command_count);
        if (use_cycles) {
            Nprg2Command c;
            CHECK(nprg2_command_at(&v->b, r.command_first, &c) == NPRG2_OK && c.opcode <= QDR_OPCODES && c.variant <= QDR_VARIANTS);
            if (r.command_count > 1) { c.opcode = 0; c.variant = 0; fused_spans++; }   /* conv + gap + epilogue */
            cost[c.opcode][c.variant] += spent;
            calls[c.opcode][c.variant]++;
            total += spent;
        }
        uint32_t ran = done + r.command_count;
        if (v->progress && (ran - logged >= step || ran == count)) {
            logged = ran;
            printf("%s: ran %lu/%lu commands (through cmd %lu), %llu %s\n", v->model, (unsigned long)ran,
                   (unsigned long)count, (unsigned long)(first + ran - 1u), (unsigned long long)(ticks() - started),
                   tick_unit());
            fflush(stdout);
        }
    }
    if (use_fuse) {
        uint32_t act = 0, dead = 0, inside = 0;
        for (uint32_t k = 0; k < v->plan.count; k++) {
            const Nprg2FusePair *pr = &v->plan.pairs[k];
            if (pr->conv < first || pr->epi >= first + count) continue;
            inside++;
            if (use_cycles || (pr->active)) { act += pr->active; }
            if (pr->active && pr->dead) { dead++; mask_conv_out(v, pr->conv); }
        }
        printf("%s: fusion pairs inside commands %lu..%lu: %lu, active %lu, dead %lu\n", v->model, (unsigned long)first,
               (unsigned long)(first + count - 1u), (unsigned long)inside, (unsigned long)act, (unsigned long)dead);
    }
    if (!use_cycles) return;
    if (fused_spans) printf("  (%lu fused spans are costed under op 0 var 0: conv + commands in between + epilogue)\n", (unsigned long)fused_spans);
    printf("%s: commands %lu..%lu cost %llu %s in handlers (validate + run)\n  share        %-7s  commands  op var\n",
           v->model, (unsigned long)first, (unsigned long)(first + count - 1u), (unsigned long long)total, tick_unit(),
           tick_unit());
    for (;;) {
        uint32_t bo = 0, bv = 0;
        for (uint32_t o = 0; o <= QDR_OPCODES; o++)
            for (uint32_t w = 0; w <= QDR_VARIANTS; w++)
                if (calls[o][w] && (!calls[bo][bv] || cost[o][w] > cost[bo][bv])) { bo = o; bv = w; }
        if (!calls[bo][bv]) break;
        printf("  %5.1f%%  %12llu  %8lu  %2lu %3lu\n", total ? 100.0 * (double)cost[bo][bv] / (double)total : 0.0,
               (unsigned long long)cost[bo][bv], (unsigned long)calls[bo][bv], (unsigned long)bo, (unsigned long)bv);
        calls[bo][bv] = 0;
    }
    if (conv.phase)
        for (int k = 0; k < 2; k++)
            printf("  near conv phases %s (%s): staging %llu, mac %llu, requant %llu, output %llu\n", k ? "u16" : "u8",
                   tick_unit(), (unsigned long long)conv_phase[4 * k], (unsigned long long)conv_phase[4 * k + 1],
                   (unsigned long long)conv_phase[4 * k + 2], (unsigned long long)conv_phase[4 * k + 3]);
    fflush(stdout);
}

static int env_pools_match(Env *v, const uint8_t *const *want) {
    for (int i = 0; i < 7; i++) {
        /* hmx: the conv workspace is pool 1's reserved tail above the writable
         * ceiling (the Frame layout), so that range is scratch, not state. */
        uint32_t len = i == 1 && hmx_vtcm && v->pools[1].bytes > POOL1_CEILING ? POOL1_CEILING : v->pools[i].bytes;
        if (!len || !memcmp(v->pools[i].base, want[i], len)) continue;
        uint32_t diff = 0;
        if (i == 1 && use_fuse) {
            int bad = 0;
            for (uint32_t b = 0; b * 2048u < len && !bad; b++) {
                uint32_t n = len - b * 2048u < 2048u ? len - b * 2048u : 2048u;
                if (v->mask[b] || !memcmp(v->pools[1].base + b * 2048u, want[1] + b * 2048u, n)) continue;
                diff = b * 2048u;
                while (v->pools[1].base[diff] == want[1][diff]) diff++;
                bad = 1;
            }
            if (!bad) continue;
        } else while (v->pools[i].base[diff] == want[i][diff]) diff++;
        fprintf(stderr, "%s pool mismatch: pool=%d off=0x%lx prepared=%u want=%u\n", v->model, i, (unsigned long)diff,
                v->pools[i].base[diff], want[i][diff]);
        uint32_t n = 0, lines = 0, last = 0, prev_line = UINT32_MAX;
        for (uint32_t k = diff; k < len; k++)
            if (v->pools[i].base[k] != want[i][k]) { n++; last = k; if (k / 128u != prev_line) { lines++; prev_line = k / 128u; } }
        fprintf(stderr, "  %lu bytes differ in %lu 128-byte lines, first 0x%lx last 0x%lx\n", (unsigned long)n, (unsigned long)lines,
                (unsigned long)diff, (unsigned long)last);
        return 0;
    }
    return 1;
}

static void read_smoke(const char *model, uint8_t **inputs, uint32_t *input_bytes, uint8_t **constants, uint32_t *cn) {
    static const char *const names[4] = {"x", "y", "hidden", "t"};
    char path[256];
    for (int j = 0; j < 4; j++) {
        snprintf(path, sizeof path, "hexecutor/native/build/native-smoke/%s.raw", names[j]);
        inputs[j] = read_file(path, &input_bytes[j]);
    }
    snprintf(path, sizeof path, "recovered/%s/constants/constant_pool_5.bin", model);
    *constants = read_file(path, cn);
}

static void run_prefix(const char *model) {
    char path[256];
    uint32_t fn;
    snprintf(path, sizeof path, "hexecutor/native/program/fixtures/%s-smoke-prefix-%d.mpfx", model,
             model_code(model) ? 100 : 132);
    uint8_t *fx = read_file(path, &fn);
    CHECK(fn >= MPFX_HEADER && !memcmp(fx, "MPFX", 4) && rd32(fx + 4) == 1u && rd32(fx + 16) == 4u);
    CHECK(rd32(fx + 8) == (uint32_t)model_code(model));
    uint32_t count = rd32(fx + 12), input_bytes[4], pool_bytes[7];
    uint8_t *inputs[4];
    const uint8_t *initial[7], *want[7];
    uint64_t at = MPFX_HEADER;
    for (int j = 0; j < 4; j++) { input_bytes[j] = rd32(fx + 20 + 4 * j); inputs[j] = fx + at; at += input_bytes[j]; }
    for (int i = 0; i < 7; i++) { pool_bytes[i] = rd32(fx + 36 + 4 * i); initial[i] = fx + at; at += pool_bytes[i]; }
    for (int i = 0; i < 7; i++) { want[i] = fx + at; at += pool_bytes[i]; }
    CHECK(at == fn);
    static Env v;
    env_open(&v, model, inputs, input_bytes);
    CHECK(count > 0 && count <= v.p.command_count);
    for (int i = 0; i < 7; i++) {
        CHECK(v.pools[i].bytes == pool_bytes[i]);
        if (pool_bytes[i]) memcpy(v.pools[i].base, initial[i], pool_bytes[i]);
    }
    env_census(&v, 0, count);
    const uint8_t *start[9] = {initial[0], initial[1], initial[2], initial[3], initial[4], initial[5], initial[6], NULL, NULL};
    env_run(&v, 0, count, start);
    if (!env_pools_match(&v, want)) exit(1);
    printf("PASS %s reference prefix: commands 0..%lu, one preflight without pool writes, one run, "
           "all seven pools identical to the NativeGraph CPU capture\n", model, (unsigned long)count - 1u);
    fflush(stdout);
    env_close(&v);
    free(fx);
}

static void run_full(const char *model) {
    uint8_t *inputs[4], *constants;
    uint32_t input_bytes[4], cn;
    read_smoke(model, inputs, input_bytes, &constants, &cn);
    static Env v;
    env_open(&v, model, inputs, input_bytes);
    CHECK(v.pools[5].bytes == cn);
    memcpy(v.pools[5].base, constants, cn);
    env_census(&v, 0, v.p.command_count);
    const uint8_t *start[9] = {0};
    start[5] = constants;
    env_run(&v, 0, v.p.command_count, start);
    if (!outputs_match_emulator(model, v.out)) exit(1);
    printf("PASS %s reference full schedule: %lu commands, one preflight without pool writes, one run, "
           "both graph outputs match the emulator smoke sha256\n", model, (unsigned long)v.p.command_count);
    fflush(stdout);
    env_close(&v);
    free(constants);
    for (int j = 0; j < 4; j++) free(inputs[j]);
}

/* RSEG v1: "RSEG", version, model code, first, count, input bytes[4],
 * pool bytes[7], output bytes[2]; then the inputs, the start pools and
 * outputs, and the end pools and outputs. */
static uint8_t *snapshot(Env *v, uint32_t *bytes) {
    uint32_t n = kOutputBytes[0] + kOutputBytes[1];
    for (int i = 0; i < 7; i++) n += v->pools[i].bytes;
    uint8_t *s = malloc(n), *at = s;
    CHECK(s);
    for (int i = 0; i < 7; i++) { if (v->pools[i].bytes) memcpy(at, v->pools[i].base, v->pools[i].bytes); at += v->pools[i].bytes; }
    for (int j = 0; j < 2; j++) { memcpy(at, v->out[j], kOutputBytes[j]); at += kOutputBytes[j]; }
    *bytes = n;
    return s;
}

static void write_segment(Env *v, const char *dir, uint32_t k, uint32_t first, uint32_t count, const uint32_t *input_bytes,
                          const uint8_t *start, const uint8_t *end, uint32_t state_bytes) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s-seg%lu.rseg", dir, v->model, (unsigned long)k);
    uint8_t h[RSEG_HEADER];
    memcpy(h, "RSEG", 4);
    wr32(h + 4, 1);
    wr32(h + 8, (uint32_t)model_code(v->model));
    wr32(h + 12, first);
    wr32(h + 16, count);
    for (int j = 0; j < 4; j++) wr32(h + 20 + 4 * j, input_bytes[j]);
    for (int i = 0; i < 7; i++) wr32(h + 36 + 4 * i, v->pools[i].bytes);
    for (int j = 0; j < 2; j++) wr32(h + 64 + 4 * j, kOutputBytes[j]);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot create %s\n", path); exit(1); }
    CHECK(fwrite(h, 1, sizeof h, f) == sizeof h);
    for (int j = 0; j < 4; j++) CHECK(fwrite(v->in[j].data, 1, input_bytes[j], f) == input_bytes[j]);
    CHECK(fwrite(start, 1, state_bytes, f) == state_bytes);
    CHECK(fwrite(end, 1, state_bytes, f) == state_bytes);
    CHECK(!fclose(f));
    printf("%s: commands %lu..%lu\n", path, (unsigned long)first, (unsigned long)(first + count - 1u));
}

static void split(const char *model, uint32_t segments, const char *dir) {
    CHECK(segments >= 1 && segments <= MAX_SEGMENTS);
    uint8_t *inputs[4], *constants;
    uint32_t input_bytes[4], cn;
    read_smoke(model, inputs, input_bytes, &constants, &cn);
    static Env v;
    env_open(&v, model, inputs, input_bytes);
    uint32_t n = v.p.command_count;
    CHECK(v.pools[5].bytes == cn && segments <= n);
    memcpy(v.pools[5].base, constants, cn);
    env_census(&v, 0, n);

    double *cost = calloc(n, sizeof *cost), total = 0;
    CHECK(cost);
    for (uint32_t i = 0; i < n; i++) {
        clock_t t0 = clock();
        env_run(&v, i, 1, NULL);
        cost[i] = (double)(clock() - t0) + 1.0;
        total += cost[i];
    }
    uint32_t bound[MAX_SEGMENTS + 1] = {0}, k = 1;
    double cum = 0;
    for (uint32_t i = 0; i < n && k < segments; i++) {
        cum += cost[i];
        if (cum >= total * k / segments && i + 1 > bound[k - 1]) bound[k++] = i + 1;
    }
    segments = k;
    bound[segments] = n;
    free(cost);

    for (int i = 0; i < 7; i++) if (v.pools[i].bytes) memset(v.pools[i].base, 0, v.pools[i].bytes);
    memcpy(v.pools[5].base, constants, cn);
    for (int j = 0; j < 2; j++) memset(v.out[j], 0, kOutputBytes[j]);
    uint32_t state_bytes;
    uint8_t *start = snapshot(&v, &state_bytes);
    for (uint32_t s = 0; s < segments; s++) {
        env_run(&v, bound[s], bound[s + 1] - bound[s], NULL);
        uint8_t *end = snapshot(&v, &state_bytes);
        write_segment(&v, dir, s, bound[s], bound[s + 1] - bound[s], input_bytes, start, end, state_bytes);
        free(start);
        start = end;
    }
    free(start);
    if (!outputs_match_emulator(model, v.out)) exit(1);
    printf("PASS %s split: %lu segments cover commands 0..%lu; final outputs match the emulator smoke sha256\n", model,
           (unsigned long)segments, (unsigned long)n - 1u);
    env_close(&v);
    free(constants);
    for (int j = 0; j < 4; j++) free(inputs[j]);
}

/* One segment covering [first, end): the host runs the schedule once, keeping the state before
 * `first`, so a simulator run can start in the late part of a schedule without simulating
 * everything before it. */
static void tail(const char *model, uint32_t first, const char *dir, uint32_t window) {
    uint8_t *inputs[4], *constants;
    uint32_t input_bytes[4], cn;
    read_smoke(model, inputs, input_bytes, &constants, &cn);
    static Env v;
    env_open(&v, model, inputs, input_bytes);
    uint32_t n = v.p.command_count;
    CHECK(v.pools[5].bytes == cn && first < n);
    /* window == 0: through the end of the schedule; else only [first, first+window), for a short
     * focused simulator run (the outputs then need not be final). */
    uint32_t count = window && window < n - first ? window : n - first;
    memcpy(v.pools[5].base, constants, cn);
    env_census(&v, 0, n);
    env_run(&v, 0, first, NULL);
    uint32_t state_bytes;
    uint8_t *start = snapshot(&v, &state_bytes);
    env_run(&v, first, count, NULL);
    uint8_t *end = snapshot(&v, &state_bytes);
    write_segment(&v, dir, 0, first, count, input_bytes, start, end, state_bytes);
    free(start);
    free(end);
    if (count == n - first && !outputs_match_emulator(model, v.out)) exit(1);
    printf("PASS %s tail: commands %lu..%lu written%s\n", model,
           (unsigned long)first, (unsigned long)(first + count - 1u),
           count == n - first ? "; final outputs match the emulator smoke sha256" : " (window)");
    env_close(&v);
    free(constants);
    for (int j = 0; j < 4; j++) free(inputs[j]);
}

/* Host time per (opcode, variant) over the full schedule, slowest first. */
static void profile(const char *model) {
    uint8_t *inputs[4], *constants;
    uint32_t input_bytes[4], cn;
    read_smoke(model, inputs, input_bytes, &constants, &cn);
    static Env v;
    env_open(&v, model, inputs, input_bytes);
    CHECK(v.pools[5].bytes == cn);
    memcpy(v.pools[5].base, constants, cn);
    env_census(&v, 0, v.p.command_count);
    static double cost[QDR_OPCODES + 1][QDR_VARIANTS + 1];
    static uint32_t calls[QDR_OPCODES + 1][QDR_VARIANTS + 1];
    memset(cost, 0, sizeof cost);
    memset(calls, 0, sizeof calls);
    double total = 0;
    for (uint32_t i = 0; i < v.p.command_count; i++) {
        Nprg2Command c;
        CHECK(nprg2_command_at(&v.b, i, &c) == NPRG2_OK && c.opcode <= QDR_OPCODES && c.variant <= QDR_VARIANTS);
        clock_t t0 = clock();
        env_run(&v, i, 1, NULL);
        double t = (double)(clock() - t0) / CLOCKS_PER_SEC;
        cost[c.opcode][c.variant] += t;
        calls[c.opcode][c.variant]++;
        total += t;
    }
    if (!outputs_match_emulator(model, v.out)) exit(1);
    printf("%s profile: %.3f s total host time, reference kernel on every key\n", model, total);
    printf("  share   seconds  commands  op var\n");
    for (;;) {
        uint32_t bo = 0, bv = 0;
        for (uint32_t o = 0; o <= QDR_OPCODES; o++)
            for (uint32_t w = 0; w <= QDR_VARIANTS; w++)
                if (calls[o][w] && cost[o][w] > cost[bo][bv]) { bo = o; bv = w; }
        if (!calls[bo][bv] || cost[bo][bv] < total * 0.001) break;
        printf("  %5.1f%%  %7.3f  %8lu  %2lu %3lu\n", 100.0 * cost[bo][bv] / total, cost[bo][bv],
               (unsigned long)calls[bo][bv], (unsigned long)bo, (unsigned long)bv);
        calls[bo][bv] = 0;
    }
    env_close(&v);
    free(constants);
    for (int j = 0; j < 4; j++) free(inputs[j]);
}

static void run_segment(const char *path, uint32_t progress) {
    uint32_t fn;
    uint8_t *fx = read_file(path, &fn);
    CHECK(fn >= RSEG_HEADER && !memcmp(fx, "RSEG", 4) && rd32(fx + 4) == 1u && rd32(fx + 8) <= 1u);
    const char *model = rd32(fx + 8) ? "alternate" : "primary";
    uint32_t first = rd32(fx + 12), count = rd32(fx + 16), input_bytes[4], pool_bytes[7];
    uint8_t *inputs[4];
    uint64_t at = RSEG_HEADER;
    for (int j = 0; j < 4; j++) { input_bytes[j] = rd32(fx + 20 + 4 * j); inputs[j] = fx + at; at += input_bytes[j]; }
    for (int j = 0; j < 2; j++) CHECK(rd32(fx + 64 + 4 * j) == kOutputBytes[j]);
    const uint8_t *state[2][9];
    for (int s = 0; s < 2; s++) {
        for (int i = 0; i < 7; i++) { pool_bytes[i] = rd32(fx + 36 + 4 * i); state[s][i] = fx + at; at += pool_bytes[i]; }
        for (int j = 0; j < 2; j++) { state[s][7 + j] = fx + at; at += kOutputBytes[j]; }
    }
    CHECK(at == fn);
    static Env v;
    env_open(&v, model, inputs, input_bytes);
    v.progress = progress;
    CHECK(count > 0 && first <= v.p.command_count && count <= v.p.command_count - first);
    for (int i = 0; i < 7; i++) {
        CHECK(v.pools[i].bytes == pool_bytes[i]);
        if (pool_bytes[i]) memcpy(v.pools[i].base, state[0][i], pool_bytes[i]);
    }
    for (int j = 0; j < 2; j++) memcpy(v.out[j], state[0][7 + j], kOutputBytes[j]);
    env_census(&v, first, count);
    env_run(&v, first, count, state[0]);
    if (!env_pools_match(&v, state[1])) exit(1);
    for (int j = 0; j < 2; j++) {
        if (memcmp(v.out[j], state[1][7 + j], kOutputBytes[j])) { fprintf(stderr, "%s output %d mismatch\n", model, j); exit(1); }
    }
    printf("PASS %s reference segment: commands %lu..%lu, one preflight without pool or output writes, %s, "
           "all seven pools and both graph outputs identical to the host end state\n", model, (unsigned long)first,
           (unsigned long)(first + count - 1u), progress ? "runs in logged chunks" : "one run");
    fflush(stdout);
    env_close(&v);
    free(fx);
}

int main(int argc, char **argv) {
    int kept = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "fast")) use_fast = 1;
        else if (!strcmp(argv[i], "cycles")) use_cycles = 1;
        else if (!strcmp(argv[i], "ew")) use_ew = 1;
        else if (!strcmp(argv[i], "fuse")) use_fuse = 1;
        else if (!strcmp(argv[i], "hmx")) {
#ifdef __hexagon__
            use_fast = 1;
            hmx_map_vtcm();
#else
            use_fast = 1; /* host (split): HMX only exists in the simulator */
#endif
        }
        else if (!strcmp(argv[i], "par") || !strcmp(argv[i], "parrev") || !strcmp(argv[i], "parshuf") || !strcmp(argv[i], "thr")) {
            use_par = 1;
            par_hook.max_parts = 3;
            par_hook.order = !strcmp(argv[i], "parrev") ? 1u : !strcmp(argv[i], "parshuf") ? 2u : 0u;
            par_hook.seed = 99;
#ifdef __hexagon__
            if (!strcmp(argv[i], "thr")) sj_start();
#else
            if (!strcmp(argv[i], "thr")) { fprintf(stderr, "thr needs the simulator\n"); return 2; }
#endif
        }
        else argv[kept++] = argv[i];
    }
    argc = kept;
    if (argc > 2 && !strcmp(argv[1], "segment")) {
        run_segment(argv[2], argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 0);
        return 0;
    }
    const char *which = argc > 1 ? argv[1] : "both";
    const char *mode = argc > 2 ? argv[2] : "prefix";
    CHECK(!strcmp(which, "both") || !strcmp(which, "primary") || !strcmp(which, "alternate"));
    if (!strcmp(mode, "profile") || !strcmp(mode, "split") || !strcmp(mode, "tail")) use_cycles = 0;
    if (!strcmp(mode, "profile")) {
        if (strcmp(which, "alternate")) profile("primary");
        if (strcmp(which, "primary")) profile("alternate");
        return 0;
    }
    if (!strcmp(mode, "split")) {
        CHECK(argc > 4 && strcmp(which, "both"));
        split(which, (uint32_t)strtoul(argv[3], NULL, 10), argv[4]);
        return 0;
    }
    if (!strcmp(mode, "tail")) {
        CHECK(argc > 4 && strcmp(which, "both"));
        tail(which, (uint32_t)strtoul(argv[3], NULL, 10), argv[4], argc > 5 ? (uint32_t)strtoul(argv[5], NULL, 10) : 0);
        return 0;
    }
    CHECK(!strcmp(mode, "prefix") || !strcmp(mode, "full"));
    void (*run)(const char *) = strcmp(mode, "full") ? run_prefix : run_full;
    if (strcmp(which, "alternate")) run("primary");
    if (strcmp(which, "primary")) run("alternate");
    return 0;
}
