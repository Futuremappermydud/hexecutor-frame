/* Host lockstep test: the prepared executor against qd_graph's CPU backend over
 * every command of an actual model schedule.
 *
 *   test_prepared_full_model <primary|alternate> [reference|mixed|fast|mixedfast] [lockstep|whole|both]
 *
 * reference registers prepared_reference for every key. mixed registers the
 * host-runnable specialized handlers first, chains the reference kernel under
 * them, and fills the remaining keys. The fast modes first register the fast
 * exact convolution with a VTCM-tail-sized workspace. lockstep runs one command at a time and
 * compares all seven pools after each command; whole preflights and runs the
 * complete schedule as one request. Both modes check the graph outputs against
 * the recovered emulator smoke hashes. Run from the depth-model directory. */
#include "prepared_reference.h"
#include "prepared_abs.h"
#include "prepared_addsub_flat.h"
#include "prepared_block_zap.h"
#include "prepared_flat_vtcm.h"
#include "prepared_io.h"
#include "prepared_mean.h"
#include "prepared_mul_flat.h"
#include "prepared_quantize.h"
#include "prepared_reshape_cb.h"
#include "prepared_sign.h"
#include "prepared_spill_fill.h"
#include "prepared_fuse.h"
#include "prepared_dma_ring.h"
#include "prepared_executor.h"
#include "../qd_graph.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

typedef struct { uint64_t seq;void *dst;const void *src;size_t bytes;uint8_t queued; } MockDmaDesc;
typedef struct { MockDmaDesc d[NPRG2_DMA_RING_SLOTS];uint64_t clock;uint8_t active;uint32_t issued,completed; } MockDma;
static int mock_dma_begin(void *ctx){MockDma *m=ctx;if(m->active)return 0;m->active=1;return 1;}
static int mock_dma_issue(void *ctx,uint64_t seq,uint32_t slot,void *dst,const void *src,size_t bytes){
    MockDma *m=ctx;if(!m->active||slot>=NPRG2_DMA_RING_SLOTS||!bytes)return -1;
    if((((uintptr_t)dst|(uintptr_t)src|bytes)&127u)!=0)return 0;
    MockDmaDesc *d=&m->d[slot];if(d->queued)return -1;
    *d=(MockDmaDesc){seq,dst,src,bytes,1};m->issued++;return 1;
}
static int mock_dma_done(void *ctx,uint64_t seq,uint32_t slot){
    MockDma *m=ctx;if(!m->active||slot>=NPRG2_DMA_RING_SLOTS)return -1;
    MockDmaDesc *d=&m->d[slot];if(!d->queued||d->seq!=seq)return -1;
    memmove(d->dst,d->src,d->bytes);d->queued=0;m->completed++;return 1;
}
static int mock_dma_finish(void *ctx,int had_error){
    MockDma *m=ctx;
    if(had_error){for(uint32_t i=0;i<NPRG2_DMA_RING_SLOTS;i++)m->d[i].queued=0;m->active=0;return -1;}
    for(uint32_t i=0;i<NPRG2_DMA_RING_SLOTS;i++)if(m->d[i].queued)return 0;
    m->active=0;return 1;
}
static uint64_t mock_dma_clock(void *ctx){return ((MockDma *)ctx)->clock;}
static void mock_dma_relax(void *ctx){((MockDma *)ctx)->clock++;}
static const Nprg2DmaRingOps mock_dma_ops={mock_dma_begin,mock_dma_issue,mock_dma_done,
    mock_dma_finish,mock_dma_clock,mock_dma_relax};
static void mock_dma_init(Nprg2WeightsAsync *a,Nprg2DmaRing *r,MockDma *m,uint32_t commands){
    memset(a,0,sizeof *a);memset(m,0,sizeof *m);
    CHECK(nprg2_dma_ring_init(r,&mock_dma_ops,m,100000000u));
    a->command_count=commands;a->ring=r;a->ring_mode=1;a->timeout_cycles=100000000u;
}

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

typedef struct {
    const char *model;
    uint8_t *qdrx, *nprg, *constants;
    uint32_t qdrx_bytes, nprg_bytes, constants_bytes;
    uint8_t *inputs[4];
    uint32_t input_bytes[4];
    Nprg2Pool pools[7];
    uint8_t *initial[7];
    Nprg2Program program;
    Nprg2Bound bound;
    uint8_t *hot;
    Nprg2Executor executor;
    Nprg2ReferenceChain chains[64];
    uint8_t *scratch;
    uint32_t scratch_bytes;
    uint32_t *need;
    int reference_only;
    Nprg2ReferenceConv conv;
    Nprg2FusePlan plan;
    int fuse;
    uint8_t mask[4096];     /* pool-1 blocks a fused dead conv deliberately left stale */
} Session;

/* The Frame VTCM tail less its 8 KiB control area, and the Frame's L2-TCM
 * near slot for the fast convolution. */
enum { CONV_WORKSPACE = 0x7e000, CONV_NEAR = 0x8000 };

/* QD_PAR=0|1|2: the parallel hook with 3 parts and no worker, so every
 * parallel job runs serially in ascending (0), descending (1) or shuffled (2)
 * order; helper conv slots of QD_PAR_SLOT bytes (default 0x3e00, the Frame's).
 * The pools must come out identical whatever the part order. */
static Nprg2Parallel g_par;
typedef struct { Nprg2ParallelFn fn; void *ctx; uint32_t part, parts; } PtJob;
static void *pt_main(void *a) { PtJob *j = a; j->fn(j->ctx, j->part, j->parts); return NULL; }
/* QD_PAR=3: real concurrent threads (parts 1.. on new pthreads). */
static int pt_run(void *impl, uint32_t parts, Nprg2ParallelFn fn, void *ctx) {
    (void)impl;
    pthread_t th[8]; PtJob job[8];
    for (uint32_t i = 1; i < parts; i++) {
        job[i] = (PtJob){fn, ctx, i, parts};
        CHECK(!pthread_create(&th[i], NULL, pt_main, &job[i]));
    }
    fn(ctx, 0, parts);
    for (uint32_t i = 1; i < parts; i++) pthread_join(th[i], NULL);
    return 1;
}
static Nprg2Parallel *par_hook(void) {
    const char *e = getenv("QD_PAR");
    if (!e) return NULL;
    g_par.max_parts = 3;
    g_par.order = (uint32_t)atoi(e);
    if (g_par.order == 3) g_par.run = pt_run;
    g_par.seed = 12345;
    return &g_par;
}

static const char *const kInputNames[4] = {"x", "y", "hidden", "t"};

static void pools_reset(Session *s) {
    for (int i = 0; i < 7; i++) if (s->pools[i].bytes) memcpy(s->pools[i].base, s->initial[i], s->pools[i].bytes);
}

static void open_session(Session *s, const char *model) {
    char path[256];
    memset(s, 0, sizeof *s);
    s->model = model;
    snprintf(path, sizeof path, "hexecutor/native/build/resident-%s.qdrx", model);
    s->qdrx = read_file(path, &s->qdrx_bytes);
    snprintf(path, sizeof path, "depth-model/deployment/programs/%s-v2.nprg", model);
    s->nprg = read_file(path, &s->nprg_bytes);
    snprintf(path, sizeof path, "recovered/%s/constants/constant_pool_5.bin", model);
    s->constants = read_file(path, &s->constants_bytes);
    for (int i = 0; i < 4; i++) {
        /* QD_INPUTS=dir: another input case (x/y/hidden/t.raw), e.g. the random case. */
        snprintf(path, sizeof path, "%s/%s.raw", getenv("QD_INPUTS") ? getenv("QD_INPUTS") : "hexecutor/native/build/native-smoke",
                 kInputNames[i]);
        s->inputs[i] = read_file(path, &s->input_bytes[i]);
    }
    CHECK(s->input_bytes[0] == 81920 && s->input_bytes[1] == 81920 && s->input_bytes[2] == 655360 && s->input_bytes[3] == 128);
    for (uint32_t i = 0; i < 7; i++) {
        s->pools[i].bytes = rd32(s->nprg + 144 + i * 4);
        if (!s->pools[i].bytes) continue;
        s->pools[i].base = aligned_alloc(2048, (s->pools[i].bytes + 2047u) & ~2047u);
        s->initial[i] = calloc(1, s->pools[i].bytes);
        CHECK(s->pools[i].base && s->initial[i]);
    }
    CHECK(s->pools[5].bytes == s->constants_bytes);
    memcpy(s->initial[5], s->constants, s->constants_bytes);
    pools_reset(s);

    uint8_t qsha[32], nsha[32];
    nprg2_sha256(s->qdrx, s->qdrx_bytes, qsha);
    nprg2_sha256(s->nprg, s->nprg_bytes, nsha);
    Nprg2ExpectedHashes exp = {s->qdrx + 32, s->qdrx + 64, qsha, nsha};
    CHECK(nprg2_open(&s->program, s->nprg, s->nprg_bytes, &exp, s->pools, (Nprg2Allocator){0}) == NPRG2_OK);
    s->hot = calloc(1, 300000);
    CHECK(s->hot);
    Nprg2BindConfig bc = {.hot_memory = s->hot, .hot_capacity = 300000, .l2_available_bytes = 786432,
                          .pool1_is_verified_vtcm = 1, .resolve = resolve_all};
    CHECK(nprg2_bind_mock(&s->program, s->pools, &bc, &s->bound) == NPRG2_OK);
}

static void register_handlers(Session *s, int mixed, int fast) {
    Nprg2Executor *e = &s->executor;
    nprg2_executor_init(e);
    if (fast) {
        s->conv.workspace = aligned_alloc(2048, CONV_WORKSPACE);
        s->conv.workspace_bytes = CONV_WORKSPACE;
        s->conv.near = aligned_alloc(128, CONV_NEAR);
        s->conv.near_bytes = CONV_NEAR;
        CHECK(s->conv.workspace && s->conv.near);
        if (getenv("QD_PAR")) {
            const char *sb = getenv("QD_PAR_SLOT");
            s->conv.par_near_bytes = sb ? (uint32_t)atoi(sb) : 0x3e00;
            for (int k = 0; k < 2; k++) s->conv.par_near[k] = aligned_alloc(128, 0x10000);
        }
        /* QD_HMX=1: eligible u8 convs take the HMX tiling with its exact scalar model. */
        static const Nprg2HmxOps host_hmx = {0};
        if (getenv("QD_HMX") && atoi(getenv("QD_HMX"))) s->conv.hmx = &host_hmx;
        uint32_t added = 0;
        CHECK(nprg2_executor_register_reference_conv(e, &s->conv, &added) == NPRG2_EXEC_OK);
        printf("%s: %lu conv keys registered to the fast exact convolution\n", s->model, (unsigned long)added);
    }
    if (mixed) {
        CHECK(nprg2_executor_register_weights(e) == NPRG2_EXEC_OK);
        CHECK(nprg2_executor_register_core_safe_ops(e) == NPRG2_EXEC_OK);
        CHECK(nprg2_executor_register_block_zap(e) == NPRG2_EXEC_OK);
        CHECK(nprg2_executor_register_spill_fill(e) == NPRG2_EXEC_OK);
        CHECK(nprg2_executor_register_flat_vtcm(e) == NPRG2_EXEC_OK);
        CHECK(nprg2_executor_register_abs(e) == NPRG2_EXEC_OK);
        CHECK(nprg2_executor_register_mean(e) == NPRG2_EXEC_OK);
        CHECK(nprg2_executor_register_sign(e) == NPRG2_EXEC_OK);
        CHECK(nprg2_executor_register_addsub_flat(e) == NPRG2_EXEC_OK);
        CHECK(nprg2_executor_register_mul_flat(e) == NPRG2_EXEC_OK);
        CHECK(nprg2_executor_register_reshape_cb(e) == NPRG2_EXEC_OK);
        CHECK(nprg2_executor_register_quantize(e) == NPRG2_EXEC_OK);
        uint32_t chained = 0;
        for (uint32_t i = 0; i < NPRG2_EXEC_REGISTRY_CAPACITY; i++) {
            Nprg2ExecHandler *h = &e->handlers[i];
            if (!h->used || h->opcode == QDR_NOP || h->validate == nprg2_exec_reference_validate) continue;
            if (h->variant > QDR_VARIANTS || qdr_variant_ops[h->variant] != h->opcode) continue;
            CHECK(chained < sizeof s->chains / sizeof s->chains[0]);
            CHECK(nprg2_executor_chain_reference(e, h->opcode, h->variant, &s->chains[chained]) == NPRG2_EXEC_OK);
            chained++;
        }
        printf("%s: %lu specialized keys chained over the reference kernel\n", s->model, (unsigned long)chained);
    }
    uint32_t added = 0;
    CHECK(nprg2_executor_register_reference_missing(e, &added) == NPRG2_EXEC_OK);
    printf("%s: %lu keys registered to the reference kernel, %lu total\n", s->model, (unsigned long)added, (unsigned long)e->count);
}

static Nprg2ExecRequest request(Session *s, Nprg2ExecBuffer *in, Nprg2ExecBuffer *out, uint32_t first, uint32_t count,
                                Nprg2ExecStats *stats) {
    Nprg2ExecRequest r = {.fuse = s->fuse ? &s->plan : NULL, .program = &s->program, .bound = &s->bound, .pool1_writable_bytes = s->pools[1].bytes,
                          .inputs = in, .input_count = 4, .outputs = out, .output_count = 2,
                          .scratch = s->scratch, .scratch_bytes = s->scratch_bytes,
                          .command_first = first, .command_count = count, .stats = stats, .parallel = par_hook()};
    memcpy(r.pools, s->pools, sizeof s->pools);
    /* QD_NEAR=1: a 32 KiB near scratch region (the Frame's L2-TCM slot); commands whose scratch need fits use it. */
    if (getenv("QD_NEAR")) {
        static uint8_t *near_region;
        if (!near_region) near_region = aligned_alloc(4096, 0x8000);
        r.extra_scratch[0] = (Nprg2ExecBuffer){near_region, 0x8000};
    }
    return r;
}

/* Every command must be covered by a validator; reports the reference
 * kernel's exact scratch peak and the commands writing the VTCM tail. */
static void census(Session *s) {
    uint32_t peak = 0, peak_cmd = 0, peak_op = 0, peak_var = 0, measured = 0, tail = 0, over_ddr = 0;
    uint32_t convs = 0, fast = 0, near = 0, ws_peak = 0;
    uint32_t histogram[QDR_OPCODES + 1] = {0};
    static uint8_t out0[163840], out1[655360];
    s->need = calloc(s->program.command_count, sizeof *s->need);
    CHECK(s->need);
    Nprg2ExecBuffer in[4], ob[2] = {{out0, sizeof out0}, {out1, sizeof out1}};
    for (int k = 0; k < 4; k++) in[k] = (Nprg2ExecBuffer){s->inputs[k], s->input_bytes[k]};
    for (uint32_t i = 0; i < s->program.command_count; i++) {
        Nprg2ExecCall call = {.command_index = i, .program = &s->program, .bound = &s->bound, .pools = s->pools,
                              .pool1_writable_bytes = s->pools[1].bytes,
                              .inputs = in, .input_count = 4, .outputs = ob, .output_count = 2};
        CHECK(nprg2_command_at(&s->bound, i, &call.command) == NPRG2_OK);
        uint32_t need = 0, ws = 0;
        if (!nprg2_exec_reference_conv_measure(&call, s->conv.workspace ? &s->conv : NULL, &need, &ws)) {
            fprintf(stderr, "%s census reject cmd=%lu op=%u var=%u line=%lu\n", s->model, (unsigned long)i,
                    call.command.opcode, call.command.variant, (unsigned long)nprg2_exec_reference_error_line());
            histogram[call.command.opcode]++;
            continue;
        }
        measured++;
        s->need[i] = need;
        if (call.command.opcode == QDR_HMX_CONV) {
            convs++;
            if (ws) fast++;
            if (nprg2_exec_reference_conv_path() == 2) near++;
            if (ws > ws_peak) ws_peak = ws;
        }
        if (need > 0x1000000u) {
            over_ddr++;
            if (over_ddr <= 8)
                fprintf(stderr, "%s census: cmd=%lu op=%u var=%u needs %lu scratch bytes\n", s->model, (unsigned long)i,
                        call.command.opcode, call.command.variant, (unsigned long)need);
        }
        if (need > peak) { peak = need; peak_cmd = i; peak_op = call.command.opcode; peak_var = call.command.variant; }
        call.pool1_writable_bytes = 0x780000u;
        if (!nprg2_exec_reference_scratch(&call, &need)) tail++;
    }
    uint32_t rejected = s->program.command_count - measured;
    printf("%s census: %lu/%lu commands accepted by the reference validator; scratch peak %lu bytes at cmd %lu "
           "(op %lu var %lu); %lu commands need more than 16 MiB scratch; "
           "%lu commands write above the 0x780000 pool-1 ceiling\n",
           s->model, (unsigned long)measured, (unsigned long)s->program.command_count, (unsigned long)peak,
           (unsigned long)peak_cmd, (unsigned long)peak_op, (unsigned long)peak_var, (unsigned long)over_ddr,
           (unsigned long)tail);
    if (s->conv.workspace)
        printf("%s census: %lu/%lu convolutions on the fast path (%lu with near memory), workspace peak %lu of %lu bytes\n",
               s->model, (unsigned long)fast, (unsigned long)convs, (unsigned long)near, (unsigned long)ws_peak,
               (unsigned long)s->conv.workspace_bytes);
    for (uint32_t op = 0; op <= QDR_OPCODES; op++)
        if (histogram[op]) fprintf(stderr, "  opcode %lu: %lu rejected\n", (unsigned long)op, (unsigned long)histogram[op]);
    CHECK(rejected == 0);
    s->scratch_bytes = (peak + 4095u) & ~4095u;
    s->scratch = aligned_alloc(128, s->scratch_bytes);
    CHECK(s->scratch);
}

static int same_pools(Session *s, uint8_t *const *ref, uint32_t cmd) {
    for (int p = 0; p < 7; p++) {
        if (!s->pools[p].bytes) continue;
        uint32_t diff = 0;
        if (p == 1 && s->fuse) {
            /* Dead conv outputs stay stale until rewritten: allowed in masked 2 KiB blocks only. */
            int bad = 0;
            for (uint32_t b = 0; b * 2048u < s->pools[1].bytes; b++) {
                uint32_t n = s->pools[1].bytes - b * 2048u < 2048u ? s->pools[1].bytes - b * 2048u : 2048u;
                if (!memcmp(s->pools[1].base + b * 2048u, ref[1] + b * 2048u, n)) { s->mask[b] = 0; continue; }
                if (s->mask[b]) continue;
                diff = b * 2048u;
                while (s->pools[1].base[diff] == ref[1][diff]) diff++;
                bad = 1;
                break;
            }
            if (!bad) continue;
        } else {
            if (!memcmp(s->pools[p].base, ref[p], s->pools[p].bytes)) continue;
            while (s->pools[p].base[diff] == ref[p][diff]) diff++;
        }
        Nprg2Command c;
        nprg2_command_at(&s->bound, cmd, &c);
        fprintf(stderr, "%s pool mismatch after cmd=%lu op=%u var=%u seq=%u: pool=%d off=0x%lx prepared=%u cpu=%u\n",
                s->model, (unsigned long)cmd, c.opcode, c.variant, c.sequence, p, (unsigned long)diff,
                s->pools[p].base[diff], ref[p][diff]);
        return 0;
    }
    return 1;
}

/* Marks the pool-1 blocks of command cmd's first output (a u8 CH tensor). */
static void mask_conv_out(Session *s, uint32_t cmd) {
    Nprg2Command c;
    CHECK(nprg2_command_at(&s->bound, cmd, &c) == NPRG2_OK && c.output_count == 1);
    uint32_t ix = s->bound.refs.data[c.output_first * 2u] | (uint32_t)s->bound.refs.data[c.output_first * 2u + 1u] << 8;
    Nprg2TensorView v;
    CHECK(nprg2_bound_tensor_at(&s->program, &s->bound, ix, &v) == NPRG2_OK && v.storage_kind == 2);
    const uint8_t *g = v.geometry_record;
    int m1 = (int16_t)(g[16] | g[17] << 8), m2 = (int16_t)(g[18] | g[19] << 8), m3 = (int16_t)(g[20] | g[21] << 8);
    uint64_t n = (uint64_t)(m1 / 8) * (m2 / 8) * (m3 / 32);
    for (uint64_t i = 0; i < n; i++) {
        uint32_t e = v.blocks[v.slice_origin + i];
        if ((e & 127u) == 1u) s->mask[(e & ~127u) / 2048u] = 1;
    }
}

static void check_outputs(Session *s, uint8_t *const *out, const char *what) {
    static const char *expect[2][2] = {
        {"dfacee192b7a534ef79634993d03a1c870e1550e5ba1c485af5ab17466616ad3",
         "a978ee8291ea05c0b482b72cdc54f09c1ad8294a366f3ba215bb32653cb3f32b"},
        {"dfacee192b7a534ef79634993d03a1c870e1550e5ba1c485af5ab17466616ad3",
         "98d28b66c3489d1424569aad3e1c2b9bfa12fac3aba5b11e67b0fde10d8bc36a"}};
    static const uint32_t bytes[2] = {163840, 655360};
    int alt = !strcmp(s->model, "alternate");
    /* QD_EXPECT="hex0,hex1" overrides the emulator hashes (the silicon-rounding
     * oracle from a reference-mode run); QD_EXPECT=print prints them instead. */
    const char *env = getenv("QD_EXPECT");
    char ov[2][65] = {{0}};
    int print = env && !strcmp(env, "print");
    if (env && !print) {
        CHECK(strlen(env) == 129 && env[64] == ',');
        memcpy(ov[0], env, 64); memcpy(ov[1], env + 65, 64);
    }
    if (getenv("QD_DUMP")) for (int k = 0; k < 2; k++) {
        char dp[512]; snprintf(dp, sizeof dp, "%s/out%d.raw", getenv("QD_DUMP"), k);
        FILE *df = fopen(dp, "wb"); CHECK(df && fwrite(out[k], 1, bytes[k], df) == bytes[k]); fclose(df);
    }
    for (int k = 0; k < 2; k++) {
        uint8_t sha[32];
        char hex[65];
        nprg2_sha256(out[k], bytes[k], sha);
        for (int j = 0; j < 32; j++) snprintf(hex + 2 * j, 3, "%02x", sha[j]);
        const char *want = env && !print ? ov[k] : expect[alt][k];
        if (print) { printf("%s %s output %d sha256 %s\n", s->model, what, k, hex); continue; }
        if (strcmp(hex, want)) {
            fprintf(stderr, "%s %s output %d sha256 %s, expected %s\n", s->model, what, k, hex, want);
            exit(1);
        }
    }
    if (g_par.max_parts)
        printf("parallel hook order=%u: jobs parallel %lu/%lu/%lu serial conv/ew/dm %lu/%lu/%lu parts %lu/%lu/%lu\n", g_par.order,
               (unsigned long)g_par.jobs_parallel[0], (unsigned long)g_par.jobs_parallel[1], (unsigned long)g_par.jobs_parallel[2],
               (unsigned long)g_par.jobs_serial[0], (unsigned long)g_par.jobs_serial[1], (unsigned long)g_par.jobs_serial[2],
               (unsigned long)g_par.parts_run[0], (unsigned long)g_par.parts_run[1], (unsigned long)g_par.parts_run[2]);
    printf("PASS %s %s: both graph outputs match the %s smoke sha256\n", s->model, what, env ? "expected" : "emulator");
}

static void lockstep(Session *s) {
    uint8_t *ref[7] = {0};
    uint64_t ref_bytes[7] = {0};
    for (int p = 0; p < 7; p++) {
        if (!s->pools[p].bytes) continue;
        ref[p] = aligned_alloc(2048, (s->pools[p].bytes + 2047u) & ~2047u);
        CHECK(ref[p]);
        memcpy(ref[p], s->initial[p], s->pools[p].bytes);
        ref_bytes[p] = s->pools[p].bytes;
    }
    pools_reset(s);
    int error = 0;
    QdgGraph *g = qdg_create(s->qdrx, s->qdrx_bytes, ref, ref_bytes, &error);
    CHECK(g);
    qdg_set_cpu(g);
    qdg_set_hoist(g, 0);
    qdg_set_block_output(g, 0);
    uint8_t *ref_out[2] = {calloc(1, 163840), calloc(1, 655360)}, *out[2] = {calloc(1, 163840), calloc(1, 655360)};
    CHECK(ref_out[0] && ref_out[1] && out[0] && out[1]);
    const void *qin[4] = {s->inputs[0], s->inputs[1], s->inputs[2], s->inputs[3]};
    uint64_t qin_bytes[4] = {81920, 81920, 655360, 128}, qout_bytes[2] = {163840, 655360};
    void *qout[2] = {ref_out[0], ref_out[1]};
    Nprg2ExecBuffer in[4], ob[2] = {{out[0], 163840}, {out[1], 655360}};
    for (int k = 0; k < 4; k++) in[k] = (Nprg2ExecBuffer){s->inputs[k], s->input_bytes[k]};
    CHECK((uint32_t)qdg_commands(g) == s->program.command_count);
    uint32_t steps = 0, fused_steps = 0, fused_active = 0, fused_dead = 0;
    for (uint32_t i = 0, end; i < s->program.command_count; i = end + 1) {
        /* A step is one command, or the span of overlapping fusion pairs (compared after its last command). */
        end = i;
        if (s->fuse)
            for (uint32_t k = i; k <= end; k++) {
                uint16_t role = s->plan.role[k];
                if (role && !(role & NPRG2_FUSE_EPI) && s->plan.pairs[role - 1u].epi > end) end = s->plan.pairs[role - 1u].epi;
            }
        steps++;
        if (end > i) fused_steps++;
        int rc = qdg_run(g, qin, qin_bytes, qout, qout_bytes, i, end + 1);
        if (rc) { fprintf(stderr, "qd_graph failed cmd=%lu rc=%d line=%u\n", (unsigned long)i, rc, qdg_stats(g)->error_detail); exit(1); }
        Nprg2ExecStats st;
        Nprg2ExecRequest r = request(s, in, ob, i, end + 1 - i, &st);
        /* The run must fit in exactly the scratch its own dry pass measured. */
        if (s->reference_only) r.scratch_bytes = s->need[i];
        rc = nprg2_executor_run(&s->executor, &r);
        if (rc != NPRG2_EXEC_OK) {
            Nprg2Command c;
            nprg2_command_at(&s->bound, i, &c);
            fprintf(stderr, "%s prepared failed cmd=%lu op=%u var=%u rc=%d reference_line=%lu\n", s->model,
                    (unsigned long)i, c.opcode, c.variant, rc, (unsigned long)nprg2_exec_reference_error_line());
            exit(1);
        }
        if (s->fuse)
            for (uint32_t k = i; k <= end; k++) {
                uint16_t role = s->plan.role[k];
                if (role && !(role & NPRG2_FUSE_EPI) && s->plan.pairs[role - 1u].epi <= end && s->plan.pairs[role - 1u].active) {
                    fused_active++;
                    if (s->plan.pairs[role - 1u].dead) { fused_dead++; mask_conv_out(s, k); }
                }
            }
        if (!same_pools(s, ref, end)) exit(1);
        if (memcmp(out[0], ref_out[0], 163840) || memcmp(out[1], ref_out[1], 655360)) {
            fprintf(stderr, "%s graph output mismatch after cmd=%lu\n", s->model, (unsigned long)end);
            exit(1);
        }
    }
    printf("PASS %s lockstep: %lu commands, all seven pools identical to qd_graph CPU after every command\n",
           s->model, (unsigned long)s->program.command_count);
    if (s->fuse)
        printf("%s lockstep fusion: %lu steps (%lu spanning fused pairs), %lu pairs active (%lu dead, raw output not stored)\n",
               s->model, (unsigned long)steps, (unsigned long)fused_steps, (unsigned long)fused_active, (unsigned long)fused_dead);
    check_outputs(s, out, "lockstep");
    qdg_destroy(g);
    for (int p = 0; p < 7; p++) free(ref[p]);
    for (int k = 0; k < 2; k++) { free(ref_out[k]); free(out[k]); }
}

static void whole(Session *s) {
    pools_reset(s);
    uint8_t *out[2] = {calloc(1, 163840), calloc(1, 655360)};
    CHECK(out[0] && out[1]);
    Nprg2ExecBuffer in[4], ob[2] = {{out[0], 163840}, {out[1], 655360}};
    for (int k = 0; k < 4; k++) in[k] = (Nprg2ExecBuffer){s->inputs[k], s->input_bytes[k]};
    Nprg2ExecStats st;
    Nprg2ExecRequest r = request(s, in, ob, 0, s->program.command_count, &st);
    int rc = nprg2_executor_preflight(&s->executor, &r);
    if (rc != NPRG2_EXEC_OK) {
        fprintf(stderr, "%s whole preflight rc=%d cmd=%lu op=%u var=%u line=%lu\n", s->model, rc,
                (unsigned long)st.failed_command, st.failed_opcode, st.failed_variant,
                (unsigned long)nprg2_exec_reference_error_line());
        exit(1);
    }
    for (int p = 0; p < 7; p++) CHECK(!s->pools[p].bytes || !memcmp(s->pools[p].base, s->initial[p], s->pools[p].bytes));
    rc = nprg2_executor_run(&s->executor, &r);
    if (rc != NPRG2_EXEC_OK) {
        fprintf(stderr, "%s whole run rc=%d cmd=%lu op=%u var=%u line=%lu\n", s->model, rc,
                (unsigned long)st.failed_command, st.failed_opcode, st.failed_variant,
                (unsigned long)nprg2_exec_reference_error_line());
        exit(1);
    }
    CHECK(st.executed_commands == s->program.command_count && st.validated_commands == s->program.command_count);
    if (s->fuse) {
        uint32_t act = 0, dead = 0;
        for (uint32_t k = 0; k < s->plan.count; k++) { act += s->plan.pairs[k].active; dead += s->plan.pairs[k].active && s->plan.pairs[k].dead; }
        printf("%s whole fusion: %lu of %lu pairs active (%lu dead)\n", s->model, (unsigned long)act, (unsigned long)s->plan.count, (unsigned long)dead);
    }
    printf("PASS %s whole schedule: one preflight validated %lu commands without pool writes, one run executed all\n",
           s->model, (unsigned long)st.executed_commands);
    if (st.approx_cost_volume_commands || st.approx_cost_volume_interpolated_planes)
        printf("%s cost-volume step %d: skipped %u handlers (%u conv planes), interpolated %u planes\n",
               s->model, NPRG2_APPROX_COST_VOLUME_STEP, st.approx_cost_volume_commands, st.approx_cost_volume_conv_planes,
               st.approx_cost_volume_interpolated_planes);
    check_outputs(s, out, "whole");
    free(out[0]);
    free(out[1]);
}

/* QD_WRITESET=path: after each replayed command, diff the mutable pools against a
 * shadow and log the 2 KiB blocks it really changed plus a hash of the new bytes. */
static uint8_t *ws_shadow[7];
static FILE *ws_file;
static void ws_begin(Session *s) {
    const char *p = getenv("QD_WRITESET");
    if (!p) return;
    ws_file = fopen(p, "w"); CHECK(ws_file);
    for (int i = 0; i < 7; i++) if (s->pools[i].bytes) {
        if (!ws_shadow[i]) ws_shadow[i] = malloc(s->pools[i].bytes);
        memcpy(ws_shadow[i], s->pools[i].base, s->pools[i].bytes);
    }
}
static void ws_step(Session *s, uint32_t step, uint32_t ix) {
    if (!ws_file) return;
    uint64_t h = 14695981039346656037ull; int any = 0;
    uint32_t wg = getenv("QD_GRAN") ? atoi(getenv("QD_GRAN")) : 2048;
    fprintf(ws_file, "%u %u", step, ix);
    for (int i = 0; i < 7; i++) {
        if (!s->pools[i].bytes) continue;
        uint32_t n = s->pools[i].bytes; int64_t lastb = -1;
        for (uint32_t o = 0; o < n; o += 64) {
            uint32_t c = n - o < 64 ? n - o : 64;
            if (!memcmp(s->pools[i].base + o, ws_shadow[i] + o, c)) continue;
            for (uint32_t k = 0; k < c; k++) if (s->pools[i].base[o + k] != ws_shadow[i][o + k])
                h = (h ^ (uint64_t)(((uint64_t)i << 40) | ((uint64_t)(o + k) << 8) | s->pools[i].base[o + k])) * 1099511628211ull;
            if ((int64_t)(o / wg) != lastb) { fprintf(ws_file, " %d:%u", i, o / wg); lastb = o / wg; }
            memcpy(ws_shadow[i] + o, s->pools[i].base + o, c); any = 1;
        }
    }
    fprintf(ws_file, " #%016llx\n", (unsigned long long)(any ? h : 0));
}
static void ws_end(void) { if (ws_file) { fclose(ws_file); ws_file = NULL; } }

/* QD_READSET=spec: for each command, rerun it with every 2 KiB block outside its declared
 * footprint flipped; a changed result means an undeclared read. Bisects to the blocks.
 * spec lines: "cmd pool:block pool:block ..." (declared reads and writes). Result on stdout. */
static uint8_t *rs_spec[8192]; /* per command, per pool-block bitmap index */
static uint32_t rs_base[8], rs_total, rs_gran = 2048;
static uint8_t *rs_save[7], *rs_good[7];
static void rs_load(Session *s, const char *path) {
    rs_total = 0;
    if (getenv("QD_GRAN")) rs_gran = atoi(getenv("QD_GRAN"));
    for (int i = 0; i < 7; i++) { rs_base[i] = rs_total; rs_total += (s->pools[i].bytes + rs_gran - 1u) / rs_gran; }
    FILE *f = fopen(path, "r"); CHECK(f);
    static char line[1 << 20];
    while (fgets(line, sizeof line, f)) {
        char *p = line; uint32_t ix = strtoul(p, &p, 10);
        CHECK(ix < 8192); rs_spec[ix] = calloc(rs_total, 1);
        while (*p == ' ') {
            uint32_t pool = strtoul(p + 1, &p, 10); CHECK(*p == ':');
            uint32_t b = strtoul(p + 1, &p, 10);
            if (pool < 7 && rs_base[pool] + b < rs_total) rs_spec[ix][rs_base[pool] + b] = 1;
        }
    }
    fclose(f);
    for (int i = 0; i < 7; i++) if (s->pools[i].bytes) { rs_save[i] = malloc(s->pools[i].bytes); rs_good[i] = malloc(s->pools[i].bytes); }
}
static void rs_copy(Session *s, uint8_t **d, int to_pool) {
    for (int i = 0; i < 7; i++) if (s->pools[i].bytes && i != 5) {
        if (to_pool) memcpy(s->pools[i].base, d[i], s->pools[i].bytes); else memcpy(d[i], s->pools[i].base, s->pools[i].bytes);
    }
}
static void rs_flip(Session *s, uint32_t idx) {
    for (int i = 6; i >= 0; i--) if (s->pools[i].bytes && idx >= rs_base[i]) {
        uint32_t o = (idx - rs_base[i]) * rs_gran, n = s->pools[i].bytes - o < rs_gran ? s->pools[i].bytes - o : rs_gran;
        for (uint32_t k = 0; k < n; k++) s->pools[i].base[o + k] ^= 0x5a;
        return;
    }
}
/* Run with the listed blocks flipped; 1 if every block the good run wrote still matches. */
static int rs_try(Session *s, Nprg2ExecBuffer *in, Nprg2ExecBuffer *ob, uint32_t ix, const uint32_t *list, uint32_t n) {
    rs_copy(s, rs_save, 1);
    for (uint32_t i = 0; i < n; i++) rs_flip(s, list[i]);
    Nprg2ExecStats st; Nprg2ExecRequest r = request(s, in, ob, ix, 1, &st);
    if (s->reference_only) r.scratch_bytes = s->need[ix];
    if (nprg2_executor_run(&s->executor, &r) != NPRG2_EXEC_OK) return 0;
    const uint8_t *sp = rs_spec[ix];
    for (int i = 0; i < 7; i++) if (s->pools[i].bytes && i != 5) {
        for (uint32_t b = 0; b * rs_gran < s->pools[i].bytes; b++) {
            if (!sp[rs_base[i] + b]) continue;   /* flipped or untouched: only declared blocks are results */
            uint32_t o = b * rs_gran, m = s->pools[i].bytes - o < rs_gran ? s->pools[i].bytes - o : rs_gran;
            if (memcmp(s->pools[i].base + o, rs_good[i] + o, m)) return 0;
        }
    }
    return 1;
}
static void rs_step(Session *s, Nprg2ExecBuffer *in, Nprg2ExecBuffer *ob, uint32_t ix) {
    /* Called after the good run: rs_save holds the state before it. */
    rs_copy(s, rs_good, 0);
    uint32_t *list = malloc(sizeof(uint32_t) * rs_total), n = 0;
    for (uint32_t b = 0; b < rs_total; b++) if (!rs_spec[ix][b] && !(b >= rs_base[5] && b < rs_base[6])) list[n++] = b;
    if (!rs_try(s, in, ob, ix, list, n)) {
        printf("HIDDEN cmd=%u", ix);
        /* Bisect: repeatedly find the first single block that matters. */
        uint32_t lo = 0, hi = n, found = 0;
        while (lo < hi && found < 24) {
            uint32_t a = lo, b = hi;
            if (rs_try(s, in, ob, ix, list + a, b - a)) break;
            while (b - a > 1) { uint32_t m = a + (b - a) / 2; if (!rs_try(s, in, ob, ix, list + a, m - a)) b = m; else a = m; }
            for (int i = 6; i >= 0; i--) if (s->pools[i].bytes && list[a] >= rs_base[i]) { printf(" %d:%u", i, list[a] - rs_base[i]); break; }
            found++; lo = a + 1;
        }
        printf("\n");
    }
    free(list);
    rs_copy(s, rs_good, 1);
}

/* Host-only QNN run-list replay: events are generated from the metadata-only
 * x86 capture and carry command index, wait CP, done CP, and source list. */
static void runlists(Session *s, const char *order) {
    const char *path = getenv("QD_RUNLIST_EVENTS");
    CHECK(path && *path);
    uint32_t bytes = 0;
    uint8_t *raw = read_file(path, &bytes);
    CHECK(bytes >= 4);
    uint32_t cp_count = rd32(raw);
    CHECK(cp_count > 1 && cp_count <= 8192);
    uint64_t need_bytes = 4ull + 4ull * cp_count + 24ull * s->program.command_count;
    CHECK(need_bytes == bytes);
    uint32_t *cp = calloc(cp_count, sizeof *cp);
    CHECK(cp);
    for (uint32_t i = 0; i < cp_count; i++) cp[i] = rd32(raw + 4u + i * 4u);
    const uint8_t *events = raw + 4u + cp_count * 4u;
    Nprg2WeightsAsync dma;Nprg2DmaRing ring;MockDma mock;
    mock_dma_init(&dma,&ring,&mock,s->program.command_count);
    pools_reset(s);
    uint8_t *out[2] = {calloc(1, 163840), calloc(1, 655360)};
    CHECK(out[0] && out[1]);
    Nprg2ExecBuffer in[4], ob[2] = {{out[0], 163840}, {out[1], 655360}};
    for (int k = 0; k < 4; k++) in[k] = (Nprg2ExecBuffer){s->inputs[k], s->input_bytes[k]};
    ws_begin(s);
    for (uint32_t i = 0; i < s->program.command_count; i++) {
        const uint8_t *ev = events + i * 24u;
        uint32_t ix = rd32(ev), wait = rd32(ev + 4), done = rd32(ev + 8), lid = rd32(ev + 12);
        uint32_t dma_event_id=rd32(ev+16),dma_event_kind=rd32(ev+20);
        CHECK(ix < s->program.command_count && wait < cp_count && done < cp_count && lid < 2);
        if (cp[wait] != 0) {
            fprintf(stderr, "%s %s blocked at step=%lu command=%lu list=%lu wait=%lu value=%lu\n",
                    s->model, order, (unsigned long)i, (unsigned long)ix, (unsigned long)lid,
                    (unsigned long)wait, (unsigned long)cp[wait]);
            exit(1);
        }
        if(dma_event_kind==2u)CHECK(nprg2_exec_weights_dma_checkpoint_wait(&dma,dma_event_id)==NPRG2_EXEC_OK);
        Nprg2ExecStats st;
        if (getenv("QD_READSET")) { if (!rs_total) rs_load(s, getenv("QD_READSET")); rs_copy(s, rs_save, 0); }
        Nprg2ExecRequest r = request(s, in, ob, ix, 1, &st);
        r.weights_async=&dma;
        if (s->reference_only) r.scratch_bytes = s->need[ix];
        int rc = nprg2_executor_run(&s->executor, &r);
        if (rc != NPRG2_EXEC_OK || st.validated_commands != 1 || st.executed_commands != 1) {
            Nprg2Command c = {0}; nprg2_command_at(&s->bound, ix, &c);
            fprintf(stderr, "%s %s failed step=%lu command=%lu op=%u var=%u rc=%d line=%lu\n",
                    s->model, order, (unsigned long)i, (unsigned long)ix, c.opcode, c.variant, rc,
                    (unsigned long)nprg2_exec_reference_error_line());
            exit(1);
        }
        ws_step(s, i, ix);
        if (rs_total && rs_spec[ix]) rs_step(s, in, ob, ix);
        if(dma_event_kind==1u)CHECK(nprg2_exec_weights_dma_checkpoint_set(&dma,dma_event_id)==NPRG2_EXEC_OK);
        if (done) {
            CHECK(cp[done] != 0);
            cp[done]--;
        }
    }
    ws_end();
    CHECK(nprg2_exec_weights_dma_drain_all(&dma)==NPRG2_EXEC_OK);
    for (uint32_t i = 0; i < cp_count; i++) CHECK(cp[i] == 0);
    printf("PASS %s QNN %s run-list replay: %lu prepared commands, %lu CPU CPs and %lu DMA copies drained\n",
           s->model, order, (unsigned long)s->program.command_count, (unsigned long)cp_count,
           (unsigned long)mock.completed);
    check_outputs(s, out, order);
    free(out[0]); free(out[1]); free(cp); free(raw);
}

typedef struct { uint32_t command, wait_cp, done_cp, list, dma_event_id, dma_event_kind; } RunlistEvent;
typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t changed;
    uint32_t *cp;
    uint32_t cp_count;
    int failed;
    int preferred_list, other_started;
    char failure[192];
    Session *session;
    Nprg2ExecBuffer *inputs, *outputs;
    Nprg2ExecRequest lane_request[2];
    Nprg2Parallel *worker_parallel;
    Nprg2WeightsAsync *dma;
    uint8_t *scratch[2];
    uint8_t *near_scratch[2];
} RunlistShared;
typedef struct { RunlistShared *shared; RunlistEvent *events; uint32_t count, list; } RunlistThread;

static void runlist_fail(RunlistShared *sh, const char *msg) {
    pthread_mutex_lock(&sh->lock);
    if (!sh->failed) { sh->failed = 1; snprintf(sh->failure, sizeof sh->failure, "%s", msg); }
    pthread_cond_broadcast(&sh->changed);
    pthread_mutex_unlock(&sh->lock);
}
static void *runlist_thread(void *arg) {
    RunlistThread *t = (RunlistThread *)arg;
    RunlistShared *sh = t->shared;
    Session *s = sh->session;
    pthread_mutex_lock(&sh->lock);
    while (!sh->failed && (int)t->list != sh->preferred_list && !sh->other_started)
        pthread_cond_wait(&sh->changed, &sh->lock);
    pthread_mutex_unlock(&sh->lock);
    for (uint32_t i = 0; i < t->count; i++) {
        RunlistEvent *ev = &t->events[i];
        pthread_mutex_lock(&sh->lock);
        if ((int)t->list == sh->preferred_list && sh->cp[ev->wait_cp] != 0 && !sh->other_started) {
            sh->other_started = 1;
            pthread_cond_broadcast(&sh->changed);
        }
        while (!sh->failed && sh->cp[ev->wait_cp] != 0)
            pthread_cond_wait(&sh->changed, &sh->lock);
        int stop = sh->failed;
        pthread_mutex_unlock(&sh->lock);
        if (stop) return NULL;

        if (t->list==0u && ev->dma_event_kind==2u &&
            nprg2_exec_weights_dma_checkpoint_wait(sh->dma,ev->dma_event_id)!=NPRG2_EXEC_OK) {
            runlist_fail(sh,"DMA checkpoint wait failed");return NULL;
        }
        int rc = nprg2_executor_run_preflighted_command_prepared(&s->executor,
            &sh->lane_request[t->list],ev->command);
        if (rc != NPRG2_EXEC_OK) {
            char msg[192];
            Nprg2Command c={0};nprg2_command_at(&s->bound,ev->command,&c);
            snprintf(msg, sizeof msg, "list %lu command %lu op=%u var=%u failed rc=%d async=%lu fallback=%lu ringfail=%u issued=%lu completed=%lu line=%lu",
                     (unsigned long)t->list, (unsigned long)ev->command,c.opcode,c.variant,rc,
                     (unsigned long)sh->dma->starts,(unsigned long)sh->dma->cpu_fallbacks,
                     sh->dma->ring->failed,(unsigned long)sh->dma->ring->issued,(unsigned long)sh->dma->ring->completed,
                     (unsigned long)nprg2_exec_reference_error_line());
            runlist_fail(sh, msg);
            return NULL;
        }
        if(ev->dma_event_kind==1u &&
           nprg2_exec_weights_dma_checkpoint_set(sh->dma,ev->dma_event_id)!=NPRG2_EXEC_OK) {
            runlist_fail(sh,"DMA checkpoint set failed");return NULL;
        }
        pthread_mutex_lock(&sh->lock);
        if (ev->done_cp && !sh->cp[ev->done_cp]) {
            if (!sh->failed) {
                sh->failed = 1;
                snprintf(sh->failure, sizeof sh->failure, "list %lu command %lu underflowed CP %lu",
                         (unsigned long)t->list, (unsigned long)ev->command, (unsigned long)ev->done_cp);
            }
        } else if (ev->done_cp) {
            sh->cp[ev->done_cp]--;
        }
        pthread_cond_broadcast(&sh->changed);
        int failed = sh->failed;
        pthread_mutex_unlock(&sh->lock);
        if (failed) return NULL;
    }
    pthread_mutex_lock(&sh->lock);
    if ((int)t->list == sh->preferred_list && !sh->other_started) {
        sh->other_started = 1;
        pthread_cond_broadcast(&sh->changed);
    }
    pthread_mutex_unlock(&sh->lock);
    return NULL;
}

static void runlists_parallel(Session *s, int preferred_list) {
    CHECK(!s->fuse);
    const char *path = getenv("QD_RUNLIST_EVENTS");
    CHECK(path && *path);
    uint32_t bytes = 0;
    uint8_t *raw = read_file(path, &bytes);
    CHECK(bytes >= 4);
    uint32_t cp_count = rd32(raw);
    CHECK(cp_count > 1 && cp_count <= 8192);
    uint64_t need_bytes = 4ull + 4ull * cp_count + 24ull * s->program.command_count;
    CHECK(need_bytes == bytes);
    uint32_t *cp = calloc(cp_count, sizeof *cp);
    RunlistEvent *events[2] = {calloc(s->program.command_count, sizeof(RunlistEvent)),
                               calloc(s->program.command_count, sizeof(RunlistEvent))};
    CHECK(cp && events[0] && events[1]);
    for (uint32_t i = 0; i < cp_count; i++) cp[i] = rd32(raw + 4u + i * 4u);
    uint32_t counts[2] = {0, 0};
    const uint8_t *encoded = raw + 4u + cp_count * 4u;
    for (uint32_t i = 0; i < s->program.command_count; i++) {
        const uint8_t *p = encoded + i * 24u;
        uint32_t lid = rd32(p + 12);
        CHECK(lid < 2);
        events[lid][counts[lid]++] = (RunlistEvent){rd32(p), rd32(p + 4), rd32(p + 8), lid,
                                                    rd32(p + 16), rd32(p + 20)};
    }
    CHECK(counts[0] == 2668 && counts[1] == 2400);

    /* Complete the normal whole-program preflight before either list starts. */
    pools_reset(s);
    Nprg2ExecBuffer out_buf[2] = {{calloc(1, 163840), 163840}, {calloc(1, 655360), 655360}};
    CHECK(out_buf[0].data && out_buf[1].data);
    Nprg2ExecBuffer in_buf[4];
    for (int k = 0; k < 4; k++) in_buf[k] = (Nprg2ExecBuffer){s->inputs[k], s->input_bytes[k]};
    Nprg2ExecStats preflight_stats;
    Nprg2ExecRequest preflight = request(s, in_buf, out_buf, 0, s->program.command_count, &preflight_stats);
    Nprg2WeightsAsync dma;Nprg2DmaRing ring;MockDma mock;
    mock_dma_init(&dma,&ring,&mock,s->program.command_count);
    preflight.weights_async=&dma;
    preflight.parallel = NULL; preflight.fuse = NULL;
    preflight.extra_scratch[0] = (Nprg2ExecBuffer){NULL, 0};
    CHECK(nprg2_executor_preflight(&s->executor, &preflight) == NPRG2_EXEC_OK);
    CHECK(preflight_stats.validated_commands == s->program.command_count);
    uint32_t qweights=0;for(uint32_t wi=0;wi<s->program.command_count;wi++){Nprg2Command wc;if(!nprg2_command_at(&s->bound,wi,&wc)&&wc.opcode==QDR_WEIGHTS_TO_VTCM)qweights++;}
    fprintf(stderr,"host DMA mock observed %u weights commands\n",qweights);
    CHECK(nprg2_exec_reference_clone_lane(0, 1));
    for (int p = 0; p < 7; p++)
        CHECK(!s->pools[p].bytes || !memcmp(s->pools[p].base, s->initial[p], s->pools[p].bytes));

    RunlistShared sh;
    memset(&sh, 0, sizeof sh);
    CHECK(!pthread_mutex_init(&sh.lock, NULL)); CHECK(!pthread_cond_init(&sh.changed, NULL));
    sh.cp = cp; sh.cp_count = cp_count; sh.session = s; sh.inputs = in_buf; sh.outputs = out_buf;sh.dma=&dma;
    memset(&g_par,0,sizeof g_par);g_par.max_parts=2;g_par.run=pt_run;
    sh.worker_parallel=&g_par;
    sh.preferred_list = preferred_list;
    sh.scratch[0] = aligned_alloc(128, (s->scratch_bytes + 127u) & ~127u);
    sh.scratch[1] = aligned_alloc(128, (s->scratch_bytes + 127u) & ~127u);
    sh.near_scratch[0] = aligned_alloc(128, 0x8000u);
    sh.near_scratch[1] = aligned_alloc(128, 0x8000u);
    CHECK(sh.scratch[0] && sh.scratch[1] && sh.scratch[0] != sh.scratch[1] &&
          sh.near_scratch[0] && sh.near_scratch[1] && sh.near_scratch[0] != sh.near_scratch[1]);
    CHECK(nprg2_executor_prepare_preflighted_request(&s->executor,&preflight,0,NULL,
        sh.scratch[0],s->scratch_bytes,sh.near_scratch[0],0x8000u,&sh.lane_request[0])==NPRG2_EXEC_OK);
    CHECK(nprg2_executor_prepare_preflighted_request(&s->executor,&preflight,1,sh.worker_parallel,
        sh.scratch[1],s->scratch_bytes,sh.near_scratch[1],0x8000u,&sh.lane_request[1])==NPRG2_EXEC_OK);
    pthread_t th[2];
    RunlistThread tasks[2] = {{&sh, events[0], counts[0], 0}, {&sh, events[1], counts[1], 1}};
    CHECK(!pthread_create(&th[preferred_list], NULL, runlist_thread, &tasks[preferred_list]));
    CHECK(!pthread_create(&th[1u-(uint32_t)preferred_list], NULL, runlist_thread, &tasks[1u-(uint32_t)preferred_list]));
    CHECK(!pthread_join(th[0], NULL)); CHECK(!pthread_join(th[1], NULL));
    if(nprg2_exec_weights_dma_drain_all(&dma)!=NPRG2_EXEC_OK)runlist_fail(&sh,"DMA drain failed after both lists joined");
    if (sh.failed) { fprintf(stderr, "parallel QNN run-list failure: %s\n", sh.failure); exit(1); }
    for (uint32_t i = 0; i < cp_count; i++) CHECK(cp[i] == 0);
    printf("PASS %s concurrent QNN run lists (%s preferred): %lu main + %lu worker entries, %lu CPU CPs, %lu DMA starts/%lu fallback (%lu/%lu copies queued/completed)\n",
           s->model, preferred_list ? "worker" : "main", (unsigned long)counts[0],
           (unsigned long)counts[1], (unsigned long)cp_count,(unsigned long)dma.starts,(unsigned long)dma.cpu_fallbacks,
           (unsigned long)mock.issued,(unsigned long)mock.completed);
    check_outputs(s, (uint8_t *const[]){out_buf[0].data, out_buf[1].data}, "concurrent-runlists");
    pthread_cond_destroy(&sh.changed); pthread_mutex_destroy(&sh.lock);
    free(sh.scratch[0]); free(sh.scratch[1]); free(out_buf[0].data); free(out_buf[1].data);
    free(sh.near_scratch[0]); free(sh.near_scratch[1]);
    free(events[0]); free(events[1]); free(cp); free(raw);
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : "primary";
    const char *mode = argc > 2 ? argv[2] : "reference";
    const char *what = argc > 3 ? argv[3] : "both";
    CHECK(!strcmp(model, "primary") || !strcmp(model, "alternate"));
    CHECK(!strcmp(mode, "reference") || !strcmp(mode, "mixed") || !strcmp(mode, "fast") || !strcmp(mode, "mixedfast"));
    static Session s;
    open_session(&s, model);
    int mixed = !strncmp(mode, "mixed", 5), fast = !!strstr(mode, "fast");
    s.reference_only = !mixed;
    s.fuse = getenv("QD_FUSE") && atoi(getenv("QD_FUSE")) && fast;
    if (s.fuse) {
        int frc = nprg2_fuse_plan_build(&s.plan, &s.program, &s.bound, (Nprg2Allocator){0});
        CHECK(frc == NPRG2_FUSE_OK);
        printf("%s fusion plan: %lu pairs\n", model, (unsigned long)s.plan.count);
        CHECK(s.plan.count > 0);
    }
    register_handlers(&s, mixed, fast);
    census(&s);
    if (!strcmp(what, "runlist-main") || !strcmp(what, "runlist-worker")) {
        runlists(&s, !strcmp(what, "runlist-main") ? "main-first" : "worker-first");
        return 0;
    }
    if (!strcmp(what, "runlists-parallel-main")) { runlists_parallel(&s, 0); return 0; }
    if (!strcmp(what, "runlists-parallel-worker")) { runlists_parallel(&s, 1); return 0; }
    if (strcmp(what, "whole")) lockstep(&s);
    /* QD_REPEAT=n: n whole runs (pools reset each time), so state carried
     * between runs (memoised per-command decisions, caches) is exercised. */
    int repeat = getenv("QD_REPEAT") ? atoi(getenv("QD_REPEAT")) : 1;
    if (strcmp(what, "lockstep")) for (int k = 0; k < (repeat > 0 ? repeat : 1); k++) whole(&s);
    return 0;
}
