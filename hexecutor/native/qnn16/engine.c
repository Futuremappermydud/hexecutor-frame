/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Q16P engine: runs lowered QNN FP16 run lists (program.h) on the shared
 * guarded tensor bootstrap and transport. Worker stacks, mappings, the VTCM
 * snapshot and restoration remain bootstrap-owned.
 *
 * Every program uses QNN's own placements across the whole 8 MiB VTCM lease
 * and re-stages its weights each run, so runs are independent and graphs can
 * alternate freely. Operations run serially on hardware thread 0; the other
 * three workers only take part in the status/exit protocol. */
#include "program.h"
#include "../tensor/graph.h"
#include "../qd_conv.h"
static uint64_t cycles(void);
#include "../qd_hmx_power.h"
#include "../qd_owned_memory.h"
#define QDS_OP_RUN 3u
#include "../qd_protocol.h"
#include <stddef.h>
#include <string.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#define SERVICE ((volatile uint32_t *)0xf1800000u)
#define SHARED ((Shared *)0x300f0000u)
#define VTCM ((uint8_t *)0x31000000u)
/* Scratch lives in private DDR page 3 (L1 write-back; page 1, which holds
 * the host exchange, is uncached). The host keeps low constants below 56 MiB and
 * guards from 64 MiB - 4 KiB. HVX bypasses L1, so nothing vectorized may
 * touch this range: the engine builds with -fno-vectorize and memcpy/memset
 * take the scalar path here. */
#define SCRATCH_OFFSET 0x3800000u
#define SCRATCH_BYTES 0x7ff000u
#define SCRATCH ((uint8_t *)(QDO_BASE + SCRATCH_OFFSET))
static int in_scratch(const void *p, size_t n) {
    uintptr_t a = (uintptr_t)p, lo = (uintptr_t)SCRATCH;
    return a < lo + SCRATCH_BYTES && lo < a + n;
}
#define Q16S_RESULT_MAGIC 0x52363151u /* "Q16R" */
#define PARTS 4u
#define JOB_OP 0xfffffffdu
#define EXIT_OP 0xfffffffeu
#define STATUS_OP 0xffffffffu

static const char q16_engine_feature[] __attribute__((used, section(".rodata.hexecutor_features"))) =
    "HXQ16E01";
static const char q16_profile_feature[] __attribute__((used, section(".rodata.hexecutor_features"))) =
    "HXQ16PC1";
static const char q16_low_spill_feature[] __attribute__((used, section(".rodata.hexecutor_features"))) =
    "HXQ16SP1";
#if QDO_SLOTS > 8
#define Q16_PROFILE_CYCLES Q16P_WIDE_PROFILE_CYCLES
_Static_assert(Q16P_PROFILE_OFFSET + Q16_PROFILE_CYCLES * sizeof(uint64_t) <= QDO_STATE,
               "Q16P profiling must leave the JTLB restoration state intact");
#else
#define Q16_PROFILE_CYCLES Q16P_PROFILE_CYCLES
#endif
static const char q16_spill_feature[] __attribute__((used, section(".rodata.hexecutor_features"))) =
    "HXQ16SP3";
static const uint8_t q16_engine_window[] __attribute__((used, section(".rodata.hexecutor_features"))) =
    QDO_WINDOW_FEATURE;
static const char q16_engine_parts[] __attribute__((used, section(".rodata.hexecutor_features"))) =
    "HXTPART4";

/* The tensor bootstrap validates the HxSessionConfig at SERVICE + 4096 before
 * entry, so Q16P sessions use the same memory contract: per graph an image,
 * a constant pool and an arena holding the exchange area then scratch. */
typedef struct {
    volatile uint32_t ready, generation, done[4], op, sequence;
    Q16Task volatile task;
    void *volatile ctx;
} Shared;

static Q16Program programs[HXFG_MAX_GRAPHS];

static inline uint32_t part_of(uint32_t tid) { return tid == 5u ? 3u : tid; }
static void pause_wait(void) { __asm__ volatile("pause(#255)" ::: "memory"); }
#ifndef Q16_WORKER_PAUSE
#define Q16_WORKER_PAUSE 255
#endif
#define Q16_STR2(x) #x
#define Q16_STR(x) Q16_STR2(x)
/* Idle workers waiting for the next job. Shorter pauses and adaptive
 * spinning both measured slower (2026-11-09): idle polling takes issue slots
 * from the busy threads. */
static void worker_wait(void) { __asm__ volatile("pause(#" Q16_STR(Q16_WORKER_PAUSE) ")" ::: "memory"); }
static void barrier(void) { __asm__ volatile("barrier" ::: "memory"); }
static uint64_t cycles(void) {
    uint64_t c;
    __asm__ volatile("%0=upcycle" : "=r"(c));
    return c;
}
static uint64_t timer_ticks(void) {
    uint64_t ticks;
    __asm__ volatile("%0=c31:30" : "=r"(ticks));
    return ticks;
}
int *_Geterrno(void) {
    static int e;
    return &e;
}
void *memcpy(void *dst, const void *src, size_t n) {
    uint8_t *d = dst;
    const uint8_t *s = src;
    const int scalar = in_scratch(dst, n) || in_scratch(src, n);
    while (!scalar && n >= 128) {
        *(HVX_UVector *)d = *(const HVX_UVector *)s;
        d += 128;
        s += 128;
        n -= 128;
    }
    if (!(((uintptr_t)d | (uintptr_t)s) & 3u))
        while (n >= 4) {
            *(uint32_t *)d = *(const uint32_t *)s;
            d += 4;
            s += 4;
            n -= 4;
        }
    while (n--)
        *d++ = *s++;
    return dst;
}
void *memset(void *dst, int c, size_t n) {
    uint8_t *d = dst;
    HVX_Vector v = Q6_V_vsplat_R((uint32_t)(uint8_t)c * 0x01010101u);
    const int scalar = in_scratch(dst, n);
    while (!scalar && n >= 128) {
        *(HVX_UVector *)d = v;
        d += 128;
        n -= 128;
    }
    while (n--)
        *d++ = (uint8_t)c;
    return dst;
}
static void status(uint32_t tid, uint32_t seq) {
    volatile QdStatus *s = (volatile QdStatus *)((uint8_t *)SERVICE + 128) + tid;
    s->error = 0;
    s->rows = 0;
    s->reserved = 0;
    barrier();
    s->done_sequence = seq;
}
/* Publish an operation to workers 1..3, run part 0 of a job here, then
 * wait for all of them. */
static void broadcast(uint32_t op, uint32_t seq) {
    Shared *s = SHARED;
    s->op = op;
    s->sequence = seq;
    uint32_t gen = s->generation + 1;
    barrier();
    s->generation = gen;
    if (op == JOB_OP)
        s->task(s->ctx, 0, PARTS);
    for (uint32_t t = 1; t < PARTS; t++)
        while (s->done[t] != gen)
            pause_wait();
    barrier();
}
static void parallel(Q16Task task, void *ctx) {
    Shared *s = SHARED;
    s->task = task;
    s->ctx = ctx;
    broadcast(JOB_OP, s->sequence);
}

__attribute__((section(".text.resident_entry"))) void resident_entry(uint32_t tid) {
    Shared *s = SHARED;
    if (tid) {
        uint32_t seen = 0;
        const uint32_t part = part_of(tid);
        while (!s->ready)
            pause_wait();
        for (;;) {
            uint32_t gen;
while ((gen = s->generation) == seen)
                worker_wait();
            barrier();
            uint32_t op = s->op;
            if (op == JOB_OP)
                s->task(s->ctx, part, PARTS);
            else if (op == STATUS_OP)
                status(tid, s->sequence);
            barrier();
            s->done[part] = gen;
            seen = gen;
            if (op == EXIT_OP)
                return;
        }
    }
    const HxSessionConfig *cfg = (const HxSessionConfig *)((const uint8_t *)SERVICE + 4096);
    volatile uint32_t *result = (volatile uint32_t *)((uint8_t *)SERVICE + 2048);
    memset((void *)s, 0, sizeof *s);
    uint32_t init_error = hx_session_config_valid(cfg) ? Q16_OK : Q16_HEADER, bad = Q16P_NONE;
    for (uint32_t i = 0; !init_error && i < cfg->graphs; i++) {
        const HxGraphConfig *c = &cfg->graph[i];
        init_error = q16_program_open(&programs[i], (const void *)(QDO_BASE + c->image), c->image_bytes,
                                      c->constant_bytes, &bad);
        /* The bootstrap validates the generic regions. Keep scratch disjoint
         * and reject high constants outside this build's mapped window. */
        const uint64_t constant_end = (uint64_t)c->constants + c->constant_bytes;
        const int constants_fit = c->constants >= HXFG_HIGH_CONSTANT_START
                                      ? constant_end <= QDO_BYTES
                                      : constant_end <= SCRATCH_OFFSET;
        if (!init_error) {
            const Q16Header *h = programs[i].header;
            uint32_t spill_bytes = h->reserved[1];
            const uint64_t state_end = (uint64_t)cfg->state + cfg->state_bytes;
            const int low_spill = cfg->state >= Q16P_LOW_SPILL_START && state_end <= Q16P_LOW_SPILL_END;
            const int high_spill = cfg->state >= HXFG_HIGH_CONSTANT_START && state_end <= QDO_BYTES;
            int spill_fits = !spill_bytes ||
                (spill_bytes <= cfg->state_bytes && (low_spill || high_spill));
            if (h->scratch_bytes > SCRATCH_BYTES || !constants_fit || !spill_fits)
                init_error = Q16_RANGE;
        }
    }
    barrier();
    s->ready = 1;
    QdHmxState *hmx = (QdHmxState *)((uint8_t *)SERVICE + 512);
    uint32_t last = 0;
    for (;;) {
        uint32_t seq;
        while (!(seq = SERVICE[2]) || seq == last)
            pause_wait();
        barrier();
        last = seq;
        if (SERVICE[3] == QDC_OP_EXIT)
            break;
        uint32_t error = init_error, graph = SERVICE[4], completed = 0, matrix = 0;
        uint32_t fail = init_error ? bad : Q16P_NONE;
        uint64_t start = cycles(), timer_start = timer_ticks();
        if (!error && (SERVICE[0] != QDC_MAGIC || SERVICE[1] != QDC_VERSION || SERVICE[3] != QDS_OP_RUN ||
                       graph >= cfg->graphs || SERVICE[5] != cfg->graphs || SERVICE[6]))
            error = Q16_HEADER;
        if (!error && hmx_start(hmx))
            error = Q16_OPERATOR;
        if (!error) {
            const HxGraphConfig *c = &cfg->graph[graph];
            const Q16Program *p = &programs[graph];
            uint8_t *arena = (uint8_t *)(QDO_BASE + c->arena);
            uint8_t *spill = p->header->reserved[1] ? (uint8_t *)(QDO_BASE + cfg->state) : NULL;
            if (spill)
                memset(spill, 0, p->header->reserved[1]);
            Q16Exec e = {p, VTCM, arena, SCRATCH,
                         (const uint8_t *)(QDO_BASE + c->constants), qnn16_conv1x1, qnn16_hmx_finish,
                         parallel, 0, 0, spill};
            s->sequence = seq;
            qnn16_hmx_reset();
            uint32_t since_check = 8u;
            for (uint32_t op = 0; op < p->header->ops;) {
                /* Uncached control reads; a few operations of latency is enough. */
                if (since_check >= 8u) {
                    since_check = 0;
                    if (SERVICE[3] == QDC_OP_EXIT || *(volatile uint32_t *)((uint8_t *)SERVICE + 3840)) {
                        error = Q16_CANCELLED;
                        fail = op;
                        break;
                    }
                }
                uint64_t began = cycles();
                uint32_t ran = 1;
                /* A run of weight copies, or of independent GroupNorm-chain tile ops, is one parallel job;
                 * its cycles go to its first op. */
                error = q16_exec_run(&e, op, &ran);
#ifndef Q16_NO_OP_PROFILE
                for (uint32_t k = 0; k < ran && op + k < Q16_PROFILE_CYCLES; k++)
                    ((volatile uint64_t *)((uint8_t *)SERVICE + Q16P_PROFILE_OFFSET))[op + k] =
                        k ? 0 : cycles() - began;
#else
                (void)began;
#endif
                if (error) {
                    fail = op;
                    break;
                }
                completed += ran;
                op += ran;
                since_check += ran;
            }
            matrix = e.matrix_ops;
        }
        uint64_t elapsed = cycles() - start, ticks = timer_ticks() - timer_start;
        result[0] = Q16S_RESULT_MAGIC;
        result[1] = error;
        result[2] = fail;
        result[3] = completed;
        result[6] = matrix;
        result[7] = 0;
        result[8] = (uint32_t)ticks;
        result[9] = (uint32_t)(ticks >> 32);
        result[4] = (uint32_t)elapsed;
        result[5] = (uint32_t)(elapsed >> 32);
        broadcast(STATUS_OP, seq);
        status(0, seq);
    }
    broadcast(EXIT_OP, last);
    if (hmx_shutdown(hmx))
        result[1] = Q16_OPERATOR;
}
