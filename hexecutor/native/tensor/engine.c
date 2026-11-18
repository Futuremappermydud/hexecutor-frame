/* Generic tensor engine on the shared guarded bootstrap/transport. Worker
 * stacks, mapping, VTCM snapshot and restoration remain bootstrap-owned. */
#include "graph.h"
#include "../qd_conv.h"
#include "hmx.h"
#include "fp16.h"
static uint64_t cycles(void);
#include "../qd_hmx_power.h"
#define QDO_BASE 0xa0000000u
#define QDS_OP_RUN 3u
#include "../qd_protocol.h"
#include <stddef.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#define SERVICE ((volatile uint32_t *)0xf1800000u)
#define SHARED ((Shared *)0x300f0000u)
#define GRAPHS ((HxGraph *)0x300c0000u)
static const char hx_engine_feature[] __attribute__((used, section(".rodata.hexecutor_features"))) =
    "HXTENG01";
/* Physical hthreads 0, 1, 2 and 5. Work indices stay dense because the fourth
 * physical id is sparse. The host checks that the bootstrap launches them all. */
#define HX_PARTS 4u
static const char hx_engine_parts[] __attribute__((used, section(".rodata.hexecutor_features"))) =
    "HXTPART4";
static inline uint32_t part_of(uint32_t tid) { return tid == 5u ? 3u : tid; }
typedef struct {
    volatile uint32_t ready, generation, done[4], op, graph, sequence, fp16, resident;
    const HxFp16Job *volatile job;
} Shared;
/* Park idle hardware threads so polling does not steal issue slots from
 * the active HMX/packing thread. Wake latency is bounded to 256 cycles. */
static void pause_wait(void) { __asm__ volatile("pause(#255)" ::: "memory"); }
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
    while (n >= 128) {
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
    while (n >= 128) {
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
#define HXFG_JOB (HXFG_NONE - 2u)
static void dispatch(uint32_t graph, uint32_t op, uint32_t seq) {
    Shared *s = SHARED;
    s->graph = graph;
    s->op = op;
    s->sequence = seq;
    uint32_t gen = s->generation + 1;
    barrier();
    s->generation = gen;
    if (op == HXFG_JOB)
        hx_fp16_job(s->job, 0, HX_PARTS);
    else if (op < HXFG_MAX_OPS) {
        if (s->fp16)
            hx_fp16_op(&GRAPHS[graph], op, 0, HX_PARTS);
        else
            hx_execute_op(&GRAPHS[graph], op, 0, HX_PARTS);
    }
    for (uint32_t t = 1; t < HX_PARTS; t++)
        while (s->done[t] != gen) {
            pause_wait();
        }
    barrier();
}
static void parallel(const HxFp16Job *job) {
    SHARED->job = job;
    dispatch(0, HXFG_JOB, SHARED->sequence);
}
__attribute__((section(".text.resident_entry"))) void resident_entry(uint32_t tid) {
    Shared *s = SHARED;
    if (tid) {
        const uint32_t part = part_of(tid);
        uint32_t seen = 0;
        while (!s->ready) {
            pause_wait();
        }
        for (;;) {
            uint32_t gen;
            while ((gen = s->generation) == seen) {
                pause_wait();
            }
            barrier();
            uint32_t op = s->op;
            if (op == HXFG_JOB)
                hx_fp16_job(s->job, part, HX_PARTS);
            else if (op < HXFG_MAX_OPS) {
                if (s->fp16)
                    hx_fp16_op(&GRAPHS[s->graph], op, part, HX_PARTS);
                else
                    hx_execute_op(&GRAPHS[s->graph], op, part, HX_PARTS);
            } else if (op == HXFG_NONE)
                status(tid, s->sequence);
            barrier();
            s->done[part] = gen;
            seen = gen;
            if (op == HXFG_NONE - 1u)
                return;
        }
    }
    const HxSessionConfig *cfg = (const HxSessionConfig *)((const uint8_t *)SERVICE + 4096);
    /* Per wake: VTCM records never survive an exit, and SHARED is cleared below. */
    uint32_t shared_records = 0;
    volatile uint32_t *result = (volatile uint32_t *)((uint8_t *)SERVICE + 2048);
    memset(s, 0, sizeof *s);
    uint32_t init_error = hx_session_config_valid(cfg) ? HX_OK : HX_HEADER, bad = HXFG_NONE;
    if (!init_error) {
        if (cfg->state_bytes) {
            memset((void *)(QDO_BASE + cfg->state), 0, cfg->state_bytes);
        }
        for (uint32_t i = 0; i < cfg->graphs; i++) {
            const HxGraphConfig *c = &cfg->graph[i];
            /* hx_graph_open keeps a state pointer set beforehand and checks persistent tensors
             * against it; set it first so they are bounded by state_bytes, not the arena. */
            hx_graph_set_state(&GRAPHS[i],
                               cfg->state_bytes ? (void *)(QDO_BASE + cfg->state) : (void *)0,
                               cfg->state_bytes);
            init_error =
                hx_graph_open(&GRAPHS[i], (const void *)(QDO_BASE + c->image), c->image_bytes,
                              (const void *)(QDO_BASE + c->constants), c->constant_bytes,
                              (void *)(QDO_BASE + c->arena), c->arena_bytes, &bad);
            if (init_error)
                break;
            if (cfg->state_bytes) {
                GRAPHS[i].state = (uint8_t *)(QDO_BASE + cfg->state);
                GRAPHS[i].state_bytes = cfg->state_bytes;
            }
        }
    }
    /* Each selected graph reuses the same preserved VTCM activation arena.
     * DDR remains the host exchange area. The upper MiB is disjoint kernel
     * scratch, never a live tensor alias. Large graphs keep their DDR arena.
     * The FP16 plan and every metadata table live in L2 for the session. */
    if (!init_error) {
        uint8_t *metadata = (uint8_t *)0x30000000u;
        for (uint32_t i = 0; i < cfg->graphs; i++) {
            HxGraph *g = &GRAPHS[i];
            if (g->header->arena_bytes <= 0x700000u)
                g->arena = (uint8_t *)0x31000000u;
            uint32_t bytes = g->header->bytes;
            if (((bytes + 127u) & ~127u) <= 0x30060000u - (uint32_t)metadata) {
                memcpy(metadata, g->header, bytes);
                g->header = (const HxHeader *)metadata;
                g->tensors = (const HxTensor *)(metadata + g->header->tensor_offset);
                g->ops = (const HxOp *)(metadata + g->header->op_offset);
                g->io = (const uint32_t *)(metadata + g->header->io_offset);
                metadata += (bytes + 127u) & ~127u;
            }
            uint32_t words = hx_fp16_table_words(g);
            uint32_t need = (sizeof(struct HxFp16Plan) + words * 4u + 127u) & ~127u;
            if (g->arena == (uint8_t *)0x31000000u && (uint32_t)g->header >= 0x30000000u &&
                (uint32_t)g->header < 0x30060000u && need <= 0x30060000u - (uint32_t)metadata) {
                struct HxFp16Plan *plan = (struct HxFp16Plan *)metadata;
                if (hx_fp16_plan(g, plan, (uint32_t *)(plan + 1), (uint8_t *)0x31800000u)) {
                    g->fp16 = plan;
                    metadata += need;
                }
            }
        }
        /* With several FP16 graphs, give each its own resident-record window so
         * a graph switch does not re-stage records that are still in VTCM. */
        HxGraph *blocked_graphs[HXFG_MAX_GRAPHS];
        uint32_t count = 0;
        for (uint32_t i = 0; i < cfg->graphs; i++)
            if (GRAPHS[i].fp16)
                blocked_graphs[count++] = &GRAPHS[i];
        shared_records = hx_fp16_share(blocked_graphs, count, (uint8_t *)0x31800000u);
    }
    barrier();
    s->ready = 1;
    QdHmxState *hmx = (QdHmxState *)((uint8_t *)SERVICE + 512);
    uint32_t hmx_ops = 0;
    uint32_t last = 0;
    for (;;) {
        uint32_t seq;
        while (!(seq = SERVICE[2]) || seq == last) {
            pause_wait();
        }
        barrier();
        last = seq;
        if (SERVICE[3] == QDC_OP_EXIT)
            break;
        if (SERVICE[3] == 4u /* QDS_OP_RESET_STATE */) {
            if (cfg->state_bytes)
                memset((void *)(QDO_BASE + cfg->state), 0, cfg->state_bytes);
            result[0] = HXFG_RESULT_MAGIC;
            result[1] = HX_OK;
            result[2] = 0;
            result[3] = 0;
            result[4] = 0;
            result[5] = 0;
            result[6] = 0;
            result[7] = 0;
            result[8] = 0;
            result[9] = 0;
            dispatch(0, HXFG_NONE, seq);
            status(0, seq);
            continue;
        }
        uint32_t error = init_error, graph = SERVICE[4], completed = 0;
        uint64_t start = cycles(), timer_start = timer_ticks();
        if (!error &&
            (SERVICE[0] != QDC_MAGIC || SERVICE[1] != QDC_VERSION || SERVICE[3] != QDS_OP_RUN ||
             graph >= cfg->graphs || SERVICE[5] != cfg->graphs || (SERVICE[6] & ~3u)))
            error = HX_HEADER;
        hmx_ops = 0;
        s->fp16 = 0;
        if (!error && !(SERVICE[6] & 1u))
            error = hmx_start(hmx) ? HX_OPERATOR : HX_OK;
        if (!error) {
            HxGraph *g = &GRAPHS[graph];
            const uint32_t blocked = !(SERVICE[6] & 3u) && g->fp16;
            s->fp16 = blocked;
            s->sequence = seq;
            uint8_t *const workspace = (uint8_t *)0x31700000u;
            uint8_t *exchange = (uint8_t *)(QDO_BASE + cfg->graph[graph].arena);
            /* s->resident is a bitmask of graphs whose resident records are in
             * VTCM. Without shared windows the graphs' records overlap, so
             * staging one evicts the rest; any FP32 run evicts everything. */
            if (blocked && !(s->resident & (1u << graph))) {
                hx_fp16_stage(g, parallel);
                s->resident = shared_records ? s->resident | (1u << graph) : 1u << graph;
            } else if (!blocked)
                s->resident = 0;
            if (g->arena != exchange)
                for (uint32_t i = 0; i < g->header->inputs; i++) {
                    const HxTensor *t = &g->tensors[g->io[i]];
                    if (blocked)
                        hx_fp16_import(g, g->io[i], (const float *)(exchange + t->offset), parallel);
                    else
                        memcpy(g->arena + t->offset, exchange + t->offset, hx_tensor_bytes(t));
                }
            for (uint32_t op = 0; op < GRAPHS[graph].header->ops; op++) {
                /* These are uncached DDR reads; a few operators of latency is enough. */
                if (!(op & 7u) && (SERVICE[3] == QDC_OP_EXIT ||
                                   *(volatile uint32_t *)((uint8_t *)SERVICE + 3840))) {
                    error = HX_CANCELLED;
                    bad = op;
                    break;
                }
                uint64_t began = cycles();
                const HxOp *o = &g->ops[op];
                if (blocked) {
                    if (o->kind == HX_CONV) {
                        hx_fp16_conv(g, op, parallel);
                        hmx_ops++;
                    } else
                        dispatch(graph, op, seq);
                    hx_fp16_complete(g, op);
                } else if (!(SERVICE[6] & 1u) && o->kind == HX_CONV && o->p[10] &&
                           o->z == HXFG_NONE) {
                    /* hx_hmx_conv gathers undilated windows; dilated convs take hx_execute_op. */
                    hx_hmx_conv(g, op, workspace);
                    hmx_ops++;
                } else {
                    /* Small constant vectors and depthwise filters are reused
                     * for every pixel. Stage once before publishing the job. */
                    const uint8_t *original = g->constants;
                    const HxOp *original_ops = g->ops;
                    const HxTensor *t = &g->tensors[o->out];
                    HxOp staged = *o;
                    uint32_t bytes = 0;
                    if (o->kind == HX_DEPTHWISE) {
                        uint32_t wb = o->p[0] * o->p[1] * t->stride * 4u;
                        if (wb + t->stride * 4u <= 0x100000u) {
                            memcpy(workspace, original + o->p[6], wb);
                            memcpy(workspace + wb, original + o->p[7], t->stride * 4u);
                            staged.p[6] = 0;
                            staged.p[7] = wb;
                            bytes = wb + t->stride * 4u;
                        }
                    }
                    if (o->kind == HX_PRELU) {
                        bytes = t->stride * 4u;
                        memcpy(workspace, original + o->p[0], bytes);
                        staged.p[0] = 0;
                    }
                    if (bytes) {
                        g->constants = workspace;
                        g->ops = &staged;
                        dispatch(graph, 0, seq);
                        g->constants = original;
                        g->ops = original_ops;
                    } else
                        dispatch(graph, op, seq);
                }
                ((volatile uint64_t *)((uint8_t *)SERVICE + 16384))[op] = cycles() - began;
                completed++;
            }
            if (g->arena != exchange && !error)
                for (uint32_t i = 0; i < g->header->outputs; i++) {
                    const uint32_t id = g->io[g->header->inputs + i];
                    const HxTensor *t = &g->tensors[id];
                    if (blocked)
                        hx_fp16_export(g, id, (float *)(exchange + t->offset), parallel);
                    else
                        memcpy(exchange + t->offset, g->arena + t->offset, hx_tensor_bytes(t));
                }
        }
        uint64_t elapsed = cycles() - start, ticks = timer_ticks() - timer_start;
        result[0] = HXFG_RESULT_MAGIC;
        result[1] = error;
        result[2] = bad;
        result[3] = completed;
        result[6] = hmx_ops;
        result[7] = s->fp16;
        result[8] = (uint32_t)ticks;
        result[9] = (uint32_t)(ticks >> 32);
        result[4] = (uint32_t)elapsed;
        result[5] = (uint32_t)(elapsed >> 32);
        dispatch(0, HXFG_NONE, seq);
        status(0, seq);
    }
    dispatch(0, HXFG_NONE - 1u, last);
    if (hmx_shutdown(hmx))
        result[1] = HX_OPERATOR;
}
