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
#include "../qd_graph.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

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
} Session;

/* The Frame VTCM tail less its 8 KiB control area, and the Frame's L2-TCM
 * near slot for the fast convolution. */
enum { CONV_WORKSPACE = 0x7e000, CONV_NEAR = 0x8000 };

static const char *const kInputNames[4] = {"x", "y", "hidden", "t"};

static void pools_reset(Session *s) {
    for (int i = 0; i < 7; i++) if (s->pools[i].bytes) memcpy(s->pools[i].base, s->initial[i], s->pools[i].bytes);
}

static void open_session(Session *s, const char *model) {
    char path[256];
    memset(s, 0, sizeof *s);
    s->model = model;
    snprintf(path, sizeof path, "frame/build/resident-%s.qdrx", model);
    s->qdrx = read_file(path, &s->qdrx_bytes);
    snprintf(path, sizeof path, "frame/native_program/examples/%s-v2.nprg", model);
    s->nprg = read_file(path, &s->nprg_bytes);
    snprintf(path, sizeof path, "recovered/%s/constants/constant_pool_5.bin", model);
    s->constants = read_file(path, &s->constants_bytes);
    for (int i = 0; i < 4; i++) {
        snprintf(path, sizeof path, "frame/build/native-smoke/%s.raw", kInputNames[i]);
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
    Nprg2ExecRequest r = {.program = &s->program, .bound = &s->bound, .pool1_writable_bytes = s->pools[1].bytes,
                          .inputs = in, .input_count = 4, .outputs = out, .output_count = 2,
                          .scratch = s->scratch, .scratch_bytes = s->scratch_bytes,
                          .command_first = first, .command_count = count, .stats = stats};
    memcpy(r.pools, s->pools, sizeof s->pools);
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
        if (!s->pools[p].bytes || !memcmp(s->pools[p].base, ref[p], s->pools[p].bytes)) continue;
        uint32_t diff = 0;
        while (s->pools[p].base[diff] == ref[p][diff]) diff++;
        Nprg2Command c;
        nprg2_command_at(&s->bound, cmd, &c);
        fprintf(stderr, "%s pool mismatch after cmd=%lu op=%u var=%u seq=%u: pool=%d off=0x%lx prepared=%u cpu=%u\n",
                s->model, (unsigned long)cmd, c.opcode, c.variant, c.sequence, p, (unsigned long)diff,
                s->pools[p].base[diff], ref[p][diff]);
        return 0;
    }
    return 1;
}

static void check_outputs(Session *s, uint8_t *const *out, const char *what) {
    static const char *expect[2][2] = {
        {"dfacee192b7a534ef79634993d03a1c870e1550e5ba1c485af5ab17466616ad3",
         "a978ee8291ea05c0b482b72cdc54f09c1ad8294a366f3ba215bb32653cb3f32b"},
        {"dfacee192b7a534ef79634993d03a1c870e1550e5ba1c485af5ab17466616ad3",
         "98d28b66c3489d1424569aad3e1c2b9bfa12fac3aba5b11e67b0fde10d8bc36a"}};
    static const uint32_t bytes[2] = {163840, 655360};
    int alt = !strcmp(s->model, "alternate");
    for (int k = 0; k < 2; k++) {
        uint8_t sha[32];
        char hex[65];
        nprg2_sha256(out[k], bytes[k], sha);
        for (int j = 0; j < 32; j++) snprintf(hex + 2 * j, 3, "%02x", sha[j]);
        if (strcmp(hex, expect[alt][k])) {
            fprintf(stderr, "%s %s output %d sha256 %s, emulator %s\n", s->model, what, k, hex, expect[alt][k]);
            exit(1);
        }
    }
    printf("PASS %s %s: both graph outputs match the emulator smoke sha256\n", s->model, what);
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
    for (uint32_t i = 0; i < s->program.command_count; i++) {
        int rc = qdg_run(g, qin, qin_bytes, qout, qout_bytes, i, i + 1);
        if (rc) { fprintf(stderr, "qd_graph failed cmd=%lu rc=%d line=%u\n", (unsigned long)i, rc, qdg_stats(g)->error_detail); exit(1); }
        Nprg2ExecStats st;
        Nprg2ExecRequest r = request(s, in, ob, i, 1, &st);
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
        if (!same_pools(s, ref, i)) exit(1);
        if (memcmp(out[0], ref_out[0], 163840) || memcmp(out[1], ref_out[1], 655360)) {
            fprintf(stderr, "%s graph output mismatch after cmd=%lu\n", s->model, (unsigned long)i);
            exit(1);
        }
    }
    printf("PASS %s lockstep: %lu commands, all seven pools identical to qd_graph CPU after every command\n",
           s->model, (unsigned long)s->program.command_count);
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
    printf("PASS %s whole schedule: one preflight validated %lu commands without pool writes, one run executed all\n",
           s->model, (unsigned long)st.executed_commands);
    check_outputs(s, out, "whole");
    free(out[0]);
    free(out[1]);
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
    register_handlers(&s, mixed, fast);
    census(&s);
    if (strcmp(what, "whole")) lockstep(&s);
    if (strcmp(what, "lockstep")) whole(&s);
    return 0;
}
