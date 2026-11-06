/* Isolated Frame probe for the adapter + QNN 1x1 leaf. Worker 0 only. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "../qd_protocol.h"
#include "../qd_conv.h"
#include "qd_native_1x1_adapter.h"

#define ARENA 0xf1800000u
#define MAGIC 0x31514151u
#define FIXTURE 0x10000u
#define W_COPY 0x26000u
#define FI_COPY 0x2a000u
#define POOL 0x40000u
#define OUTPUT_POOL_OFFSET 0x10000u
#define OUTPUT_COPY 0x90000u
#define INPUT_ECHO 0xa0000u
#define BLOCK_TABLES 0x100000u
#define BACKUP 0x300000u
#define WORKSPACE_OFFSET 0x700000u

static uint64_t cycles(void) {
    uint64_t value;
    __asm__ volatile("%0=upcycle" : "=r"(value));
    return value;
}
#include "../qd_axim.h"
#include "../qd_hmx_power.h"

extern void qd_hmx_finish(void *, void *);
extern void qds_on_stack(void (*entry)(uint32_t), uint32_t tid, uint32_t stack);

static uint32_t pointer_tables[512] __attribute__((aligned(128)));
static uint32_t abi_storage[30] __attribute__((aligned(128)));
static uint32_t pool_sizes[7] = {0, 0x20000, 0, 0, 0, 0, 0};

void *memset(void *dst, int value, uint32_t bytes) {
    uint8_t *p = dst;
    for (uint32_t i = 0; i < bytes; ++i) p[i] = (uint8_t)value;
    return dst;
}

static void copybytes(void *dst, const void *src, uint32_t bytes) {
    uint8_t *d = dst;
    const uint8_t *s = src;
    for (uint32_t i = 0; i < bytes; ++i) d[i] = s[i];
}

static void copy128(void *dst, const void *src, uint32_t bytes) {
    for (uint32_t i = 0; i < bytes; i += 128)
        *(HVX_Vector *)((uint8_t *)dst + i) = *(const HVX_Vector *)((const uint8_t *)src + i);
}

static void fill128(void *dst, uint32_t bytes, uint32_t value) {
    HVX_Vector v = Q6_V_vsplat_R(value);
    for (uint32_t i = 0; i < bytes; i += 128)
        *(HVX_Vector *)((uint8_t *)dst + i) = v;
}

static int same128(const uint8_t *a, const uint8_t *b, uint32_t bytes) {
    HVX_Vector diff = Q6_V_vzero();
    for (uint32_t i = 0; i < bytes; i += 128)
        diff = Q6_V_vor_VV(diff, Q6_V_vxor_VV(*(const HVX_Vector *)(a + i),
                                               *(const HVX_Vector *)(b + i)));
    uint32_t any = 0;
    /* The caller is back on the firmware stack after run_adapter returns.
     * Avoid HVX stores followed by potentially cached scalar stack reads. */
    for (uint32_t i = 0; i < 32; ++i) any |= (uint32_t)Q6_R_vextract_VR(diff, i * 4u);
    return any == 0;
}

static int drain_hmx(int32_t *zero_words, uint8_t *done) {
    *(volatile uint32_t *)done = 0xffffffffu;
    qd_hmx_finish(zero_words, done);
    uint64_t deadline = cycles() + 200000000u;
    while (*(volatile uint32_t *)done)
        if (cycles() > deadline) return 1;
    return 0;
}

static int check_vtcm_tlb(volatile uint32_t *result, uint32_t before_restore) {
    uint64_t e8, e9;
    __asm__ volatile("%0=tlbr(%1)" : "=r"(e8) : "r"(8));
    __asm__ volatile("%0=tlbr(%1)" : "=r"(e9) : "r"(9));
    uint32_t at = before_restore ? 23u : 19u;
    result[at] = (uint32_t)e8;
    result[at + 1u] = (uint32_t)(e8 >> 32);
    result[at + 2u] = (uint32_t)e9;
    result[at + 3u] = (uint32_t)(e9 >> 32);
    return e8 == 0xc0031000f7062020ull && e9 == 0xc0031400f7062820ull;
}

static void run_adapter(uint32_t unused) {
    (void)unused;
    volatile uint32_t *command = (volatile uint32_t *)ARENA;
    volatile uint32_t *result = (volatile uint32_t *)(ARENA + 2048);
    const uint32_t h = command[3], w = command[4], ci = command[5], co = command[6];
    const uint32_t repeats = command[7];
    uint8_t *arena = (uint8_t *)ARENA;
    const uint8_t *fixture = arena + FIXTURE;
    const uint32_t *fh = (const uint32_t *)fixture;
    const uint32_t input_blocks = (h / 8u) * (w / 8u) * (ci / 32u);
    const uint32_t output_blocks = (h / 8u) * (w / 8u) * (co / 32u);
    const uint32_t expected_input_blocks = 32u;
    const uint32_t expected_weights_bytes = ci * co;
    const uint32_t expected_fi_bytes = co * 8u;

    /* Validate every byte range and supported shape before staging anything. */
    if (fh[0] != 0x31585151u || fh[1] != 1195 || fh[2] != h || fh[3] != w ||
        fh[4] != ci || fh[5] != co || fh[6] != expected_input_blocks ||
        fh[7] != expected_weights_bytes || fh[8] != expected_fi_bytes ||
        h != 8 || w != 64 || ci != 128 || co != 128 || repeats == 0 || repeats > 100) {
        result[1] = 21;
        return;
    }
    const uint32_t fixture_bytes = 36u + fh[6] * 2048u + fh[7] + fh[8];
    if (FIXTURE + fixture_bytes > 0x30000u || POOL + pool_sizes[1] > 0x60000u ||
        OUTPUT_COPY + output_blocks * 2048u > INPUT_ECHO ||
        INPUT_ECHO + input_blocks * 2048u > BLOCK_TABLES) {
        result[1] = 22;
        return;
    }

    const uint8_t *input = fixture + 36u;
    const uint8_t *weights_src = input + fh[6] * 2048u;
    const uint8_t *fi_src = weights_src + fh[7];
    uint8_t *pool = arena + POOL;
    uint8_t *weights = arena + W_COPY;
    uint8_t *fi = arena + FI_COPY;
    uint32_t *input_blocks_table = (uint32_t *)(arena + BLOCK_TABLES);
    uint32_t *output_blocks_table = input_blocks_table + 256;

    /* All live operand copies are 128-byte aligned; publish them before HMX. */
    copybytes(pool, input, input_blocks * 2048u);
    copybytes(weights, weights_src, fh[7]);
    copybytes(fi, fi_src, fh[8]);
    fill128(pool + OUTPUT_POOL_OFFSET, output_blocks * 2048u, 0xa5a5a5a5u);
    for (uint32_t i = 0; i < 256; ++i) {
        input_blocks_table[i] = 1u;
        output_blocks_table[i] = 1u | OUTPUT_POOL_OFFSET;
    }
    for (uint32_t i = 0; i < input_blocks; ++i)
        input_blocks_table[i] = i * 2048u | 1u;
    for (uint32_t i = 0; i < output_blocks; ++i)
        output_blocks_table[i] = (i * 2048u + OUTPUT_POOL_OFFSET) | 1u;
    for (uintptr_t p = (uintptr_t)pool; p < (uintptr_t)pool + (input_blocks + output_blocks) * 2048u; p += 32)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    for (uintptr_t p = (uintptr_t)weights; p < (uintptr_t)weights + fh[7]; p += 32)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    for (uintptr_t p = (uintptr_t)fi; p < (uintptr_t)fi + fh[8]; p += 32)
        __asm__ volatile("dccleaninva(%0)" :: "r"(p) : "memory");
    __asm__ volatile("barrier" ::: "memory");

    QdN1x1Request request;
    memset(&request, 0, sizeof request);
    request.pool_count = 7;
    request.pool[1] = pool;
    request.pool_bytes = pool_sizes;
    request.input = (QdN1x1Blocked){
        .dims = {1, h, w, ci}, .max_dims = {1, h, w, ci},
        .block_count = input_blocks, .blocks = input_blocks_table};
    request.output = (QdN1x1Blocked){
        .dims = {1, h, w, co}, .max_dims = {1, h, w, co},
        .block_count = output_blocks, .blocks = output_blocks_table};
    request.weights = weights;
    request.weights_bytes = fh[7];
    request.weight_ci = ci;
    request.weight_co = co;
    request.weight_kernel_h = 1;
    request.weight_kernel_w = 1;
    request.input_dtype = QD_N1X1_DTYPE_U8;
    request.output_dtype = QD_N1X1_DTYPE_U8;
    request.weight_dtype = QD_N1X1_DTYPE_I8;
    request.fi_dtype = QD_N1X1_DTYPE_I32;
    request.weight_format = QD_N1X1_WEIGHT_XPB;
    request.fi = (const int32_t *)fi;
    request.fi_bytes = fh[8];
    request.workspace = (void *)(QDC_VTCM_BASE + WORKSPACE_OFFSET);
    request.workspace_bytes = QD_N1X1_WORKSPACE_MAX;
    request.table_storage = pointer_tables;
    request.table_storage_bytes = sizeof pointer_tables;
    request.abi_storage = abi_storage;
    request.abi_storage_bytes = sizeof abi_storage;

    uint64_t started = cycles();
    QdN1x1Prepared prepared;
    int status = qd_native_1x1_prepare(&request, &prepared);
    if (status) { result[1] = 30 + status; return; }
    result[4] = (uint32_t)(cycles() - started);
    uint8_t *done = (uint8_t *)request.workspace + 0x52000u;
    uint64_t elapsed = 0;
    for (uint32_t r = 0; r < repeats; ++r) {
        __asm__ volatile("mxclracc" ::: "memory");
        uint64_t t = cycles();
        qd_qnn_hmx_1x1(prepared.output_desc, prepared.input_desc, prepared.weights,
                       prepared.fi, prepared.cfg, prepared.mode);
        if (drain_hmx(prepared.zero_words, done)) { result[1] = 40; return; }
        elapsed += cycles() - t;
        status = qd_native_1x1_finish(&prepared);
        if (status) { result[1] = 50 + status; return; }
    }
    result[5] = (uint32_t)elapsed;
    result[6] = (uint32_t)(elapsed >> 32);
    started = cycles();
    for (uint32_t i = 0; i < output_blocks; ++i)
        copy128(arena + OUTPUT_COPY + i * 2048u, pool + OUTPUT_POOL_OFFSET + i * 2048u, 2048u);
    for (uint32_t i = 0; i < input_blocks; ++i)
        copy128(arena + INPUT_ECHO + i * 2048u, pool + i * 2048u, 2048u);
    result[7] = (uint32_t)(cycles() - started);
    result[8] = output_blocks * 2048u;
    result[9] = input_blocks * 2048u;
    result[2] = 3;
}

__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;
    __asm__ volatile("%0=htid" : "=r"(tid));
    if (tid >= 3) return;
    volatile uint32_t *command = (volatile uint32_t *)ARENA;
    volatile uint32_t *result = (volatile uint32_t *)(ARENA + 2048);
    uint32_t sequence = command[2];
    if (tid == 0) {
        for (uint32_t i = 0; i < 32; ++i) result[i] = 0;
        result[0] = MAGIC;
        result[2] = 1;
        uint64_t deadline = cycles() + 2000000000u;
        while (*(volatile uint32_t *)(ARENA + 144) != sequence ||
               *(volatile uint32_t *)(ARENA + 160) != sequence)
            if (cycles() > deadline) { result[1] = 1; goto done; }
        if (command[0] != MAGIC || command[1] != 1 || command[3] != 8 ||
            command[4] != 64 || command[5] != 128 || command[6] != 128 ||
            command[7] == 0 || command[7] > 100 || command[9] != WORKSPACE_OFFSET) {
            result[1] = 2;
            goto done;
        }
        volatile uint32_t *clocks = (volatile uint32_t *)(ARENA + QDC_CLOCK_OFFSET);
        QdHmxState *hmx = (QdHmxState *)(ARENA + QDC_HMX_STATE_OFFSET);
        result[1] = clock_raise(clocks);
        if (!result[1]) result[1] = hmx_start(hmx);
        if (!result[1]) {
            if (!check_vtcm_tlb(result, 0)) {
                result[1] = 6;
            } else {
                uint64_t t = cycles();
                copy128((void *)(ARENA + BACKUP),
                        (void *)(QDC_VTCM_BASE + WORKSPACE_OFFSET), 0x80000u);
                __asm__ volatile("syncht" ::: "memory");
                result[15] = (uint32_t)(cycles() - t);
                result[18] = 1;
                qds_on_stack(run_adapter, 0, 0x30090000u);
            }
        }
        result[10] = hmx_shutdown(hmx);
        if (result[18] && !result[10]) {
            if (!check_vtcm_tlb(result, 1)) {
                /* Mapping changed: preserve the DDR backup and never write VTCM. */
                result[1] = 6;
            } else {
                uint64_t t = cycles();
                copy128((void *)(QDC_VTCM_BASE + WORKSPACE_OFFSET),
                        (void *)(ARENA + BACKUP), 0x80000u);
                __asm__ volatile("syncht" ::: "memory");
                result[16] = (uint32_t)(cycles() - t);
                result[17] = same128((const uint8_t *)(QDC_VTCM_BASE + WORKSPACE_OFFSET),
                                     (const uint8_t *)(ARENA + BACKUP), 0x80000u);
                if (!result[17]) result[1] = 4;
            }
        }
        result[11] = clock_restore(clocks);
        if (result[10] || result[11]) result[1] = 5;
        result[2] = 4;
    }
done:
    __asm__ volatile("barrier" ::: "memory");
    volatile QdStatus *status = (volatile QdStatus *)(ARENA + QD_STATUS_OFFSET) + tid;
    status->error = tid ? 0 : result[1];
    status->rows = 0;
    status->reserved = QDC_EXITED;
    __asm__ volatile("barrier" ::: "memory");
    status->done_sequence = sequence;
}
