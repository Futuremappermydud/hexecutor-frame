#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "qd_native_1x1_adapter.h"

#define FIXTURE "/src/native_adapter/padded-ci.bin"
#ifndef PADDED_CI_POISON_TAIL
#define PADDED_CI_POISON_TAIL 0
#endif
#if PADDED_CI_POISON_TAIL
#define OUTPUT  "/src/native_adapter/padded-ci-poison-out.bin"
#else
#define OUTPUT  "/src/native_adapter/padded-ci-out.bin"
#endif
#define MAGIC   0x31585151u

extern void qd_hmx_finish(void *, void *);
static uint8_t pool1[0x100000] __attribute__((aligned(128)));
static uint32_t tables[512] __attribute__((aligned(128)));
static uint32_t abi[30] __attribute__((aligned(128)));
static uint32_t pool_sizes[7] = {0, sizeof(pool1), 0, 0, 0, 0, 0};
static uint8_t payload[0x100000] __attribute__((aligned(128)));

int main(void) {
    FILE *file = fopen(FIXTURE, "rb");
    if (!file) { puts("FAIL open padded fixture"); return 1; }
    uint32_t h[32];
    if (fread(h, sizeof(h), 1, file) != 1 || h[0] != MAGIC ||
        (h[1] != 115 && h[1] != 3416)) {
        puts("FAIL padded fixture header"); fclose(file); return 2;
    }
    uint32_t input_bytes = h[7] * QD_N1X1_BLOCK_BYTES;
    uint32_t output_bytes = h[13] * QD_N1X1_BLOCK_BYTES;
    uint64_t total = (uint64_t)input_bytes + output_bytes + h[17] + h[18] + h[25] + h[26];
    if (!h[7] || !h[13] || input_bytes + output_bytes > sizeof(pool1) ||
        h[14] != 32 || h[17] != 32u * h[15] || h[18] != h[15] * 8u ||
        h[25] != h[24] * h[15] || h[26] != output_bytes ||
        (h[24] == 12 ? h[27] != 9 : h[24] == 20 ? h[27] != 18 : 1) ||
        total > sizeof(payload)) {
        puts("FAIL padded fixture sizes"); fclose(file); return 3;
    }
    if (fread(payload, (size_t)total, 1, file) != 1 || fgetc(file) != EOF) {
        puts("FAIL padded fixture payload"); fclose(file); return 4;
    }
    fclose(file);

    uint8_t *input = payload;
    uint8_t *output_before = input + input_bytes;
    uint8_t *weights32 = output_before + output_bytes;
    uint8_t *fi = weights32 + h[17];
    uint8_t *weights_original = fi + h[18];
    uint8_t *cpu_reference = weights_original + h[25];
    memcpy(pool1, input, input_bytes);
    memcpy(pool1 + input_bytes, output_before, output_bytes);
#if PADDED_CI_POISON_TAIL
    /* The graph's logical input depth is 9 or 18, but the original Xpb
     * command consumes 12 or 20 physical channels. Leave those channels
     * intact and poison only lanes that the zero-expanded filter must ignore. */
    for (uint32_t block = 0; block < h[7]; ++block)
        for (uint32_t row = 0; row < 8; ++row)
            for (uint32_t col = 0; col < 8; ++col)
                for (uint32_t channel = h[24]; channel < 32; ++channel)
                    pool1[block * QD_N1X1_BLOCK_BYTES + row * 256u + col * 32u + channel] = 0xa5;
#endif

    uint8_t *base = (uint8_t *)(__rdcfg(__vtcm_base) << 16);
    for (uint32_t offset = 0; offset < QD_N1X1_WORKSPACE_MAX; offset += 4096)
        add_translation(base + offset, base + offset, 6);
    memset(base, 0xa5, QD_N1X1_WORKSPACE_MAX);

    uint32_t input_blocks[QD_N1X1_TABLE_ENTRIES];
    uint32_t output_blocks[QD_N1X1_TABLE_ENTRIES];
    if (h[7] > QD_N1X1_TABLE_ENTRIES || h[13] > QD_N1X1_TABLE_ENTRIES) {
        puts("FAIL padded table capacity"); return 5;
    }
    for (uint32_t i = 0; i < h[7]; ++i) input_blocks[i] = i * 2048u | 1u;
    for (uint32_t i = 0; i < h[13]; ++i)
        output_blocks[i] = (input_bytes + i * 2048u) | 1u;

    QdN1x1Request request;
    memset(&request, 0, sizeof(request));
    request.pool_count = 7;
    request.pool[1] = pool1;
    request.pool_bytes = pool_sizes;
    request.input = (QdN1x1Blocked){
        .dims = {1, h[2], h[3], h[27]}, .max_dims = {1, h[4], h[5], 32},
        .padding = {0, h[19], h[20], 0}, .slice_offset = h[6],
        .block_count = h[7], .blocks = input_blocks
    };
    request.output = (QdN1x1Blocked){
        .dims = {1, h[8], h[9], h[15]}, .max_dims = {1, h[10], h[11], h[15]},
        .padding = {0, h[21], h[22], 0}, .slice_offset = h[12],
        .block_count = h[13], .blocks = output_blocks
    };
    request.weights = weights_original;
    request.weights_bytes = h[25];
    request.weight_ci = h[24];
    request.weight_co = h[15];
#if QDS_NATIVE_PADDED_1X1
    request.weight_native_ci = 32;
#endif
    request.weight_kernel_h = 1;
    request.weight_kernel_w = 1;
    request.input_dtype = QD_N1X1_DTYPE_U8;
    request.output_dtype = QD_N1X1_DTYPE_U8;
    request.weight_dtype = QD_N1X1_DTYPE_I8;
    request.fi_dtype = QD_N1X1_DTYPE_I32;
    request.weight_format = QD_N1X1_WEIGHT_XPB;
    request.fi = (const int32_t *)fi;
    request.fi_bytes = h[18];
    request.workspace = base;
    request.workspace_bytes = QD_N1X1_WORKSPACE_MAX;
    request.table_storage = tables;
    request.table_storage_bytes = sizeof(tables);
    request.abi_storage = abi;
    request.abi_storage_bytes = sizeof(abi);

    QdN1x1Prepared prepared;
#if QDS_NATIVE_PADDED_1X1
    QdN1x1Request rejected = request;
    QdN1x1Prepared rejected_prepared;
    rejected.weight_native_ci = 64;
    if (qd_native_1x1_prepare(&rejected, &rejected_prepared) != QD_N1X1_UNSUPPORTED) {
        puts("FAIL unsupported padded geometry was accepted"); return 6;
    }
    for (uint32_t i = 0; i < QD_N1X1_WORKSPACE_MAX; ++i)
        if (base[i] != 0xa5) { puts("FAIL rejected prepare modified workspace"); return 6; }
#endif
    int status = qd_native_1x1_prepare(&request, &prepared);
    if (status) { printf("FAIL prepare %d\n", status); return 6; }
    if (memcmp(prepared.weights, weights32, h[17])) {
        puts("FAIL adapter Xpb CI expansion differs from fixture oracle"); return 6;
    }
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" :: "r"(ssr) : "memory");
    __asm__ volatile("mxclracc" ::: "memory");
    qd_qnn_hmx_1x1(prepared.output_desc, prepared.input_desc, prepared.weights,
                   prepared.fi, prepared.cfg, prepared.mode);
    volatile uint32_t *done = (volatile uint32_t *)(base + 0x58000u);
    *done = 0xffffffffu;
    qd_hmx_finish(prepared.zero_words, (void *)done);
    uint32_t tries = 0;
    while (*done && tries++ < 100000u) __asm__ volatile("nop");
    if (*done) { puts("FAIL native finish timeout"); return 7; }
    status = qd_native_1x1_finish(&prepared);
    if (status) { printf("FAIL scatter %d\n", status); return 8; }

    file = fopen(OUTPUT, "wb");
    if (!file) { puts("FAIL output open"); return 9; }
    if (fwrite(h, sizeof(h), 1, file) != 1 ||
        fwrite(pool1 + input_bytes, 1, output_bytes, file) != output_bytes ||
        fwrite(cpu_reference, 1, output_bytes, file) != output_bytes ||
        fwrite(weights_original, 1, h[25], file) != h[25] ||
        fwrite(fi, 1, h[18], file) != h[18]) {
        puts("FAIL output write"); fclose(file); return 10;
    }
    fclose(file);
    printf("PASS padded CI%u->32 sequence %u%s: %u/%u blocks, staged=%u\n",
           h[24], h[1], PADDED_CI_POISON_TAIL ? " (tail poisoned)" : "",
           prepared.input_blocks, prepared.output_blocks,
           prepared.workspace_used);
    return 0;
}
