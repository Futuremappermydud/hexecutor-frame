/* End-to-end simulator staging/call/download path shared with Frame payload. */
#include <stdint.h>
#include <hexagon_standalone.h>
#include <stdio.h>
#include <string.h>

static uint8_t simulator_arena[0x800000] __attribute__((aligned(128)));
#define XWB1022_SIM 1
#define ARENA ((uintptr_t)simulator_arena)
#include "xwb1022_frame_probe.c"
#include "xwb1022_actual_data.h"

int main(void) {
    memset(simulator_arena, 0, sizeof(simulator_arena));
    uint32_t *header = (uint32_t *)(simulator_arena + FIXTURE);
    header[0] = MAGIC; header[1] = 1; header[2] = 1022; header[3] = 836;
    header[4] = 102; header[5] = 32; header[6] = IN_BYTES;
    header[7] = W_BYTES; header[8] = FI_BYTES;
    header[9] = OUT_BYTES; header[10] = OUT_BYTES;
    uint8_t *data = simulator_arena + FIXTURE + FIXTURE_HEADER;
    memcpy(data, actual_input_pages, IN_BYTES);
    memcpy(data + IN_BYTES, actual_weights, W_BYTES);
    memcpy(data + IN_BYTES + W_BYTES, actual_fi, FI_BYTES);

    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    for (uint32_t offset = 0; offset < VTCM_BYTES; offset += 4096u)
        add_translation((void *)(vtcm + WORKSPACE_OFFSET + offset),
                        (void *)(vtcm + WORKSPACE_OFFSET + offset), 6);
    uint32_t ssr;
    __asm__ volatile("%0=ssr" : "=r"(ssr));
    ssr |= 1u << 26;
    __asm__ volatile("ssr=%0\n isync" :: "r"(ssr) : "memory");
    live_vtcm = (uint8_t *)(vtcm + WORKSPACE_OFFSET);
    memset(live_vtcm, 0xa5, VTCM_BYTES);

    run_kernel(0);
    volatile uint32_t *result = (volatile uint32_t *)(simulator_arena + 2048u);
    if (result[1] || result[8] != OUT_BYTES || result[9] != IN_BYTES) {
        printf("FAIL shared run_kernel status=%u output=%u input=%u\n",
               result[1], result[8], result[9]);
        return 1;
    }
    if (memcmp(simulator_arena + INPUT_ECHO, actual_input_pages, IN_BYTES) != 0) {
        puts("FAIL shared run_kernel input download differs from staged input");
        return 2;
    }
    printf("STAGING fnv32 input=%08x weights=%08x fi=%08x\n",
           result[28], result[29], result[30]);
    puts("RAW_OUTPUT_PHYSICAL_HEX_BEGIN");
    const uint8_t *output = simulator_arena + OUTPUT_COPY;
    for (uint32_t i = 0; i < OUT_BYTES; ++i) {
        printf("%02x", output[i]);
        if ((i & 255u) == 255u) putchar('\n');
    }
    puts("RAW_OUTPUT_PHYSICAL_HEX_END");
    puts("PASS shared simulator staging/call/drain/download/guards");
    return 0;
}
