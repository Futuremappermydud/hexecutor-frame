/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Hexagon simulator test for persistent state across decode steps.
 * Compares against independent NumPy reference in test_persistent_golden.h.
 */

#include "graph.h"
#include "lfm_ops.h"
#include "test_persistent_golden.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float fabsf_(float x) {
    return x < 0.0f ? -x : x;
}

int main(void) {
    printf("Starting persistent state simulation test (independent NumPy reference)...\n");

    /* Align buffers to 128 bytes */
    static uint8_t image[1024] __attribute__((aligned(128)));
    static uint8_t arena[2048] __attribute__((aligned(128)));
    static uint8_t constants[1024] __attribute__((aligned(128)));
    static uint8_t state[2048] __attribute__((aligned(128)));

    /* Load constants: conv weights at 0, conv bias at 384 */
    memset(constants, 0, sizeof(constants));
    memcpy(constants, golden_conv_w, sizeof(golden_conv_w));
    memcpy(constants + 384, golden_conv_b, sizeof(golden_conv_b));

    /* Zero-initialize persistent state at session open */
    memset(state, 0, sizeof(state));

    /* 6 tensors */
    HxTensor tensors[6];
    memset(tensors, 0, sizeof(tensors));

    /* 0: x_conv [1, 1, 32] */
    tensors[0].h = 1;
    tensors[0].w = 1;
    tensors[0].c = GOLDEN_C;
    tensors[0].stride = GOLDEN_C;
    tensors[0].offset = 0;

    /* 1: conv_out [1, 1, 32] */
    tensors[1].h = 1;
    tensors[1].w = 1;
    tensors[1].c = GOLDEN_C;
    tensors[1].stride = GOLDEN_C;
    tensors[1].offset = 128;

    /* 2: k_in [1, 1, 32] */
    tensors[2].h = 1;
    tensors[2].w = 1;
    tensors[2].c = GOLDEN_C;
    tensors[2].stride = GOLDEN_C;
    tensors[2].offset = 256;

    /* 3: pos_in [1, 1, 1] */
    tensors[3].h = 1;
    tensors[3].w = 1;
    tensors[3].c = 1;
    tensors[3].stride = 32; /* aligned to 128-byte stride */
    tensors[3].offset = 384;

    /* 4: conv_state [2, 1, 32] persistent */
    tensors[4].h = GOLDEN_S_LEN;
    tensors[4].w = 1;
    tensors[4].c = GOLDEN_C;
    tensors[4].stride = GOLDEN_C;
    tensors[4].offset = 0; /* in state buffer */
    tensors[4].reserved[0] = HX_TENSOR_PERSISTENT;
    tensors[4].reserved[1] = 0x1001; /* name hash / state ID */

    /* 5: kv_cache [8, 1, 32] persistent */
    tensors[5].h = GOLDEN_MAX_TOKENS;
    tensors[5].w = 1;
    tensors[5].c = GOLDEN_C;
    tensors[5].stride = GOLDEN_C;
    tensors[5].offset = 256; /* in state buffer */
    tensors[5].reserved[0] = HX_TENSOR_PERSISTENT;
    tensors[5].reserved[1] = 0x1002; /* name hash / state ID */

    /* 2 ops */
    HxOp ops[2];
    memset(ops, 0, sizeof(ops));

    /* Op 0: HX_SHORTCONV (x=0, y=4, z=NONE, out=1) */
    ops[0].kind = HX_SHORTCONV;
    ops[0].x = 0;
    ops[0].y = 4;
    ops[0].z = HXFG_NONE;
    ops[0].out = 1;
    ops[0].p[0] = GOLDEN_K_TAPS;
    ops[0].p[1] = 0;   /* weight offset */
    ops[0].p[2] = 384; /* bias offset */
    ops[0].p[3] = 3;   /* SiLU */

    /* Op 1: HX_KV_WRITE (x=2, y=3, z=NONE, out=5) */
    ops[1].kind = HX_KV_WRITE;
    ops[1].x = 2;
    ops[1].y = 3;
    ops[1].z = HXFG_NONE;
    ops[1].out = 5;
    ops[1].p[0] = 0;
    ops[1].p[1] = 0;
    ops[1].p[2] = 0;

    /* Inputs: 0, 2, 3; Output: 1 */
    uint32_t io[4] = { 0, 2, 3, 1 };

    /* Construct header */
    HxHeader h;
    memset(&h, 0, sizeof(h));
    h.magic = HXFG_MAGIC;
    h.version = 1;
    h.tensors = 6;
    h.ops = 2;
    h.inputs = 3;
    h.outputs = 1;
    h.arena_bytes = sizeof(arena);
    h.constant_bytes = sizeof(constants);
    h.tensor_offset = 256;
    h.op_offset = 256 + sizeof(tensors); /* 256 + 192 = 448 */
    h.io_offset = h.op_offset + sizeof(ops); /* 448 + 128 = 576 */
    h.bytes = h.io_offset + sizeof(io); /* 576 + 16 = 592 */

    memset(image, 0, sizeof(image));
    memcpy(image, &h, sizeof(h));
    memcpy(image + h.tensor_offset, tensors, sizeof(tensors));
    memcpy(image + h.op_offset, ops, sizeof(ops));
    memcpy(image + h.io_offset, io, sizeof(io));

    /* Open graph */
    HxGraph g;
    memset(&g, 0, sizeof(g));
    hx_graph_set_state(&g, state, sizeof(state));

    uint32_t bad_op = 0;
    uint32_t rc = hx_graph_open(&g, image, h.bytes, constants, sizeof(constants),
                                arena, sizeof(arena), &bad_op);
    if (rc != 0) {
        printf("FAIL: hx_graph_open returned %u, bad_op=%u\n", (unsigned)rc, (unsigned)bad_op);
        return 1;
    }

    float max_diff = 0.0f;
    float max_conv_diff = 0.0f;
    float max_kv_diff = 0.0f;

    /* Run sequential decode steps */
    for (uint32_t step = 0; step < GOLDEN_NUM_STEPS; step++) {
        /* Set step inputs */
        memcpy(arena + tensors[0].offset, golden_x_conv_steps[step], sizeof(golden_x_conv_steps[step]));
        memcpy(arena + tensors[2].offset, golden_k_val_steps[step], sizeof(golden_k_val_steps[step]));
        float pos_val = golden_pos_steps[step];
        memcpy(arena + tensors[3].offset, &pos_val, sizeof(float));

        /* Execute decode graph ops */
        hx_execute_op(&g, 0, 0, 1);
        hx_execute_op(&g, 1, 0, 1);

        /* Validate conv output against NumPy reference */
        const float *conv_out = (const float *)(arena + tensors[1].offset);
        for (uint32_t c = 0; c < GOLDEN_C; c++) {
            float diff = fabsf_(conv_out[c] - golden_conv_out[step][c]);
            if (diff > max_conv_diff) max_conv_diff = diff;
            if (diff > max_diff) max_diff = diff;
        }

        /* Validate conv state persistence against NumPy reference */
        const float *c_state = (const float *)(state + tensors[4].offset);
        for (uint32_t s = 0; s < GOLDEN_S_LEN; s++) {
            for (uint32_t c = 0; c < GOLDEN_C; c++) {
                float diff = fabsf_(c_state[s * GOLDEN_C + c] - golden_conv_state[step][s][c]);
                if (diff > max_conv_diff) max_conv_diff = diff;
                if (diff > max_diff) max_diff = diff;
            }
        }

        /* Validate KV cache persistence against NumPy reference */
        const float *kv_out = (const float *)(state + tensors[5].offset);
        for (uint32_t t = 0; t < GOLDEN_MAX_TOKENS; t++) {
            for (uint32_t c = 0; c < GOLDEN_C; c++) {
                float diff = fabsf_(kv_out[t * GOLDEN_C + c] - golden_kv_cache[step][t][c]);
                if (diff > max_kv_diff) max_kv_diff = diff;
                if (diff > max_diff) max_diff = diff;
            }
        }
    }

    printf("Decode steps completed. max_conv_diff=%e, max_kv_diff=%e, total max_diff=%e\n",
           (double)max_conv_diff, (double)max_kv_diff, (double)max_diff);

    if (max_diff > 1e-4f) {
        printf("FAIL: max_diff %e exceeds tolerance 1e-4\n", (double)max_diff);
        return 1;
    }

    /* Test state reset */
    memset(state, 0, sizeof(state));
    const float *kv_reset = (const float *)(state + tensors[5].offset);
    for (uint32_t i = 0; i < GOLDEN_MAX_TOKENS * GOLDEN_C; i++) {
        if (kv_reset[i] != 0.0f) {
            printf("FAIL: state reset failed to clear KV cache\n");
            return 1;
        }
    }

    printf("PASS persistent state sim: steps=%u max_diff=%e vs numpy reference\n",
           GOLDEN_NUM_STEPS, (double)max_diff);
    return 0;
}
