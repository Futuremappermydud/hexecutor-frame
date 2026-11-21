/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Q16P v2 persistent DDR arena: bounds, execution and scratch isolation. */
#include "program.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    Q16Header h;
    Q16Tensor t[1];
    Q16Op op[2];
    Q16Io io[2];
} Image;

int main(void) {
    Image image = {0};
    Qnn16Shape shape = {1,1,1,2};
    image.h.magic = Q16P_MAGIC; image.h.version = Q16P_ARENA_VERSION;
    image.h.bytes = sizeof image; image.h.tensors = 1; image.h.ops = 2;
    image.h.inputs = image.h.outputs = 1;
    image.h.tensor_offset = sizeof(Q16Header);
    image.h.op_offset = image.h.tensor_offset + sizeof image.t;
    image.h.word_offset = image.h.op_offset + sizeof image.op;
    image.h.io_offset = image.h.word_offset;
    image.h.constant_bytes = 8; image.h.scratch_bytes = 256;
    image.h.exchange_bytes = 256; image.h.vtcm_bytes = Q16P_VTCM_BYTES;
    image.h.reserved[0] = 128;
    image.t[0].direct = 2; image.t[0].dtype = Q16_F32; image.t[0].bytes = 8;
    image.t[0].dims = image.t[0].full = shape;
    image.op[0].kind = Q16_COPY; image.op[0].out = 0; image.op[0].p[1] = 8;
    image.op[1].kind = Q16_OUTPUT; image.op[1].x = 0;
    image.io[0].dims = image.io[1].dims = shape;
    image.io[0].bytes = image.io[1].bytes = 8; image.io[1].offset = 128;
    Q16Program p; uint32_t bad;
    assert(q16_program_open(&p, &image, sizeof image, 8, &bad) == Q16_OK);
    uint8_t *vtcm = calloc(1, Q16P_VTCM_BYTES), scratch[256] = {0}, exchange[256] = {0};
    float values[] = {1.25f, -2.5f};
    Q16Exec e = {.program=&p, .vtcm=vtcm, .scratch=scratch, .exchange=exchange,
                 .constants=(const uint8_t *)values};
    assert(q16_exec_op(&e, 0) == Q16_OK);
    assert(q16_exec_op(&e, 1) == Q16_OK);
    assert(!memcmp(scratch, values, 8) && !memcmp(exchange + 128, values, 8));
    image.t[0].offset = 124;
    assert(q16_program_open(&p, &image, sizeof image, 8, &bad) == Q16_RANGE);
    image.t[0].offset = 0; image.h.scratch_bytes = 128;
    assert(q16_program_open(&p, &image, sizeof image, 8, &bad) == Q16_SCRATCH);
    image.h.scratch_bytes = 256; image.h.version = Q16P_VERSION;
    assert(q16_program_open(&p, &image, sizeof image, 8, &bad) == Q16_HEADER);
    /* Compact U16 ingress clips and rounds; egress uses the same scale and zero point. */
    image.h.version = Q16P_ARENA_VERSION;
    image.h.constant_bytes = 0;
    image.t[0].dtype = Q16_U16;
    image.t[0].bytes = 4;
    image.op[0] = (Q16Op){.kind = Q16_INPUT_U16, .out = 0, .p = {0, 0, 0, 0, 0x3e800000u, 10}};
    image.op[1] = (Q16Op){.kind = Q16_OUTPUT_U16, .x = 0, .p = {0, 0, 0, 0, 0x3e800000u, 10}};
    float raw_input[] = {-10.0f, 1.125f};
    memcpy(exchange, raw_input, sizeof raw_input);
    assert(q16_program_open(&p, &image, sizeof image, 0, &bad) == Q16_OK);
    assert(q16_exec_op(&e, 0) == Q16_OK);
    const uint16_t expected_codes[] = {0, 15};
    assert(!memcmp(scratch, expected_codes, sizeof expected_codes));
    assert(q16_exec_op(&e, 1) == Q16_OK);
    const float expected_output[] = {-2.5f, 1.25f};
    assert(!memcmp(exchange + 128, expected_output, sizeof expected_output));
    image.op[0].p[2] = 1;
    assert(q16_program_open(&p, &image, sizeof image, 0, &bad) == Q16_OPERATOR && bad == 0);
    image.op[0].p[2] = 0;
    image.op[1].p[4] = 0x7f800000u;
    assert(q16_program_open(&p, &image, sizeof image, 0, &bad) == Q16_OPERATOR && bad == 1);
    free(vtcm);
    puts("PASS Q16P persistent arena and quantized exchange");
    return 0;
}
