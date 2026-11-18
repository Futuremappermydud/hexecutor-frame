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
    free(vtcm);
    puts("PASS Q16P persistent arena");
    return 0;
}
