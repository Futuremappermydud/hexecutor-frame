/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Host harness for Q16P programs: validates an image and runs operations with
 * the portable 1x1 reference in place of HMX. Used by test_qnn16_program.py
 * through ctypes; VTCM, exchange and scratch are caller-owned buffers. */
#include "program.h"
#include <string.h>

static Q16Program program;
static Q16Exec exec;

uint32_t q16_host_open(const void *image, uint32_t image_bytes, uint32_t constant_bytes,
                       uint32_t *bad_op) {
    return q16_program_open(&program, image, image_bytes, constant_bytes, bad_op);
}

void q16_host_bind(uint8_t *vtcm, uint8_t *exchange, uint8_t *scratch, const uint8_t *constants) {
    memset(&exec, 0, sizeof exec);
    exec.program = &program;
    exec.vtcm = vtcm;
    exec.exchange = exchange;
    exec.scratch = scratch;
    exec.constants = constants;
    exec.conv1x1 = q16_conv1x1_reference;
}

void q16_host_bind_spill(uint8_t *spill) { exec.spill = spill; }

uint32_t q16_host_step(uint32_t op) { return q16_exec_op(&exec, op); }
