/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_PROTOCOL_H
#define QD_PROTOCOL_H
#include <stdint.h>
#define QD_MAGIC 0x31564451u /* QDV1 */
#define QD_VERSION 1u
#define QD_ARENA_OFFSET 0x01800000u
#define QD_ARENA_BYTES 0x00800000u
#define QD_CODE_BASE 0x87605000u
#define QD_CHUNK_K 256u
#define QD_STATUS_OFFSET 128u
#define QD_DATA_OFFSET 4096u
#define QD_MAX_M 8192u
#define QD_MAX_N 1024u
#define QD_MAX_K 8192u
/* All addresses are offsets within the dedicated arena. Little endian.
 * A: signed int16 [M,K]. B: signed int32 [K,N], weights restricted to [-128,127].
 * C: signed int32 [ceil(K/256),M,N]. N must be a multiple of 32.
 * Partial sums cannot overflow int32; the host combines them in int64. */
typedef struct {
    uint32_t magic, version, sequence, arena_bytes;
    uint32_t m, n, k, a_offset, b_offset, c_offset;
    uint32_t reserved[6];
} QdCommand;
/* Each worker writes error, rows and reserved before publishing done_sequence. */
typedef struct { uint32_t done_sequence, error, rows, reserved; } QdStatus;
enum { QD_OK=0, QD_BAD_HEADER=1, QD_BAD_SHAPE=2, QD_BAD_RANGE=3 };
#endif
