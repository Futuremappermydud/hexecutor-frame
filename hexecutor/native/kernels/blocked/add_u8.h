/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_BLOCKED_ADD_U8_H
#define QD_BLOCKED_ADD_U8_H
#include "linearclip_u8_u16.h"
#include <stdint.h>

typedef struct { QdBlockedU8 a,b,out; uint32_t cookie; } QdBlockedAddU8;
typedef struct { uint32_t ma,mb,shift; } QdAddU8Params;

/* Prepare the static geometry/alias proof once. Pool encodings and bounds
 * must already be validated; metadata tables and storage stay immutable. */
int qd_blocked_add_u8_prepare(const QdBlockedU8 *a,const QdBlockedU8 *b,
        const QdBlockedU8 *out,QdBlockedAddU8 *prepared);
int qd_blocked_add_u8_apply(const QdBlockedAddU8 *prepared,const QdAddU8Params *p);
/* Whole-block fast form (all regions whole 8x8x32 blocks): 1 if eligible. */
int qd_blocked_add_u8_whole(const QdBlockedAddU8 *prepared);
int qd_blocked_add_u8_apply_whole(const QdBlockedAddU8 *prepared,const QdAddU8Params *p,uint32_t part,uint32_t parts);
uint8_t qd_add_u8_oracle(uint8_t a,uint8_t b,const QdAddU8Params *p);
#endif
