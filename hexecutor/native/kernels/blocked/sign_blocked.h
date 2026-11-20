/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_SIGN_BLOCKED_H
#define QD_SIGN_BLOCKED_H
#include "linearclip_u16.h"
#include "sign_levels.h"

typedef struct {
    QdBlockedU16 in,out;
    uint32_t cookie;
} QdBlockedSignU16;

/* Preparation checks only touched block entries. Touched output blocks must be
 * injective and disjoint from input, except exact same-descriptor in-place
 * operation, which is safe because each unique physical vector is read before
 * it is written. */
int qd_blocked_sign_u16_prepare(const QdBlockedU16 *in,const QdBlockedU16 *out,
        QdBlockedSignU16 *prepared);
int qd_blocked_sign_u16_apply(const QdBlockedSignU16 *prepared,int32_t input_zero,
        const QdSignLevels *levels);

#endif
