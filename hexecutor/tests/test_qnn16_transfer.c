/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "../native/qnn16/qnn16_transfer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    uint8_t src[3 * 16], a[16], b[16], c[16];
    uint8_t *dst[] = {a, b, c};
    for (uint32_t i = 0; i < sizeof src; ++i) src[i] = (uint8_t)(i * 17u + 3u);
    memset(a, 0, sizeof a);
    memset(b, 0, sizeof b);
    memset(c, 0, sizeof c);
    if (!qnn16_copy_chunks(src, 16, dst, 3, 16) ||
        memcmp(a, src, 16) || memcmp(b, src + 16, 16) || memcmp(c, src + 32, 16)) {
        puts("FAIL qnn16 strided copy");
        return 1;
    }
    qnn16_transfer_fence();
    if (!qnn16_zero_chunks(dst, 3, 16)) {
        puts("FAIL qnn16 zero");
        return 1;
    }
    for (uint32_t i = 0; i < sizeof a; ++i)
        if (a[i] || b[i] || c[i]) {
            puts("FAIL qnn16 zero contents");
            return 1;
        }
    if (qnn16_copy_chunks(src, 8, dst, 3, 16) ||
        qnn16_zero_chunks(dst, 3, 0)) {
        puts("FAIL qnn16 invalid descriptor accepted");
        return 1;
    }
    puts("PASS qnn16 transfer chunks");
    return 0;
}
