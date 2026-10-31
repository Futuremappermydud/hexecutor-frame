#ifndef QD_BLOCKED_LINEARCLIP_U8_U16_H
#define QD_BLOCKED_LINEARCLIP_U8_U16_H
#include <stdint.h>

/* QNN u8 crouton descriptor: 8x8x32, 2 KiB. */
typedef struct {
    uint8_t *const *pools;
    const uint32_t *blocks;
    uint32_t block_count, slice;
    int32_t meta[4], dims[4], origin[4];
} QdBlockedU8;

/* QNN u16 crouton descriptor: 8x4x32, adjacent W values interleaved. */
typedef struct {
    uint8_t *const *pools;
    const uint32_t *blocks;
    uint32_t block_count, slice;
    int32_t meta[4], dims[4], origin[4];
} QdBlockedU16Out;

typedef struct { uint32_t multiplier, shift; } QdLinearclipU8U16Params;
typedef struct {
    QdBlockedU8 in;
    QdBlockedU16Out out;
    uint32_t cookie;
} QdBlockedLinearclipU8U16;

/* Prepare performs the O(input blocks * output blocks) alias check once.
 * Caller validates encoded pool IDs and block ranges first; referenced pools,
 * tables and memory must remain alive/unchanged while prepared is used. */
int qd_blocked_linearclip_u8_u16_prepare(const QdBlockedU8 *in,
        const QdBlockedU16Out *out, QdBlockedLinearclipU8U16 *prepared);
/* Exact recovered rule: clip_u16((x*multiplier + 2^(shift-1)) >> shift). */
int qd_blocked_linearclip_u8_u16_apply(const QdBlockedLinearclipU8U16 *prepared,
        const QdLinearclipU8U16Params *params);
uint32_t qd_linearclip_u8_u16_oracle(uint8_t x,
        const QdLinearclipU8U16Params *params);
#endif
