#ifndef QD_BLOCKED_LINEARCLIP_U16_H
#define QD_BLOCKED_LINEARCLIP_U16_H

/* Direct blocked-storage u16 linearclip. No dense tensor is materialized.
 * Blocks use QNN's 2 KiB u16 crouton: 8x4x32, with adjacent x positions
 * interleaved as halfwords (offset = row*256 + (x%4)/2*128 + (x%2)*2 + c*4).
 * See README.md for the eligibility predicate and alias semantics. */
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t *const *pools; /* indexed by encoded block's low 7 bits */
    const uint32_t *blocks;
    uint32_t block_count;
    uint32_t slice;        /* block depth slice, as in qd_block_u16 */
    int32_t meta[4];       /* full [batch,height,width,depth] metadata */
    int32_t dims[4];       /* logical region */
    int32_t origin[4];     /* logical region origin */
} QdBlockedU16;

typedef struct {
    int32_t multiplier, input_zero, output_zero;
    uint32_t shift;
    uint8_t reserved;      /* output is u16 in this first scoped kernel */
} QdLinearclipParams;

/* Prepared objects snapshot descriptors and params. Pool pointers, block-table
 * contents, and pointed-to storage must remain immutable and alive until the
 * prepared object is discarded. Prepare once per command; apply is O(tensor)
 * and does not repeat the whole-table overlap proof. */
typedef struct {
    QdBlockedU16 in, out;
    QdLinearclipParams params;
    uint32_t cookie;
} QdBlockedU16Linearclip;

/* Returns 0 for unsupported or unsafe metadata, 1 after transforming.
 * Inputs and outputs must be non-overlapping. Missing input blocks read zero;
 * missing output blocks discard stores. Valid logical elements only are read
 * or written. Caller must prove physical block injectivity (README predicate).
 */
int qd_blocked_linearclip_u16_prepare(const QdBlockedU16 *in,
        const QdBlockedU16 *out, const QdLinearclipParams *p,
        QdBlockedU16Linearclip *prepared);
int qd_blocked_linearclip_u16_apply(const QdBlockedU16Linearclip *prepared);
/* Reuse prepared geometry/alias proof with runtime-derived parameters. This
 * repeats only the inexpensive affine range proof. */
int qd_blocked_linearclip_u16_apply_params(const QdBlockedU16Linearclip *prepared,
        const QdLinearclipParams *p);
/* Checked convenience API: prepare + apply on each call. */
int qd_blocked_linearclip_u16(const QdBlockedU16 *in,
        const QdBlockedU16 *out, const QdLinearclipParams *p);

/* Independent scalar oracle used by the ISA simulator test. */
uint32_t qd_linearclip_u16_oracle(uint16_t x, const QdLinearclipParams *p);

#endif
