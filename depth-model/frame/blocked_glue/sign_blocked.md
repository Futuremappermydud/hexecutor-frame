# Direct blocked u16 sign

`qd_blocked_sign_u16_prepare` caches a geometry and alias proof; call it once
while the pool vectors and block tables are immutable. `qd_blocked_sign_u16_apply`
then uses physical HVX vectors and freshly supplied input zero point and exact
precomputed levels. QDR block encodings and pool bounds must already have been
validated by the caller: this descriptor does not carry pool lengths.

Eligibility requires equal logical dimensions, W origins congruent mod 2, and
depth origins congruent mod 32. Metadata extents, batch/height origins, W and
depth origins otherwise may differ; each view computes its own table index and
physical offset. Missing input blocks read zero; missing output blocks skip the
store. Only logical crop lanes are stored, preserving padding. Empty crops are
successful no-ops.

The proof enumerates only block-table entries touched by the crop. Touched
output blocks must be pairwise nonoverlapping and must not overlap touched input
blocks. Exact identical descriptors are permitted in-place: injectivity means
each physical vector is read and written only for its own logical vector.
Shifted, partial, or repeated-block aliases are rejected. Untouched table
entries may alias freely.

The simulator test covers all 65,536 u16 values, guards, exact in-place,
repeated output aliases, an empty crop, a W crop crossing a 4-wide block
boundary, a depth crop crossing a 32-lane boundary, independent metadata and
origins (including W shifted by 2 and depth by 32), a missing input block,
and a missing output block. It checks that every byte outside the logical
output crop remains unchanged. Dense selector tests additionally
cover all 65,536 values at the four actual input zero points (9196, 10701,
61378, 65432). The recovered graph's large sign commands all use output levels
`(1, 32768, 65535)` and all 12 per model passed the touched-region disjoint
proof.
