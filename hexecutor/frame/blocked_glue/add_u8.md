# Direct blocked u8 add

`qd_blocked_add_u8_prepare` caches the static geometry and touched-block alias
proof. Pool encodings/bounds must already have been validated by the caller;
this descriptor has no pool-length fields. Keep pool pointers, tables, and
storage alive and immutable for the prepared object's lifetime.

The apply rule is exactly
`clip_u8((a*ma + b*mb + 2^(shift-1)) >> shift)`. It accepts shifts 1..31 and
only coefficients whose complete unsigned 32-bit epilogue cannot overflow;
otherwise the caller should use the original generic path. Missing source
blocks read zero, missing output blocks skip stores, and stores preserve all
bytes outside the logical depth crop. The three tensors may have independent
metadata, batch/height/width origins, and slices. Their depth origins must be
congruent modulo 32 and their width origins modulo 4: each aligned 128-byte
HVX vector contains four adjacent 32-byte pixels. Partial first/last
four-pixel groups use predicated stores. Empty crops are no-ops.

Preparation checks only table entries touched by the crop. Touched output
blocks must be injective and disjoint from both touched inputs. This version
rejects in-place and shifted aliases. Unused entries may alias freely.

For `Add.tcm@CB*3`, the recovered graphs have 170 commands/model, matching
input/output dimensions, aligned depth origins, and all commands passed the
touched-region disjointness proof. They use nonnegative coefficients and shift
13. The latest pre-add profile totals 37.55 ms primary and 35.43 ms alternate.
The simulator covers all 65,536 byte input pairs at one coefficient set,
independent origins/metadata/slices, width and depth tails across croutons, missing
blocks, alias rejection, an independent integer oracle, and preservation of
every untouched output byte.
