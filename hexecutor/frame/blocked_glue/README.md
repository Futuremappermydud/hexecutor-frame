# Blocked u16 linearclip prototype

`linearclip_u16.c` applies the recovered integer epilogue directly through
QNN u16 crouton addresses. It does not create a dense tensor. The standalone
entry point is deliberately limited to u16 output; the u8 crouton has a
different layout and needs its own implementation.

Arithmetic is the exact recovered rule:

```text
clip_u16(((int64(x) - input_zero) * multiplier + 2^(shift-1)) >> shift
         + output_zero)
```

The caller may use this only if logical dims agree; W origins have the same
parity; depth origins agree modulo 32; all region bounds fit their respective
metadata; `shift` is 1..31; and the whole output block table is injective. It
must either be disjoint from
every input block or be the exact same physical mapping as input. The latter
permits safe in-place operation because each vector is loaded before its own
store and no other logical vector can alias it. Shifted partial overlap and
repeated output block aliases are rejected, preserving read-all-before-write
behavior. This proof is conservative across whole tables. Input holes read as
zero; output holes discard writes. Only addressed logical lanes are touched;
crouton padding and crop-external lanes are preserved.

Call `qd_blocked_linearclip_u16_prepare` once when command metadata is
established, then cache the returned prepared object and use
`qd_blocked_linearclip_u16_apply` for each execution. The prepared object
copies descriptors and parameters; the pool pointer array, block tables, and
pointed-to storage must remain alive and unchanged until it is discarded. The
checked convenience function performs prepare and apply on every call and is
intended for one-off use or tests. Preparation currently checks overlap in
O(input block count × output block count) time.
For runtime-computed parameters, `qd_blocked_linearclip_u16_apply_params`
rechecks only the affine bounds proof and reuses cached geometry and aliasing.

The caller must first validate QDR block encodings and bounds: `pools` must
have entries for encoded pool IDs 0 through 127, every nonzero encoding must
select a present pool, and its offset plus 2048 bytes must fit that pool. The
descriptor currently carries no per-pool byte lengths, so this bound cannot be
checked here. Metadata dimensions whose ceil-to-crouton calculation or total
block index would overflow are rejected. A zero dimension is accepted as an
empty no-op and writes no bytes.

The arithmetic runs directly on physical 128-byte crouton vectors. Each
vector holds two adjacent W positions interleaved by halfword, and a predicate
limits stores to the requested W parity and depth range. It removes dense
tensor transfers but has not been benchmarked. It intentionally returns
unsupported when the exact rounding intermediates cannot be proven to fit
signed 32-bit HVX lanes; the existing exact generic path remains the fallback.

## Recovered graph eligibility

Enumerating opcode 39 in the checked-in primary and alternate QDRX images
found 79 `linearclip` commands in each graph; 30 per graph are u16→u16. With
matching logical dims, W-origin parity, D-origin modulo 32, blocked storage,
and injective output tables, the conservative whole-table alias proof accepts
14 primary commands (source sequences 5595, 5592, 5669, 5673, 5674, 5676,
5666, 5667, 5732, 5733, 5734, 5747, 5750, 5751) and 19 alternate commands
(5543, 5544, 5548, 5551, 5542, 5631, 5632, 5636, 5639, 5642, 5640, 5641,
5683, 5685, 5696, 5697, 5708, 5710, 5712). The rest fail output injectivity
or have shifted partial overlap. None of the overlapping mappings is an exact
injective in-place mapping in these graphs.
