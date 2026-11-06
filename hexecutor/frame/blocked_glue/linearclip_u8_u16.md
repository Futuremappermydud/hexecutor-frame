# Direct blocked u8→u16 linearclip

`linearclip_u8_u16.c` handles the recovered `linearclip@CH.CB.fi` rule:

```text
clip_u16((x * multiplier + 2^(shift-1)) >> shift)
```

It accepts `multiplier <= 65535` and `shift` 1..31, which keeps the positive
product and rounding addition within signed 32-bit HVX lanes. Other parameter
ranges are rejected for the exact generic fallback.

The source is the QNN 8×8×32 u8 crouton; output is the 8×4×32 u16 crouton.
Each source 128-byte vector contains four W positions. HVX unpacks it to two
halfword vectors, applies the multiply/round/shift, then shuffles each pair of
W values into one physical u16 output vector. Store predicates preserve rows,
W positions, depth tails, and crouton padding outside the logical crop.

Preparation requires matching logical dimensions, W origins congruent modulo
4, and depth origins congruent modulo 32. Each descriptor uses its own metadata,
batch/height origin, slice, and block table. Preparation enumerates only block
table entries addressed by that crop; distinct touched output entries must map
to distinct physical 2 KiB blocks, and touched input/output blocks must be
disjoint. Repeated block references outside the crop do not affect eligibility.
The check runs once; callers retain the prepared descriptor and pass current
runtime multiplier/shift to apply.

As with the u16→u16 prototype, the caller must validate QDR pool IDs and prove
each referenced 2 KiB range fits its pool before preparing. Missing source
blocks read as zero; missing output blocks discard stores. Empty regions are
no-ops. Overlapping touched input/output storage is rejected; this path has no
staging fallback.

The recovered primary and alternate QDRX images each contain 40 u8→u16
commands. The touched-region proof accepts all 40 in each graph: 32 source
blocks (65,536 bytes) and 64 output blocks per command, injective output, and
no touched cross-overlap. The simulator covers shifted metadata/origins/slices,
scattered blocks, exact scalar-oracle results, saturation, parity, tails,
guards, and output-alias rejection.
