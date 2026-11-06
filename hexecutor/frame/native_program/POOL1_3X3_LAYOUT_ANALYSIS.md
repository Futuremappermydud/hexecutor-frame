# Pool-1 layout analysis for Xwb 3x3 sequence 247

Offline analysis only. No runtime or pool mapping was changed for this study.
The target is primary-model schedule command 171 / sequence 247, `ConvLayer_s1.opt@CB*2.Xwb.Fi.t`.

## Target components

- Input X: logical `[1,10,130,64]`, physical max `[1,24,144,64]`, padded H/W by 7.
- HMX input page grid: `3 rows × 18 columns × 2 channel pages = 108 references`.
- The input block table has 108 refs, 97 unique physical pages, and 11 repeated refs. The row-start page refs are all page 1834, so row-start deltas are zero.
- Original X addresses span pages 432–3735 (6,766,592 bytes); maximum adjacent input-table address jump is 2,733 pages / 5,597,184 bytes, at reference positions 63→64 (pages 999→3732).
- The 3×3 XWB weights are 36,864 bytes at pool-1 offset 3,762,176. Fi is 512 bytes at offset 3,799,040.
- The output descriptor has a block table with 136 entries and slice origin 100; this command touches 32 output pages.

## Pinning every direct tensor span

Across the model, 906 direct pool-1 tensor records cover 14,901,202 bytes before overlap. Their union intersects 2,425 of the 3,840 active 2 KiB pages, leaving 1,415 active pages that are not covered by any direct tensor span. The 55 unique X pages that intersect these direct spans form these pinned ranges:

| Page range | Pool-1 byte range | Pages |
|---|---:|---:|
| 1680–1687 | 0x348000–0x34c000 | 8 |
| 1720–1727 | 0x35c000–0x360000 | 8 |
| 1834 | 0x395000–0x395800 | 1 |
| 2865–2870 | 0x598800–0x59b800 | 6 |
| 3002–3003 | 0x5dd000–0x5de000 | 2 |
| 3189–3196 | 0x63a800–0x63e800 | 8 |
| 3206–3212 | 0x643000–0x646800 | 7 |
| 3344–3354 | 0x688000–0x68d800 | 11 |
| 3732–3735 | 0x74a000–0x74c000 | 4 |

Keeping all direct spans at their current addresses fixes the X span to at least
pages 1680–3735: 2,056 pages / 4,210,688 bytes. It also fixes one adjacent
input-table edge: positions 1→2 are page 1834→3344, a 1,510-page / 3,092,480-byte jump.
That is a lower bound on the maximum adjacent X address delta for any global
one-to-one permutation that leaves every direct span fixed.

A constructive static permutation reaches that lower bound for the target
sequence: move the ten unique movable pages referenced at positions 54–63 to
free pages 2611–2620, and the 32 unique movable pages referenced at positions
74–105 to free pages 1752–1783. The chosen destinations are outside all direct
spans. Keep every pinned page in place; extend these assignments to a bijection
of all other unpinned active pages. Rewrite every occurrence of each source
page in the expanded block tables, and permute page contents by the same map.
This preserves global block-page aliases and every direct tensor address. It
reduces the max adjacent input-page jump from 2,733 to 1,510 pages (5.60 MiB to
2.95 MiB) and the overall X span to the pinned lower bound of 4.01 MiB.
These are address-layout bounds, not a claim that the QNN kernel accepts the
remaining jump; no hardware delta threshold is established.

This static page permutation cannot meet a compact 251-page input allocation.
It needs to transform up to all 1,415 unpinned active pages, keep all direct
spans fixed, and still has the pinned 1,510-page edge.

## Session-local tail copy option

The sequence-247 X has 97 unique physical source pages. A command-local copy
can place those unique pages in 97 consecutive 2 KiB slots in the reserved
512 KiB tail, and build the 108-entry input table by mapping repeated source
pages to the same copied destination. The table's adjacent deltas then become
0 or +2,048 bytes, while row-start deltas remain zero. The 251-page capacity
reported for the tail is sufficient for 97 unique pages (or all 108 refs if
aliases are deliberately copied separately), with no change to existing direct
tensor spans or other graph block-table views. This is a bounded per-command
copy mapping rather than a global permutation; tail authorization, cache
publication and output/tail disjointness remain runtime integration checks.

At sequence 247, only four target X pages (3732–3735) overlap direct pool-1
spans whose tensors are live at that schedule point; the other target pages
that overlap a direct span are tied to tensors live at other schedule points.
That distinction may permit a schedule-aware allocator, but proving it would
require a full block-page interference graph across every producer, consumer,
view alias, and direct tensor interval. The conservative global permutation
above assumes all direct spans stay pinned for the whole session.

## Conservative schedule live-set bound

I also computed an interference-style lower bound for a two-bank allocator.
For each tensor, the interval starts at its producer command (or command 0
for graph inputs/constants) and ends at its last scheduled input use; graph
outputs stay live to the end. At each of the 606 native 3x3 commands in each
model, I unioned the actual pool-1 pages referenced by live direct tensors and
by each live blocked tensor's max-shape active block-table slice. Physical page
aliases are counted once, including repeated references. This deliberately
keeps max-sized shapes and entire page footprints live, even where logical
padding or a partial write might allow a tighter allocator.

| Model / target point | Live direct records / pages | Live blocked records / pages | A + W + Fi pages | Other blocked pages |
|---|---:|---:|---:|---:|
| Primary, seq 247 | 29 / 462 | 1,332 / 3,840 | 116 (97 A + 19 W/Fi) | 3,724 |
| Alternate, seq 180 | 31 / 543 | 1,336 / 3,840 | 116 (97 A + 19 W/Fi) | 3,724 |
| Primary peak direct-page union, seq 3900 | 615 / 948 | — | — | — |
| Alternate peak direct-page union, seq 3751 | 704 / 734 | — | — | — |
| Primary peak A + W + Fi union, seq 3527 | — | — | 338 | — |
| Alternate peak A + W + Fi union, seq 3957 | — | — | 338 | — |
| Primary peak other blocked union, seq 1125 | — | — | — | 3,769 |
| Alternate peak other blocked union, seq 1110 | — | — | — | 3,769 |

At both named target points, all 3,840 active pool-1 pages appear in the
conservative live blocked-view union. Thus the native input plus direct W/Fi
fit comfortably in one 4 MiB bank (116 pages at the target; at most 338 pages
over the 606 commands), but the other simultaneously-live blocked views would
need 7.27 MiB (3,724 pages) at those points. That exceeds a 4 MiB
bank's 2,048 pages before accounting for any other direct tensors. This rules
out the proposed simple two-bank split under the stated fixed-lifetime model;
it does not rule out schedule-aware reuse, relocating values at producer
boundaries, or shortening lifetimes after proving all partial-write/view
semantics. A practical allocator needs a real interference graph and proof of
those transformations rather than treating this conservative union as an
achievable allocation.
