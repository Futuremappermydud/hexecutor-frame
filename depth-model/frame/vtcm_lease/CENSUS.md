# QDRX DDR/VTCM traffic census

Run `python3 frame/vtcm_lease/census_schedule.py` to decode
`frame/build/resident-primary.qdrx`. Pass an alternate QDRX path as the first
argument to census another exact schedule. The script reads the serialized
image through `resident_image.unpack`; it does not connect to hardware.

## Primary schedule

Image SHA-256: `dc8fcddcf4876d11f7a717ec4f111e1d3d3a40d4f3163758b590234e0b50b92b`
(5,068 commands; pool 1 is exactly 8 MiB).

| Scheduled operation | DDR → VTCM | VTCM → DDR | Basis |
|---|---:|---:|---|
| `weights_to_vtcm` (op 4) | 7,833,732 B | 0 B | 235 commands; each uses `min(count4(src.m)*esize, count4(dst.m)*esize)`. Pool edges: 7,178,372 B from pool 5 and 655,360 B from pool 4 into pool 1. |
| spill/fill (op 26) | 1,703,936 B | 1,703,936 B | `fv` byte counts for pool tag 1; `@Fill` is spill-DDR → pool 1, `@Spill` is pool 1 → spill-DDR. 62 fill and 26 spill commands. |
| explicit activation copy (op 17) | 6,144 B | 2,048 B | 3 pool-5 → pool-1 copies and 1 pool-1 → pool-4 copy, using full tensor extent. |
| **Scheduled DDR/VTCM total** | **9,543,812 B** | **1,705,984 B** | **11,249,796 B total transferred** |

The `op 4` and `op 17` estimates use each tensor's maximum shape because the
DSP kernels copy `m` extents. Spill/fill uses its serialized byte count per
descriptor. Other pool copies are not counted as VTCM transfers.

The graph inputs total **819,328 B** (host to private DDR); graph outputs total
**819,200 B** (private DDR to host). These are reported separately from VTCM
traffic. Including them gives 12,888,324 B of scheduled boundary traffic.

## 20 ms bandwidth context

At the measured bulk-copy rate of about 10 GB/s, the primary schedule's
11,249,796 DDR/VTCM bytes have an optimistic transfer-time floor of **1.125
ms**. Including graph I/O gives **1.289 ms**. That leaves about **18.875 ms**
of a 20 ms budget after DDR/VTCM transfers alone. This ignores latency, DMA
setup, command/barrier overhead, graph compute, HMX contention, eye-client
contention, cache effects, and imperfect overlap; it is not a 20 ms guarantee.

## Dense/HMX activation staging

The same schedule has 1,036 convolution commands. Mirroring the dense input
window calculation in `k_hmx_conv`, the script reports **67,020,544 B** of
activation materialization volume if every command takes the dense path.
Direct blocked HMX may bypass some or all of that materialization, so this is
an upper-bound staging volume, not an added required transfer total. With pool
1 resident in VTCM, those reads/materialization are local VTCM work; they must
not be counted as DDR/VTCM traffic. A native tensor-aware kernel can avoid the
dense intermediate, while HMX still needs its own packet/layout work. This
staging figure is intentionally kept outside the 11.25 MB required-transfer
sum.

For comparison, the alternate schedule contains 5,040 commands and has
10,480,772 B of scheduled DDR/VTCM traffic (1.048 ms at the same optimistic
rate), plus the same 1,638,528 B of graph I/O.
