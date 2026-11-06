# Resident layout kernel audit

Offline source and schedule audit for the exact compiled resident images:

| Image | Commands | SHA-256 | InputSlicePad commands |
|---|---:|---|---:|
| `frame/build/resident-primary.qdrx` | 5,068 | `dc8fcddcf4876d11f7a717ec4f111e1d3d3a40d4f3163758b590234e0b50b92b` | 105 |
| `frame/build/resident-alternate.qdrx` | 5,040 | `54b2949bbb25655ee459c903432f67cacd26e66519854dcb9745a91777eb7298` | 99 |

The attribution rows below come from `.work/frame-direct-kernels-20261030/resident-blocked-add-full.json`, before the recent layout bypasses. Root reports the InputSlicePad full matrix at 896/897 ms warm and the subsequent im2col full matrix at 866.35/867.03 ms. The first correct space-to-depth integration regressed its command stage from 44 ms to 101.4 ms, so root disabled it by default. The prepared/tiled prototype below targets the repeated proof, global cache prefetch, and per-pixel copy costs; it is not integrated or hardware-timed.

## Ranked layout drains

| Rank | Kernel / profile attribution | Primary / alternate commands | Exact schedule layout and next action |
|---:|---|---:|---|
| 1 | `k_im2col`: 55.7 ms; CB/FB 49.9 ms, CH/FH 5.7 ms | 80 + 20 / 80 + 20 | All 80 u8 inputs are direct `[1,10,258,1]` and outputs are blocked `[1,8,256,9]`. The 20 u16 inputs are direct `[1,32,256,2]`; outputs are blocked `[1,8,64,32]`. Current code materializes `t_read`, a second dense im2col result, then blocked `t_write`. A strict fused u8 prototype now covers the observed 3x3/stride-1 case and writes crouton padding directly. All 80 primary and alternate u8 commands pass the geometry, pool-bounds, uniqueness, and source-alias census, including output block-list slice offsets. |
| 2 | `k_input_slice_pad`: 46.5 ms | 105 / 99 | Every output is u8 direct `[1,10,258,1]` in pool 1 (270,900 B total primary; 255,420 B alternate). Commands use external graph input 0 or 1. Replace the dense scratch + full `t_write` with one direct fill and copies of contiguous valid source spans. This is the best quick exact bypass: it has full observed schedule eligibility, straightforward row semantics, and avoids an allocation, a full intermediate write, and a second full transfer per command. |
| 3 | `k_space_to_depth`: 44.0 ms | 174 / 174 | All inputs and outputs are blocked u8, 32→128 channels. A strict direct-blocked prototype now covers every primary/alternate call after proving touched input/output blocks are disjoint. It uses an 8 KiB L2 source tile and writes output croutons in place, preserving padded width and partial block contents. |
| 4 | `k_transpose`: 47.6 ms | 111 total: CB 82, CH 18, FB 10, FH 1 / same | CB/CH outputs are blocked; FB/FH are direct. The common CB case is u8 `[1,1,64,128]` → `[1,1,128,64]`; the CH cases are u16 shapes with 32/64 channel axes. `k_transpose` currently reads the entire source to dense, performs scalar element gathers, then writes the full result. Prioritize exact 2D u8 and u16 blocked tile transposes after the direct InputSlicePad bypass. The direct FB/FH set is small and uses several permutations, so guard by permutation and alignment rather than adding a broad dispatch. |

These measured rows are command-profile attribution, not exclusive kernel CPU time. Use a new latest-engine profile to judge savings; don't add the nested preparation/read/write counters from the audit to these rows.

## InputSlicePad direct bypass

`k_input_slice_pad` currently creates dense scratch, fills it with `fill`, copies the intersecting source window, then calls `t_write`. The exact copy interval is already represented by `lo[i] = start[i] - pad_before[i]`, `s0 = max(lo, 0)`, and `s1 = min(lo + output_dim, input_dim)`, clipped so `s1 >= s0`.

The prototype in `input_slice_pad_direct.h` writes the contiguous u8 direct destination in place: one fill of the full output and then one `memcpy` per intersecting batch/height row when all channels are copied, or one copy per width position for partial-channel cases. It does not allocate an intermediate. It declines before writing if capacities do not match the four-dimensional shape, pointer-range arithmetic overflows, or source and destination overlap; the current kernel remains the fallback in those cases. The runtime validates QDRX args, `QDR_SHAPE_VALUE`, scalar fill, dtype and graph input byte count first. Its fast-path guard is limited to direct non-crouton output with `d=m=output_shape` and zero tensor padding, and bounds-checks the direct pool pointer/offset against pool bytes before dispatch.

The helper leaves input/output shapes and command footprints unchanged. In-place alias cases intentionally fall back, because the dense implementation reads all input before writing the output. The helper's region math accepts crop, padding on any axis, empty intersections, and nonzero fill. Root integrated this guarded path; its full correctness matrix passed at 896/897 ms warm primary/alternate.

## HVX/compiler notes

`qd_glue_hvx.h` operates on physical 128-byte vectors. Its direct u8/u16 block movers are selected only when depth dimensions and origins meet the 32-lane alignment checks in `blocked_xfer`; otherwise the generic fragment path runs. Predicated stores use an aligned vector base, while VMEMU loads may start unaligned. A new kernel should optimize away dense staging before trying to hand-schedule the existing block transfer: the source currently performs redundant dense reads/writes at the layout operators themselves.

For a fused blocked kernel, keep short tails and missing blocks defined exactly. The u8 physical block is arranged in 32-byte width lanes; u16 uses two width positions per vector and interleaved halfwords, with width parity selecting the stored half. `t_write_region` only writes its region, so any new direct path must not touch neighboring blocks, padding, or aliases outside the current output footprint.

## Simulation

The host C simulator exhaustively compares the direct InputSlicePad helper against the current scalar reference over 5,308,416 small-shape/start/crop/pad combinations, including empty intersections, the full-channel row-copy path, and nonzero fill. It also verifies overlapping-buffer and wrong-capacity rejection leave memory untouched.

```sh
cc -O2 -std=c11 -Wall -Wextra -Werror \
  frame/layout_audit/input_slice_pad_direct_sim.c \
  -o /tmp/input_slice_pad_direct_sim
/tmp/input_slice_pad_direct_sim
```

Expected output: `PASS exhaustive crop/pad cases=5308416; alias/capacity guards`.

The separate fused u8 im2col simulator compares the blocked result against a dense reference followed by crouton packing for 32 input patterns. It verifies all 23 padded lanes are zero and checks nonzero block-list slice offsets, invalid pools 7/127 against seven-entry arrays, 128-byte but non-2048-byte-aligned offsets, overlapping physical ranges across aliasing pool bases, source/output alias rejection, duplicate blocks, and callback-failure preflight without output mutation. It also passes under AddressSanitizer.

```sh
cc -O2 -std=c11 -Wall -Wextra -Werror \
  frame/layout_audit/im2col_u8_crouton_direct_sim.c \
  -o /tmp/im2col_u8_crouton_direct_sim
/tmp/im2col_u8_crouton_direct_sim
```

Expected output: `PASS fused im2col u8 geometry, slice, zero padding, pool/bounds/alias guards`.

Root integrated this im2col path and reports its full correctness matrix at 866.35/867.03 ms warm primary/alternate.

## Space-to-depth direct blocked prototype

`space_to_depth_u8_direct.h` implements the exact observed u8 blocked family with 32 input channels and 128 output channels. `qd_layout_s2d_prepare` performs geometry, range and alias proof once for a command and stores only immutable encoded block IDs and row spans, never cache pointers. `qd_layout_s2d_apply` checks seven pool identities, gathers one output 8x8 tile's source region into an 8 KiB caller-provided L2 buffer, then writes its four output depth blocks. This removes the full `t_read_region` tensor and dense output snapshot while bounding scratch at 8 KiB. Gather now copies each contiguous horizontal row span with one `memcpy`; source fragments are copied into the local tile before asking the block cache for another block. An output cache pointer is held only while writing that block; the direct-mapped cache cannot evict a pointer the helper still uses. The profiled API reports one-time `proof_ticks`, per-call `gather_ticks` and `scatter_ticks`, plus callback counts and bytes. `prefetch_ticks` and `prefetch_block_calls` are zero because the optimized API removes global cache prefetch. The default API has no clock calls.

The helper preserves a detail that a generic stride-2 rewrite could miss: `ow = min((i.m[2] - pw) / 2, o.m[2])`, which can be larger than `o.d[2]`. For example, output shape width 66 can write width 68 into padded output max width 72. The helper writes the computed `ow`, as the existing `t_write_region` does, and leaves remaining width/height tails in partial physical blocks untouched. The S2D source loop order is `(dy, dx) = (0,0),(0,1),(1,0),(1,1)` for each 32-channel group. `S2_TAPS` is used in weight permutation code, not by this transform.

The QDRX census found all **174 primary and 174 alternate** calls match this u8/32→128 family. Their physical input read-block rectangles and output write-block rectangles are in bounds and disjoint; output blocks are physically unique. Some input views repeat a physical block, which is safe for read-only gather. The decoder-backed C fixtures cover 53 primary and 39 alternate distinct shape/view layouts, including block-list slice offsets.

The simulator compares every output physical block byte against a dense `t_read_region` + current S2D loop + blocked `t_write_region` oracle. It exercises the real 2048-byte block-cache policy, confirms input blocks remain unchanged, checks output holes and tails, and rejects bad pool IDs, bad ranges, 128-byte-only offsets, input/output aliases, duplicate output blocks, overlapping pool bases, a failing first input callback, and short scratch before writes. It also passes under AddressSanitizer. The decoded C fixtures prove that prepare accepts all 348 QDRX commands; apply then reuses each prepared plan. Run:

```sh
python3 frame/layout_audit/make_space_to_depth_cases.py
cc -O2 -std=c11 -Wall -Wextra -Werror \
  frame/layout_audit/space_to_depth_u8_direct_sim.c \
  -o /tmp/space_to_depth_u8_direct_sim
/tmp/space_to_depth_u8_direct_sim
cc -O1 -g -std=c11 -Wall -Wextra -Werror -fsanitize=address \
  frame/layout_audit/space_to_depth_u8_direct_sim.c \
  -o /tmp/space_to_depth_u8_direct_sim_asan
/tmp/space_to_depth_u8_direct_sim_asan
```

Both simulator builds print `PASS full QDRX physical oracle eligible=348/348; cache, holes/tails, alias guards`. The optimized prepare/apply API remains a standalone prototype; it is not connected to the runtime or target-timed. The original one-shot wrapper is retained for compatibility but recomputes proof each call and should not be used for production measurements.
