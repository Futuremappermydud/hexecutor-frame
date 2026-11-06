# Direct3 A allocation decision (offline)

`plan_direct3_a.py` inspects the actual primary and alternate QDRX images and the direct-3x3 eligibility census. It is read-only and performs no DSP/device access.

## Finding

A global rearrangement of pool-1 is not necessary to give each direct3 call compact A. Stage only that call's X page refs into the reserved tail, preserving aliases by mapping repeated source pages to the same compact slot. Keep W and Fi at their existing direct model addresses and keep Y at its graph output pages. Build the temporary A page table from the compact mapping, issue/drain the leaf, then reuse the tail on the next command. The source scatter experiment at sequence 247 independently exercised a non-monotone table with 129 unique pages; the actual layout note also records 97 unique pages for the 108-ref seq247 profile. These are different table/shape footprints, not contradictory counts.

The offline planner's current eligibility census covers 510 primary and 524 alternate metadata-eligible calls (it does not claim coverage of the other S1 commands or unsupported direct3 profiles). Across those exact page-table slices, the maximum is 240 table refs but only 193 unique physical A pages (sequence 676 / 679), or 395,264 bytes. Every other direct3 call needs at most 129 unique pages; 97-page calls dominate (315 primary, 331 alternate). The stated tail budget is 251 pages (514,048 bytes), so the worst A copy leaves 58 pages / 118,784 bytes below that usable budget. The physical tail is 256 pages (524,288 bytes); this leaves 63 pages / 129,024 bytes before runtime guard/metadata reservations. The planner enforces actual page alignment, pool-1 placement, and actual sliced page table length.

This capacity proof is per call, not a whole-graph allocator proof. It assumes the eligibility census's page count calculation and the current direct3 contract. At runtime preflight must revalidate tail authority/extent, exact source refs, destination disjointness from Y and all live data, guard/zero-Fi/output scratch placement, cache clean/publication, and worker completion before reuse. Any call exceeding 251 unique pages must fail closed. Do not pack duplicate refs twice; that spends 240 pages at the worst current profile and reduces the margin.

## Lifetimes and feasibility limits

The companion `POOL1_3X3_LAYOUT_ANALYSIS.md` conservatively pins all max-shape block refs through first-to-last tensor command references. At seq247 this marks all 3,840 active pool pages occupied by live blocked views, although A+W+Fi uses only 116 pages. The same analysis explicitly says this is a conservative live-view union, not proven simultaneous hardware liveness: it preserves max footprints and view aliases, and does not prove partial-write dead regions reusable. Its 3,724 other blocked pages therefore reject a simple two-bank static split, but do not prove that no schedule-aware allocator exists.

For the practical call-local strategy, the relevant capacity proof is exact table footprint, while correctness still relies on command-serial execution and reuse only after HMX completion. The staged bytes per call are 18,432–395,264 bytes (9–193 pages). Across eligible calls the planner totals 99,725,312 bytes (95.11 MiB) primary and 103,047,168 bytes (98.27 MiB) alternate; copy bandwidth and end-to-end latency have not been measured. This strategy does not establish a 20 ms result.

## Integration outline

1. Executor preflight validates every call against the same shape/table census and computes unique page count preserving first occurrence order.
2. For each call, copy each unique 2 KiB source page to tail slots, map all aliases in A's temporary table, and leave W/Fi/Y addresses intact.
3. Ensure the destination region does not intersect output/guards/zero-Fi/other tail reservations; clean/publish copied pages and table metadata as required by the runtime.
4. Invoke and drain HMX; verify guards; only then release/reuse staged A slots.
5. Reject unsupported geometry, more than 251 unique pages, unexpected aliases/offsets, or failed completion before graph progression.

The graph-wide conservative blocked/direct-page census is useful for identifying fixed-address pressure, but is not itself an allocator proof. A true relocated allocation needs producer/consumer and alias/view interference edges plus exact partial-write semantics; no such allocator is claimed here.
