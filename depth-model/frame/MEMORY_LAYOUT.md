# Primary model memory layout

How the primary program uses memory, what constrains scheduling, and what has been measured. Numbers are from `bench/lanes/mem_layout.py` and `bench/lanes/gen_qnn_lanes.py` unless marked otherwise.

## Pools

| Pool | Size | Holds | Where it lives during a run |
|---|---:|---|---|
| 1 | 8.00 MiB | Activations, per-run weight copies | VTCM |
| 3 | 0.17 MiB | Small flat tensors: scales, Fi words | DDR, cached by the L2 lease |
| 4 | 0.57 MiB | Small blocked and flat tensors | DDR, cached by the L2 lease |
| 5 | 6.88 MiB | Constants: weights, Fi tables | DDR, read-only |
| 6 | 1.69 MiB | Spill area | DDR, cached by the L2 lease |

Spill and Fill move 3.25 MB per run between pools 1 and 6.

## Two ways a tensor reaches memory

- **Blocked (`CB`, `CH`):** a table of 2 KiB blocks, each with its own address. Any block can sit anywhere on the 2 KiB grid, so these can be relocated one block at a time. An index is `((b*nbh + H/8)*nbw + W/cw)*nbd + slice + D/32`, with `cw` 8 for u8 and 4 for u16. A block is 8 rows of 256 bytes.
- **Direct (`FB`, `Xwb`, `Xpb`, `Fi`, `ff`):** one contiguous range at an offset. These can only move as a unit.

Spill, Fill and Zap declare no tensors. Their addresses are in the command's vector words.

**A block table covers a tensor's maximum dimensions, not what a command touches.** Example: tensor 1217 has dims `(1,1,64,128)`, max `(1,8,64,128)`, padding row 4: a 32-entry table, of which a command touches one row of each of its blocks. Across the program, block tables declare 1,015 MiB against 8 MiB of real memory.

## How full is VTCM?

| Measure | Value |
|---|---:|
| Distinct pool 1 blocks in block tables | 3,840 (7.5 MiB) |
| Live data, peak, in program order | 6.93 MiB, at command 1034 |
| Live data, mean | 2.08 MiB |
| Versions per 128-byte granule | 104 |
| Value lifetime in commands, median / p90 / max | 14 / 65 / 2,456 |

These use whole-table footprints, so they overstate liveness. Memory is tight only around one peak. For most of the run, far more than half of VTCM holds nothing live.

## What actually constrains the schedule

Each row uses the same measured command costs and one HMX list:

| Footprints and ordering | Conflict pairs | Critical path | Modeled finish, 2 worker lanes |
|---|---:|---:|---:|
| Whole block tables, QNN order on conflicts | 2,097,889 | 8.67 ms | 10.37 ms |
| Active blocks, QNN order on conflicts | 546,735 | 2.29 ms | 8.58 ms |
| Active rows, true data flow only | 10,042 flow edges | 1.16 ms | 7.70 ms |

- **Most false dependencies came from overstated footprints, not from QNN's memory plan.** Exact footprints gain about 1.8 ms in the model with no memory moved.
- **Reassigning memory could gain about 0.9 ms more** in the model, at most: the third row ignores capacity.
- **The critical path is no longer the limit.** At 2.29 ms it is far below the finish time. Main busy is about 7.0 ms of mostly serial conv work, so the single HMX list now sets the pace, and worker lanes wait on it rather than the reverse.

With exact footprints, QNN's order reverses no conflicting pair and leaves only 4 unordered, against 1,501 and 4,550 before. That is further evidence the footprints are right.

## Are active footprints safe?

| Check | Result |
|---|---|
| Poison test, 128-byte row footprints, smoke, host | 0 hidden reads across 5,068 commands |
| Real writes against **row** footprints, host | 38 commands write outside: 6/33, 6/32, 6/36, 17/16–18 |
| Real writes against **block** footprints, host | 0 commands write outside |
| Adversarial topological replays, block footprints, host | 18 of 18 byte-exact |

So row-level footprints are wrong for writes, and they add nothing in the model. **Use block level** (`--active block`).

**Limit:** the host runs different kernels from the Frame for some paths, including slide and plan replay. A Hexagon kernel writing outside its active blocks would be invisible to these checks, which is why the device runs adversarial serial orders too.

## Result on the Frame

Only the lane tables changed: no memory moved, and the kernels are the same.

| Test, build | Result |
|---|---|
| Smoke pairs, `r0` against `l8` | 10.93 against 9.64 ms |
| Full gate, 27 cases × 2, `l8` | **54/54 exact, 9.681 ms** |
| Jitter stress, `l8j` | 54/54 exact |
| Serial order "latest", `advl` | 54/54 exact |
| Serial order "random", `advr` | 54/54 exact |

The model predicted 10.37 -> 8.58 ms, a gain of about 1.8 ms. Measured gain: about 1.3 ms.
