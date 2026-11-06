# Executor memory layout

The executor's performance depends on where metadata, tensor values, stacks, and workspace live. These are distinct allocations. A compact program file is not a proof that all live tensors and concurrent scratch fit on the device.

## Private DDR and address ownership

`cdsp_mem` supplies private DDR at IOVA `0xa0000000`. The depth executor expects 128 MiB and installs eight 16 MiB mappings. `qd_owned_memory.h` checks for occupied mappings/aliases and saves/restores the previous TLB contents. It must not evict firmware entries to make a graph fit.

The configuration validator separates image/constants, graph pools and I/O, and the engine heap. In the current relative allocation contract, image/constants occupy the first 16 MiB, mutable pools/I/O the next 16 MiB, and the heap starts at relative `0x02000000`. Private page 7 contains L2-cacheable working metadata for the lease design. Consult the validator for exact bounds/alignment; these regions are not a general allocator for other clients.

## Tensor pools in the primary workload

| Pool | Recorded size | Meaning | During native execution |
| --- | ---: | --- | --- |
| 1 | 8.00 MiB | Activations and runtime weight copies | VTCM |
| 3 | 0.17 MiB | Small flat scales/Fi tensors | DDR, cacheable through the lease |
| 4 | 0.57 MiB | Small flat/blocked tensors | DDR, cacheable through the lease |
| 5 | 6.88 MiB | Immutable constants | Read-only DDR |
| 6 | 1.69 MiB | Spill area | DDR, cacheable through the lease |

These are measured model requirements, not universal sizes. The recorded schedule moves about 3.25 MB between pools 1 and 6 per inference. Runtime-generated weights and scales remain mutable tensors even when their addresses are stable.

## VTCM and blocked addressing

The verified pool-1 window is `0x31000000..0x31800000`, 8 MiB. `CB`/`CH` tensors use tables of 2 KiB blocks. A block has eight 256-byte rows; the logical column width differs for u8 and u16. Direct tensors (`FB`, `Xwb`, `Xpb`, `Fi`, `ff`) instead occupy a contiguous pool range. Slice origins, maximum dimensions, and padding are part of the address calculation.

Block tables describe maximum geometry, not the subset one command accesses. Summing declared block tables counts aliases repeatedly and greatly overstates both liveness and dependencies. In the primary schedule, tables mention roughly 1,015 MiB while distinct pool-1 blocks occupy 7.5 MiB. Whole-table peak live data was estimated at 6.93 MiB; that is conservative, not a grant of permanent free workspace.

All 3,840 model blocks are touched somewhere in the run. The tail has 504 KiB after its control area. Larger scratch needs a proved dead interval and ordering edges, not an assumption that quiet memory is free. Once pool 1 is VTCM, legacy convolution staging at the same base cannot coexist with it.

## L2-TCM/cache lease

The L2 array is 1 MiB at `0x30000000..0x30100000`. `QDS_L2_CACHE=1` converts the upper 512 KiB to cache and preserves the lower half as TCM. Mode 5 converts the entire array; no L2-TCM address is usable while that lease is active. Mixing these maps caused real faults and large slowdowns.

For the half-cache layout, the source defines:

| Range | Role |
| --- | --- |
| `0x30000000..0x30002000` | Native tables/ABI state |
| `0x30002000..0x3003c000` | Hot bound program; about 235 KB in the recorded primary build |
| `0x3003c000..0x3006c000` | Three 64 KiB lease stack slots |
| `0x3006c000..0x30073e00` | Two helper convolution slots |
| `0x30074000..0x30074200` | Shared work record |
| `0x30078000..0x30080000` | Main convolution near state |
| `0x30080000..0x30100000` | Upper half saved before conversion to cache |

Expanded block tables reside in private cached DDR rather than consuming the hot program interval. A fourth physical worker uses separately defined private stack/context storage. The address macros in `qd_l2_cache.h` and `qd_graph_dsp.h` are authoritative if a build selects another mode.

The lease must clean scalar cache lines, preserve the TCM region that becomes cache, switch mode with the required ordering, execute on valid stacks, publish dirty data, return to TCM mode, restore contents, and verify preservation. The backup address is `0xf1c00000`. Later recipes use a signature check; disabling verification is a separate correctness tradeoff, not an ordinary speed flag.

Repeated work showed that uncached DDR stacks and scalar near-state placement could erase the cache benefit. Scalar access cost, vector access cost, and coherency differ. Cache attributes, alias views, stack location, and transition code must be reviewed together. A byte-count budget alone does not validate the placement.

Source: [owned DDR](../../hexecutor/frame/qd_owned_memory.h), [lease map](../../hexecutor/frame/qd_l2_cache.h), [worker layout](../../hexecutor/frame/qd_graph_dsp.h), and [configuration/binding](../../hexecutor/frame/qd_graph_prepared.h).
