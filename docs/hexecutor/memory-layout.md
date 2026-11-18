# Executor memory layout

The executor's performance depends on where metadata, tensor values, stacks, and workspace live. These are distinct allocations. A compact program file is not a proof that all live tensors and concurrent scratch fit on the device.

## Private DDR and address ownership

`cdsp_mem` supplies private DDR at IOVA `0xa0000000`. The depth and hand-tracking executors operate on a 128 MiB baseline (eight 16 MiB mappings), while heavier workloads (multi-graph models, LFM3 text at 1024 MiB, and full LFM3-VL multimodal at 1536 MiB) dynamically grow the lease up to 1.5 GiB. The 1536 MiB ceiling (`0xa0000000 + 0x60000000 = 0x100000000`) spans the full remaining 32-bit IOVA space.

`WorkloadMemoryManager` manages dynamic memory growth and shrinking:
- **Growth on demand:** Automatically expands the kernel allocation in 2 MiB chunks and maps them into the CDSP SMMU domain before a workload starts.
- **Shrinking on release:** Returns memory to the kernel (down to 128 MiB baseline) when the workload finishes or the session closes, preventing idle models from starving headset system RAM.
- **JTLB slot allocation:** Hexagon JTLB entries are assigned dynamically based on slot count: 8 entries (64..71) for 128 MiB, 32 entries (64..95) for 512 MiB, 64 entries (64..127) for 1024 MiB, and 96 entries (32..127) for 1536 MiB.
- **Interfaces:** Dynamic sizing is controllable via `/sys/module/cdsp_mem/parameters/size_mb` (0644) and ioctl `CDSP_MEM_IOC_SET_SIZE` on `/dev/cdsp_mem`.

`qd_owned_memory.h` checks for occupied mappings/aliases and saves/restores the previous TLB contents. It must not evict firmware entries to make a graph fit.

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

## Stock eye-tracking thread and VTCM

Pool 1 covers all of VTCM, including the range the stock eye-tracking code uses. This comes from the disassembly of the deployed firmware (`cdsp_sm8650.mbn` sha256 `4f9fd567…`, disassembly in `depth-model/.work/frame-direct-kernels-20261030/cdsp.asm`). FrameNPU's independent VTCM map agrees with it:

- At boot, hardware thread 3 takes its own path (`0x870002f8`). It sets r19 = `0xf1522004` and polls the block at DMA offset `0x1522000`: state at +0, active at +4, job at +8.
- Whenever active = 1, it services job 2 (load: invalidate `0x10080` bytes from +0x80, then `0x870154c0`, which copies into `0x31700000`). It also services job 4 (evaluate: invalidate `0x20180` bytes, then call `0x87015500` once per eye, writing 0x80 bytes per eye at +0x20100/+0x20180).
- `0x87015500` uses VTCM `0x31700000–0x31776600` for tables and weights, and `0x31776600–~0x317d4600` as a work buffer. It runs concurrently with the master loop and with our callback wake. Nothing in our wake stops or serializes it.

So while a Hexecutor session holds VTCM, a job 2 or 4 posted to that block makes the stock eye code read tables we have overwritten and write over our activations. Ordinary sessions avoid this only by pausing the `eyetracking` process. VTCM restoration at exit puts the eye tables back for later.

Open: the broker treats `0x1522000` as the XRService FeatureNN block (job 8) and `0x445000` as the Gazenet block ([broker](broker.md#gazenet)). But the firmware's thread-3 handler serves jobs 2/4 at `0x1522000`, with Gazenet's input (+0x100) and output (+0x20100, 64 floats) layout. Which client posts jobs 2/4 to `0x1522000`, and whether the firmware ever reads `0x445000`, have not been checked on device. A search of the disassembly finds no immediate operand for `0x445000` (cached `0x87445000` or uncached `0xf0445000`).

Other firmware-owned VTCM, from FrameNPU `docs/VTCM_MAP.md`: FeatureNN worker scratch at `0x31000000 + HTID*0xf0000` (960 KiB each, reset per request), and the FeatureNN model blob at `0x311e0000–0x316eea80` (copied by op 7 and persistent after it). The only ranges nothing references are `0x316eea80–0x31700000` and `~0x317d4600–0x31800000`.

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

Source: [owned DDR](../../hexecutor/native/qd_owned_memory.h), [lease map](../../hexecutor/native/qd_l2_cache.h), [worker layout](../../hexecutor/native/qd_graph_dsp.h), and [configuration/binding](../../hexecutor/native/qd_graph_prepared.h).
