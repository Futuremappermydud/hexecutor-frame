# Native QNN 1×1 adapter

This directory contains a bounded preparation/scatter adapter and an opt-in
source-included graph bridge for the simulator-covered
`hmx_v73_convbbb1x1_stride1` leaf. The bridge dispatches eligible QDRX 1×1
commands before legacy FI/weight conversion. It requires the preserved-VTCM
lease, owned/powered HMX, and a drained worker generation. The legacy worker
and this leaf share HMX state.

## Current adapter contract

`qd_native_1x1_prepare` accepts QDRX blocked tensor tables, pool bases, a direct
raw Xpb or Xwb 1×1 filter, raw Fi rows, and an exclusive workspace. It rejects
non-u8 blocked activations, non-s8 filters, non-i32 Fi, unsupported channel
counts (other than the explicit padded-Xpb CI12/20 cases), non-1×1 filters,
unaligned crop/padding origins, oversized
tables, out-of-pool blocks, duplicate touched output blocks, bad aliases, and
workspaces over 512 KiB.

The adapter resolves `slice_offset` as a whole-block offset, stages only the
effective blocked input croutons plus the output croutons the leaf needs, and
copies raw 1×1 Xpb/Xwb and Fi bytes without unpacking or repacking. With the
separate `QDS_NATIVE_PADDED_1X1=1` build flag, it can also stage raw Xpb CI12 or
CI20 as native CI32, expanding each CO32 filter block by copying its three or
five input groups and zero-filling the rest. This opt-in branch requires
logical activation depth 9/18 and physical depth 32. It copies original
physical activation lanes unchanged; zero weights make added lanes irrelevant.
With the flag off, the adapter request layout and CI128 behavior remain
unchanged. The native
leaf's fixed descriptor partition currently reserves 0x56800 bytes (354,304 B)
inside the caller's 512 KiB upper-VTCM lease. Optional external descriptor and
pointer-table buffers are supported; these also make the simulator's native
leaf ABI explicit.

Input and output tails have separate sentinels: the unused input table points
to a zero-filled 2 KiB dummy crouton; the unused output table points to an
0xA5-filled 2 KiB guard. `qd_native_1x1_finish` checks both table tails, both
sentinels, and all active table entries after HMX completion. It scatters only
the effective `oh × ow × co` output lanes, preserving every byte outside the
old graph's write footprint, including crop-external rows and columns.
Full effective tiles use one aligned 2 KiB copy; partial edge tiles retain
masked 32-byte lane stores. The guard checker compares whole 128-byte vectors
and reduces mismatches in registers, then uses scalar lane extracts without
spilling HVX state to cached stack memory.

The graph must opt in with `QDS_NATIVE_1X1=1` and
`QDS_PRESERVE_VTCM=1`. The bridge reserves the upper 512 KiB at VTCM offset
`0x700000`, invalidates graph and resident-worker HMX keys around the leaf, and
drains HMX before scattering. Its completion word is at workspace offset
`0x58000`; the adapter initializes all 64 `qd_hmx_finish` FI words. Diagnostics
report prepared, declined, and failed commands plus phase cycles and staged
bytes. The provider leaf can differ from the independent integer graph oracle
by 0 or +1; it is not exact graph arithmetic.
The 140-command QDRX census is a metadata/preflight candidate count, not proof
that each runtime command's flags, live pool mapping, or output values have
been validated.

## Low-CI zero-expansion experiments

Commands 115 and 3416 exercise actual Xpb filters with CI12→CO32 and CI20→CO128.
`make_padded_ci_fixture.py` snapshots the real NativeGraph CPU prefix, then
expands each 1×1 Xpb filter to CI32 by copying its three or five 128-byte input
groups and leaving the additional groups zero. The simulator uses the existing
adapter/leaf with the physical blocked input and a logical channel count of 32.
It preserves the original command's first 12 or 20 physical input channels,
including channels beyond logical input depth 9 or 18.

Both simulator outputs match the independent integer oracle within `[0,+1]`.
The poison build sets all input channels above original CI to `0xA5`; its entire
output packet is byte-identical to the unpoisoned run for each command. The
graph bridge can include these two cases only when the additional
`QDS_NATIVE_PADDED_1X1=1` flag is set. Its candidate predicate lives in
`qd_native_1x1_graph_eligibility.h`; host C tests consume metadata extracted
from compiled QDRX commands 115, 3416, and 1195, and verify low-CI candidates
are declined when the flag is off. The padded simulator now passes raw Xpb
bytes to `qd_native_1x1_prepare`, compares its staged expansion against the
independent expansion, and confirms unsupported CI64 requests leave the
poisoned workspace untouched.

## Offline evidence

`census_qdrx_1x1.py` reads the compiled primary and alternate QDRX images and
checks geometry, tensor classes/dtypes, raw Fi footprint, block-table bounds,
output aliases, and the 512 KiB workspace limit. It finds 140 shape/format
candidates per image: 80 CI128→CO64 and 60 CI128→CO128. The remaining 140
1×1 commands are u16 or have padded/non-32-aligned channel geometry.

The first command-level fixture uses actual primary QDRX command 1195's raw
filter and Fi source tensors (Xpb tensor 1199; Fi tensor 1201) with deterministic
synthetic blocked activations. It runs the adapter and original assembly leaf
in Hexagon Simulator, then compares all 65,536 outputs to independent int64
matrix arithmetic. The native result differs in 552 bytes, each by +1; the
observed bound is [0,+1]. This proves this command's data layout and quantized
conversion path in the simulator. It is not hardware validation or full-graph
exactness.

To reproduce from the repository root:

```sh
python3 frame/native_adapter/make_qdrx_1195_fixture.py
docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD/frame:/src" \
  -w /src depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
  -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin \
  -fno-stack-protector -Wall -Wextra -Werror \
  native_adapter/qnn_hmx_1x1_adapter_sim.c \
  native_adapter/qd_native_1x1_adapter.c qd_qnn_hmx_1x1.S qd_hmx_ops_encoded.S \
  -o native_adapter/qnn_hmx_1x1_adapter_sim.elf
docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD/frame:/src" \
  -w /src depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim \
  -mv75 --quiet native_adapter/qnn_hmx_1x1_adapter_sim.elf
python3 frame/native_adapter/check_qdrx_1195_sim.py
python3 frame/native_adapter/census_qdrx_1x1.py \
  --output frame/native_adapter/qdrx-1x1-census.json
```

## Broader offline checks

`preflight_qdrx.c` consumes `preflight-primary.bin`, generated from the actual
primary QDRX block tables and metadata. It calls the C adapter's prepare path
for all 140 eligible commands and verifies active/tail table entries and the
workspace limit. The pool bytes are mocked for this metadata-only sweep; it does
not establish that every command's live pool values or FI are ready for the
native leaf.

`audit_representative_sources.py` records real packed filter/FI source hashes
for representative commands spanning 8×64, 8×8, 2×8, 8×16, 4×16, and 8×32
spatial shapes, with both Xpb and Xwb formats and nonzero slice offsets. It also
records command 2380's dynamically generated FI producer. Running
`evaluate_dynamic_fi.py` executes the NativeGraph CPU prefix through compiled
command index 2128 (QDRX sequence 2380) and captures that command's actual
runtime Xpb and FI hashes. This is CPU-only and makes no device connection.
The captured 2380 runtime FI/weights have been run through the HMX simulator
leaf. A standalone masked-scatter boundary test checks partial effective
output extents and preserves untouched physical rows.

`manifest.json` hashes the adapter sources, fixtures, and current evidence.
Regenerate it with `python3 frame/native_adapter/make_manifest.py` after any
change to these files.

## Frame probe (source built, not run)

`adapter_frame_probe.c` and `run_adapter_frame_probe.py` add a separate bounded
Frame request for command 1195. The payload uses only worker 0 for the adapter
and leaf, acquires the existing HMX/power controls, backs up and restores the
full 512 KiB lease at VTCM offset `0x700000`, verifies the restored bytes, and
returns output plus an input echo. The adapter checks its zero input dummy and
A5 output guard. The host runner checks input preservation and the same
independent int64 `[0,+1]` result bound. It has only been cross-compiled; it has
not been run on Frame hardware.

Build the isolated ELF with `build_adapter_frame_probe.sh`, then invoke it only
through the existing `FrameTransport` workflow:

```sh
python3 frame/native_adapter/run_adapter_frame_probe.py \
  --fixture frame/native_adapter/fixture-qdrx-1195.bin \
  --report frame/native_adapter/adapter-frame-report.json
```

The invocation requires connected Frame transport and is intentionally not
part of the offline validation commands above.

The dynamic-fi command 2380 fixture also passes end-to-end in Hexagon Simulator.
`make_dynamic_fixture.py` snapshots the CPU graph immediately before that
command, including its real runtime Xpb and Fi values, input/output blocked
pools, and the nonzero input/output slice offsets. The simulator driver runs
adapter preparation, the QNN leaf, HMX drain, and masked scatter; the independent
checker reports 4,096 output values within `[0,+1]` and verifies all inactive
output table entries remain byte-identical. This covers the sliced CO64 path.

`masked_scatter_sim.c` separately exercises an effective 2×8 output in a
physical 8×8 crouton. It passes with four channel blocks and nonzero table
slices, and verifies rows 2–7 remain unchanged in every output crouton.

The Frame probe now copies the unaligned raw fixture regions into aligned W/FI
staging, checks fixture geometry and byte ranges before writes, and retains the
Frame session if power or VTCM restoration cannot be verified. The host-only
`validate_frame_probe.py` checks the exact packet and all arena/pool region
boundaries. The Frame ELF cross-compiles and the packet/range preflight passes;
no Frame hardware invocation has been made.

The final CPU-prefix simulator sweep poisons the complete 512 KiB workspace
with `0xA5` before prepare and includes nine actual QDRX commands:
1195, 2380, 4136, 4137, 4151, 4154, 4169, 4173, and 4269. It spans six spatial geometries (8×64, 8×8, logical 2×8/max 8×8, 8×16, logical 4×16/max 8×16, and 8×32), Xpb and Xwb, CO64/CO128, and nonzero input/output table slices. Each fixture snapshots the input and prior output croutons from the NativeGraph CPU prefix and takes the command's actual runtime packed weights and Fi bytes. The per-sequence JSON reports record the independent integer `[0,+1]` result bound and inactive output table preservation. The separate effective-H2 masked scatter probe checks rows outside a cropped logical extent within a physical crouton.

Early failed sweeps used an undersized simulator payload buffer; those results were superseded after the buffer was bounded against full input/output table sizes and expanded to 1 MiB. Only the final per-sequence reports listed in `manifest.json` are acceptance evidence.

## First Frame checkpoint

The first controlled Frame run is recorded at
`.work/frame-direct-kernels-20261030/native-adapter-1195-first.json`. It passed
with output SHA256 `d985dc47f586ac92ef379273dcf9bc182812a2b281045f361690e645afc72193`,
which matched the simulator output exactly. All 65,536 bytes were within the
integer oracle bound `[0,+1]` (552 differed by +1), input bytes were unchanged,
TLB entries 8/9 matched before backup and restore, and the full 512 KiB VTCM
lease was restored. DSP control, callback, loader, payload, arena, and eye-client
state were clean afterward.

Recorded timings were 3.523 µs for the drained leaf, 198.913 µs for the full
prepare call, 12.66 µs to download and echo, 13.93 µs to save VTCM, and 53.68 µs
to restore it. The prepare figure includes validation, pool/table resolution,
input/output staging, and descriptor setup; it is not an intrinsic leaf or
staging-only measurement.

The first Frame checkpoint above used the pre-optimization adapter source. The
latest adapter optimization removed a full input-workspace zero pass that was
immediately overwritten by full crouton copies. It also initializes only
the unused pointer-table tails; active entries are set as their staged croutons
are copied. Per-lane zeroing for invalid edge pixels remains. Output croutons
still copy their original bytes because masked scatter must preserve every byte
outside the output write footprint. The full nine-command simulator matrix and
the H2 masked-scatter proof were rerun after this change and pass. The Frame
probe ELF has not been rebuilt or rerun since this adapter-only change.

After adding the padded-CI branch, the macro-off adapter was rebuilt and the
nine actual CPU-prefix simulator commands plus effective-H2 masked scatter
were rerun. All nine remain within `[0,+1]`. The three compiled-QDRX
classifier fixtures pass the pure-C bridge predicate with the expected
macro-dependent low-CI eligibility. These changes have not been run on Frame.

A geometry-plan cache is a possible later optimization. It can cache only
shape-derived values (effective/physical extents and block counts), keyed by
all dimensions, padding, channels, and format. Pool pointers, block table
contents, slice offsets, and alias/range guards must still be resolved for each
request. This cache is not implemented; keeping full preflight per request is
safer until measurement shows that validation cost warrants it.
