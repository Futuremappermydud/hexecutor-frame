# Direct VTCM leaf bindings

`qd_native_direct.h/.c` creates the descriptor, table, and guard state for the
original QNN 1x1 Xpb/Xwb leaf and stride-1 3x3 Xwb leaf without copying tensor,
weight, or Fi bytes. The caller owns command decoding, HMX issue/drain, and the
VTCM lease.

Production calls pass the exact pool1 VTCM base as both `vtcm` and `request.pool[1]`,
`active_bytes = 0x780000`, and the final 512 KiB tail at `vtcm + 0x780000`.
Page tables and the 120-byte leaf ABI are caller-supplied L2 storage. They must
not overlap pool metadata or tensor bytes. `qd_direct_validate_*` performs the
same metadata, range, shape, and alias checks without writing; call it during
executor preflight. `qd_direct_bind_*` repeats the checks, then writes tables,
ABI, and the tail's A5 output guard, zero input guard, completion word, and
zero-Fi words. The `tail_bytes` argument must equal the reserved tail capacity
exactly. Validation result storage must not overlap the request's seven-word
`pool_bytes` array.

The currently accepted shapes require complete 8x8 output tiles. The 1x1 path
also requires fully active input/output spatial extents and 32-lane channel
blocks. The 3x3 path is restricted to the measured S1 geometry: input height
10/max-height 24 with seven-pixel origins, output height 8, full output tiles,
and the known CO2 physical-CO32 exception. Each accessed page must resolve to
pool1, be 2 KiB aligned, and end before the active-range boundary. Input,
output, W, and Fi extents are checked for overlap before any writes.

After binding, clean the L2 metadata as required by the executor, copy ABI
fields to the leaf's normal scalar argument storage, enable/own HMX, call the
original leaf, then call `qd_hmx_finish(binding.zero_fi, binding.done)` and
verify guards before the graph proceeds. Do not call the old adapter finish:
the leaf output already resides in the graph's output pages.

## Offline checks

`run_direct_binding_test.sh` checks read-only validation, direct page pointers,
pool/range/alias rejection, full-tile rejection, and tail initialization on the
host. `run_direct_1x1_sim.sh` exercises actual command-1195 weights/Fi and
compares the direct output with both an integer oracle and the staged leaf.
`run_direct_3x3_sim.sh` does the same with the real command-247 CPU-prefix
fixture and the staged leaf. The SDK simulator has only 512 KiB VTCM; these
simulators explicitly override the production 8 MiB layout to a 384 KiB active
region plus a 128 KiB tail. These runs validate leaf binding and dataflow, not
the production physical VTCM address or hardware lease itself.

`run_direct_3x3_scatter_sim.sh` is a stronger sequence-247 check: it reads the
actual QDRX pool1 block offsets, preserves repeated X aliases, and maps the
129 unique touched pages through a coprime physical permutation. The resulting
table has 57 nonmonotone transitions. This checks that the generic leaf follows
table entries rather than relying on contiguous physical pages. The run also
compares output with the staged leaf and CPU-prefix fixture; it remains offline
SDK simulation evidence, not a hardware lease test.
