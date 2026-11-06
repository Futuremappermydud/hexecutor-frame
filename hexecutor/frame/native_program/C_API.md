# NPRG v2 C reader and preflight

`nprg2_runtime.c/.h` is a freestanding-style C foundation for the prepared
schedule. It has no DSP register, VTCM, transport, or loader calls. It only
accepts caller-provided memory pointers, so the host tests exercise pointer and
budget rules without binding real hardware addresses.

## Lifecycle

1. The caller supplies manifest hashes for context, constants, source QDRX and
   the prepared NPRG artifact itself. `nprg2_open` validates NPRG magic/version/ABI, CRC, hashes, canonical section
   bounds, geometry/storage fields, all command/ref/call ranges, pool sizes and
   direct/block pointer ranges. Pool mappings must be separate, non-overlapping
   caller-provided buffers at least as large as the image's pool requirements.
2. `nprg2_open` expands each unique compressed block list once into a caller
   allocator (default: `malloc`). Duplicate tensor views reuse the same
   expanded list while preserving each tensor's separate slice origin. It
   rejects noncanonical stream gaps/overlap, malformed varints, foreign pools,
   out-of-pool 2 KiB blocks and slice origins outside the view's block span.
   Current primary sizing is 481,320 B for expanded blocks plus 106,720 B for
   runtime tensor locations. Open temporarily uses a 327,680 B dedup hash
   table; peak allocator payload is about 916 KB, then the hash table is freed.
3. `nprg2_bind_mock` requires the caller to mark pool1 as already verified
   VTCM, supply a hot-memory buffer and budget, and provide a resolver for every
   scheduled command. The resolver receives full tensor storage/geometry and
   command fields. Missing/unsupported coverage, `may_fallback`, or excess
   local plus concurrent-worker scratch rejects before any hot-memory copy.
   The API has no legacy fallback path and does not write graph pools.
4. On success, the bound command/geometry/tensor/ref/vector/call pointers refer
   to the caller's hot buffer. `nprg2_command_at` reads a command from that hot
   buffer. At bind, blocked tensor rows are rewritten once: their 24-bit field
   becomes an expanded-block index, replacing the compressed-stream offset in
   the original prepared image. `nprg2_bound_tensor_at` reads all tensor storage
   and geometry from hot memory and returns the
   once-expanded block list from private heap. The compressed block stream is
   not touched in the command loop. The setup-only decoded tensor table is
   also absent from command-time tensor lookups. Raw NPRG bytes remain unchanged.
   Hot memory may not alias the source image, expanded block arrays, decoded
   tensor table, or graph pools.

The resolver used by the C tests deliberately accepts every command to test the
mechanics; this is not a numerical proof or production capability registry.
An actual native entry point must use proof-backed resolvers and reject the
whole requested command range before executing command zero. `pool1_is_verified_vtcm`
is a caller assertion for tests, not physical mapping evidence. Real address
binding remains gated on the preserved-L2/firmware map in `RUNTIME_PLAN.md`.

## Host equivalence and malformed-input checks

Build and run from the repository root:

```sh
cc -std=c11 -O2 -Wall -Wextra -Werror \
  frame/native_program/nprg2_runtime.c \
  frame/native_program/test_nprg2_runtime.c \
  -o /tmp/test_nprg2_runtime
/tmp/test_nprg2_runtime \
  frame/build/resident-primary.qdrx frame/native_program/examples/primary-v2.nprg \
  frame/build/resident-alternate.qdrx frame/native_program/examples/alternate-v2.nprg
```

The C test independently hashes each real source QDRX, opens each NPRG v2,
compares every tensor shape/type/quantization/direct pointer/block list/slice
origin and every command/ref/vector/call against QDRX, then tests CRC, image
section bounds, hash mismatch, malformed delta varint, invalid tensor ref,
invalid slice origin, and direct-pool overflow. It confirms missing coverage,
legacy fallback, insufficient scratch and unverified pool1 leave both the hot
buffer and graph pools untouched. An allow-all mock resolver verifies only the
bind/copy/accessor mechanics. ASan/UBSan passed the same two-image driver.

Root follow-up validation checks every bound tensor while the program’s setup tensor pointer is NULL, then restores it for cleanup. This proves the execution accessor does not depend on the setup DDR tensor array. Both actual models pass strict compilation and ASan/UBSan after the rewrite.

## Prepared blocked elementwise subset

`prepared_elementwise.c/.h` adds `nprg2_executor_register_elementwise`, which
registers only ADD_U8 variant 13 and blocked u8-to-u16 LinearClip variant 68.
The handlers consume the executor's caller scratch; on Hexagon it must be at
least 30,748 bytes. Pool-1 block references and writes are bounded by the
request's active writable limit, so the reserved tail is excluded. Unsupported
LinearClip variants and unsupported tensor geometry remain absent from the
registry and fail preflight.

Run the NativeGraph CPU-prefix capture, then the prepared-image fixtures and
HVX glue on the V75 simulator with:

```sh
frame/native_program/run_prepared_elementwise_sim.sh
```

The capture stage runs the actual CPU graph through the command immediately
before the first ADD_U8 variant 13 and LinearClip variant 68 in each model,
records every graph pool plus the live Fi, executes that single CPU command,
and stores its physical output pages. The Hexagon harness reloads those
captured pools, dispatches the same prepared command, and compares every pool
byte to the captured CPU step. It then checks overlapping output and
active-tail metadata are rejected before pool writes.
