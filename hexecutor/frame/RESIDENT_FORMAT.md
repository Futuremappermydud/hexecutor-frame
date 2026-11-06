# Resident decoder checkpoint

QDRX v1 is a compact, relative-address image for a native DSP reader. It is not
a DSP inference backend. `resident_program.py` remains the full source-auditable
QDRP IR; `resident_image.py` lowers its execution metadata to QDRX and compares
every reconstructed tensor, command, argument, I/O reference and pool size with
the source. No runtime-produced tensor or dynamic weight/Fi tensor is cached as
a constant. The companion JSON records the image and constant-file SHA256s.

## Tested on Oct 29, 2026

| Model | QDRP bytes | QDRX bytes | Tensors | Commands | Shared block entries |
| --- | ---: | ---: | ---: | ---: | ---: |
| primary | 31,279,510 | 1,738,088 | 5,336 | 5,068 | 120,330 |
| alternate | 30,058,243 | 1,736,920 | 5,343 | 5,040 | 120,592 |

The original tensor tables contain 519,572 / 492,852 block entries. Exact table
interning reduces those counts without changing any pool, offset, slice offset
or alias. The images retain all 75 scheduled operation variants and 44 opcodes.
They do not contain the 7.2 MB constant pools.

`qd_resident.c` is a freestanding, allocation-free reader and serial dispatcher
core. It validates image integrity, ABI, canonical sections, tensor layouts,
table spans, relative pool ranges, serial dependencies, required arithmetic
calls, spill/fill and block-zap ranges. Both image size and aggregate validation
work are bounded. Pointer placeholders remain tagged placeholders; recovered
64-bit register/stack values are diagnostic immediates, never DSP addresses.

Dispatch checks the command budget and availability of every exact operation
variant before invoking a kernel. Cancellation is checked between commands;
kernel errors retain the failing command and diagnostic value. Only explicit
no-ops are built in. There is no worker-barrier protocol, mapped-pool allocator,
or numerical graph kernel in this core. Future kernels must validate their
arity, shapes, arithmetic parameters, mapped ownership and scratch requirements.

Local checks: 11 QDRP tests and 13 QDRX/native tests pass. An ASan/UBSan host
driver checks 512 checksummed mutations per graph, including unaligned image
bases, truncation and native dispatch attempts. Host callback tests verify
serial order, cancellation, budget rejection, kernel errors and preflight
rejection of a missing late variant before any callback executes. These are
control-flow tests, not graph numerical-output tests.

The Hexagon decoder-only probe passed seven cases per graph across two separate
Frame sessions (28 requests): valid decode, missing kernel, insufficient command
budget, bad CRC, invalid tensor reference, invalid block range and truncation.
`build/resident-decoder-hardware.json` records image/payload hashes, diagnostics,
cycle counts and locked state snapshots. The probe payload contains 9,176 bytes
of text/rodata, fits the existing 32 KB payload region and imports no runtime
symbols. It uses the unchanged validated loader ELF.

Valid decode takes 3.3846–3.3888 seconds wall time, including command upload and
completion, with the current scalar reader operating through the uncached arena.
This is a setup diagnostic, not model inference latency. No graph operations
execute, no graph pools are allocated, and no AXI/HMX registers are changed.
After both sessions controls were `[1,0,1]`, callback was zero, loader/payload/
arena regions were zero and an eye client was running. No reboot was needed.

The original pool requirements remain 18,146,816 / 18,234,112 bytes, excluding
the image, input/output staging and 2,949,120-byte HMX scratch. Compact metadata
does not solve the full-graph memory problem. `hardware_layout_verified` remains
false. v5 is still the usable inference backend; no resident output-exactness,
cold/warm inference, FPS or new versioned inference package is claimed.

## Build and verify

From the source root:

```sh
python3 frame/resident_image.py recovered/primary --output frame/build/resident-primary.qdrx
python3 frame/resident_image.py recovered/alternate --output frame/build/resident-alternate.qdrx
python3 frame/tests/test_resident_program.py
python3 frame/tests/test_resident_image.py
make -C frame HEX_CC=clang-14 HEX_LD=ld.lld-14 build/qd_resident_probe.elf
```

The existing `depth-model-hvx-build` Docker image can supply the Hexagon
compiler/linker on a Mac. The probe target builds only the new payload and reader;
it never rebuilds the fixed loader. If intentionally changing the registry,
regenerate `qd_resident_schema.h` with
`python3 frame/resident_image.py --write-schema-header frame/qd_resident_schema.h`.
Registry changes alter the ABI fingerprint and require an ABI review.

Host sanitizer check:

```sh
cc -std=c11 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined \
  frame/qd_resident.c frame/tests/fuzz_resident_decoder.c \
  -o frame/build/fuzz_resident_decoder
frame/build/fuzz_resident_decoder frame/build/resident-primary.qdrx
frame/build/fuzz_resident_decoder frame/build/resident-alternate.qdrx
```

On Frame, after checking active clients and acquiring the existing lock for a
read-only state snapshot, use the isolated development copy:

```sh
~/frame-dsp/depth-model/.venv/bin/python frame/bench/verify_resident_decoder.py \
  --image frame/build/resident-primary.qdrx \
  --image frame/build/resident-alternate.qdrx --sessions 2 \
  --report frame/build/resident-decoder-hardware.json
```

The runner uses `FrameTransport` for exclusive ownership and cleanup, verifies
the exact loader ELF and image manifest hashes, and takes locked state snapshots
before and after sessions. It does not load a QNN context or modify v5.

## QDRX v1 wire ABI

All words are little-endian unsigned u32 unless specified. Section starts are
8-byte aligned; alignment padding and reserved words are zero. Sections are in
the order below, with no trailing bytes. At most 8 MiB is accepted. Tensor and
command counts are additionally capped at 65,536. No physical pointers appear
in the image. `QDR_NONE = 0xffffffff` denotes an absent I/O index or the first
command's dependency.

The header is 64 words (256 bytes):

| Words | Contents |
| --- | --- |
| 0–5 | `QDRX`, version 1, header bytes, total bytes, CRC32, ABI fingerprint |
| 6–7 | reserved zero; no hardware-ownership flag is assigned |
| 8–15 | source context SHA256 bytes |
| 16–23 | separate constant file SHA256 bytes |
| 24–30 | reference pool bytes: 0, 1, 2, 3, 4, 5, spill; pools 0 and 2 are zero |
| 31–32 | HMX scratch requirement; historically verified arena size |
| 33–36 | graph input ref first/count, output ref first/count |
| 37–48 | six section offset/count pairs |
| 49–63 | reserved zero |

CRC32 covers the entire image with bytes 16–19 zeroed. It detects corruption;
the host verifies the SHA256 manifest for source identity. Neither a checksum
nor a reference-pool size proves physical memory ownership.

| Section | Record bytes | Contents |
| --- | ---: | --- |
| tensors | 128 | descriptors below |
| blocks | 4 | aligned byte offset OR pool number in low 7 bits |
| commands | 96 | descriptors below |
| refs | 4 | zero-based tensor descriptor indices |
| words | 4 | shared checkpoints and framework vectors/words |
| calls | 88 | call ID, 10-bit pointer-placeholder mask, ten u64 argument words |

The call arguments preserve six register values followed by four stack values.
Masked arguments encode zero; the tag, rather than a raw value, means `P`.
IDs and their fingerprint are declared in `resident_schema.py` and the generated
`qd_resident_schema.h`. Opcodes retain QDRP's stable IDs. Variant IDs are needed
because kernels sharing an opcode can have different numerical semantics.

Tensor descriptor words:

| Words | Contents |
| --- | --- |
| 0–3 | original tensor ID, dtype ID, zero-padded ASCII class FourCC, flags |
| 4–15 | dimensions[4], max_dimensions[4], padding[4] |
| 16–17 | direct storage pool, byte offset |
| 18–20 | shared block first/count, slice offset |
| 21 | source shape flags |
| 22–24 | signed i32 zero point, bit-exact f64 scale |
| 25 | bit-exact scalar word (i32 or f32) |
| 26–29 | four shape values |
| 30–31 | reserved zero |

Flags are HAS_SHAPE=1, BLOCKED=2, DIRECT=4, SHAPE_VALUE=8, SCALAR=16,
QUANT=32 and CROUTON=64. Exactly one of HAS_SHAPE, SHAPE_VALUE and SCALAR is
present. Absent fields are zero. External graph inputs/outputs may have a shape
without direct or blocked storage; they require separate verified staging.
Blocks retain original 128-byte constant alignment or 2,048-byte other-pool
alignment, and each referenced block must fit completely within its pool.

Command descriptor words:

| Words | Contents |
| --- | --- |
| 0–5 | source sequence, opcode, variant ID, duplicate-input-pad flag, preceding command, graph I/O index |
| 6–9 | input refs first/count, output refs first/count |
| 10–15 | checkpoint words first/count, framework words first/count, framework vector first/count |
| 16–17 | calls first/count |
| 18–23 | reserved zero |

Repeated InputSlicePad commands have explicit NOP opcodes and retain the source
variant and I/O metadata. Serial dependencies retain all checkpoint and alias
ordering. Strings, source offsets, debug names and serialized blocktable links
are omitted; the fully resolved execution tables and relative aliases remain.

## Next gates

Prove a pool/staging layout within memory whose ownership is established, or
implement a bounded tiled allocation plan. Then implement bounded tensor access
and a numerical kernel preflight, prove one convolution plus blocked write/read,
and extend to a residual block. Worker barriers, cancellation/restart and power
restoration must be integrated before a persistent graph transport command is
enabled. All full-graph exact-output, latency and point-cloud gates in
`DSP_RESIDENT_HANDOFF.md` remain required.
