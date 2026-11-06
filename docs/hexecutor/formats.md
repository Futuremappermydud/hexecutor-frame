# Program formats and host API

Program images describe execution metadata and storage; kernels implement arithmetic. A valid image is not evidence that every command has a supported native implementation.

## Representation pipeline

| Representation | Purpose |
| --- | --- |
| QDRP | Source-auditable graph/program IR produced by `resident_program.py` |
| QDRX v1 | Compact relative-address image produced by `resident_image.py`; retains audit/provenance fields |
| NPRG v2 | Offline-lowered prepared tables with hot/cold metadata split, consumed by `nprg2_runtime` |

QDRX interns exact block tables without changing aliases, pool offsets, or slice origins. It does not package the separate constant pool or snapshot runtime-produced weights. Its allocation-free reader validates ABI, integrity, sections, layouts, pool bounds, and dispatch coverage. Historical decoder-only timing was setup work, not inference performance.

## NPRG v2 wire contract

The current C reader defines a 256-byte header, 44-byte geometry records, 12-byte tensor-location records, and 20-byte commands. References are u16; framework vector words are u32; call records are 88 bytes. Sections are canonical, little-endian, aligned, and checked for bounds/padding. The header binds the source QDRX ABI and context/constants/source hashes, counts, section sizes, pool requirements, and integrity data.

Geometry is shared separately from storage locations. Block lists are compressed in the file, then expanded once into private memory. Tensor views retain their independent slice origins even when they share an expanded list. Pointer-placeholder call operands remain tagged placeholders; recovered register values are not trusted DSP addresses.

The older NPRG v1 used a 240-byte header and 52-byte tensor records and did not fit the same hot-memory budget. Do not implement a v2 reader using v1 record sizes or assume that a table format removes the need for runtime scratch.

## Open, bind, preflight, run

1. `nprg2_open` verifies the image against caller-supplied trusted hashes, validates canonical sections/fields/pools, and expands the block stream. Supplied pools must be distinct and large enough.
2. Binding validates hot-memory placement and budgets. Hot memory must not alias source bytes, decoded/expanded setup storage, or graph pools. Bound tensor locations refer to expanded lists rather than decoding compressed streams in the command loop.
3. The resolver checks each command's opcode/variant, shapes, layout, quantization, alignment, aliases, and local plus concurrent-worker scratch. Missing support, permitted legacy fallback, or an exceeded budget rejects before execution.
4. The runner executes the accepted range, handles cancellation/errors and optional parallel hooks, and reports the failing command when appropriate.

`pool1_is_verified_vtcm` in a host fixture is an assertion used to exercise the API. It does not establish physical ownership on a device. Real binding also needs the bootstrap's preservation and mapping checks.

## Size and lifetime implications

The recorded primary graph has 5,336 tensors and 5,068 commands; alternate has 5,343 and 5,040. Primary hot metadata is about 235 KB. Earlier v2 reader accounting used about 481 KB for expanded block entries and 107 KB for decoded locations, with a temporary 328 KB dedup table during open. These describe particular images and setup implementation, not a fixed ABI allocation requirement.

Read [memory layout](memory-layout.md) before assigning hot metadata or worker scratch addresses. When changing the registry, ABI fingerprint, or serialized fields, update producer and consumer together and reject mismatched packages explicitly.

Source: [QDRX producer](../../depth-model/python/depth_model/lowering/resident_image.py), [NPRG producer](../../hexecutor/native/program/), [wire constants/API](../../hexecutor/native/program/nprg2_runtime.h), [reader](../../hexecutor/native/program/nprg2_runtime.c), and [runner](../../hexecutor/native/program/prepared_runner.c).
