# Kernel adapters and support boundaries

The adapter directories bridge a prepared tensor description to a particular native/provider kernel ABI. They are focused implementations and investigations, not a promise that every model with the same operation name is supported.

`native_adapter/` contains the 1×1/padded-channel adapter work. `native_kernels/` covers u8 stride-1 and related HMX leaf layouts. `native_stride2/` and `native_runtime/stride2/` examine stride-2 callers and staging. `native_u16/` covers halfword activations, origins, channel starts, and provider repacking. `native_quantize/` separates quantization semantics from complete convolution.

## The acceptance contract

Before invoking an adapter, validate dtype, kernel size, stride, logical/max geometry, padding, channel count, direct/blocked storage, alignment, slice origin, quantization, weight packing, and scratch lifetime. Validate input as well as output aliases. A provider's register/stack layout is an ABI that must be established for the selected leaf, not inferred from one symbol name.

Repeated basis, poisoned-padding, boundary, and full-pool comparisons were useful because an apparently correct output tile can conceal a wrong stride or an out-of-bounds read. The durable conclusions are the supported feature tuple and its rejected cases. A long list of passing individual probes is not a replacement for those guards.

Captured arrays, generated fixture headers, disassembly dumps, and provider binaries remain local and ignored. Generators/checkers remain source. `make_manifest.py` can include this guide with a locally assembled adapter package; that package's captured fixtures and outputs are not checked into this repository.

The prepared resolver must opt into a proved adapter. Missing geometry/quantization coverage rejects before executing the model range. An experimental adapter does not silently fall back to old VTCM scratch while pool 1 occupies VTCM.

Source locations: `hexecutor/native/kernels/adapters/`, `native_kernels/`, `native_stride2/`, `native_runtime/stride2/`, and `native_u16/`. See the canonical memory and kernel guides in `docs/hexecutor/` for the shared contracts.
