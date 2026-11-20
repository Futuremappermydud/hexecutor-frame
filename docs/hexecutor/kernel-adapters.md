# Kernel adapters and support boundaries

The adapter directories bridge a prepared tensor description to a particular native/provider kernel ABI. They are focused implementations and investigations, not a promise that every model with the same operation name is supported.

The source layout follows kernel families. The 1×1 and padded-channel ABI work lives under [adapters](../../hexecutor/native/kernels/adapters/). Use [convolution](../../hexecutor/native/kernels/convolution/) for HMX leaf layouts, [stride-2](../../hexecutor/native/kernels/stride2/) for caller and staging work, [u16](../../hexecutor/native/kernels/u16/) for halfword layouts and [quantization](../../hexecutor/native/kernels/quantize/) for isolated conversion semantics. Provider-specific probes live under [provider runtime research](../../hexecutor/research/provider-runtime/).

## The acceptance contract

Before invoking an adapter, validate dtype, kernel size, stride, logical/max geometry, padding, channel count, direct/blocked storage, alignment, slice origin, quantization, weight packing, and scratch lifetime. Validate input as well as output aliases. A provider's register/stack layout is an ABI that must be established for the selected leaf, not inferred from one symbol name.

Repeated basis, poisoned-padding, boundary, and full-pool comparisons were useful because an apparently correct output tile can conceal a wrong stride or an out-of-bounds read. The durable conclusions are the supported feature tuple and its rejected cases. A long list of passing individual probes is not a replacement for those guards.

Captured arrays, generated fixture headers, disassembly dumps, and provider binaries remain local and ignored. Generators/checkers remain source. `make_manifest.py` can include this guide with a locally assembled adapter package; that package's captured fixtures and outputs are not checked into this repository.

The prepared resolver must opt into a proved adapter. Missing geometry/quantization coverage rejects before executing the model range. An experimental adapter does not silently fall back to old VTCM scratch while pool 1 occupies VTCM.

See the [kernel guide](kernels.md) and [memory layout](memory-layout.md) for the shared contracts.
