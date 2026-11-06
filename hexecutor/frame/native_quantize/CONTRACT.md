# Prepared Quantize prefix contract

This is an offline proof for the first production `Quantize@FH.Ff` command in
each QDRX. It does not authorize other Quantize nodes or device execution.

| Model | command / sequence | opcode / variant | input → output | pool1 byte offsets |
|---|---:|---|---|---:|
| primary | 60 / 107 | opcode 28, variant ID 48 | tensor 66 Ff f32 → tensor 106 FH u16 | 1,089,536 → 7,589,888 |
| alternate | 54 / 100 | opcode 28, variant ID 48 | tensor 54 Ff f32 → tensor 94 FH u16 | 3,997,696 → 2,775,040 |

Both tensors are direct in pool 1, dimensions and max dimensions `[1,2,4,4]`,
zero padding, and 32 elements. Input is 128 bytes and output is 64 bytes; the
intervals are disjoint. Output metadata is zero point 32768 and float32 scale
`0.0003051804378628731`. The sole leaf call is `quantize_sf_to_u16`, with
zero-point 32768 and count 32. All other call fields and the output scale must
be checked against the selected QDRX binding before a runtime handler accepts.

The decoded CPU reference computes `inv = (double)(float)(1.0 / scale)`;
for each finite float32 input `x`, it computes `floor((double)x * inv + 0.5) +
zero_point`, then clamps to `[0,65535]` and stores uint16. The reciprocal is
rounded to float32 before widening; the multiplication and rounding expression
are evaluated in double. Fixture source and output hashes match across both
models because these early prefixes have identical values.

`capture_cpu_prefix.py` runs NativeGraph CPU commands `[0,index)` and then the
single Quantize command. It writes the source and reference output plus the
entire 8 MiB pool1 preimage. The strict C harness loads that preimage, executes
the independent numerical core in-place at the decoded offsets, checks all
pool1 bytes outside the 64-byte output, and compares the output exactly. It
also checks canaries and that invalid length, overlap, or nonfinite input
rejects before any destination write.

The reusable C core deliberately accepts only this proven 32-element profile.
Before adding the dispatcher adapter, validate model-specific command/sequence,
opcode/variant, tensor IDs, types/classes, exact shape/padding, direct pool/offset,
leaf call identity/arguments, zp and exact float32 scale bits, bounds, pointer
overflow, and non-overlap. Preflight must reject all other Quantize metadata;
there is no fallback path.

## Reproduction

From the project root:

```sh
cd depth-model
.venv-cpu/bin/python frame/native_quantize/capture_cpu_prefix.py
cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
  frame/native_quantize/quantize_u16.c \
  frame/native_quantize/test_quantize_cpu_prefix.c -lm -o /tmp/test-native-quantize
/tmp/test-native-quantize frame/native_quantize/fixtures primary
/tmp/test-native-quantize frame/native_quantize/fixtures alternate
```

The fixtures are tied to the QDRX hashes recorded in each JSON manifest.
