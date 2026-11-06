# Generic V75 spatial leaf caller trace

## Full provider search

Decoded the complete local V75 provider, not just saved snippets:

```sh
docker run --rm --platform linux/amd64 \
  -v /Users/owenelliott/Downloads/qairt/2.50.0.260828/lib/hexagon-v75/unsigned:/qnn:ro \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" \
  depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump \
  -d --mcpu=hexagonv75 --mattr=+hmx,+hvx,+hvx-length128b \
  /qnn/libQnnHtpV75.so > frame/native_kernels/full_provider_v75_decoded.asm
```

Provider SHA256: `4282ff5880c0b0b609da31769a65a101f9a2beb6b536554cbb14753370c2ab1a`.
Decoded dump SHA256: `58c55f12507180f58f23187a35d40bb781532349c7fc874ae75bd3333be7e1a3`.

The full dump contains five direct calls to `hmx_v73_convbbb_stride1` at `0x495634`, `0x495ac0`, `0x495e74`, `0x496234`, and `0x4965f4`. They share the same wrapper pattern. The first call sequence around `0x495608` is:

```asm
495608: { r1:r0 = combine(r19,r20)
49560c:   if (!p0.new) r3:r2 = combine(r18,r17)
495610:   if (!p0.new) jump 0x495634 }
...
495634: { call 0x388cc0 <hmx_v73_convbbb_stride1>
495638:   r1:r0 = combine(r19,r20)
49563c:   r5:r4 = combine(r16,r21) }
```

Because the call and both register pairs share the packet braces, the callee observes the packet’s writes. The leaf receives six register arguments: `r0=r20`, `r1=r19`, `r2/r3` from the preceding packet when that path is taken, `r4=r21`, and `r5=r16`. The leaf reads its configuration from `r4` (`+0x30`, `+0x18`, etc.), matching `r4=r21` as the generated config pointer.

## Mode-pointer source chain

The outer helper starts at `0x48a420`; `r16` is its incoming `this` pointer. At `0x48a48c`, `r17 = r16 + 0x20`. The call packet at `0x48a49c` invokes the inner helper at `0x4943e0` with `r0=r17` (the same packet also sets the other arguments). The inner helper begins with `r16=r0` at `0x4943e0`. At the generic-leaf call packet, it sets `r5=r16`. Therefore the generic leaf’s fifth argument points to the outer object’s `this+0x20` subobject.

The generated config pointer `r21` is sourced from `memw(memw(r3+8)+8)` in the inner helper and is stored into the config object at `+0x84`. It is passed in `r4`, not `r5`.

## Remaining question

The leaf reads word 0 from the `r5` object at `0x388d94`. If it is `1`, it loads word 1 (offset +4) into `r18`; the enabled path later consumes this second value as the `cvt.ub=acc(r18)` accumulator selector. So `{1,0}` has the right two-word shape (enable flag plus selector), but the graph-call selector value and its variation for CI64 still need proof. The direct simulator fixture does not itself establish those runtime field values; CI64 probes made with guessed values remain inconclusive.
