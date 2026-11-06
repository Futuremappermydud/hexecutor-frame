# Native stride-2 convolution investigation

This offline proof package covers the remaining `ConvLayer_s2.opt@CB*2.Xwb.Fi` family. It makes no changes to the graph runtime or existing native adapters.

## Command census and CPU oracle

`census_s2.py` parses both compiled QDRX images: each contains 80 S2 commands. Their weight/input channel families are 32→64 with input CI 128 (40 commands), 64→128 with input CI 256 (20), and 128→128 with input CI 512 (20). Spatial sizes vary within these groups.

`make_s2_fixture.py [sequence]` captures the blocked activation, raw Xwb, FI, and full physical output block table after the CPU prefix and target command. `check_s2_oracle.py` independently unpacks Xwb, applies `S2_TAPS` and the exact 2x2 channel lowering, then computes the int64 convolution and requantization. The three representative primary commands all match the CPU output exactly in the host oracle:

| Sequence | Source weight CI→CO | S2 input CI | Output shape | Host int64 oracle |
|---:|---:|---:|---:|---|
| 224 | 32→64 | 128 | 8×128×64 | exact |
| 363 | 64→128 | 256 | 8×64×128 | exact |
| 3407 | 128→128 | 512 | 8×32×128 | exact |

The stored `s2_fixture_<sequence>.bin` and `s2_oracle_<sequence>.json` files preserve each source fixture and oracle report. The latter sequence has output `slice_offset=52`; its simulator comparison starts at that physical output block, which is a required detail for native output table construction.

## Generic HMX simulation proof

The 2x2 HMX setter/configuration did not prove the last input row and is not used. Instead, the exact lowered 2x2 weights are embedded in a zero-filled 3x3 kernel and sent through the already-validated generic stride1 leaf with canonical `(0x700, 3, 3, CI, 7, 1, 12)` setup. Raw Xwb positions map as `raw_kx=2 → logical dx=0`, `raw_kx=1 → logical dx=1`; raw `kx=0` and `ky=2` are zero. The input is padded at `(+7,+7)` and the otherwise-unused third input block row and final block column point to a zero page. This compact physical block map keeps the CI512 proof inside the simulator's 1 MiB HMX VTCM.

Run `run_s2_family_matrix.sh` to regenerate each CPU-prefix fixture, run its independent oracle, build the three simulator variants, and compare every active physical output byte. Results are:

| Sequence | Active physical bytes | Mismatches | Difference |
|---:|---:|---:|---|
| 224 | 65,536 | 1,357 | all +1 |
| 363 | 65,536 | 2,283 | all +1 |
| 3407 | 32,768 | 781 | all +1 |

All three V75 simulator runs pass page, weight, and table-tail guards. The generic leaf's established rounding envelope is 0/+1. The physical model command remains `ConvLayer_s2`; the HMX candidate is mathematically the same lowered convolution and uses the proven stride1 leaf. No hardware, SSH, or ADB was used.

The simulation source and outputs are proof artifacts only. A production adapter still needs a metadata/liveness gate, safe compact VTCM placement, output block-slice handling, and full-model exactness/performance validation by the root integration owner.
