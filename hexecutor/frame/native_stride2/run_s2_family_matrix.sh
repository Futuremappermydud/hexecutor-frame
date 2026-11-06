#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."

run_case() {
  seq="$1"; ci="$2"; co="$3"; ih="$4"; iw="$5"; imw="$6"; oh="$7"; ow="$8"
  .venv-cpu/bin/python frame/native_stride2/make_s2_fixture.py "$seq"
  cp frame/native_stride2/s2_cpu_prefix_fixture.bin "frame/native_stride2/s2_fixture_${seq}.bin"
  .venv-cpu/bin/python frame/native_stride2/check_s2_oracle.py > "frame/native_stride2/s2_oracle_${seq}.json"
  cp frame/native_stride2/s2_oracle_report.json "frame/native_stride2/s2_oracle_${seq}.json"
  docker run --rm --platform linux/amd64 \
    -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD/frame:/src" -w /src \
    depth-model-hmx-tools:latest \
    /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
    -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin \
    -fno-stack-protector -Wall -Wextra \
    -DCI="$ci" -DCO="$co" -DSEQUENCE="$seq" -DIN_H="$ih" -DIN_W="$iw" \
    -DIN_MAX_W="$imw" -DOUT_H="$oh" -DOUT_W="$ow" \
    native_stride2/qnn_hmx_generic_s2_family_sim.c \
    native_kernels/qd_qnn_hmx_stride1_generic.S \
    native_kernels/qd_set_hmx_params_stride1.S qd_hmx_ops.S \
    -o "native_stride2/qnn_hmx_generic_s2_family_sim_${seq}.elf"
  docker run --rm --platform linux/amd64 \
    -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD/frame:/src" -w /src \
    depth-model-hmx-tools:latest \
    /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim -mv75 --quiet \
    "native_stride2/qnn_hmx_generic_s2_family_sim_${seq}.elf" \
    > "frame/native_stride2/s2_sim_${seq}.log"
  grep 'PASS actual S2 lowered 2x2 generic leaf' "frame/native_stride2/s2_sim_${seq}.log"
  tail -1 "frame/native_stride2/s2_sim_${seq}.log"
}

run_case 224 128 64 9 130 136 8 128
run_case 363 256 128 9 66 72 8 64
run_case 3407 512 128 9 34 40 8 32
