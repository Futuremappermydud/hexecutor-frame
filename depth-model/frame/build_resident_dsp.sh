#!/bin/sh
# Build extended code and its small bootstrap only. Never rebuild the loader.
set -eu
cd "$(dirname "$0")/.."
task_sdk_root="$PWD/.work/hexagon-sdk/6.6.0.0"
docker run --rm --platform linux/amd64 -v "$task_sdk_root:/sdk:ro" -v "$PWD/frame:/src" -w /src \
   -e QDS_VECTORIZE="${QDS_VECTORIZE:-1}" -e QDS_SCALAR_CACHE="${QDS_SCALAR_CACHE:-0}" \
   -e QDS_DYNAMIC_TCM="${QDS_DYNAMIC_TCM:-0}" -e QDS_FALLBACK_TCM="${QDS_FALLBACK_TCM:-0}" \
   -e QDS_PRESERVE_VTCM="${QDS_PRESERVE_VTCM:-0}" \
   -e QDS_BLOCKED_GLUE="${QDS_BLOCKED_GLUE:-0}" \
   -e QDS_FAST_SPILL="${QDS_FAST_SPILL:-0}" \
   -e QDS_BLOCKED_S2D="${QDS_BLOCKED_S2D:-0}" \
   -e QDS_NATIVE_1X1="${QDS_NATIVE_1X1:-0}" \
   -e QDS_NATIVE_PADDED_1X1="${QDS_NATIVE_PADDED_1X1:-0}" \
   -e QDS_NATIVE_3X3="${QDS_NATIVE_3X3:-0}" \
   -e QDS_NATIVE_3X3_SLICES="${QDS_NATIVE_3X3_SLICES:-0}" \
   -e QDS_NATIVE_DIRECT_RANGE="${QDS_NATIVE_DIRECT_RANGE:-0}" \
   -e QDS_NATIVE_PREPARED="${QDS_NATIVE_PREPARED:-0}" \
   -e QDS_DIRECT_BIND_ONLY="${QDS_DIRECT_BIND_ONLY:-0}" \
   -e QDS_DIRECT_COMPACT_INPUT="${QDS_DIRECT_COMPACT_INPUT:-0}" \
   -e QDS_PREPARED_COMPACT_3X3="${QDS_PREPARED_COMPACT_3X3:-0}" \
   -e QDS_PREPARED_PREPROCESS="${QDS_PREPARED_PREPROCESS:-0}" \
   -e QDS_PREPARED_QUANTIZE="${QDS_PREPARED_QUANTIZE:-0}" \
   -e QDS_PREPARED_ELEMENTWISE="${QDS_PREPARED_ELEMENTWISE:-0}" \
   depth-model-hmx-tools:latest sh -eu -c '
cc=/sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang
flags="-mv75 -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin -fno-vectorize -fno-slp-vectorize -ffp-contract=off -fno-strict-aliasing -Wall -Wextra -DQDO_SCALAR_CACHE=$QDS_SCALAR_CACHE"
case "$QDS_SCALAR_CACHE" in 0|7) ;; *) echo "QDS_SCALAR_CACHE must be 0 or 7" >&2; exit 1;; esac
case "$QDS_PRESERVE_VTCM" in 0|1) ;; *) echo "QDS_PRESERVE_VTCM must be 0 or 1" >&2; exit 1;; esac
case "$QDS_BLOCKED_GLUE" in 0|1) ;; *) echo "QDS_BLOCKED_GLUE must be 0 or 1" >&2; exit 1;; esac
case "$QDS_FAST_SPILL" in 0|1) ;; *) echo "QDS_FAST_SPILL must be 0 or 1" >&2; exit 1;; esac
case "$QDS_BLOCKED_S2D" in 0|1) ;; *) echo "QDS_BLOCKED_S2D must be 0 or 1" >&2; exit 1;; esac
case "$QDS_NATIVE_1X1" in 0|1) ;; *) echo "QDS_NATIVE_1X1 must be 0 or 1" >&2; exit 1;; esac
case "$QDS_NATIVE_PADDED_1X1" in 0|1) ;; *) echo "QDS_NATIVE_PADDED_1X1 must be 0 or 1" >&2; exit 1;; esac
if [ "$QDS_NATIVE_PADDED_1X1" = 1 ] && [ "$QDS_NATIVE_1X1" != 1 ]; then
  echo "QDS_NATIVE_PADDED_1X1 requires QDS_NATIVE_1X1=1" >&2; exit 1
fi
case "$QDS_NATIVE_3X3" in 0|1) ;; *) echo "QDS_NATIVE_3X3 must be 0 or 1" >&2; exit 1;; esac
case "$QDS_NATIVE_3X3_SLICES" in 0|1) ;; *) echo "QDS_NATIVE_3X3_SLICES must be 0 or 1" >&2; exit 1;; esac
case "$QDS_NATIVE_DIRECT_RANGE" in 0|1) ;; *) echo "QDS_NATIVE_DIRECT_RANGE must be 0 or 1" >&2; exit 1;; esac
case "$QDS_NATIVE_PREPARED" in 0|1) ;; *) echo "QDS_NATIVE_PREPARED must be 0 or 1" >&2; exit 1;; esac
case "$QDS_DIRECT_BIND_ONLY" in 0|1) ;; *) echo "QDS_DIRECT_BIND_ONLY must be 0 or 1" >&2; exit 1;; esac
case "$QDS_DIRECT_COMPACT_INPUT" in 0|1|2) ;; *) echo "QDS_DIRECT_COMPACT_INPUT must be 0, 1 or 2" >&2; exit 1;; esac
if [ "$QDS_DIRECT_COMPACT_INPUT" != 0 ] && { [ "$QDS_NATIVE_DIRECT_RANGE" != 1 ] || [ "$QDS_NATIVE_PREPARED" != 0 ] || [ "$QDS_DIRECT_BIND_ONLY" != 0 ]; }; then
  echo "QDS_DIRECT_COMPACT_INPUT requires direct-range and excludes prepared/bind-only" >&2; exit 1
fi
case "$QDS_PREPARED_COMPACT_3X3" in 0|1) ;; *) echo "QDS_PREPARED_COMPACT_3X3 must be 0 or 1" >&2; exit 1;; esac
if [ "$QDS_PREPARED_COMPACT_3X3" = 1 ] && { [ "$QDS_NATIVE_PREPARED" != 1 ] || [ "$QDS_DIRECT_COMPACT_INPUT" != 0 ]; }; then
  echo "QDS_PREPARED_COMPACT_3X3 requires prepared dispatch and excludes direct input controls" >&2; exit 1
fi
case "$QDS_PREPARED_PREPROCESS" in 0|1) ;; *) echo "QDS_PREPARED_PREPROCESS must be 0 or 1" >&2; exit 1;; esac
if [ "$QDS_PREPARED_PREPROCESS" = 1 ] && [ "$QDS_NATIVE_PREPARED" != 1 ]; then
  echo "QDS_PREPARED_PREPROCESS requires prepared dispatch" >&2; exit 1
fi
case "$QDS_PREPARED_QUANTIZE" in 0|1) ;; *) echo "QDS_PREPARED_QUANTIZE must be 0 or 1" >&2; exit 1;; esac
if [ "$QDS_PREPARED_QUANTIZE" = 1 ] && [ "$QDS_PREPARED_PREPROCESS" != 1 ]; then
  echo "QDS_PREPARED_QUANTIZE requires prepared preprocess" >&2; exit 1
fi
case "$QDS_PREPARED_ELEMENTWISE" in 0|1) ;; *) echo "QDS_PREPARED_ELEMENTWISE must be 0 or 1" >&2; exit 1;; esac
if [ "$QDS_PREPARED_ELEMENTWISE" = 1 ] && { [ "$QDS_NATIVE_PREPARED" != 1 ] || [ "$QDS_BLOCKED_GLUE" != 1 ]; }; then
  echo "QDS_PREPARED_ELEMENTWISE requires prepared and blocked glue" >&2; exit 1
fi
if [ "$QDS_DIRECT_BIND_ONLY" = 1 ] && { [ "$QDS_NATIVE_DIRECT_RANGE" != 1 ] || [ "$QDS_NATIVE_PREPARED" != 0 ]; }; then
  echo "QDS_DIRECT_BIND_ONLY requires direct-range and disables prepared dispatch" >&2; exit 1
fi
if [ "$QDS_NATIVE_PREPARED" = 1 ] && [ "$QDS_NATIVE_DIRECT_RANGE" != 1 ]; then
  echo "QDS_NATIVE_PREPARED requires QDS_NATIVE_DIRECT_RANGE=1" >&2; exit 1
fi
if [ "$QDS_NATIVE_DIRECT_RANGE" = 1 ] && [ "$QDS_NATIVE_3X3" != 1 ]; then
  echo "QDS_NATIVE_DIRECT_RANGE requires QDS_NATIVE_3X3=1" >&2; exit 1
fi
if [ "$QDS_NATIVE_3X3_SLICES" = 1 ] && [ "$QDS_NATIVE_3X3" != 1 ]; then
  echo "QDS_NATIVE_3X3_SLICES requires QDS_NATIVE_3X3=1" >&2; exit 1
fi
if [ "$QDS_NATIVE_3X3" = 1 ] && [ "$QDS_NATIVE_1X1" != 1 ]; then
  echo "QDS_NATIVE_3X3 requires QDS_NATIVE_1X1=1" >&2; exit 1
fi
if [ "$QDS_NATIVE_1X1" = 1 ] && [ "$QDS_PRESERVE_VTCM" != 1 ]; then
  echo "QDS_NATIVE_1X1 requires QDS_PRESERVE_VTCM=1" >&2; exit 1
fi
if [ "$QDS_BLOCKED_S2D" = 1 ] && [ "$QDS_BLOCKED_GLUE" != 1 ]; then
  echo "QDS_BLOCKED_S2D requires QDS_BLOCKED_GLUE=1" >&2; exit 1
fi
case "$QDS_VECTORIZE:$QDS_DYNAMIC_TCM:$QDS_FALLBACK_TCM" in
  [01]:[01]:[012]) ;; *) echo "Vectorize/dynamic must be 0/1; fallback must be 0/1/2" >&2; exit 1;;
esac
bootstrap=build/qd_graph_boot-c$QDS_SCALAR_CACHE.elf
probe=build/qd_owned_mem_probe-c$QDS_SCALAR_CACHE.elf
engine=build/qd_graph_dsp-scalar.elf
engine_flags="$flags"
if [ "$QDS_VECTORIZE" = 1 ]; then
  engine_flags="$flags -O3 -fvectorize -fslp-vectorize"
  engine=build/qd_graph_dsp.elf
fi
if [ "$QDS_SCALAR_CACHE" = 7 ]; then engine="${engine%.elf}-cached.elf"; fi
engine_flags="$engine_flags -DQDS_DYNAMIC_TCM=$QDS_DYNAMIC_TCM -DQDS_FALLBACK_TCM=$QDS_FALLBACK_TCM"
engine_flags="$engine_flags -DQDS_BLOCKED_GLUE=$QDS_BLOCKED_GLUE"
engine_flags="$engine_flags -DQDS_FAST_SPILL=$QDS_FAST_SPILL"
engine_flags="$engine_flags -DQDS_BLOCKED_S2D=$QDS_BLOCKED_S2D"
engine_flags="$engine_flags -DQDS_NATIVE_1X1=$QDS_NATIVE_1X1"
engine_flags="$engine_flags -DQDS_NATIVE_PADDED_1X1=$QDS_NATIVE_PADDED_1X1"
engine_flags="$engine_flags -DQDS_NATIVE_3X3=$QDS_NATIVE_3X3"
engine_flags="$engine_flags -DQDS_NATIVE_3X3_SLICES=$QDS_NATIVE_3X3_SLICES"
engine_flags="$engine_flags -DQDS_NATIVE_DIRECT_RANGE=$QDS_NATIVE_DIRECT_RANGE"
engine_flags="$engine_flags -DQDS_NATIVE_PREPARED=$QDS_NATIVE_PREPARED"
engine_flags="$engine_flags -DQDS_DIRECT_BIND_ONLY=$QDS_DIRECT_BIND_ONLY"
engine_flags="$engine_flags -DQDS_DIRECT_COMPACT_INPUT=$QDS_DIRECT_COMPACT_INPUT"
engine_flags="$engine_flags -DQDS_PREPARED_COMPACT_3X3=$QDS_PREPARED_COMPACT_3X3"
engine_flags="$engine_flags -DQDS_PREPARED_PREPROCESS=$QDS_PREPARED_PREPROCESS"
engine_flags="$engine_flags -DQDS_PREPARED_QUANTIZE=$QDS_PREPARED_QUANTIZE"
engine_flags="$engine_flags -DQDS_PREPARED_ELEMENTWISE=$QDS_PREPARED_ELEMENTWISE"
engine_flags="$engine_flags -DQDS_PRESERVE_VTCM=$QDS_PRESERVE_VTCM"
glue_src=
native_src=
if [ "$QDS_DYNAMIC_TCM" = 1 ]; then engine="${engine%.elf}-dynamic-tcm.elf"; fi
if [ "$QDS_FALLBACK_TCM" = 1 ]; then engine="${engine%.elf}-fallback-tcm.elf"; fi
if [ "$QDS_FALLBACK_TCM" = 2 ]; then engine="${engine%.elf}-fallback-l2.elf"; fi
if [ "$QDS_BLOCKED_GLUE" = 1 ]; then
  engine="${engine%.elf}-blocked-glue.elf"
  glue_src="blocked_glue/linearclip_u16.c blocked_glue/linearclip_u8_u16.c blocked_glue/sign_levels.c blocked_glue/sign_blocked.c blocked_glue/add_u8.c"
fi
if [ "$QDS_FAST_SPILL" = 1 ]; then engine="${engine%.elf}-fast-spill.elf"; fi
if [ "$QDS_BLOCKED_S2D" = 1 ]; then engine="${engine%.elf}-blocked-s2d.elf"; fi
if [ "$QDS_NATIVE_1X1" = 1 ]; then
  engine_flags="$engine_flags -mhmx"
  engine="${engine%.elf}-native-1x1.elf"
  native_src="native_adapter/qd_native_1x1_adapter.c qd_qnn_hmx_1x1.S"
fi
if [ "$QDS_NATIVE_3X3" = 1 ]; then
  engine="${engine%.elf}-native-3x3.elf"
  native_src="$native_src native_kernels/qd_native_3x3_adapter.c native_kernels/qd_qnn_hmx_stride1_generic.S native_kernels/qd_set_hmx_params_stride1.S"
fi
if [ "$QDS_NATIVE_3X3_SLICES" = 1 ]; then engine="${engine%.elf}-native-slices.elf"; fi
if [ "$QDS_NATIVE_PADDED_1X1" = 1 ]; then engine="${engine%.elf}-native-padded.elf"; fi
if [ "$QDS_NATIVE_DIRECT_RANGE" = 1 ]; then
  engine="${engine%.elf}-direct-range.elf"
  native_src="$native_src native_runtime/qd_native_direct.c"
fi
if [ "$QDS_NATIVE_PREPARED" = 1 ]; then
  engine="${engine%.elf}-prepared.elf"
  native_src="$native_src native_program/nprg2_runtime.c native_program/prepared_executor.c native_program/prepared_native.c native_program/prepared_io.c native_program/prepared_runner.c"
fi
if [ "$QDS_DIRECT_COMPACT_INPUT" != 0 ]; then engine="${engine%.elf}-compact-input.elf"; fi
if [ "$QDS_DIRECT_COMPACT_INPUT" = 2 ]; then engine="${engine%.elf}-aliases.elf"; fi
if [ "$QDS_DIRECT_BIND_ONLY" = 1 ]; then engine="${engine%.elf}-bind-only.elf"; fi
if [ "$QDS_PREPARED_ELEMENTWISE" = 1 ]; then
  engine="${engine%.elf}-elementwise.elf"
  native_src="$native_src native_program/prepared_elementwise.c"
fi
if [ "$QDS_PREPARED_PREPROCESS" = 1 ]; then
  engine="${engine%.elf}-preprocess.elf"
  native_src="$native_src native_program/prepared_preprocess.c"
fi
if [ "$QDS_PREPARED_QUANTIZE" = 1 ]; then
  engine="${engine%.elf}-quantize.elf"
  native_src="$native_src native_program/prepared_quantize.c native_program/prepared_quantize_core.c"
fi
if [ "$QDS_PREPARED_COMPACT_3X3" = 1 ]; then engine="${engine%.elf}-compact-3x3.elf"; fi
if [ "$QDS_PRESERVE_VTCM" = 1 ]; then
  engine="${engine%.elf}-vtcm-lease.elf"
  bootstrap="${bootstrap%.elf}-preserve-vtcm-guarded-entry.elf"
fi
$cc $engine_flags -ffunction-sections -fdata-sections -nostdlib -Wl,--gc-sections,-T,qd_graph_dsp.ld \
  qd_graph_dsp.c qd_resident.c qd_resident_conv.c qd_hmx_ops_encoded.S $glue_src $native_src \
  -Wl,--start-group -lhexagon -lgcc -lc -lm -Wl,--end-group -o "$engine"
# Experiments only write a separate engine. Use the existing mode-specific
# bootstrap; never replace the normal bootstrap/probe during these builds.
if [ "$QDS_PRESERVE_VTCM" = 1 ]; then
  $cc $flags -DQDS_PRESERVE_VTCM=1 -DQDS_ENTRY_GUARD=1 -nostdlib -Wl,-T,qd_payload.ld qd_graph_boot.c qd_graph_stack.S -o "$bootstrap"
  echo "Experimental engine: $engine; preserving bootstrap: $bootstrap"
  exit 0
fi
if [ "$QDS_DYNAMIC_TCM" = 1 ] || [ "$QDS_FALLBACK_TCM" != 0 ] || [ "$QDS_BLOCKED_GLUE" = 1 ] || [ "$QDS_FAST_SPILL" = 1 ]; then
  echo "Experimental engine: $engine; use existing bootstrap: $bootstrap"
  exit 0
fi
$cc $flags -nostdlib -Wl,-T,qd_payload.ld qd_graph_boot.c qd_graph_stack.S -o "$bootstrap"
$cc $flags -nostdlib -Wl,-T,qd_payload.ld bench/qd_owned_mem_probe.c -o "$probe"
if [ "$QDS_SCALAR_CACHE" = 0 ]; then
  cp "$bootstrap" build/qd_graph_boot.elf
  cp "$probe" build/qd_owned_mem_probe.elf
fi
echo "Engine: $engine; bootstrap: $bootstrap; ownership probe: $probe"
'
