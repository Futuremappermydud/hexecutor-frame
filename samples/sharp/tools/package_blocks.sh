#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
# From the repository root, lower exported trained blocks in a local artifact tree.
set -eu
: "${SHARP_QAIRT_SDK:?set SHARP_QAIRT_SDK to the QAIRT installation directory}"
: "${SHARP_QAIRT_IMAGE:=hexecutor-qairt-tools}"
: "${SHARP_PYTHON:=python3}"
if [ "$#" -ne 4 ]; then
    printf '%s\n' "usage: package_blocks.sh RELATIVE_ARTIFACT_DIR pe|ie FIRST COUNT" >&2
    exit 2
fi
base=$1
encoder=$2
first=$3
count=$4
case "$base" in /*|*..*) exit 2 ;; esac
case "$encoder" in pe|ie) ;; *) exit 2 ;; esac
case "$first:$count" in *[!0-9:]*) exit 2 ;; esac
if [ "$first" -ge 24 ] || [ "$count" -le 0 ] || [ "$((first + count))" -gt 24 ]; then
    exit 2
fi

container() {
    docker run --rm --pull never --platform linux/amd64 \
        -v "$SHARP_QAIRT_SDK:/sdk:ro" -v "$PWD:/workspace" -w /workspace \
        "$SHARP_QAIRT_IMAGE" "$@"
}
mkdir -p "$base/qa"
hook="$base/qa/qnn_trace_hook_${encoder}_${first}.so"
container clang++ -shared -fPIC -O2 samples/depth-model/sim/qnn_trace_hook.cpp \
    -o "$hook" -ldl -pthread
worker="$base/qa/known_${encoder}_${first}.json"
printf '{}\n' > "$worker"
index=$first
while [ "$index" -lt "$((first + count))" ]; do
    stem=$(printf 'block_%02d' "$index")
    source="$base/$encoder/$stem"
    out="$base/qa/$encoder/$stem"
    mkdir -p "$out"
    lock="$out/.build-lock"
    if ! mkdir "$lock"; then
        printf '%s\n' "another build owns $out" >&2
        exit 2
    fi
    trap 'rmdir "$lock"' 0
    trap 'exit 1' HUP INT TERM
    if [ -e "$out/package" ] || [ -e "$out/program.q16p" ] || [ -e "$out/program.q16c" ]; then
        printf '%s\n' "package exists: $out" >&2
        exit 2
    fi
    work=$(mktemp -d "$out/work.XXXXXX")
    trace="$work/trace"
    mkdir "$trace"
    printf '%s\n' "QAIRT work directory: $work"
    printf 'tokens:=%s_input_00.f32\n' "$source" > "$trace/input_list.txt"
    container sh -eu -c '
        export PYTHONPATH=/sdk/lib/python LD_LIBRARY_PATH=/sdk/lib/x86_64-linux-clang
        bins=/sdk/bin/x86_64-linux-clang
        "$bins/qairt-converter" --input_network "$1.onnx" --output_path "$2/block_fp32.dlc"
        "$bins/qairt-quantizer" --input_dlc "$2/block_fp32.dlc" --output_dlc "$2/block_w8a16.dlc" \
            --input_list "${1}_calibration.txt" --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32
    ' sh "$source" "$work" > "$work/quantize.log" 2>&1
    container python hexecutor/offline/compile_qnn.py --sdk /sdk \
        --model "$work/block_w8a16.dlc" --output "$work/compiled" --name block \
        --graph-name block_fp32 --context-only
    context="$work/compiled/block-v75.bin"
    decoded="$work/decoded"
    "$SHARP_PYTHON" samples/depth-model/tools/htp_graph.py "$context" "$decoded" --no-source-names

    run_qnn() {
        container sh -eu -c '
            export LD_LIBRARY_PATH=/sdk/lib/x86_64-linux-clang
            export ADSP_LIBRARY_PATH="/sdk/lib/hexagon-v75/unsigned"
            export LD_PRELOAD="/workspace/$1"
            shift
            exec "$@"
        ' sh "$hook" env "$@"
    }
    simulate() {
        run_qnn "$@" /sdk/bin/x86_64-linux-clang/qnn-net-run \
            --backend /sdk/lib/x86_64-linux-clang/libQnnHtp.so --retrieve_context "$context" \
            --input_list "$trace/input_list.txt" --output_dir "$trace" --log_level warn
    }
    simulate "QD_TRACE_LOG=$trace/trace.log" QD_WORKER_CHECKPOINTS=1 > "$work/trace.log" 2>&1
    "$SHARP_PYTHON" hexecutor/offline/qnn16_lower.py "$decoded" "$trace" --known "$worker" \
        --write-tensor-config "$trace/tensor-config.txt"
    simulate "QD_TENSOR_CONFIG=$trace/tensor-config.txt" "QD_TENSORS=$trace/tensors.bin" \
        > "$work/capture.log" 2>&1
    "$SHARP_PYTHON" hexecutor/offline/qnn16_program.py "$decoded" "$trace" --known "$worker" \
        --tensors "$trace/tensors.bin" --context-binary "$context" --output "$work/program"
    "$SHARP_PYTHON" hexecutor/tests/test_qnn16_program.py "$work/program" --run "$trace" --inputs "$base/$encoder"
    mkdir "$work/package"
    mv "$work/program.q16p" "$work/program.q16c" "$work/program.q16.json" "$work/package/"
    cp "$trace/Result_0/next_tokens.raw" "$work/package/qa-next_tokens.f32"
    cp "$trace/execution_metadata.yaml" "$work/package/qa-run.yaml"
    mv "$work/package" "$out/package"
    rm -rf "$work"
    rmdir "$lock"
    trap - 0 HUP INT TERM
    index=$((index + 1))
done
