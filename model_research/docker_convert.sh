#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. Generating Dummy ONNX Model ==="
cd /workspace/model_research
python3 create_dummy_model.py

echo "=== 2. Setting up QNN Environment ==="
# The SDK is mounted at /qairt
: "${PYTHONPATH:=}"
: "${LD_LIBRARY_PATH:=}"
source /qairt/bin/envsetup.sh

echo "=== 3. Converting ONNX to QNN Graph (.cpp / .bin) ==="
# We use qnn-onnx-converter instead of qairt-converter so we can get the .so library
qnn-onnx-converter -i hitnet_dummy.onnx -d input 1,2,616,616 -o qnn_hitnet_dummy.cpp

echo "=== 4. Compiling Shared Library (.so) for 0-boilerplate loading ==="
# This will output to x86_64-linux-clang/libqnn_hitnet_dummy.so or aarch64-linux/lib... depending on the container architecture
qnn-model-lib-generator -c qnn_hitnet_dummy.cpp -b qnn_hitnet_dummy.bin -o qnn_hitnet_dummy_lib

echo "=== 5. Moving Assets ==="
mkdir -p /workspace/bin/models
# Find the generated .so file and move it to models
find . -name "libqnn_hitnet_dummy.so" -exec cp {} /workspace/bin/models/ \;

echo "Successfully generated /workspace/bin/models/libqnn_hitnet_dummy.so"
