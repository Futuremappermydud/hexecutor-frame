#!/bin/sh
# Build only the isolated Android vendor-RPC client. No Frame payloads change.
set -eu
cd "$(dirname "$0")/.."
: "${QDM_NDK_ROOT:=$HOME/Library/Android/sdk/ndk/27.3.13750724}"
case "$(uname -s)" in
  Darwin) qdm_host=darwin-x86_64 ;;
  Linux) qdm_host=linux-x86_64 ;;
  *) echo 'Unsupported NDK host' >&2; exit 1 ;;
esac
mkdir -p frame/build
"$QDM_NDK_ROOT/toolchains/llvm/prebuilt/$qdm_host/bin/aarch64-linux-android29-clang" \
  -Wall -Wextra -Werror -O2 frame/bench/model_metadata_qnn_probe.c -ldl \
  -o frame/build/model_metadata_qnn_probe
