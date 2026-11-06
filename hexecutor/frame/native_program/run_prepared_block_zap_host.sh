#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
clang -std=c11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -Wall -Wextra -Werror -Iframe/native_program \
  frame/native_program/test_prepared_block_zap.c \
  frame/native_program/prepared_block_zap.c \
  frame/native_program/prepared_executor.c \
  frame/native_program/nprg2_runtime.c \
  -o /tmp/test_prepared_block_zap
/tmp/test_prepared_block_zap
