#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
set -eu
cd "$(dirname "$0")/../../.."
clang -std=c11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -Wall -Wextra -Werror -Ihexecutor/native/program \
  hexecutor/native/program/test_prepared_block_zap.c \
  hexecutor/native/program/prepared_block_zap.c \
  hexecutor/native/program/prepared_executor.c \
  hexecutor/native/program/nprg2_runtime.c \
  -o /tmp/test_prepared_block_zap
/tmp/test_prepared_block_zap
