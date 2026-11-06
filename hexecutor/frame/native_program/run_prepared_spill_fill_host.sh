#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
cc -std=c11 -Wall -Wextra -Werror -Iframe/native_program \
  frame/native_program/test_prepared_spill_fill.c \
  frame/native_program/prepared_spill_fill.c \
  frame/native_program/prepared_executor.c \
  frame/native_program/nprg2_runtime.c \
  -o /tmp/test_prepared_spill_fill
/tmp/test_prepared_spill_fill
