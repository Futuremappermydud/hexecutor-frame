#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
cc -std=c11 -Wall -Wextra -Werror -Ihexecutor/native/program \
  hexecutor/native/program/test_prepared_spill_fill.c \
  hexecutor/native/program/prepared_spill_fill.c \
  hexecutor/native/program/prepared_executor.c \
  hexecutor/native/program/nprg2_runtime.c \
  -o /tmp/test_prepared_spill_fill
/tmp/test_prepared_spill_fill
