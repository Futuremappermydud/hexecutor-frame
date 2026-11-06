#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
clang -std=c11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -Wall -Wextra -Werror -Ihexecutor/native/program -Ihexecutor/native -Ihexecutor/native/kernels/blocked \
  hexecutor/native/program/test_prepared_elementwise_host.c \
  hexecutor/native/program/prepared_elementwise.c \
  hexecutor/native/program/prepared_executor.c \
  hexecutor/native/program/prepared_dma_ring.c \
  hexecutor/native/program/nprg2_runtime.c \
  hexecutor/native/program/elementwise_host_glue_stubs.c \
  hexecutor/native/kernels/blocked/sign_levels.c \
  -o /tmp/test_prepared_elementwise_host
/tmp/test_prepared_elementwise_host
