#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
clang -std=c11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -Wall -Wextra -Werror -Iframe/native_program -Iframe -Iframe/blocked_glue \
  frame/native_program/test_prepared_elementwise_host.c \
  frame/native_program/prepared_elementwise.c \
  frame/native_program/prepared_executor.c \
  frame/native_program/nprg2_runtime.c \
  frame/native_program/elementwise_host_glue_stubs.c \
  -o /tmp/test_prepared_elementwise_host
/tmp/test_prepared_elementwise_host
