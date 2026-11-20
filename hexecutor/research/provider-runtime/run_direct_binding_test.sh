#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
set -eu
cd "$(dirname "$0")/."
cc -std=c11 -O2 -Wall -Wextra -Werror -DQD_DIRECT_ALLOW_MOCK_BASE=1 \
  direct_binding_test.c qd_native_direct.c -o /tmp/qd_direct_binding_test
/tmp/qd_direct_binding_test
