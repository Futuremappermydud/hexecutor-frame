// SPDX-License-Identifier: BSL-1.0
// Stand-in for Monado's util/u_misc.h; the vendored Mercury sources use none of its helpers.
#pragma once
#ifndef ARRAY_SIZE
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#endif
