// SPDX-License-Identifier: BSL-1.0
// Stand-in for Monado's os/os_time.h and the U_TIME constants the optimizer uses.
#pragma once
#include <stdint.h>
#include <time.h>

#define U_TIME_1MS_IN_NS (1000 * 1000)
#define U_TIME_1S_IN_NS (1000 * 1000 * 1000)

static inline uint64_t
os_monotonic_get_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * U_TIME_1S_IN_NS + (uint64_t)ts.tv_nsec;
}
