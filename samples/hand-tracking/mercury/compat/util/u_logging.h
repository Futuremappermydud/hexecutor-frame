// Copyright 2020-2026, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Minimal stand-in for Monado's util/u_logging.h: same level order and
 *        macro names, printing to stderr.
 */
#pragma once

#include <stdio.h>

enum u_logging_level
{
	U_LOGGING_TRACE,
	U_LOGGING_DEBUG,
	U_LOGGING_INFO,
	U_LOGGING_WARN,
	U_LOGGING_ERROR,
	U_LOGGING_RAW,
};

#define U_LOG_PRINT_(tag, ...)                                                                                         \
	do {                                                                                                           \
		fprintf(stderr, "[mercury %s] ", tag);                                                                 \
		fprintf(stderr, __VA_ARGS__);                                                                          \
		fputc('\n', stderr);                                                                                   \
	} while (false)

#define U_LOG_IFL_(min, cond_level, tag, ...)                                                                          \
	do {                                                                                                           \
		if ((cond_level) <= (min)) {                                                                           \
			U_LOG_PRINT_(tag, __VA_ARGS__);                                                                \
		}                                                                                                      \
	} while (false)

#define U_LOG_IFL_T(level, ...) U_LOG_IFL_(U_LOGGING_TRACE, level, "T", __VA_ARGS__)
#define U_LOG_IFL_D(level, ...) U_LOG_IFL_(U_LOGGING_DEBUG, level, "D", __VA_ARGS__)
#define U_LOG_IFL_I(level, ...) U_LOG_IFL_(U_LOGGING_INFO, level, "I", __VA_ARGS__)
#define U_LOG_IFL_W(level, ...) U_LOG_IFL_(U_LOGGING_WARN, level, "W", __VA_ARGS__)
#define U_LOG_IFL_E(level, ...) U_LOG_IFL_(U_LOGGING_ERROR, level, "E", __VA_ARGS__)
#define U_LOG_W(...) U_LOG_PRINT_("W", __VA_ARGS__)
#define U_LOG_E(...) U_LOG_PRINT_("E", __VA_ARGS__)
