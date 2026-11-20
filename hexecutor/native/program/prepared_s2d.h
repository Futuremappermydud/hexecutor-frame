/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef NPRG2_PREPARED_S2D_H
#define NPRG2_PREPARED_S2D_H

#include "prepared_executor.h"

/* Strict CB u8 stride-2 space-to-depth path using the audited blocked layout. */
int nprg2_executor_register_s2d(Nprg2Executor *executor);

#endif
