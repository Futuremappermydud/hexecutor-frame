/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef NPRG2_PREPARED_IO_H
#define NPRG2_PREPARED_IO_H

#include "prepared_executor.h"

/* Registers only side-effect-free QDR NOP records and the strict u8 direct
 * OutputSlicePad profile implemented by the resident copy loop. */
int nprg2_executor_register_core_safe_ops(Nprg2Executor *executor);

#endif
