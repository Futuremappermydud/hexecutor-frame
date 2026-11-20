/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef NPRG2_PREPARED_ABS_H
#define NPRG2_PREPARED_ABS_H

#include "prepared_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

int nprg2_exec_abs_validate(const Nprg2ExecCall *call, void *ctx);
int nprg2_exec_abs_run(const Nprg2ExecCall *call, void *ctx);

int nprg2_executor_register_abs(Nprg2Executor *executor);

#ifdef __cplusplus
}
#endif

#endif /* NPRG2_PREPARED_ABS_H */
