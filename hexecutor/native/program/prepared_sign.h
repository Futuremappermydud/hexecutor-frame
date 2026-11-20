/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef NPRG2_PREPARED_SIGN_H
#define NPRG2_PREPARED_SIGN_H

#include "prepared_executor.h"

/* Prepared u16 ElementWiseSign for direct FH and validated blocked CH tensors. */
int nprg2_exec_sign_validate(const Nprg2ExecCall *call, void *ctx);
int nprg2_exec_sign_run(const Nprg2ExecCall *call, void *ctx);
int nprg2_executor_register_sign(Nprg2Executor *executor);

#endif
