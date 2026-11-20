/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef NPRG2_PREPARED_FLAT_VTCM_H
#define NPRG2_PREPARED_FLAT_VTCM_H

#include "prepared_executor.h"

int nprg2_exec_flat_vtcm_validate(const Nprg2ExecCall *call, void *ctx);
int nprg2_exec_flat_vtcm_run(const Nprg2ExecCall *call, void *ctx);
int nprg2_executor_register_flat_vtcm(Nprg2Executor *executor);

#endif
