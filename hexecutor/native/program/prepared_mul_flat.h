/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef NPRG2_PREPARED_MUL_FLAT_H
#define NPRG2_PREPARED_MUL_FLAT_H
#include "prepared_executor.h"
#ifdef __cplusplus
extern "C" {
#endif
int nprg2_exec_mul_flat_validate(const Nprg2ExecCall *, void *);
int nprg2_exec_mul_flat_run(const Nprg2ExecCall *, void *);
int nprg2_executor_register_mul_flat(Nprg2Executor *);
#ifdef __cplusplus
}
#endif
#endif
