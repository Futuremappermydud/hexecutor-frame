#ifndef NPRG2_PREPARED_ADDSUB_FLAT_H
#define NPRG2_PREPARED_ADDSUB_FLAT_H
#include "prepared_executor.h"
#ifdef __cplusplus
extern "C" {
#endif
int nprg2_exec_addsub_flat_validate(const Nprg2ExecCall *, void *);
int nprg2_exec_addsub_flat_run(const Nprg2ExecCall *, void *);
int nprg2_executor_register_addsub_flat(Nprg2Executor *);
#ifdef __cplusplus
}
#endif
#endif
