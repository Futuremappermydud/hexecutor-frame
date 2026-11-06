#ifndef NPRG2_PREPARED_RESHAPE_CB_H
#define NPRG2_PREPARED_RESHAPE_CB_H

#include "prepared_executor.h"

/* Reshape.hw@CB*2: [1,1,64,128] -> [1,8,8,128] blocked u8. */
int nprg2_exec_reshape_cb_validate(const Nprg2ExecCall *call, void *ctx);
int nprg2_exec_reshape_cb_run(const Nprg2ExecCall *call, void *ctx);
int nprg2_executor_register_reshape_cb(Nprg2Executor *executor);

#endif
