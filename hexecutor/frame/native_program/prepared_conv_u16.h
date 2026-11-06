#ifndef NPRG2_PREPARED_CONV_U16_H
#define NPRG2_PREPARED_CONV_U16_H

#include "prepared_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Validates bound tensor metadata and storage for Opcode 6 (QDR_HMX_CONV)
 * Variant 34 (CH*2.Xpb.Fi.t) and Variant 35 (CH*2.Xwb.Fi.t). */
int nprg2_exec_conv_u16_validate(const Nprg2ExecCall *call, void *ctx);

/* Executes bit-exact HVX vectorized software convolution directly into Crouton blocks. */
int nprg2_exec_conv_u16_run(const Nprg2ExecCall *call, void *ctx);

/* Registers variants 34 and 35 into the NPRG2 executor registry. */
int nprg2_executor_register_conv_u16(Nprg2Executor *executor);

#ifdef __cplusplus
}
#endif

#endif /* NPRG2_PREPARED_CONV_U16_H */
