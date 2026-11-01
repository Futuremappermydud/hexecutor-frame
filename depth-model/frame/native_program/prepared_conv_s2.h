#ifndef NPRG2_PREPARED_CONV_S2_H
#define NPRG2_PREPARED_CONV_S2_H

#include "prepared_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Exact V73/V75 tensor object descriptors for Stride-2 HMX convolution
 * (Opcode 6, Variant 36: ConvLayer_s2.opt@CB*2.Xwb.Fi).
 * Worker entry: 0x49b660 / 0x49bd60
 * Leaf: 0x49bf18 (hmx_v73_convbbb_stride2)
 * Registers: r0=&output_obj, r1=&input_obj, r2=&weight_obj, r3=&fi_obj, r4=0, r5=0 */

typedef struct {
    uint32_t flags;
    uint32_t dims[4];
    uint32_t max_dims[4];
    uint8_t padding[4];
} Nprg2Shape4;

typedef struct {
    uint32_t vtable;
    uint32_t interface_ptr;
    uint32_t shape;   /* pointer to Nprg2Shape4 */
    uint32_t data;    /* pointer to block table (Crouton) or raw buffer (Direct) */
} Nprg2TensorObj;

typedef struct {
    Nprg2Shape4 input_shape;
    Nprg2Shape4 output_shape;
    Nprg2Shape4 weight_shape;
    Nprg2Shape4 fi_shape;
    Nprg2TensorObj input_obj;
    Nprg2TensorObj output_obj;
    Nprg2TensorObj weight_obj;
    Nprg2TensorObj fi_obj;
    uint32_t input_block_table[256];
    uint32_t output_block_table[256];
} Nprg2ConvS2Descriptors;

/* Validates bound tensor metadata and storage for Opcode 6 (QDR_HMX_CONV)
 * Variant 36 (ConvLayer_s2.opt@CB*2.Xwb.Fi). Read-only. */
int nprg2_exec_conv_s2_validate(const Nprg2ExecCall *call, void *ctx);

/* Executes bit-exact stride-2 convolution directly into Crouton blocks.
 * Re-validates before executing. */
int nprg2_exec_conv_s2_run(const Nprg2ExecCall *call, void *ctx);

/* Sets up the exact V73/V75 tensor object descriptors from an execution call */
int nprg2_conv_s2_setup_descriptors(const Nprg2ExecCall *call, Nprg2ConvS2Descriptors *desc);

/* Registers variant 36 into the NPRG2 executor registry. */
int nprg2_executor_register_conv_s2(Nprg2Executor *executor);

#ifdef __cplusplus
}
#endif

#endif /* NPRG2_PREPARED_CONV_S2_H */
