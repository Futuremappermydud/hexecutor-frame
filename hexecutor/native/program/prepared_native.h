/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef NPRG2_PREPARED_NATIVE_H
#define NPRG2_PREPARED_NATIVE_H

#include "prepared_executor.h"
#include "../../research/provider-runtime/qd_native_direct.h"

typedef int (*Nprg2NativeSubmit)(void *ctx,const Nprg2ExecCall *call,
                                 const QdDirectBinding *binding,uint8_t family);
#define NPRG2_NATIVE_ALLOW_1X1 0x01u
#define NPRG2_NATIVE_ALLOW_3X3 0x02u
typedef struct {
    uint32_t active_vtcm_bytes;
    uint32_t *tables;
    uint32_t table_bytes;
    uint32_t *abi;
    uint32_t abi_bytes;
    void *tail;
    uint32_t tail_bytes;
    Nprg2NativeSubmit submit_and_drain;
    void *submit_ctx;
    /* Zero preserves both families; a nonzero mask narrows the complete
     * preflight/run surface (e.g. allow 1x1 while 3x3 is under review). */
    uint8_t family_mask;
    uint8_t reserved[3];
    /* Optional preflight cap for a bounded pool-1 input compaction. Zero
     * preserves ordinary family coverage; applies only to 3x3 page count. */
    uint32_t max_3x3_input_pages;
} Nprg2NativeConfig;

/* Adds only the proof-approved QDR HMX variants: CB/Xpb 1x1, CB/Xwb 1x1 or
 * 3x3. Every callback validates the exact bound tensor metadata; no command
 * can be redirected to the legacy graph dispatcher. */
int nprg2_executor_register_native_conv(Nprg2Executor *executor,
                                        Nprg2NativeConfig *config);

#endif
