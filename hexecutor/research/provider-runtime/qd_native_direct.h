/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_NATIVE_DIRECT_H
#define QD_NATIVE_DIRECT_H

/* Zero-copy binding for graph tensors already resident in leased VTCM.
 * Metadata and pointer tables live in caller-supplied L2 memory. The reserved
 * final 512 KiB holds completion/guards and optional padded 1x1 weights. */
#include "../../native/kernels/adapters/qd_native_1x1_adapter.h"
#include "../../native/kernels/convolution/qd_native_3x3_adapter.h"

#ifndef QD_DIRECT_VTCM_BASE
#define QD_DIRECT_VTCM_BASE       0x31000000u
#endif
#ifndef QD_DIRECT_VTCM_BYTES
#define QD_DIRECT_VTCM_BYTES      0x00800000u
#endif
#ifndef QD_DIRECT_ACTIVE_BYTES
#define QD_DIRECT_ACTIVE_BYTES    0x00780000u
#endif
#ifndef QD_DIRECT_TAIL_OFFSET
#define QD_DIRECT_TAIL_OFFSET     0x00780000u
#endif
#ifndef QD_DIRECT_TAIL_BYTES
#define QD_DIRECT_TAIL_BYTES      0x00080000u
#endif
#define QD_DIRECT_PAGE_BYTES      2048u
#define QD_DIRECT_TABLE_BYTES     (2u*QD_N1X1_TABLE_ENTRIES*4u)
#define QD_DIRECT_ABI_BYTES       (30u*4u)

enum { QD_DIRECT_OK=0, QD_DIRECT_BAD_ARGUMENT=1, QD_DIRECT_RANGE=2,
       QD_DIRECT_ALIAS=3, QD_DIRECT_UNSUPPORTED=4 };

typedef struct {
    uint32_t *in_desc,*out_desc,*cfg,*mode;
    uint32_t *in_table,*out_table;
    void *weights,*fi,*zero_fi;
    volatile uint32_t *done;
    void *input_guard,*output_guard;
    uint32_t in_pages,out_pages;
    uint32_t tail_publish_bytes,staged_weight_bytes;
    const void *packed_weight_guard;
} QdDirectBinding;

/* pool[1] must be the actual VTCM virtual base. active_bytes must be exactly
 * 7.5 MiB. scratch/abi are caller-owned L2 storage and must be disjoint from
 * all tensor pages and the reserved VTCM tail. Ordinary bindings copy no
 * tensor bytes; opt-in CI12/20 1x1 expands only weights into the tail. Publish
 * tail_publish_bytes and check packed_weight_guard when present. Call only
 * after the VTCM lease is active and HMX is exclusively
 * owned; submit/drain remains the caller's responsibility. */
int qd_direct_bind_1x1(const QdN1x1Request *,void *vtcm,uint32_t active_bytes,
                       uint32_t *tables,uint32_t table_bytes,
                       uint32_t *abi,uint32_t abi_bytes,
                       void *tail,uint32_t tail_bytes,QdDirectBinding *);
int qd_direct_validate_1x1(const QdN1x1Request *,void *vtcm,uint32_t active_bytes,
                           uint32_t *tables,uint32_t table_bytes,
                           uint32_t *abi,uint32_t abi_bytes,
                           void *tail,uint32_t tail_bytes,QdDirectBinding *result_storage);
int qd_direct_bind_3x3(const QdN3x3Request *,void *vtcm,uint32_t active_bytes,
                       uint32_t *tables,uint32_t table_bytes,
                       uint32_t *abi,uint32_t abi_bytes,
                       void *tail,uint32_t tail_bytes,QdDirectBinding *);
int qd_direct_validate_3x3(const QdN3x3Request *,void *vtcm,uint32_t active_bytes,
                           uint32_t *tables,uint32_t table_bytes,
                           uint32_t *abi,uint32_t abi_bytes,
                           void *tail,uint32_t tail_bytes,QdDirectBinding *result_storage);

#endif
