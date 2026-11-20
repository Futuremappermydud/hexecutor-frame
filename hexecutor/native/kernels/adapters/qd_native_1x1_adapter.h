/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_NATIVE_1X1_ADAPTER_H
#define QD_NATIVE_1X1_ADAPTER_H

/* Preparation-only adapter for the simulator-covered hmx_v73_convbbb1x1 leaf.
 * Caller owns pools and supplies the preserved exclusive upper-VTCM workspace.
 * Caller must first drain every prior HMX request, hold HMX/power ownership for
 * the full prepare -> leaf -> HMX-finish -> adapter-finish interval, and must
 * not resume graph work until that interval completes. This API does not make
 * global HMX state independent of the legacy worker. */
#include <stdint.h>
#include <stddef.h>

#ifndef QDS_NATIVE_PADDED_1X1
#define QDS_NATIVE_PADDED_1X1 0
#endif

#define QD_N1X1_WORKSPACE_MAX 0x80000u
#define QD_N1X1_TABLE_ENTRIES 256u
#define QD_N1X1_BLOCK_BYTES 2048u
enum { QD_N1X1_DTYPE_U8=1, QD_N1X1_DTYPE_I8=2, QD_N1X1_DTYPE_I32=5 };
enum { QD_N1X1_WEIGHT_XPB=1, QD_N1X1_WEIGHT_XWB=2 };

typedef struct {
    uint32_t dims[4];       /* QDRX dimensions (runtime logical extent) */
    uint32_t max_dims[4];   /* physical allocated extent */
    uint32_t padding[4];
    uint32_t slice_offset;
    uint32_t block_count;
    const uint32_t *blocks; /* encoded QDRX pool | aligned byte offset */
} QdN1x1Blocked;

typedef struct {
    uint32_t pool_count;    /* must be at least 7 */
    const uint8_t *pool[7];
    const uint32_t *pool_bytes;
    QdN1x1Blocked input, output;
    const uint8_t *weights; /* raw serialized 1x1 Xpb or Xwb bytes */
    uint32_t weights_bytes;
    uint32_t weight_ci, weight_co;
#if QDS_NATIVE_PADDED_1X1
    uint32_t weight_native_ci; /* 0 means raw CI; padded support currently maps 12/20 to 32. */
#endif
    uint32_t weight_kernel_h, weight_kernel_w;
    uint32_t input_dtype, output_dtype, weight_dtype, fi_dtype, weight_format;
    const int32_t *fi;      /* raw serialized Fi rows: [CO/32][64] int32 */
    uint32_t fi_bytes;
    void *workspace;        /* 128-byte aligned upper-VTCM base */
    uint32_t workspace_bytes;
    void *table_storage;    /* optional 128-byte aligned external tables (2048B) */
    uint32_t table_storage_bytes;
    void *abi_storage;      /* optional external descriptors/config/mode (120B) */
    uint32_t abi_storage_bytes;
} QdN1x1Request;

typedef struct {
    uint32_t effective_h, effective_w;
    uint32_t physical_h, physical_w;
    uint32_t input_blocks, output_blocks;
    uint32_t input_channels, output_channels;
    uint32_t workspace_used;
    uint32_t *input_desc, *output_desc, *cfg, *mode;
    void *weights, *fi, *zero_words;
    uint8_t *guard_block, *input_guard_block;
    uint32_t *input_table, *output_table;
    uint8_t *input_data, *output_data;
    const QdN1x1Request *request;
} QdN1x1Prepared;

enum {
    QD_N1X1_OK = 0,
    QD_N1X1_UNSUPPORTED = 1,
    QD_N1X1_RANGE = 2,
    QD_N1X1_ALIAS = 3,
    QD_N1X1_WORKSPACE = 4,
    QD_N1X1_GUARD = 5
};

/* Validates eligibility, complete physical block-table bounds and workspace
 * aliases, then stages only touched croutons plus packed weights/FI. */
int qd_native_1x1_prepare(const QdN1x1Request *request, QdN1x1Prepared *out);

/* Check table/guard integrity after the native leaf and scatter active output
 * croutons to their original blocked-pool destinations. Call only after HMX
 * completion. */
int qd_native_1x1_finish(const QdN1x1Prepared *prepared);

/* Native leaf ABI used after prepare:
 * qd_qnn_hmx_1x1(output_desc,input_desc,weights,fi,cfg,mode). */
void qd_qnn_hmx_1x1(uint32_t *, uint32_t *, void *, void *, uint32_t *, uint32_t *);

#endif
