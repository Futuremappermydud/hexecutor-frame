/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_NATIVE_3X3_ADAPTER_H
#define QD_NATIVE_3X3_ADAPTER_H

/* Bounded preparation/execution adapter for the simulator-proven generic
 * V75 3x3 stride-1 Xwb HMX leaf. Caller supplies an exclusive, preserved
 * upper-VTCM workspace (up to 1 MiB), drains prior HMX work, and holds HMX and
 * power ownership from prepare through finish. This API does not coordinate
 * with the legacy worker or manage VTCM leasing/restoration. */
#include <stddef.h>
#include <stdint.h>

#define QD_N3X3_PAGE_BYTES 2048u
#define QD_N3X3_TABLE_ENTRIES 256u
#define QD_N3X3_WORKSPACE_MAX 0x100000u
enum { QD_N3X3_DTYPE_U8=1, QD_N3X3_DTYPE_I8=2, QD_N3X3_DTYPE_I32=5 };
enum { QD_N3X3_WEIGHT_XWB=2 };

typedef struct {
    uint32_t dims[4];
    uint32_t max_dims[4];
    uint32_t padding[4];
    uint32_t slice_offset;
    uint32_t block_count;
    const uint32_t *blocks; /* encoded QDRX pool index | aligned byte offset */
} QdN3x3Blocked;

typedef struct {
    uint32_t pool_count;
    const uint8_t *pool[7];
    const uint32_t *pool_bytes;
    QdN3x3Blocked input, output;
    const uint8_t *weights;
    uint32_t weights_bytes;
    uint32_t weight_ci, weight_co, weight_kernel_h, weight_kernel_w;
    uint32_t input_dtype, output_dtype, weight_dtype, fi_dtype, weight_format;
    const int32_t *fi;
    uint32_t fi_bytes;
    uint32_t input_zero_point, output_zero_point;
    void *workspace;
    uint32_t workspace_bytes;
} QdN3x3Request;

typedef struct {
    uint32_t tables[2u*QD_N3X3_TABLE_ENTRIES] __attribute__((aligned(128)));
    uint32_t abi[30];
    uint32_t input_pages, output_pages;
    uint32_t input_page_rows, input_page_cols;
    uint32_t output_page_rows, output_page_cols;
    uint32_t logical_h, logical_w, physical_h, physical_w;
    uint32_t workspace_used;
    uint32_t *input_desc, *output_desc, *cfg, *mode;
    uint32_t *input_table, *output_table;
    uint8_t *input_data, *output_data, *weights, *weight_guard;
    int32_t *fi, *zero_fi;
    uint8_t *input_guard, *output_guard;
    volatile uint32_t *done;
    const QdN3x3Request *request;
    uint32_t issued, executed, saved_ssr;
} QdN3x3Prepared;

enum {
    QD_N3X3_OK=0, QD_N3X3_UNSUPPORTED=1, QD_N3X3_RANGE=2,
    QD_N3X3_ALIAS=3, QD_N3X3_WORKSPACE=4, QD_N3X3_GUARD=5,
    QD_N3X3_TIMEOUT=6
};

/* Validate all metadata/ranges/aliases before workspace writes, then stage the
 * complete max-dimension crouton grids and build the generic-leaf ABI. */
int qd_native_3x3_prepare(const QdN3x3Request *request, QdN3x3Prepared *out);

/* Caller must already own HMX/power and have drained prior commands. Issue
 * and drain are separate so the graph bridge can account for both phases. */
int qd_native_3x3_issue(QdN3x3Prepared *prepared);
int qd_native_3x3_drain(QdN3x3Prepared *prepared);
/* Convenience wrapper for isolated simulator fixtures. */
int qd_native_3x3_execute(QdN3x3Prepared *prepared);

/* Verify guards/table tails and masked-scatter logical output pixels. The
 * explicitly bounded CO2/maxCO32 profile writes its full physical 32-lane
 * crouton because the old engine writes those aliased pool bytes as well. */
int qd_native_3x3_finish(const QdN3x3Prepared *prepared);

void qd_qnn_hmx_stride1_generic(uint32_t *, uint32_t *, void *, void *, uint32_t *, uint32_t *);

#endif
