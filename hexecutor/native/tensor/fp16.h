/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef HEXECUTOR_TENSOR_FP16_H
#define HEXECUTOR_TENSOR_FP16_H
#include "graph.h"
/* Private DSP representation, planned at session open and never exposed:
 * [32-channel block][pixel pair][32 channels][2 pixels]. Rows are padded to an
 * even physical width so a pair never straddles rows; the pad column is kept
 * zero. Each block plane holds whole 32-pixel HMX tiles, so pointwise matrix
 * operations read and write activations in place. External I/O and the file
 * format remain FP32. A compact RGB input is staged as exact half values in word
 * lanes, inside zero padding that gives every window a contiguous source, and
 * occupies the arena only until its one convolution has consumed it. */
struct HxFp16Plan {
    uint32_t *offset;  /* per tensor: arena-relative plane offset */
    uint32_t *weights; /* per op: resident HMX weights, or 0 to copy on use */
    uint32_t *aux;     /* per op: resident bias, filter or slope record */
    uint8_t *workspace; /* kernel scratch, disjoint from tensors and records */
    uint32_t peak, workspace_bytes;
    uint32_t image_rows, image_row_words, image_top; /* compact input geometry */
};
enum { HX_JOB_COPY = 1, HX_JOB_IMAGE_HALF, HX_JOB_RGB_PACK, HX_JOB_ACTIVATE, HX_JOB_EXPORT,
       HX_JOB_IM2COL };
typedef struct {
    uint32_t kind;
    const HxGraph *g;
    const HxTensor *t;
    const void *src;
    void *dst;
    uint32_t bytes, index, first, count;
} HxFp16Job;
/* Runs a job on every worker and joins; NULL runs it on the caller. */
typedef void (*HxFp16Parallel)(const HxFp16Job *);

uint32_t hx_fp16_table_words(const HxGraph *g);
/* Fills plan from the graph; returns 0 when the FP16 path cannot serve it.
 * g->arena must already address VTCM ending at vtcm_end. Records fill all but
 * HX_FP16_MIN_WORKSPACE; the workspace takes whatever remains. */
#define HX_FP16_MIN_WORKSPACE 0x40000u
int hx_fp16_plan(const HxGraph *g, struct HxFp16Plan *plan, uint32_t *tables, uint8_t *vtcm_end);
/* Re-places resident records of several planned graphs sharing one VTCM arena
 * so they do not overlap; see fp16.c. Returns 0 (plans unchanged) if they cannot. */
int hx_fp16_share(HxGraph *const *graphs, uint32_t count, uint8_t *vtcm_end);
/* Loads every resident record for g. Required after another user of VTCM. */
void hx_fp16_stage(const HxGraph *g, HxFp16Parallel parallel);
void hx_fp16_import(const HxGraph *g, uint32_t tensor, const float *src, HxFp16Parallel parallel);
void hx_fp16_export(const HxGraph *g, uint32_t tensor, float *dst, HxFp16Parallel parallel);
void hx_fp16_op(const HxGraph *g, uint32_t index, uint32_t part, uint32_t parts);
void hx_fp16_conv(const HxGraph *g, uint32_t index, HxFp16Parallel parallel);
/* Caller joins all parts first. Restores the zero pad column of odd widths. */
void hx_fp16_complete(const HxGraph *g, uint32_t index);
void hx_fp16_job(const HxFp16Job *job, uint32_t part, uint32_t parts);
#endif
