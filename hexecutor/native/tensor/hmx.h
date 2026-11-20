/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef HEXECUTOR_TENSOR_HMX_H
#define HEXECUTOR_TENSOR_HMX_H
#include "graph.h"
/* Worker zero owns HMX; followers are quiescent during a matrix op. The
 * complete scratch footprint is checked during graph preflight. */
void hx_hmx_conv(const HxGraph *, uint32_t op, uint8_t *vtcm_workspace);
#endif
