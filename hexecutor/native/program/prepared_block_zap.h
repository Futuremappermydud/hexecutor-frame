/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef NPRG2_PREPARED_BLOCK_ZAP_H
#define NPRG2_PREPARED_BLOCK_ZAP_H

#include "prepared_executor.h"

/* The observed @BlockZap profile copies up to eight 2 KiB pool-1 pages,
 * replacing a 256-byte multiple at the end of each destination page. */
int nprg2_executor_register_block_zap(Nprg2Executor *executor);

#endif
