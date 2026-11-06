#ifndef NPRG2_PREPARED_SPILL_FILL_H
#define NPRG2_PREPARED_SPILL_FILL_H

#include "prepared_executor.h"

/* Registers strict QDR_SPILL_FILL variants @Fill (8) and @Spill (9). */
int nprg2_executor_register_spill_fill(Nprg2Executor *executor);

#endif
