#ifndef NPRG2_PREPARED_PREPROCESS_H
#define NPRG2_PREPARED_PREPROCESS_H
#include "prepared_executor.h"
/* Exact early graph profiles: Ff InputSlice, CB activation DMA, and audited u8 CB 3x3 im2col. */
int nprg2_executor_register_preprocess(Nprg2Executor *executor);
#endif
