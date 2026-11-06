#ifndef NPRG2_PREPARED_SLICE_PAD_H
#define NPRG2_PREPARED_SLICE_PAD_H

#include "prepared_executor.h"

/* Registers strict u8 direct InputSlicePad operator (QDR_INPUT_SLICE_PAD, opcode 2, variants 2 and 8). */
int nprg2_executor_register_slice_pad(Nprg2Executor *executor);

#endif
