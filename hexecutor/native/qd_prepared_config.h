/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_PREPARED_CONFIG_H
#define QD_PREPARED_CONFIG_H
#include <stdint.h>
#define QDS_PREPARED_MAGIC 0x31504e51u
/* Optional extension immediately after the original 33-word QdsConfig. */
typedef struct {
    uint32_t magic,version,offset,bytes,sha256[8];
} QdsPreparedConfig;
#endif
