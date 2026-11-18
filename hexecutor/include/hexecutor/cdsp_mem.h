#ifndef HEXECUTOR_CDSP_MEM_H
#define HEXECUTOR_CDSP_MEM_H

#include <stdint.h>

#define CDSP_MEM_DEV_PATH "/dev/cdsp_mem"
#define CDSP_MEM_SYSFS_SIZE "/sys/module/cdsp_mem/parameters/size_mb"
#define CDSP_MEM_SYSFS_IOVA "/sys/module/cdsp_mem/parameters/iova"

#define CDSP_MEM_MIN_SIZE_MB 128u
#define CDSP_MEM_MAX_SIZE_MB 1536u /* 1.5 GiB (0x60000000 bytes, fits 32-bit IOVA at 0xa0000000) */
#define CDSP_MEM_CHUNK_MB 2u
#define CDSP_MEM_CHUNK_BYTES (CDSP_MEM_CHUNK_MB * 1024u * 1024u)
#define CDSP_MEM_BASE_IOVA 0xa0000000u

#define CDSP_MEM_IOC_MAGIC 'C'
#define CDSP_MEM_IOC_SET_SIZE _IOW(CDSP_MEM_IOC_MAGIC, 1, uint32_t)
#define CDSP_MEM_IOC_GET_SIZE _IOR(CDSP_MEM_IOC_MAGIC, 2, uint32_t)

struct cdsp_mem_info {
    uint32_t size_mb;
    uint32_t min_size_mb;
    uint32_t max_size_mb;
    uint32_t chunk_mb;
    uint64_t iova;
};
#define CDSP_MEM_IOC_GET_INFO _IOR(CDSP_MEM_IOC_MAGIC, 3, struct cdsp_mem_info)
#define CDSP_MEM_IOC_PIN      _IOW(CDSP_MEM_IOC_MAGIC, 4, uint32_t)
#define CDSP_MEM_IOC_UNPIN    _IO(CDSP_MEM_IOC_MAGIC, 5)

#endif /* HEXECUTOR_CDSP_MEM_H */
