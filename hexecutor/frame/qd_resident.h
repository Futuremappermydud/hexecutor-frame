#ifndef QD_RESIDENT_H
#define QD_RESIDENT_H
#include <stdint.h>
#include "qd_resident_schema.h"

/* QDRX is little endian and uses byte reads, including at unaligned addresses.
 * All offsets are image/pool relative; this module never maps physical memory.
 * The caller must keep the image immutable through validation and dispatch.
 * Pool sizes describe reference requirements, NOT verified hardware ownership.
 * CRC32 catches corruption; the host must verify the SHA256 package identity.
 */
#define QDR_HEADER_BYTES 256u
#define QDR_MAX_BYTES 0x800000u
#define QDR_NONE 0xffffffffu
enum { QDR_TENSORS, QDR_BLOCKS, QDR_COMMANDS, QDR_REFS, QDR_WORDS, QDR_CALL_TABLE, QDR_TABLES };
enum { QDR_OK, QDR_BAD_HEADER, QDR_BAD_CRC, QDR_BAD_SECTION, QDR_BAD_TENSOR,
       QDR_BAD_BLOCK, QDR_BAD_COMMAND, QDR_BAD_REF, QDR_BAD_CALL, QDR_BAD_POOL,
       QDR_UNSUPPORTED, QDR_CANCELLED, QDR_BUDGET, QDR_KERNEL_ERROR };
enum { QDR_HAS_SHAPE=1, QDR_BLOCKED=2, QDR_DIRECT=4, QDR_SHAPE_VALUE=8,
       QDR_SCALAR=16, QDR_QUANT=32, QDR_CROUTON=64 };
typedef struct {
    const uint8_t *data;
    uint32_t bytes, offset[QDR_TABLES], count[QDR_TABLES], pool_bytes[7];
} QdrImage;
typedef struct { uint32_t code, command, tensor, detail; } QdrStatus;
uint32_t qdr_u32(const void *data);
uint64_t qdr_u64(const void *data);
const uint8_t *qdr_record(const QdrImage *image, uint32_t table, uint32_t index);
uint32_t qdr_open(const void *data, uint32_t bytes, QdrImage *out, QdrStatus *status);

/* Future kernel interface. An implementation is registered for each exact
 * variant, not just each opcode. All required variants and the total command
 * budget are checked BEFORE the first callback. Serial checkpoints remain in
 * the stream. Only explicit no-ops are built in. This is a single-worker core;
 * transport publication, worker barriers, mapped pools and kernels are separate.
 * Neither a null handler nor a kernel error falls back to the host.
 */
typedef uint32_t (*QdrKernel)(const QdrImage *, uint32_t command, void *context);
typedef uint32_t (*QdrCancel)(void *context);
uint32_t qdr_dispatch(const void *data, uint32_t bytes,
                      const QdrKernel *variants, uint32_t variant_count,
                      uint32_t command_budget, QdrCancel cancel,
                      void *context, QdrStatus *status);
#endif
