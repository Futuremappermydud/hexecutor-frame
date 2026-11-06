/* Read-only dump of QDSP6SS RSCC sequencer registers and program memory,
 * into arena+0x10000 (worker 0 only). */
#include <stdint.h>

#define ARENA_UNCACHED 0xf1800000u
#define QDSP6SS 0x32300000u

/* out[0] holds the offset about to be read, so a stalled read is identifiable. */
static void dump(volatile uint32_t *out, uint32_t *n, uint32_t off, uint32_t words) {
    for (uint32_t i = 0; i < words; i++) {
        out[0] = off + 4 * i;
        __asm__ volatile("barrier" ::: "memory");
        out[1 + (*n)++] = *(volatile uint32_t *)(QDSP6SS + off + 4 * i);
    }
}

__attribute__((section(".text.entry"))) void payload(void) {
    uint32_t tid;
    __asm__ volatile("%0 = htid" : "=r"(tid));
    if (tid >= 3) return;
    uint32_t seq = *(volatile uint32_t *)(ARENA_UNCACHED + 8);
    if (tid == 0) {
        volatile uint32_t *out = (volatile uint32_t *)(ARENA_UNCACHED + 0x10000);
        uint32_t n = 0;
        dump(out, &n, 0x0, 8);         /* VERSION, RST_EVB (+0x10) */
        dump(out, &n, 0x400, 4);       /* BOOT_CORE_START, BOOT_CMD, BOOT_STATUS */
        out[0] = 0xffffffffu;          /* finished */
    }
    __asm__ volatile("barrier" ::: "memory");
    *(volatile uint32_t *)(ARENA_UNCACHED + 128 + tid * 16 + 12) = 0x54495845u;
    *(volatile uint32_t *)(ARENA_UNCACHED + 128 + tid * 16) = seq;
    __asm__ volatile("barrier" ::: "memory");
}
