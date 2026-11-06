#ifndef QD_GRAPH_H
#define QD_GRAPH_H
#include <stddef.h>
#include <stdint.h>

/* Native ARM executor for a QDRX image. Every scheduled operation runs in C;
 * convolutions go to a backend: exact int64 CPU arithmetic or the persistent
 * v6 DSP payload (qd_conv.c), driven from C. Semantics mirror tools/htp_emu.py
 * and frame/fast_depth.py bit for bit. Pools are caller-owned byte arrays in the
 * emulator's layout, so tests can compare them with the Python reference. */
enum { QDG_POOL_VTCM=1, QDG_POOL_DDR3=3, QDG_POOL_DDR4=4, QDG_POOL_CONST=5, QDG_POOL_SPILL=6, QDG_POOLS=7 };
enum { QDG_BACKEND_CPU=0, QDG_BACKEND_DSP=1 };
enum { QDG_OK=0, QDG_ERR_IMAGE=1, QDG_ERR_TENSOR=2, QDG_ERR_SHAPE=3, QDG_ERR_ARGS=4, QDG_ERR_RANGE=5,
       QDG_ERR_UNSUPPORTED=6, QDG_ERR_MEMORY=7, QDG_ERR_DSP=8, QDG_ERR_TIMEOUT=9, QDG_ERR_WAKE=10,
       QDG_ERR_CALLBACK=11 };

typedef struct QdgGraph QdgGraph;
/* phase 0: all workers left the persistent loop; release them.
 * phase 1: a command is published and workers are not running; wake them. */
typedef int (*QdgWake)(int phase);

typedef struct {
    uint64_t calls, hmx_calls, bytes_uploaded, hmx_phase_cycles[4];
    uint64_t prepare_ns, wait_ns, collect_ns, conv_ns, total_ns, zero_ns;
    uint64_t variant_ns[80], variant_count[80];
    uint32_t sequence, persistent, error_command, error_detail;
    uint64_t bytes_x[2], bytes_w[2], bytes_out[2]; /* [0] HVX, [1] HMX requests */
    uint64_t weights_reused;
    uint64_t hvx_phase_cycles[2][3]; /* [u8, u16]: copy, barrier, compute */
    uint64_t hmx_block_calls; /* requests collected as complete croutons */
} QdgStats;

QdgGraph *qdg_create(const void *image, uint32_t bytes, uint8_t *const *pools, const uint64_t *pool_bytes, int *error);
void qdg_destroy(QdgGraph *g);
/* DSP backend: arena is the mapped 8 MiB arena, callback_word the WORK+24 word. */
void qdg_set_dsp(QdgGraph *g, uint8_t *arena, const volatile uint32_t *callback_word, uint32_t callback_value,
                 uint32_t sequence, uint32_t persistent, uint32_t raise_clock, uint32_t use_hmx,
                 double timeout, QdgWake wake);
void qdg_set_cpu(QdgGraph *g);
void qdg_set_profile(QdgGraph *g, int enable);
/* Reorder independent convolutions ahead of commands waiting on the DSP (default on). */
void qdg_set_native_hmx(QdgGraph *,int enable);
void qdg_set_hoist(QdgGraph *g, int enable);
/* Experimental direct HMX crouton output/collection (default off). Requires
 * the payload with QDC_CROUTON8 support; preserves the convolution footprint. */
void qdg_set_block_output(QdgGraph *g, int enable);
/* Helper threads (pinned to cpus[i] when given) for bulk copies out of the arena. */
int qdg_set_helpers(QdgGraph *g, int count, const int *cpus);
/* inputs: x u8[81920], y u8[81920], hidden f32[163840], t f32[32] (native).
 * outputs: out0 u8 [1,2,320,256], out1 f32 [1,2,320,256]. */
int qdg_run(QdgGraph *g, const void *const *inputs, const uint64_t *input_bytes,
            void *const *outputs, const uint64_t *output_bytes, uint32_t first, uint32_t last);
int qdg_commands(const QdgGraph *g);
const QdgStats *qdg_stats(const QdgGraph *g);
void qdg_reset_stats(QdgGraph *g);
#endif
