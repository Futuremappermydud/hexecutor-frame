#ifndef HEXECUTOR_TENSOR_LFM_OPS_H
#define HEXECUTOR_TENSOR_LFM_OPS_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* RMSNorm: out = x * rsqrt(mean(x^2) + eps) * gamma + (beta ? beta : 0) */
void lfm_rmsnorm(float *out, const float *x, const float *gamma, const float *beta,
                 uint32_t count, float eps);

/* Rotary Position Embedding (NeoX style) */
void lfm_rope(float *out, const float *x, uint32_t head_dim, uint32_t num_heads,
              uint32_t pos, uint32_t mode);

/* Causal depthwise short conv decode step with persistent state update */
void lfm_shortconv(float *out, const float *x, float *state_ring, const float *gate,
                   const float *weights, const float *bias, uint32_t channels,
                   uint32_t k_taps, uint32_t act);

/* Quantized linear matrix-vector multiplication (Q4_0 / Q8_0) */
void lfm_linear_q(float *out, const float *x, const uint8_t *weights, const float *bias,
                  uint32_t k, uint32_t n, uint32_t fmt, uint32_t act);

#ifdef __cplusplus
}
#endif

#endif /* HEXECUTOR_TENSOR_LFM_OPS_H */
