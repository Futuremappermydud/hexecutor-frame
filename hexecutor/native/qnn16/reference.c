/* Portable float reference for Q16_CONV1X1, for host tests only: the DSP
 * engine uses the HMX leaf. Kept apart from exec.c because Hexagon clang
 * cannot lower this _Float16 code. */
#include "program.h"
#include <string.h>

void q16_conv1x1_reference(const Qnn16Conv1x1 *j) {
    for (uint32_t mi = 0; mi < j->m; mi++)
        for (uint32_t ni = 0; ni < j->n; ni++) {
            const uint8_t *table = j->fi + ni * j->tables * 256u;
            uint8_t *out = j->c[mi * j->n + ni];
            for (uint32_t px = 0; px < 32; px++)
                for (uint32_t co = 0; co < 32; co++) {
                    float sum = 0;
                    for (uint32_t ki = 0; ki < j->k; ki++) {
                        const _Float16 *a = (const _Float16 *)j->a[mi * j->k + ki];
                        const _Float16 *w = (const _Float16 *)(j->w + (ni * j->k + ki) * 2048u);
                        /* Activation tile [16 pixel pairs][32 ci][2]; weight [16 ci pairs][32 co][2]. */
                        for (uint32_t ci = 0; ci < 32; ci++) {
                            float av = (float)a[((px / 2) * 32 + ci) * 2 + px % 2];
                            float wv = (float)w[((ci / 2) * 32 + co) * 2 + ci % 2];
                            if (av != 0 && wv != 0)
                                sum += av * wv;
                        }
                    }
                    uint32_t sw, bw;
                    memcpy(&sw, table + co * 4u, 4);
                    memcpy(&bw, table + (32 + co) * 4u, 4);
                    _Float16 sh, bh;
                    uint16_t s16 = (uint16_t)sw, b16 = (uint16_t)(bw >> 16);
                    memcpy(&sh, &s16, 2);
                    memcpy(&bh, &b16, 2);
                    float v = sum * (float)sh + (float)bh;
                    if (j->tables == 2) {
                        uint32_t hw;
                        memcpy(&hw, table + 256u + co * 4u, 4);
                        uint16_t h16 = (uint16_t)(hw >> 16);
                        _Float16 hh;
                        memcpy(&hh, &h16, 2);
                        if (v < 0)
                            v = 0;
                        if (v > (float)hh)
                            v = (float)hh;
                    }
                    _Float16 r = (_Float16)v;
                    memcpy(out + (((px / 2) * 32 + co) * 2 + px % 2) * 2u, &r, 2);
                }
        }
}
