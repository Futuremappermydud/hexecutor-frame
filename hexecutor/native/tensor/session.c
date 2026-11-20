/* SPDX-License-Identifier: LGPL-3.0-or-later */
#include "graph.h"
static int span(uint32_t at, uint64_t n, uint32_t bytes) { return at <= bytes && n <= bytes - at; }
static int overlap(uint32_t a, uint32_t an, uint32_t b, uint32_t bn) {
    return a < b + bn && b < a + an;
}
int hx_session_config_valid(const HxSessionConfig *c) {
    if (c->magic != HXFG_CONFIG_MAGIC || c->version != 1 || !c->code_bytes ||
        c->code_bytes > 0x100000u || !c->graphs || c->graphs > HXFG_MAX_GRAPHS)
        return 0;
    for (uint32_t i = 0; i < c->graphs; i++) {
        const HxGraphConfig *g = &c->graph[i];
        if ((g->image | g->constants | g->arena) & 127u || g->image < 0x100000u ||
            g->image_bytes < 256 || !span(g->image, g->image_bytes, 0x1000000u) ||
            g->constants < 0x2000000u || !g->constant_bytes ||
            !span(g->constants, g->constant_bytes, HXFG_CONSTANT_REGION_END) || g->arena < 0x1000000u ||
            !g->arena_bytes || !span(g->arena, g->arena_bytes, 0x2000000u))
            return 0;
        for (uint32_t j = 0; j < i; j++) {
            const HxGraphConfig *b = &c->graph[j];
            if (overlap(g->image, g->image_bytes, b->image, b->image_bytes) ||
                overlap(g->constants, g->constant_bytes, b->constants, b->constant_bytes) ||
                overlap(g->arena, g->arena_bytes, b->arena, b->arena_bytes))
                return 0;
        }
    }
    if (c->state_bytes) {
        if ((c->state & 127u) || c->state < 0x2000000u)
            return 0;
        for (uint32_t i = 0; i < c->graphs; i++) {
            const HxGraphConfig *g = &c->graph[i];
            if (overlap(c->state, c->state_bytes, g->image, g->image_bytes) ||
                overlap(c->state, c->state_bytes, g->constants, g->constant_bytes) ||
                overlap(c->state, c->state_bytes, g->arena, g->arena_bytes))
                return 0;
        }
    }
    return 1;
}
