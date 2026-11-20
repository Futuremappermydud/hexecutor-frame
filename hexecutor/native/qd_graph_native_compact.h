/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef QD_GRAPH_NATIVE_COMPACT_H
#define QD_GRAPH_NATIVE_COMPACT_H
/* Experimental A-only staging: isolate input-page scatter/span from W/Fi/Y.
 * It copies A into reserved lease tail, never into the active model arena. */
static int graph_direct_compact_fits(const QdDirectBinding *b) {
    return b->in_pages && b->in_pages<=(QD_DIRECT_TAIL_BYTES-0x2000u-128u)/2048u;
}
static void graph_direct_compact_input(const QdDirectBinding *b) {
    uint8_t *a=(uint8_t *)(QD_DIRECT_VTCM_BASE+QD_DIRECT_TAIL_OFFSET+0x2000u);
#if QDS_DIRECT_COMPACT_INPUT == 2
    /* Preserve every repeated source-page reference. This isolates the
     * address span from alias/zero-dY behavior in the original input table. */
    uint32_t source[QD_N3X3_TABLE_ENTRIES];
    for(uint32_t q=0;q<b->in_pages;q++)source[q]=b->in_table[q];
    for(uint32_t q=0;q<b->in_pages;q++) {
        uint32_t first=0;while(first<q && source[first]!=source[q])first++;
        if(first==q)dsp_packet_copy(a+q*2048u,(const void *)(uintptr_t)source[q],2048u);
        b->in_table[q]=(uint32_t)(uintptr_t)(a+first*2048u);
    }
#else
    for(uint32_t q=0;q<b->in_pages;q++) {
        dsp_packet_copy(a+q*2048u,(const void *)(uintptr_t)b->in_table[q],2048u);
        b->in_table[q]=(uint32_t)(uintptr_t)(a+q*2048u);
    }
#endif
    uint32_t *guard=(uint32_t *)(a+b->in_pages*2048u);
    for(uint32_t q=0;q<32;q++)guard[q]=0xa5a5a5a5u;
    dsp_clean_range(a,b->in_pages*2048u+128u);
}
static int graph_direct_compact_guard(const QdDirectBinding *b) {
    HVX_Vector v=*(const HVX_Vector *)(QD_DIRECT_VTCM_BASE+QD_DIRECT_TAIL_OFFSET+0x2000u+b->in_pages*2048u);
    v=Q6_V_vxor_VV(v,Q6_V_vsplat_R(0xa5a5a5a5u));
    uint32_t any=0;for(uint32_t q=0;q<32;q++)any|=(uint32_t)Q6_R_vextract_VR(v,q*4u);
    return !any;
}
#endif
