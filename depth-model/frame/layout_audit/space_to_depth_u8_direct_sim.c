#include "space_to_depth_u8_direct.h"
#include "space_to_depth_cases.h"
#include <stdio.h>
#include <stdlib.h>

#define TEST_POOL_BYTES (8u*1024u*1024u)
#define BLOCK_BYTES 2048u
#define MAX_TOUCHED QD_LAYOUT_S2D_MAX_TOUCHED

typedef struct {
    uint32_t tags[256];
    uint8_t dirty[256];
    uint8_t data[256][BLOCK_BYTES];
    uint8_t *const *pools;
    uint32_t stores[1024];
    uint32_t store_count;
    uint32_t fail_id;
    int fail_enabled;
} BlockCacheSim;

static uint8_t pool_memory[TEST_POOL_BYTES];
static uint8_t *pools[QD_LAYOUT_S2D_POOL_COUNT];
static size_t pool_bytes[QD_LAYOUT_S2D_POOL_COUNT];
static uint8_t tile[QD_LAYOUT_S2D_TILE_BYTES];
static uint8_t input_before[MAX_TOUCHED][BLOCK_BYTES];
static uint8_t output_before[MAX_TOUCHED][BLOCK_BYTES];
static uint8_t output_expected[MAX_TOUCHED][BLOCK_BYTES];
static uint32_t input_ids[MAX_TOUCHED], output_ids[MAX_TOUCHED];
static uintptr_t input_addresses[MAX_TOUCHED], output_addresses[MAX_TOUCHED];
static uint8_t dense_input[16u*264u*32u];
static uint8_t dense_output[8u*136u*128u];
static BlockCacheSim cache_sim;
static uint32_t mutated_in_blocks[256];
static uint32_t mutated_out_blocks[4096];
static uint64_t fake_ticks;

static uint64_t fake_clock(void *ctx) { (void)ctx;return ++fake_ticks; }

static uint32_t pool_id(uint32_t e) { return e&127u; }
static uint32_t pool_offset(uint32_t e) { return e&~127u; }
static uint8_t *physical(uint32_t e) {
    return pools[pool_id(e)]+pool_offset(e);
}
static uint32_t block_slot(uint32_t e) { return ((e>>11)+(e&127u)*17u)&255u; }

static void cache_begin(void) {
    memset(cache_sim.tags,0,sizeof cache_sim.tags);
    memset(cache_sim.dirty,0,sizeof cache_sim.dirty);
    cache_sim.pools=pools;cache_sim.store_count=0;cache_sim.fail_enabled=0;
}

static void cache_flush(uint32_t slot) {
    uint32_t e=cache_sim.tags[slot];
    if (e && cache_sim.dirty[slot]) {
        memcpy(cache_sim.pools[pool_id(e)]+pool_offset(e),cache_sim.data[slot],BLOCK_BYTES);
        cache_sim.dirty[slot]=0;
    }
}

static uint8_t *cache_block(uint32_t e,int store,void *ctx) {
    (void)ctx;
    if (cache_sim.fail_enabled && e==cache_sim.fail_id) return NULL;
    uint32_t slot=block_slot(e);
    if (cache_sim.tags[slot]!=e) {
        cache_flush(slot);
        memcpy(cache_sim.data[slot],pools[pool_id(e)]+pool_offset(e),BLOCK_BYTES);
        cache_sim.tags[slot]=e;
    }
    if (store) {
        cache_sim.dirty[slot]=1;
        if (cache_sim.store_count<1024) cache_sim.stores[cache_sim.store_count++]=e;
    }
    return cache_sim.data[slot];
}

static void cache_end(void) { for (uint32_t i=0;i<256;i++) cache_flush(i); }

static int unique_add(uint32_t e,uint32_t *ids,uintptr_t *addresses,uint32_t *count) {
    uintptr_t addr=(uintptr_t)physical(e);
    for (uint32_t i=0;i<*count;i++) {
        if (addr==addresses[i]) return 2;
        if (addr<addresses[i]+BLOCK_BYTES && addresses[i]<addr+BLOCK_BYTES) return 0;
    }
    if (*count>=MAX_TOUCHED) return 0;
    ids[*count]=e;addresses[*count]=addr;(*count)++;return 1;
}

static int collect_touched(const LayoutS2dCase *c,uint32_t *ni,uint32_t *no) {
    int32_t ow=(c->in_m[2]-c->in_p[2])/2;
    if (ow>c->out_m[2]) ow=c->out_m[2];
    uint32_t in_nbw=((uint32_t)c->in_m[2]+7u)/8u;
    uint32_t out_nbw=((uint32_t)c->out_m[2]+7u)/8u;
    uint32_t in_bh0=(uint32_t)c->in_p[1]/8u;
    uint32_t in_bh1=(uint32_t)(c->in_p[1]+2*c->out_d[1]-1)/8u;
    uint32_t in_bw0=(uint32_t)(c->in_p[2]-1)/8u;
    uint32_t in_bw1=(uint32_t)(c->in_p[2]-1+2*ow-1)/8u;
    uint32_t in_count=0,out_count=0;
    for (uint32_t bh=in_bh0;bh<=in_bh1;bh++) for (uint32_t bw=in_bw0;bw<=in_bw1;bw++) {
        uint64_t idx=((uint64_t)bh*in_nbw+bw)+c->in_slice;
        if (idx>=c->in_nblocks) return 0;
        int rc=unique_add(layout_s2d_blocks[c->in_start+idx],input_ids,input_addresses,&in_count);
        if (!rc && rc!=2) return 0;
    }
    uint32_t out_bh1=(uint32_t)(c->out_d[1]-1)/8u;
    uint32_t out_bw1=(uint32_t)(ow-1)/8u;
    for (uint32_t bh=0;bh<=out_bh1;bh++) for (uint32_t bw=0;bw<=out_bw1;bw++)
        for (uint32_t d=0;d<4;d++) {
            uint64_t idx=((uint64_t)bh*out_nbw+bw)*4u+c->out_slice+d;
            if (idx>=c->out_nblocks) return 0;
            int rc=unique_add(layout_s2d_blocks[c->out_start+idx],output_ids,output_addresses,&out_count);
            if (rc!=1) return 0;
        }
    for (uint32_t a=0;a<in_count;a++) for (uint32_t b=0;b<out_count;b++)
        if (input_addresses[a]<output_addresses[b]+BLOCK_BYTES
                && output_addresses[b]<input_addresses[a]+BLOCK_BYTES) return 0;
    *ni=in_count;*no=out_count;return 1;
}

static int block_list_has(const uint32_t *ids,uint32_t count,uint32_t e) {
    uintptr_t addr=(uintptr_t)physical(e);
    for (uint32_t i=0;i<count;i++) if ((uintptr_t)physical(ids[i])==addr) return 1;
    return 0;
}

static int setup_case(const LayoutS2dCase *c,uint32_t *ni,uint32_t *no) {
    if (!collect_touched(c,ni,no)) return 0;
    for (uint32_t i=0;i<*ni;i++) {
        uint8_t *p=physical(input_ids[i]);
        for (uint32_t j=0;j<BLOCK_BYTES;j++) p[j]=(uint8_t)(j*19u+i*31u+c->sequence*7u+c->image);
        memcpy(input_before[i],p,BLOCK_BYTES);
    }
    for (uint32_t i=0;i<*no;i++) {
        uint8_t *p=physical(output_ids[i]);
        for (uint32_t j=0;j<BLOCK_BYTES;j++) p[j]=(uint8_t)(0x93u+j*11u+i*23u+c->sequence);
        memcpy(output_before[i],p,BLOCK_BYTES);
        memcpy(output_expected[i],p,BLOCK_BYTES);
    }
    return 1;
}

static int block_index(const LayoutS2dCase *c,uint32_t h,uint32_t w,
        int output,uint32_t depth_block,uint32_t *encoded) {
    uint64_t idx;
    if (output) {
        uint32_t nbw=((uint32_t)c->out_m[2]+7u)/8u;
        idx=((uint64_t)(h/8u)*nbw+w/8u)*4u+c->out_slice+depth_block;
        if (idx>=c->out_nblocks) return 0;
        *encoded=layout_s2d_blocks[c->out_start+idx];
    } else {
        uint32_t nbw=((uint32_t)c->in_m[2]+7u)/8u;
        idx=((uint64_t)(h/8u)*nbw+w/8u)+c->in_slice;
        if (idx>=c->in_nblocks) return 0;
        *encoded=layout_s2d_blocks[c->in_start+idx];
    }
    return 1;
}

/* Byte-for-byte reference: t_read_region into dense `xf`, existing scalar
 * k_space_to_depth loop, then t_write_region into mapped blocks. */
static int reference(const LayoutS2dCase *c,uint32_t ni,uint32_t no) {
    int32_t ow=(c->in_m[2]-c->in_p[2])/2;
    if (ow>c->out_m[2]) ow=c->out_m[2];
    uint32_t oh=(uint32_t)c->out_d[1],ci=(uint32_t)c->in_m[3],oc=(uint32_t)c->out_d[3];
    uint32_t xf_w=2u*(uint32_t)ow;
    size_t xf_bytes=(size_t)(2u*oh)*xf_w*ci;
    if (xf_bytes>sizeof dense_input) return 0;
    for (uint32_t h=0;h<2u*oh;h++) for (uint32_t w=0;w<xf_w;w++) {
        uint32_t H=(uint32_t)c->in_p[1]+h,W=(uint32_t)c->in_p[2]-1u+w,e;
        if (!block_index(c,H,W,0,0,&e)) return 0;
        const uint8_t *src=physical(e)+(H%8u)*256u+(W%8u)*32u;
        memcpy(dense_input+((size_t)h*xf_w+w)*ci,src,ci);
    }
    size_t out_bytes=(size_t)oh*(uint32_t)ow*oc;
    if (out_bytes>sizeof dense_output) return 0;
    memset(dense_output,0,out_bytes);
    for (uint32_t dy=0;dy<2;dy++) for (uint32_t dx=0;dx<2;dx++) {
        uint32_t j=2u*dy+dx;
        for (uint32_t h=0;h<oh;h++) for (uint32_t w=0;w<(uint32_t)ow;w++) {
            const uint8_t *src=dense_input+((size_t)(2*h+dy)*xf_w+(2*w+1u-dx))*ci;
            uint8_t *dst=dense_output+((size_t)h*(uint32_t)ow+w)*oc+32u*j;
            memcpy(dst,src,32);
        }
    }
    for (uint32_t h=0;h<oh;h++) for (uint32_t w=0;w<(uint32_t)ow;w++)
        for (uint32_t j=0;j<4;j++) {
            uint32_t e;if (!block_index(c,h,w,1,j,&e)) return 0;
            uintptr_t addr=(uintptr_t)physical(e);uint8_t *dst=NULL;
            for (uint32_t q=0;q<no;q++) if (output_addresses[q]==addr) {dst=output_expected[q];break;}
            if (!dst) return 0;
            memcpy(dst+(h%8u)*256u+(w%8u)*32u,
                   dense_output+((size_t)h*(uint32_t)ow+w)*oc+32u*j,32);
        }
    for (uint32_t i=0;i<ni;i++) if (memcmp(physical(input_ids[i]),input_before[i],BLOCK_BYTES)) return 0;
    return 1;
}

static int check_outputs(uint32_t no) {
    for (uint32_t i=0;i<no;i++)
        if (memcmp(physical(output_ids[i]),output_expected[i],BLOCK_BYTES)) return 0;
    return 1;
}

static int check_stores(uint32_t no) {
    if (cache_sim.store_count!=no) return 0;
    for (uint32_t i=0;i<cache_sim.store_count;i++)
        if (!block_list_has(output_ids,no,cache_sim.stores[i])) return 0;
    return 1;
}

int main(void) {
    pools[1]=pool_memory;pool_bytes[1]=sizeof pool_memory;
    uint32_t eligible=0;
    for (uint32_t n=0;n<LAYOUT_S2D_CASE_COUNT;n++) {
        const LayoutS2dCase *c=&layout_s2d_cases[n];uint32_t ni=0,no=0;
        QdLayoutS2dPlan plan;QdLayoutS2dStats prepare_stats,apply_stats;
        if (!setup_case(c,&ni,&no) || !reference(c,ni,no)) {
            fprintf(stderr,"fixture/reference failure image=%u seq=%u\n",c->image,c->sequence);return 1;
        }
        if (!qd_layout_s2d_prepare_profiled(c->in_m,c->in_p,c->out_d,c->out_m,c->out_p,
                layout_s2d_blocks+c->in_start,c->in_nblocks,c->in_slice,
                layout_s2d_blocks+c->out_start,c->out_nblocks,c->out_slice,
                pools,pool_bytes,&plan,fake_clock,NULL,&prepare_stats)
                || !prepare_stats.proof_ticks) {
            fprintf(stderr,"prepare failure image=%u seq=%u\n",c->image,c->sequence);return 2;
        }
        cache_begin();
        int handled=qd_layout_s2d_apply_profiled(&plan,pools,pool_bytes,
                tile,sizeof tile,cache_block,NULL,fake_clock,NULL,&apply_stats);
        uint64_t ow=(uint64_t)(c->in_m[2]-c->in_p[2])/2u;
        if (ow>(uint64_t)c->out_m[2]) ow=(uint64_t)c->out_m[2];
        uint64_t expected_source=(uint64_t)2*c->out_d[1]*2u*ow*32u;
        uint64_t expected_output=(uint64_t)c->out_d[1]*ow*128u;
        if (!handled || !apply_stats.gather_ticks || !apply_stats.scatter_ticks
                || apply_stats.prefetch_ticks || apply_stats.prefetch_block_calls
                || apply_stats.source_bytes!=expected_source
                || apply_stats.output_bytes!=expected_output) {
            fprintf(stderr,"helper fallback image=%u seq=%u geometry=%dx%d\n",
                    c->image,c->sequence,c->out_d[1],c->out_d[2]);return 3;
        }
        cache_end();
        if (!check_outputs(no) || !check_stores(no)) {
            fprintf(stderr,"physical output mismatch image=%u seq=%u\n",c->image,c->sequence);return 4;
        }
        for (uint32_t i=0;i<ni;i++) if (memcmp(physical(input_ids[i]),input_before[i],BLOCK_BYTES)) {
            fprintf(stderr,"input changed image=%u seq=%u\n",c->image,c->sequence);return 5;
        }
        eligible++;
    }

    /* Guard checks use a real decoded case and require rejection before any
     * callback store, with the entire touched output blocks unchanged. */
    {
        const LayoutS2dCase *c=&layout_s2d_cases[0];uint32_t ni=0,no=0;
        uint32_t in_nbw=((uint32_t)c->in_m[2]+7u)/8u;
        uint32_t in_map=((uint32_t)(c->in_p[1]/8)*in_nbw
                       +(uint32_t)((c->in_p[2]-1)/8))+c->in_slice;
        uint32_t out_map=c->out_slice;
        if (c->in_nblocks>sizeof mutated_in_blocks/sizeof mutated_in_blocks[0]
                || c->out_nblocks>sizeof mutated_out_blocks/sizeof mutated_out_blocks[0]) return 5;

#define REJECT_WITH_MAPS(IMAP,OMAP,FAIL_EN,FAIL_ID) do { \
            if (!setup_case(c,&ni,&no)) return 6; \
            cache_begin();cache_sim.fail_enabled=(FAIL_EN);cache_sim.fail_id=(FAIL_ID); \
            int handled=qd_layout_space_to_depth_u8_direct(c->in_m,c->in_p, \
                    c->out_d,c->out_m,c->out_p, \
                    (IMAP),c->in_nblocks,c->in_slice,(OMAP),c->out_nblocks,c->out_slice, \
                    pools,pool_bytes,tile,sizeof tile,cache_block,NULL); \
            cache_end(); \
            if (handled || cache_sim.store_count || !check_outputs(no)) return 7; \
            for (uint32_t z=0;z<ni;z++) if (memcmp(physical(input_ids[z]),input_before[z],BLOCK_BYTES)) return 8; \
        } while (0)

        memcpy(mutated_in_blocks,layout_s2d_blocks+c->in_start,c->in_nblocks*sizeof(uint32_t));
        memcpy(mutated_out_blocks,layout_s2d_blocks+c->out_start,c->out_nblocks*sizeof(uint32_t));
        uint32_t original_in=mutated_in_blocks[in_map-c->in_slice];
        uint32_t original_out=mutated_out_blocks[out_map];
        uint32_t original_out1=mutated_out_blocks[out_map+1];
        mutated_in_blocks[in_map-c->in_slice]=(original_in&~127u)|7u;
        REJECT_WITH_MAPS(mutated_in_blocks,mutated_out_blocks,0,0);
        mutated_in_blocks[in_map-c->in_slice]=(original_in&~127u)|127u;
        REJECT_WITH_MAPS(mutated_in_blocks,mutated_out_blocks,0,0);
        mutated_in_blocks[in_map-c->in_slice]=1u|((original_in&~127u)+128u);
        REJECT_WITH_MAPS(mutated_in_blocks,mutated_out_blocks,0,0);
        mutated_in_blocks[in_map-c->in_slice]=1u|(uint32_t)pool_bytes[1];
        REJECT_WITH_MAPS(mutated_in_blocks,mutated_out_blocks,0,0);
        memcpy(mutated_in_blocks,layout_s2d_blocks+c->in_start,c->in_nblocks*sizeof(uint32_t));

        mutated_out_blocks[out_map]=(original_out&~127u)|7u;
        REJECT_WITH_MAPS(mutated_in_blocks,mutated_out_blocks,0,0);
        mutated_out_blocks[out_map]=(original_out&~127u)|127u;
        REJECT_WITH_MAPS(mutated_in_blocks,mutated_out_blocks,0,0);
        mutated_out_blocks[out_map]=1u|((original_out&~127u)+128u);
        REJECT_WITH_MAPS(mutated_in_blocks,mutated_out_blocks,0,0);
        mutated_out_blocks[out_map]=1u|(uint32_t)pool_bytes[1];
        REJECT_WITH_MAPS(mutated_in_blocks,mutated_out_blocks,0,0);
        memcpy(mutated_out_blocks,layout_s2d_blocks+c->out_start,c->out_nblocks*sizeof(uint32_t));
        mutated_out_blocks[out_map]=original_in;
        REJECT_WITH_MAPS(mutated_in_blocks,mutated_out_blocks,0,0);
        mutated_out_blocks[out_map]=original_out;
        mutated_out_blocks[out_map+1]=original_out;
        REJECT_WITH_MAPS(mutated_in_blocks,mutated_out_blocks,0,0);
        memcpy(mutated_out_blocks,layout_s2d_blocks+c->out_start,c->out_nblocks*sizeof(uint32_t));

        /* Distinct pool IDs with overlapping bases must also be rejected. */
        pools[2]=pool_memory+(original_out1&~127u);
        pool_bytes[2]=sizeof pool_memory-(original_out1&~127u);
        mutated_out_blocks[out_map]=2u;
        REJECT_WITH_MAPS(mutated_in_blocks,mutated_out_blocks,0,0);
        pools[2]=NULL;pool_bytes[2]=0;
        memcpy(mutated_out_blocks,layout_s2d_blocks+c->out_start,c->out_nblocks*sizeof(uint32_t));

        cache_begin();cache_sim.fail_enabled=1;cache_sim.fail_id=original_in;
        if (qd_layout_space_to_depth_u8_direct(c->in_m,c->in_p,c->out_d,c->out_m,c->out_p,
                mutated_in_blocks,c->in_nblocks,c->in_slice,
                mutated_out_blocks,c->out_nblocks,c->out_slice,
                pools,pool_bytes,tile,sizeof tile,cache_block,NULL)) return 9;
        cache_end();
        if (cache_sim.store_count || !check_outputs(no)) return 10;

        if (qd_layout_space_to_depth_u8_direct(c->in_m,c->in_p,c->out_d,c->out_m,c->out_p,
                mutated_in_blocks,c->in_nblocks,c->in_slice,
                mutated_out_blocks,c->out_nblocks,c->out_slice,
                pools,pool_bytes,tile,sizeof tile-1,cache_block,NULL)) return 11;
        if (!check_outputs(no)) return 12;
#undef REJECT_WITH_MAPS
    }
    printf("PASS full QDRX physical oracle eligible=%u/%u; cache, holes/tails, alias guards\n",eligible,LAYOUT_S2D_CASE_COUNT);
    return 0;
}
