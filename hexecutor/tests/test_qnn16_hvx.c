#include "qnn16_hvx.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int conversions_exact(void) {
    for (uint32_t h = 0; h < 65536u; h++) {
        union { uint16_t u; __fp16 f; } v = {(uint16_t)h};
        float want = (float)v.f, got = qnn16_half_to_float((uint16_t)h);
        if (want == want ? memcmp(&want, &got, 4) != 0 : got == got)
            return printf("half %04x -> %g, want %g\n", (unsigned)h, (double)got, (double)want), 0;
    }
    /* Every float bit pattern near the binary16 range, stepping finely
     * enough to cover each rounding tie, plus subnormal and overflow edges. */
    for (uint64_t bits = 0; bits < 0x100000000ull; bits += 4093u) {
        for (uint32_t d = 0; d < 3; d++) {
            uint32_t u = (uint32_t)bits + d * 0x1000u;
            float f;
            memcpy(&f, &u, 4);
            if (f != f)
                continue;
            union { uint16_t u; __fp16 f; } want = {.f = (__fp16)f};
            uint16_t got = qnn16_float_to_half(f);
            if (got != want.u)
                return printf("float %08x -> %04x, want %04x\n", (unsigned)u, got, want.u), 0;
        }
    }
    return 1;
}

int main(void) {
    assert(conversions_exact());
    /* Use values exactly representable as binary16. */
    uint16_t x[8], y[8], z[8]; float f[2]={1.5f,-2.0f}, back[2];
    qnn16_f32_to_f16(x,f,2); qnn16_f16_to_f32(back,x,2);
    assert(back[0]==1.5f && back[1]==-2.0f);
    for(int i=0;i<8;i++){x[i]=(uint16_t[]){0x3c00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800}[i];y[i]=0x3800;}
    qnn16_mul_f16(z,x,y,8); assert(z[0]==0x3800 && z[1]==0x3c00);
    qnn16_minmax_f16(z,x,8,0.0f,3.0f); assert(z[0]==0x3c00 && z[3]==0x4200);
    Qnn16Shape sh={1,2,2,1}; uint16_t img[]={0x3c00,0x4000,0x4200,0x4400}, out[4];
    Qnn16Shape x2s_shape={1,1,1,4};
    qnn16_x2s_f16(out,img,x2s_shape);
    assert(out[0]==img[0]&&out[1]==img[1]&&out[2]==img[2]&&out[3]==img[3]);
    qnn16_transpose_hw_f16(out,img,sh); assert(out[0]==img[0]&&out[1]==img[2]&&out[2]==img[1]&&out[3]==img[3]);
    qnn16_avgpool2x2_f16(out,img,sh,1,0); assert(out[0]==0x4100); /* 2.5 */
    const uint16_t a[]={1,2},b[]={3,4}; const uint16_t *parts[]={a,b};
    Qnn16Shape cs[]={{1,1,1,2},{1,1,1,2}}; qnn16_concat_f16(out,parts,cs,2,3);
    assert(out[0]==1&&out[1]==2&&out[2]==3&&out[3]==4);
    /* QAIRT graph tensor storage: padded 2x2x3 logical region inside one
       crouton tile. The adapter must honor padding and preserve poison lanes. */
    uint8_t tile[2048]; for(int i=0;i<2048;i++)tile[i]=0xa5;
    uint32_t offs[]={0}; uint16_t logical[]={0x3c00,0x4000,0x4200,0x4400,0x4500,0x4600};
    Qnn16Tensor t={.base=tile,.block_offsets=offs,.block_count=1,.base_bytes=sizeof tile,
        .dims={1,2,3,1},.full={1,8,4,32},.padding={0,1,1,2},.direct=0};
    assert(qnn16_scatter_f16(&t,logical)==0);
    uint16_t roundtrip[6]={0}; assert(qnn16_gather_f16(roundtrip,&t)==0);
    for(int i=0;i<6;i++)assert(roundtrip[i]==logical[i]);
    /* Padding lane (full y=0,x=0,c=0) remains untouched. */
    assert(tile[0]==0xa5 && tile[1]==0xa5);
    /* Swish uses logical coordinates and supports in-place crouton tensors. */
    float sw_in[]={-4.f,-1.f,0.f,1.f,4.f,8.f};
    qnn16_f32_to_f16(logical,sw_in,6);
    assert(qnn16_scatter_f16(&t,logical)==0);
    assert(qnn16_tensor_swish(&t,&t)==0);
    assert(qnn16_gather_f16(roundtrip,&t)==0);
    for(int i=0;i<6;i++) {
        float expected=sw_in[i]/(1.f+expf(-sw_in[i]));
        assert(fabsf(qnn16_half_to_float(roundtrip[i])-expected)<0.002f);
    }
    uint32_t bad_offsets[]={2048}; t.block_offsets=bad_offsets;
    assert(qnn16_gather_f16(roundtrip,&t)<0);

    /* Input band window with a one-pixel leading pad. */
    float image[]={1,2,3,4,5,6}, band[12];
    assert(qnn16_window_f32(band,(Qnn16Shape){1,3,4,1},image,(Qnn16Shape){1,2,3,1},-1,-1)==0);
    assert(band[0]==0&&band[3]==0&&band[4]==0&&band[5]==1&&band[7]==3&&band[9]==4&&band[11]==6);
    /* Output tile placement. */
    float heat[16]={0}, part[]={7,8};
    assert(qnn16_place_f32(heat,(Qnn16Shape){1,4,4,1},part,(Qnn16Shape){1,1,2,1},2,1)==0);
    assert(heat[9]==7&&heat[10]==8&&heat[8]==0);
    assert(qnn16_place_f32(heat,(Qnn16Shape){1,4,4,1},part,(Qnn16Shape){1,1,2,1},2,3)<0);

    /* Space-to-depth phase q reads row 2y + q/2, column 2x - q%2. */
    uint16_t src[2*4*32], s2d[1*3*128];
    for(int i=0;i<2*4*32;i++)src[i]=(uint16_t)(i+1);
    assert(qnn16_space_to_depth_stride2_f16(s2d,src,(Qnn16Shape){1,2,4,32},1,3)==0);
    assert(s2d[0]==src[0]&&s2d[32]==0&&s2d[64]==src[4*32]&&s2d[96]==0);
    assert(s2d[128+32]==src[1*32]&&s2d[128+64+5]==src[(4+2)*32+5]);

    /* Zap codes clear trailing rows or columns of an FP16 crouton block. */
    uint32_t rs, colstart;
    assert(qnn16_zap_region(0xc3,&rs,&colstart)==0&&rs==5&&colstart==0);
    assert(qnn16_zap_region(0x54,&rs,&colstart)==0&&rs==0&&colstart==1);
    assert(qnn16_zap_region(0x12,&rs,&colstart)<0);
    for(int i=0;i<2048;i++)tile[i]=0xa5;
    uint8_t *zap[]={tile};
    assert(qnn16_zap_region(0x4c,&rs,&colstart)==0&&qnn16_zap_blocks(zap,1,rs,colstart)==0);
    /* Rows are [column pair][32 channels][2]: columns 2-3 clear, 0-1 stay. */
    assert(tile[(7*2+1)*128]==0&&tile[(7*2+1)*128+2]==0&&tile[7*2*128+2]==0xa5&&tile[0]==0xa5);
    puts("PASS qnn16 hvx scalar semantics"); return 0;
}
