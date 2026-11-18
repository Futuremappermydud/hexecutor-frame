/* Portable scalar semantics for QNN FP16 vector/data-movement leaves.
 * Buffers are logical NHWC, as decoded from the QAIRT graph/trace oracle.
 */
#include "qnn16_hvx.h"
#include <math.h>
#include <string.h>

#include "qnn16_half.h"
float qnn16_half_to_float(uint16_t h){return hf(h);}
uint16_t qnn16_float_to_half(float f){return fh(f);}
static uint32_t dim(Qnn16Shape s,uint32_t a){return a==0?s.b:a==1?s.h:a==2?s.w:s.c;}
static uint64_t block_count64(Qnn16Shape f){return (uint64_t)f.b*(((uint64_t)f.h+7)/8)*(((uint64_t)f.w+3)/4)*(((uint64_t)f.c+31)/32);}
static uint32_t block_count(Qnn16Shape f){return (uint32_t)block_count64(f);}
static uint32_t block_index(Qnn16Shape f,uint32_t b,uint32_t y,uint32_t x,uint32_t c){
 uint32_t nh=(f.h+7)/8,nw=(f.w+3)/4,nc=(f.c+31)/32;
 return ((b*nh+y/8)*nw+x/4)*nc+c/32;
}
static uint32_t tile_byte(uint32_t y,uint32_t x,uint32_t c){
 /* QNN FP16 tile is [8,2,32,2], equivalent logical [8,4,32]. */
 return (((y%8)*2+x%4/2)*32+c%32)*4+(x%2)*2;
}
static int valid_tensor(const Qnn16Tensor *t){
 if(!t||!t->base)return 0;
 if(block_count64(t->full)>UINT32_MAX)return 0;
 for(uint32_t a=0;a<4;a++)if(dim(t->dims,a)>dim(t->full,a)||dim(t->padding,a)+dim(t->dims,a)>dim(t->full,a))return 0;
 if(t->direct){uint64_t n=(uint64_t)t->full.b*t->full.h*t->full.w*t->full.c*2;return t->direct_offset<=t->base_bytes&&n<=t->base_bytes-t->direct_offset;}
 if(!t->block_offsets||t->block_count<block_count(t->full))return 0;
 for(uint32_t i=0;i<block_count(t->full);i++){uint32_t off=t->block_offsets[i];if(off!=UINT32_MAX&&(off>t->base_bytes||t->base_bytes-off<2048))return 0;}
 return 1;
}
/* Per-pixel cursor: one block lookup per 32-channel group, not per element.
 * Within a group, channels are 4 bytes apart in a tile and 2 bytes apart in
 * direct storage. */
typedef struct{const Qnn16Tensor *t;uint8_t *p;uint32_t fb,fy,fx,g,stride;}Cur;
static inline void cur_pixel(Cur *k,const Qnn16Tensor *t,uint32_t b,uint32_t y,uint32_t x){
 k->t=t;k->fb=b+t->padding.b;k->fy=y+t->padding.h;k->fx=x+t->padding.w;k->g=UINT32_MAX;k->p=0;k->stride=t->direct?2:4;
}
static inline uint8_t *cur_at(Cur *k,uint32_t c){
 const Qnn16Tensor *t=k->t;uint32_t fc=c+t->padding.c,g=fc>>5;
 if(g!=k->g){
  k->g=g;
  if(t->direct)k->p=t->base+t->direct_offset+2*(((k->fb*t->full.h+k->fy)*t->full.w+k->fx)*t->full.c+g*32);
  else{uint32_t off=t->block_offsets[block_index(t->full,k->fb,k->fy,k->fx,g*32)];
   k->p=off==UINT32_MAX?0:t->base+off+tile_byte(k->fy,k->fx,0);}
 }
 return k->p?k->p+(fc&31)*k->stride:0;
}
int qnn16_gather_f16(uint16_t *out,const Qnn16Tensor *t){
 if(!out||!valid_tensor(t))return -1;
 Cur k;
 for(uint32_t b=0;b<t->dims.b;b++)for(uint32_t y=0;y<t->dims.h;y++)for(uint32_t x=0;x<t->dims.w;x++){
  uint16_t *o=out+((b*t->dims.h+y)*t->dims.w+x)*t->dims.c;
  cur_pixel(&k,t,b,y,x);
  for(uint32_t c=0;c<t->dims.c;c++){const uint8_t *p=cur_at(&k,c);o[c]=p?*(const uint16_t *)p:0;}
 }
 return 0;
}
int qnn16_scatter_f16(Qnn16Tensor *t,const uint16_t *in){
 if(!in||!valid_tensor(t))return -1;
 Cur k;
 for(uint32_t b=0;b<t->dims.b;b++)for(uint32_t y=0;y<t->dims.h;y++)for(uint32_t x=0;x<t->dims.w;x++){
  const uint16_t *v=in+((b*t->dims.h+y)*t->dims.w+x)*t->dims.c;
  cur_pixel(&k,t,b,y,x);
  for(uint32_t c=0;c<t->dims.c;c++){uint8_t *p=cur_at(&k,c);if(p)*(uint16_t *)p=v[c];}
 }
 return 0;
}
static int same_shape(Qnn16Shape a,Qnn16Shape b){return a.b==b.b&&a.h==b.h&&a.w==b.w&&a.c==b.c;}
static int scalar_shape(Qnn16Shape s){return s.b==1&&s.h==1&&s.w==1&&s.c==1;}
static uint8_t *tensor_ptr(Qnn16Tensor *t,uint32_t b,uint32_t y,uint32_t x,uint32_t c){
 uint32_t fb=b+t->padding.b,fy=y+t->padding.h,fx=x+t->padding.w,fc=c+t->padding.c;
 if(t->direct)return t->base+t->direct_offset+2*(((fb*t->full.h+fy)*t->full.w+fx)*t->full.c+fc);
 uint32_t off=t->block_offsets[block_index(t->full,fb,fy,fx,fc)];
 return off==UINT32_MAX?0:t->base+off+tile_byte(fy,fx,fc);
}
static uint16_t load_h(const uint8_t *p){return p?*(const uint16_t *)p:0;}
static void store_h(uint8_t *p,uint16_t x){if(p)*(uint16_t *)p=x;}
int qnn16_tensor_copy_format(Qnn16Tensor *out,const Qnn16Tensor *in){
 if(!out||!in||!same_shape(out->dims,in->dims)||!valid_tensor(out)||!valid_tensor(in))return -1;
 Cur ko,ki;
 for(uint32_t b=0;b<in->dims.b;b++)for(uint32_t y=0;y<in->dims.h;y++)for(uint32_t x=0;x<in->dims.w;x++){
  cur_pixel(&ko,out,b,y,x);cur_pixel(&ki,in,b,y,x);
  for(uint32_t c=0;c<in->dims.c;c++)store_h(cur_at(&ko,c),load_h(cur_at(&ki,c)));
 }
 return 0;
}
int qnn16_tensor_f16_to_f32(float *out,const Qnn16Tensor *in){
 if(!out||!valid_tensor(in))return -1;
 Cur k;
 for(uint32_t b=0;b<in->dims.b;b++)for(uint32_t y=0;y<in->dims.h;y++)for(uint32_t x=0;x<in->dims.w;x++){
  float *o=out+((b*in->dims.h+y)*in->dims.w+x)*in->dims.c;
  cur_pixel(&k,in,b,y,x);
  for(uint32_t c=0;c<in->dims.c;c++)o[c]=hf(load_h(cur_at(&k,c)));
 }
 return 0;
}
int qnn16_tensor_im2col_cast_rows(Qnn16Tensor *out,const float *in,Qnn16Shape sh,uint32_t part,uint32_t parts){
 if(!out||!in||!valid_tensor(out)||sh.c!=1||sh.h<3||sh.w<3||parts==0||
    out->dims.b!=sh.b||out->dims.h!=(sh.h-3)/2+1||
    out->dims.w!=(sh.w-3)/2+1||out->dims.c!=9)return -1;
 /* Blocked output with the 9 channels inside one 32-channel group: one block
  * lookup per pixel pair, and both pixels of a tile pair written as one word
  * per channel (channels are 4 bytes apart). */
 const int fast=!out->direct&&out->padding.c+9<=32;
 for(uint32_t r=part;r<sh.b*out->dims.h;r+=parts){
  uint32_t b=r/out->dims.h,y=r%out->dims.h;
  for(uint32_t x=0;x<out->dims.w;){
   const float *src=in+(b*sh.h+y*2)*sh.w+x*2;
   if(fast){
    uint8_t *p=tensor_ptr(out,b,y,x,0);
    const uint32_t fx=x+out->padding.w;
    if(!(fx&1u)&&x+1<out->dims.w){
     if(p)for(uint32_t ky=0;ky<3;ky++)for(uint32_t kx=0;kx<3;kx++)
      *(uint32_t *)(p+(ky*3+kx)*4)=(uint32_t)fh(src[ky*sh.w+kx])|(uint32_t)fh(src[ky*sh.w+kx+2])<<16;
     x+=2;continue;
    }
    if(p)for(uint32_t ky=0;ky<3;ky++)for(uint32_t kx=0;kx<3;kx++)
     *(uint16_t *)(p+(ky*3+kx)*4)=fh(src[ky*sh.w+kx]);
    x++;continue;
   }
   for(uint32_t ky=0;ky<3;ky++)for(uint32_t kx=0;kx<3;kx++)
    store_h(tensor_ptr(out,b,y,x,ky*3+kx),fh(src[ky*sh.w+kx]));
   x++;
  }
 }
 return 0;
}
int qnn16_tensor_im2col_cast(Qnn16Tensor *out,const float *in,Qnn16Shape sh){
 return qnn16_tensor_im2col_cast_rows(out,in,sh,0,1);
}
int qnn16_tensor_sigmoid(Qnn16Tensor *out,const Qnn16Tensor *in,uint16_t *scratch){
 (void)scratch;if(!out||!in||!same_shape(out->dims,in->dims)||!valid_tensor(out)||!valid_tensor(in))return -1;
 Cur ko,ki;
 for(uint32_t b=0;b<in->dims.b;b++)for(uint32_t y=0;y<in->dims.h;y++)for(uint32_t x=0;x<in->dims.w;x++){
  cur_pixel(&ko,out,b,y,x);cur_pixel(&ki,in,b,y,x);
  for(uint32_t c=0;c<in->dims.c;c++){float v=hf(load_h(cur_at(&ki,c)));store_h(cur_at(&ko,c),fh(1.f/(1.f+expf(-v))));}
 }
 return 0;
}
int qnn16_tensor_softmax(Qnn16Tensor *out,const Qnn16Tensor *in){
 if(!out||!in||!same_shape(out->dims,in->dims)||!valid_tensor(out)||!valid_tensor(in))return -1;
 for(uint32_t b=0;b<in->dims.b;b++)for(uint32_t y=0;y<in->dims.h;y++)for(uint32_t x=0;x<in->dims.w;x++){
  Cur a,o;cur_pixel(&a,in,b,y,x);cur_pixel(&o,out,b,y,x);float max=-INFINITY,sum=0;
  for(uint32_t c=0;c<in->dims.c;c++){float v=hf(load_h(cur_at(&a,c)));if(v>max)max=v;}
  for(uint32_t c=0;c<in->dims.c;c++)sum+=expf(hf(load_h(cur_at(&a,c)))-max);
  for(uint32_t c=0;c<in->dims.c;c++)store_h(cur_at(&o,c),fh(expf(hf(load_h(cur_at(&a,c)))-max)/sum));
 }
 return 0;
}
int qnn16_tensor_resize_nearest(Qnn16Tensor *out,const Qnn16Tensor *in,uint32_t oy,uint32_t ox){
 if(!out||!in||!valid_tensor(out)||!valid_tensor(in)||in->dims.b!=out->dims.b||in->dims.c!=out->dims.c||oy>=in->dims.h||ox>=in->dims.w||(out->dims.h-1)/2>=in->dims.h-oy||(out->dims.w-1)/2>=in->dims.w-ox)return -1;
 for(uint32_t b=0;b<out->dims.b;b++)for(uint32_t y=0;y<out->dims.h;y++)for(uint32_t x=0;x<out->dims.w;x++){
  Cur a,o;cur_pixel(&a,in,b,oy+y/2,ox+x/2);cur_pixel(&o,out,b,y,x);
  for(uint32_t c=0;c<out->dims.c;c++)store_h(cur_at(&o,c),load_h(cur_at(&a,c)));
 }
 return 0;
}
int qnn16_tensor_resize(Qnn16Tensor *out,const Qnn16Tensor *in,float sy,float oy,float sx,float ox){
 if(!out||!in||!valid_tensor(out)||!valid_tensor(in)||in->dims.b!=out->dims.b||in->dims.c!=out->dims.c||!in->dims.h||!in->dims.w||!isfinite(sy)||!isfinite(oy)||!isfinite(sx)||!isfinite(ox))return -1;
 for(uint32_t b=0;b<out->dims.b;b++)for(uint32_t y=0;y<out->dims.h;y++)for(uint32_t x=0;x<out->dims.w;x++){
  float fy=y*sy+oy,fx=x*sx+ox; if(fy<0)fy=0;if(fx<0)fx=0;if(fy>in->dims.h-1)fy=in->dims.h-1;if(fx>in->dims.w-1)fx=in->dims.w-1;
  uint32_t y0=(uint32_t)fy,x0=(uint32_t)fx,y1=y0+1<in->dims.h?y0+1:y0,x1=x0+1<in->dims.w?x0+1:x0;fy-=y0;fx-=x0;
  Cur a,b0,c,d,o;cur_pixel(&a,in,b,y0,x0);cur_pixel(&b0,in,b,y0,x1);cur_pixel(&c,in,b,y1,x0);cur_pixel(&d,in,b,y1,x1);cur_pixel(&o,out,b,y,x);
  for(uint32_t ch=0;ch<out->dims.c;ch++){
   float v0=hf(load_h(cur_at(&a,ch)))*(1-fx)+hf(load_h(cur_at(&b0,ch)))*fx;
   float v1=hf(load_h(cur_at(&c,ch)))*(1-fx)+hf(load_h(cur_at(&d,ch)))*fx;
   store_h(cur_at(&o,ch),fh(v0*(1-fy)+v1*fy));
  }
 }
 return 0;
}
int qnn16_tensor_decimate(Qnn16Tensor *out,const Qnn16Tensor *in,uint32_t sy,uint32_t sx){
 if(!out||!in||!sy||!sx||!valid_tensor(out)||!valid_tensor(in)||in->dims.b!=out->dims.b||in->dims.c!=out->dims.c||out->dims.h!=(in->dims.h+sy-1)/sy||out->dims.w!=(in->dims.w+sx-1)/sx)return -1;
 Cur ko,ki;
 for(uint32_t b=0;b<out->dims.b;b++)for(uint32_t y=0;y<out->dims.h;y++)for(uint32_t x=0;x<out->dims.w;x++){
  cur_pixel(&ko,out,b,y,x);cur_pixel(&ki,in,b,y*sy,x*sx);
  for(uint32_t c=0;c<out->dims.c;c++)store_h(cur_at(&ko,c),load_h(cur_at(&ki,c)));
 }
 return 0;
}
int qnn16_tensor_add_swish(Qnn16Tensor *out,const Qnn16Tensor *in){
 if(!out||!in||!valid_tensor(out)||!valid_tensor(in)||in->dims.b!=out->dims.b||in->dims.h!=out->dims.h||in->dims.w!=out->dims.w||in->dims.c!=64||out->dims.c!=32)return -1;
 Cur ko,ki;
 for(uint32_t b=0;b<in->dims.b;b++)for(uint32_t y=0;y<in->dims.h;y++)for(uint32_t x=0;x<in->dims.w;x++){
  cur_pixel(&ko,out,b,y,x);cur_pixel(&ki,in,b,y,x);
  for(uint32_t c=0;c<32;c++){float v=hf(load_h(cur_at(&ki,c)))+hf(load_h(cur_at(&ki,c+32)));store_h(cur_at(&ko,c),fh(v/(1.f+expf(-v))));}
 }
 return 0;
}
int qnn16_tensor_swish(Qnn16Tensor *out,const Qnn16Tensor *in){
 if(!out||!in||!same_shape(out->dims,in->dims)||!valid_tensor(out)||!valid_tensor(in))return -1;
 Cur ko,ki;
 for(uint32_t b=0;b<in->dims.b;b++)for(uint32_t y=0;y<in->dims.h;y++)for(uint32_t x=0;x<in->dims.w;x++){
  cur_pixel(&ko,out,b,y,x);cur_pixel(&ki,in,b,y,x);
  for(uint32_t c=0;c<in->dims.c;c++){float v=hf(load_h(cur_at(&ki,c)));store_h(cur_at(&ko,c),fh(v/(1.f+expf(-v))));}
 }
 return 0;
}
int qnn16_tensor_minmax(Qnn16Tensor *out,const Qnn16Tensor *in,uint16_t *scratch,float lo,float hi){
 (void)scratch;if(!out||!in||!same_shape(out->dims,in->dims)||!valid_tensor(out)||!valid_tensor(in))return -1;
 Cur ko,ki;
 for(uint32_t b=0;b<in->dims.b;b++)for(uint32_t y=0;y<in->dims.h;y++)for(uint32_t x=0;x<in->dims.w;x++){
  cur_pixel(&ko,out,b,y,x);cur_pixel(&ki,in,b,y,x);
  for(uint32_t c=0;c<in->dims.c;c++){float v=hf(load_h(cur_at(&ki,c)));if(v<lo)v=lo;if(v>hi)v=hi;store_h(cur_at(&ko,c),fh(v));}
 }
 return 0;
}
int qnn16_tensor_mul(Qnn16Tensor *out,const Qnn16Tensor *a,const Qnn16Tensor *b,uint16_t *sa,uint16_t *sb){
 (void)sa;(void)sb;if(!out||!a||!b||!same_shape(out->dims,a->dims)||
  !(b->dims.b==1||b->dims.b==a->dims.b)||!(b->dims.h==1||b->dims.h==a->dims.h)||
  !(b->dims.w==1||b->dims.w==a->dims.w)||!(b->dims.c==1||b->dims.c==a->dims.c)||
  !valid_tensor(out)||!valid_tensor(a)||!valid_tensor(b))return -1;
 const int scalar=scalar_shape(b->dims);
 const float bs=scalar?hf(load_h(tensor_ptr((Qnn16Tensor*)b,0,0,0,0))):0.f;
 Cur ko,ka,kb;
 for(uint32_t n=0;n<a->dims.b;n++)for(uint32_t y=0;y<a->dims.h;y++)for(uint32_t x=0;x<a->dims.w;x++){
  cur_pixel(&ko,out,n,y,x);cur_pixel(&ka,a,n,y,x);
  if(!scalar)cur_pixel(&kb,b,b->dims.b==1?0:n,b->dims.h==1?0:y,b->dims.w==1?0:x);
  for(uint32_t c=0;c<a->dims.c;c++){
   float av=hf(load_h(cur_at(&ka,c))),bv=scalar?bs:hf(load_h(cur_at(&kb,b->dims.c==1?0:c)));
   store_h(cur_at(&ko,c),fh(av*bv));
  }
 }
 return 0;
}
void qnn16_copy(void *d, const void *s, uint32_t n) { memmove(d, s, n); }
void qnn16_zero(void *d, uint32_t n) { memset(d, 0, n); }
void qnn16_f32_to_f16(uint16_t *d, const float *s, uint32_t n) { for(uint32_t i=0;i<n;i++) d[i]=fh(s[i]); }
void qnn16_f16_to_f32(float *d, const uint16_t *s, uint32_t n) { for(uint32_t i=0;i<n;i++) d[i]=hf(s[i]); }
void qnn16_mul_f16(uint16_t *d,const uint16_t *a,const uint16_t *b,uint32_t n) { for(uint32_t i=0;i<n;i++) d[i]=fh(hf(a[i])*hf(b[i])); }
void qnn16_minmax_f16(uint16_t *d,const uint16_t *s,uint32_t n,float lo,float hi) { for(uint32_t i=0;i<n;i++){float x=hf(s[i]); if(x<lo)x=lo; if(x>hi)x=hi; d[i]=fh(x);} }
void qnn16_sigmoid_f16(uint16_t *d,const uint16_t *s,uint32_t n) { for(uint32_t i=0;i<n;i++){float x=hf(s[i]); d[i]=fh(1.0f/(1.0f+expf(-x)));} }
static uint16_t fi_half(const uint8_t *fi,uint32_t word,uint32_t high){
 uint32_t v=*(const uint32_t *)(fi+word*4);return (uint16_t)(high?v>>16:v);
}
static float clip_table(float acc,const uint8_t *table,uint32_t lane,uint32_t tables){
 /* The second table's words 0-31 hold the upper clamp in their high half. */
 if(tables==2){float hi=hf(fi_half(table+256,lane,1));if(acc<0)acc=0;if(acc>hi)acc=hi;}
 return acc;
}
void qnn16_depthwise1x1_f16(uint16_t *d,const uint16_t *s,Qnn16Shape sh,
 uint32_t oc,const uint16_t *w,const uint8_t *fi,uint32_t tables){
 if(!d||!s||!w||!fi||sh.c!=oc*2||(tables!=1&&tables!=2))return;
 for(uint32_t i=0;i<sh.b*sh.h*sh.w;i++)for(uint32_t co=0;co<oc;co++){
  uint32_t group=co/32,lane=co%32;
  const uint8_t *table=fi+group*tables*256;
  float scale=hf(fi_half(table,lane,0));
  float acc=hf(fi_half(table,32+lane,1));
  for(uint32_t j=0;j<2;j++)
   acc+=hf(s[i*sh.c+co+j*32])*hf(w[group*1024+(lane/2)*4+j*3])*scale;
  d[i*oc+co]=fh(clip_table(acc,table,lane,tables));
 }
}
void qnn16_depthwise3x3_f16(uint16_t *d,const uint16_t *s,Qnn16Shape sh,
 const uint16_t *w,const uint8_t *fi,uint32_t tables){
 if(!d||!s||!w||!fi||sh.h<3||sh.w<3||(tables!=1&&tables!=2))return;
 uint32_t oh=sh.h-2,ow=sh.w-2;
 for(uint32_t b=0;b<sh.b;b++)for(uint32_t y=0;y<oh;y++)for(uint32_t x=0;x<ow;x++)
  for(uint32_t co=0;co<sh.c;co++){
   uint32_t group=co/32,lane=co%32;
   const uint8_t *table=fi+group*tables*256;
   float scale=hf(fi_half(table,lane,0));
   float acc=hf(fi_half(table,32+lane,1));
   /* [group][tap q = ky*3 + 2 - kx][16 channel pairs][even, 0, 0, odd]. */
   uint32_t base=group*9*64+(lane/2)*4+(lane%2)*3;
   for(uint32_t ky=0;ky<3;ky++)for(uint32_t kx=0;kx<3;kx++)
    acc+=hf(s[((b*sh.h+y+ky)*sh.w+x+kx)*sh.c+co])*hf(w[base+(ky*3+2-kx)*64])*scale;
   d[((b*oh+y)*ow+x)*sh.c+co]=fh(clip_table(acc,table,lane,tables));
  }
}
int qnn16_space_to_depth_stride2_f16(uint16_t *d,const uint16_t *s,Qnn16Shape sh,
 uint32_t oh,uint32_t ow){
 if(!d||!s||!sh.c||sh.c%32)return -1;
 uint32_t oc=sh.c*4;
 for(uint32_t b=0;b<sh.b;b++)for(uint32_t y=0;y<oh;y++)for(uint32_t x=0;x<ow;x++)
  for(uint32_t q=0;q<4;q++){
   /* Phase q: row 2y + q/2, column 2x - q%2; outside the tensor is zero. */
   uint32_t r=2*y+(q>>1);int64_t c=(int64_t)2*x-(q&1);
   int in=r<sh.h&&c>=0&&c<sh.w;
   for(uint32_t ch=0;ch<sh.c;ch++){
    uint32_t o=((b*oh+y)*ow+x)*oc+((ch/32)*4+q)*32+ch%32;
    d[o]=in?s[((b*sh.h+r)*sh.w+(uint32_t)c)*sh.c+ch]:0;
   }
  }
 return 0;
}
int qnn16_depthwise3x3_stride2_f16(uint16_t *d,const uint16_t *s,Qnn16Shape sh,
 uint32_t oh,uint32_t ow,const uint16_t *w,const uint8_t *fi,uint32_t tables){
 /* Xwe tile (group-major, 9 per 32-channel group) for tap ky*3 + kx. */
 static const uint8_t tile_of_tap[9]={1,4,0,7,8,6,3,5,2};
 /* Space-to-depth row/column offset and phase bit for k = 0, 1, 2. */
 static const uint8_t off[3]={0,1,1},phase[3]={0,1,0};
 if(!d||!s||!w||!fi||!sh.c||sh.c%128||(tables!=1&&tables!=2)||
    oh+1>sh.h||ow+1>sh.w)return -1;
 uint32_t c=sh.c/4;
 for(uint32_t b=0;b<sh.b;b++)for(uint32_t y=0;y<oh;y++)for(uint32_t x=0;x<ow;x++)
  for(uint32_t co=0;co<c;co++){
   uint32_t group=co/32,lane=co%32;
   const uint8_t *table=fi+group*tables*256;
   float scale=hf(fi_half(table,lane,0));
   float acc=hf(fi_half(table,32+lane,1));
   for(uint32_t ky=0;ky<3;ky++)for(uint32_t kx=0;kx<3;kx++){
    /* Tap ky selects row phase ky%2 at row y + ky/2; tap kx column phase
     * kx%2 at column x + (kx+1)/2 (phase 1 holds the column to the left). */
    uint32_t q=phase[ky]*2+(kx==1);
    uint32_t iy=y+(ky==2),ix=x+off[kx];
    uint32_t ci=(group*4+q)*32+lane;
    uint32_t wi=(group*9+tile_of_tap[ky*3+kx])*1024+(lane/2)*64+lane*2+lane%2;
    acc+=hf(s[((b*sh.h+iy)*sh.w+ix)*sh.c+ci])*hf(w[wi])*scale;
   }
   d[((b*oh+y)*ow+x)*c+co]=fh(clip_table(acc,table,lane,tables));
  }
 return 0;
}
void qnn16_pointwise_f16(uint16_t *d,const uint16_t *s,Qnn16Shape sh,
 uint32_t ic,uint32_t oc,const uint16_t *w,const uint8_t *fi,uint32_t tables){
 if(!d||!s||!w||!fi||!ic||!oc||sh.c>ic||(tables!=1&&tables!=2))return;
 /* Xpe: per 32-output group, ceil(ic/2) input-channel pairs of [32 co][2]. */
 uint32_t pairs=(ic+1)/2;
 for(uint32_t i=0;i<sh.b*sh.h*sh.w;i++)for(uint32_t co=0;co<oc;co++){
  uint32_t group=co/32,lane=co%32;
  const uint8_t *table=fi+group*tables*256;
  float scale=hf(fi_half(table,lane,0));
  float acc=hf(fi_half(table,32+lane,1));
  for(uint32_t ci=0;ci<sh.c;ci++)
   acc+=hf(s[i*sh.c+ci])*hf(w[(group*pairs+ci/2)*64+lane*2+ci%2])*scale;
  d[i*oc+co]=fh(clip_table(acc,table,lane,tables));
 }
}
static void conv3x3(uint16_t *d,const uint16_t *s,Qnn16Shape sh,
 uint32_t ic,uint32_t oc,const uint16_t *w,const uint8_t *fi,uint32_t tables){
 if(!d||!s||!w||!fi||sh.h<3||sh.w<3||!ic||!oc||(tables!=1&&tables!=2))return;
 uint32_t oh=sh.h-2,ow=sh.w-2,kg=(ic+31)/32;
 for(uint32_t b=0;b<sh.b;b++)for(uint32_t y=0;y<oh;y++)for(uint32_t x=0;x<ow;x++)
  for(uint32_t co=0;co<oc;co++){
   uint32_t group=co/32,lane=co%32;
   const uint8_t *table=fi+group*tables*256;
   float scale=hf(fi_half(table,lane,0));
   float acc=hf(fi_half(table,32+lane,1));
   for(uint32_t ci=0;ci<sh.c;ci++)for(uint32_t ky=0;ky<3;ky++)for(uint32_t kx=0;kx<3;kx++){
    uint32_t q=ky*3+2-kx,cb=ci/32,pair=(ci%32)/2,cc=ci%2;
    uint32_t wi=(((group*kg+cb)*9+q)*16+pair)*64+lane*2+cc;
    acc+=hf(s[((b*sh.h+y+ky)*sh.w+x+kx)*sh.c+ci])*hf(w[wi])*scale;
   }
   d[((b*oh+y)*ow+x)*oc+co]=fh(clip_table(acc,table,lane,tables));
  }
}
void qnn16_conv3x3_f16(uint16_t *d,const uint16_t *s,Qnn16Shape sh,
 uint32_t ic,uint32_t oc,const uint16_t *w,const uint8_t *fi,uint32_t tables){
 conv3x3(d,s,sh,ic,oc,w,fi,tables);
}

/* Generic axis copy: row-major NHWC and arbitrary batch/spatial/channel axis. */
static uint32_t outer(Qnn16Shape s,uint32_t a){uint32_t n=1;for(uint32_t i=0;i<a;i++)n*=dim(s,i);return n;}
static uint32_t inner(Qnn16Shape s,uint32_t a){uint32_t n=1;for(uint32_t i=a+1;i<4;i++)n*=dim(s,i);return n;}
void qnn16_concat_f16(uint16_t *d,const uint16_t *const *src,const Qnn16Shape *ss,uint32_t count,uint32_t axis){
 if(axis>3||!count)return; uint32_t o=outer(ss[0],axis), in=inner(ss[0],axis), dp=0;
 for(uint32_t x=0;x<o;x++) for(uint32_t k=0;k<count;k++){uint32_t n=dim(ss[k],axis)*in; memcpy(d+dp,src[k]+x*n,n*2);dp+=n;}
}
void qnn16_slice_f16(uint16_t *d,const uint16_t *s,Qnn16Shape sh,uint32_t axis,uint32_t start,uint32_t len){
 if(axis>3||start+len>dim(sh,axis))return; uint32_t o=outer(sh,axis),in=inner(sh,axis),full=dim(sh,axis)*in,take=len*in;
 for(uint32_t x=0;x<o;x++)memcpy(d+x*take,s+x*full+start*in,take*2);
}
void qnn16_transpose_hw_f16(uint16_t *d,const uint16_t *s,Qnn16Shape sh){
 for(uint32_t b=0;b<sh.b;b++)for(uint32_t y=0;y<sh.h;y++)for(uint32_t x=0;x<sh.w;x++)for(uint32_t c=0;c<sh.c;c++) d[((b*sh.w+x)*sh.h+y)*sh.c+c]=s[((b*sh.h+y)*sh.w+x)*sh.c+c];
}
void qnn16_transpose_f16(uint16_t *d,const uint16_t *s,Qnn16Shape sh,const uint32_t perm[4]){
 if(!perm)return;
 uint32_t seen=0,src_dim[4]={sh.b,sh.h,sh.w,sh.c},dst_dim[4];
 for(uint32_t i=0;i<4;i++){
  if(perm[i]>3||(seen&(1u<<perm[i])))return;
  seen|=1u<<perm[i];dst_dim[i]=src_dim[perm[i]];
 }
 for(uint32_t b=0;b<sh.b;b++)for(uint32_t h=0;h<sh.h;h++)
  for(uint32_t w=0;w<sh.w;w++)for(uint32_t c=0;c<sh.c;c++){
   uint32_t src[4]={b,h,w,c};
   uint32_t o=((src[perm[0]]*dst_dim[1]+src[perm[1]])*dst_dim[2]
               +src[perm[2]])*dst_dim[3]+src[perm[3]];
   d[o]=s[((b*sh.h+h)*sh.w+w)*sh.c+c];
  }
}
void qnn16_channel_shuffle_f16(uint16_t *d,const uint16_t *s,Qnn16Shape sh,uint32_t g){
 if(!g||sh.c%g)return; uint32_t per=sh.c/g;
 for(uint32_t p=0;p<sh.b*sh.h*sh.w;p++)for(uint32_t c=0;c<sh.c;c++){uint32_t src=(c%g)*per+c/g;d[p*sh.c+c]=s[p*sh.c+src];}
}
void qnn16_avgpool2x2_f16(uint16_t *d,const uint16_t *s,Qnn16Shape sh,uint32_t stride,uint32_t ceil_mode){
 if(!stride)return; uint32_t oh=ceil_mode?(sh.h+stride-1)/stride:(sh.h<2?0:(sh.h-2)/stride+1),ow=ceil_mode?(sh.w+stride-1)/stride:(sh.w<2?0:(sh.w-2)/stride+1);
 for(uint32_t b=0;b<sh.b;b++)for(uint32_t y=0;y<oh;y++)for(uint32_t x=0;x<ow;x++)for(uint32_t c=0;c<sh.c;c++){float sum=0;uint32_t n=0;for(uint32_t dy=0;dy<2;dy++)for(uint32_t dx=0;dx<2;dx++){uint32_t iy=y*stride+dy,ix=x*stride+dx;if(iy<sh.h&&ix<sh.w){sum+=hf(s[((b*sh.h+iy)*sh.w+ix)*sh.c+c]);n++;}}d[((b*oh+y)*ow+x)*sh.c+c]=fh(n?sum/n:0);}
}
void qnn16_avgpool_valid_f16(uint16_t *d,const uint16_t *s,Qnn16Shape sh,uint32_t kernel,uint32_t stride){
 if(!kernel||!stride||sh.h<kernel||sh.w<kernel)return;
 uint32_t oh=(sh.h-kernel)/stride+1,ow=(sh.w-kernel)/stride+1;
 for(uint32_t b=0;b<sh.b;b++)for(uint32_t y=0;y<oh;y++)for(uint32_t x=0;x<ow;x++)
  for(uint32_t c=0;c<sh.c;c++){
   float sum=0;
   for(uint32_t ky=0;ky<kernel;ky++)for(uint32_t kx=0;kx<kernel;kx++)
    sum+=hf(s[((b*sh.h+y*stride+ky)*sh.w+x*stride+kx)*sh.c+c]);
   d[((b*oh+y)*ow+x)*sh.c+c]=fh(sum/(float)(kernel*kernel));
  }
}
int qnn16_window_f32(float *d,Qnn16Shape out,const float *s,Qnn16Shape in,
 int32_t y0,int32_t x0){
 if(!d||!s||out.b!=in.b||out.c!=in.c)return -1;
 /* Columns [xs, xe) of each output row come from one contiguous source run. */
 int64_t xs=x0<0?-(int64_t)x0:0,xe=(int64_t)in.w-x0;
 if(xe>(int64_t)out.w)xe=out.w;
 if(xs>(int64_t)out.w)xs=out.w;
 if(xe<xs)xe=xs;
 const size_t px=out.c*sizeof(float);
 for(uint32_t b=0;b<out.b;b++)for(uint32_t y=0;y<out.h;y++){
  int64_t iy=(int64_t)y+y0;
  float *o=d+(size_t)(b*out.h+y)*out.w*out.c;
  if(iy<0||iy>=in.h){memset(o,0,out.w*px);continue;}
  if(xs)memset(o,0,(size_t)xs*px);
  if(xe>xs)memcpy(o+(size_t)xs*out.c,s+((size_t)(b*in.h+(uint32_t)iy)*in.w+(size_t)(xs+x0))*in.c,(size_t)(xe-xs)*px);
  if(xe<(int64_t)out.w)memset(o+(size_t)xe*out.c,0,(size_t)(out.w-xe)*px);
 }
 return 0;
}
int qnn16_place_f32(float *d,Qnn16Shape out,const float *s,Qnn16Shape tile,
 uint32_t y0,uint32_t x0){
 if(!d||!s||out.b!=tile.b||out.c!=tile.c||y0>out.h||x0>out.w||
    tile.h>out.h-y0||tile.w>out.w-x0)return -1;
 for(uint32_t b=0;b<tile.b;b++)for(uint32_t y=0;y<tile.h;y++)
  memcpy(d+((b*out.h+y0+y)*out.w+x0)*out.c,s+(b*tile.h+y)*tile.w*tile.c,
         (size_t)tile.w*tile.c*sizeof(float));
 return 0;
}
int qnn16_zap_region(uint32_t code,uint32_t *row_start,uint32_t *col_start){
 if(!row_start||!col_start)return -1;
 if((code&0xf0)==0xc0&&(code&0xf)>=1&&(code&0xf)<=8){*row_start=8-(code&0xf);*col_start=0;return 0;}
 /* Only the column codes Mercury's graphs use are decoded. */
 if(code==0x50||code==0x4c||code==0x54){*row_start=0;*col_start=code==0x50?3:code==0x4c?2:1;return 0;}
 return -1;
}
int qnn16_zap_blocks(uint8_t *const *blocks,uint32_t count,uint32_t row_start,uint32_t col_start){
 if(!blocks||row_start>8||col_start>4)return -1;
 for(uint32_t i=0;i<count;i++){
  if(!blocks[i])return -1;
  for(uint32_t y=row_start;y<8;y++)for(uint32_t x=col_start;x<4;x++)for(uint32_t c=0;c<32;c++)
   memset(blocks[i]+tile_byte(y,x,c),0,2);
 }
 return 0;
}
/* x2s_opt is depth-to-space, DCR channel order. */
void qnn16_x2s_f16(uint16_t *d,const uint16_t *s,Qnn16Shape sh){
 if(sh.c%4)return; uint32_t oc=sh.c/4,oh=sh.h*2,ow=sh.w*2;
 for(uint32_t b=0;b<sh.b;b++)for(uint32_t y=0;y<sh.h;y++)for(uint32_t x=0;x<sh.w;x++)
  for(uint32_t c=0;c<sh.c;c++){
   uint32_t q=c/oc;
   d[((b*oh+y*2+q/2)*ow+x*2+q%2)*oc+c%oc]=s[((b*sh.h+y)*sh.w+x)*sh.c+c];
  }
}
