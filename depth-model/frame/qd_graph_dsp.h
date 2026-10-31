#ifndef QD_GRAPH_DSP_H
#define QD_GRAPH_DSP_H
#include "qd_conv.h"
#include "qd_owned_memory.h"
#define QDS_MAGIC 0x31534451u
#define QDS_CONFIG 4096u
#define QDS_WORK 8192u
#define QDS_WORK_L2 0x300f0000u
#define QDS_RESULT 2048u
#define QDS_PROFILE 16384u
#define QDS_OP_RUN 3u
typedef struct {
    uint32_t magic,version,image,image_bytes,pool[7],pool_bytes[7];
    uint32_t input[4],input_bytes[4],output[2],output_bytes[2];
    uint32_t heap,heap_bytes,code_bytes;
} QdsConfig;
/* Check the complete allocation contract before initializing the heap or
 * touching any graph-owned bytes. Scratch and worker staging have fixed slots. */
static uint32_t qds_config_valid(const QdsConfig *c) {
    uint32_t offsets[15],bytes[15],n=0;
    if(c->magic!=QDS_MAGIC || c->version!=1 || !c->code_bytes || c->code_bytes>0x100000u ||
       c->heap!=0x2000000u || c->heap_bytes!=0x2000000u-4096u)return 0;
    offsets[n]=c->image;bytes[n++]=c->image_bytes;
    for(uint32_t i=0;i<7;i++)if(c->pool_bytes[i]) {
        offsets[n]=c->pool[i];bytes[n++]=c->pool_bytes[i];
        uint32_t lo=i==5?0x100000u:0x1000000u,hi=i==5?0x1000000u:0x2000000u;
        if(c->pool[i]<lo || c->pool[i]>hi || c->pool_bytes[i]>hi-c->pool[i])return 0;
    }
    for(uint32_t i=0;i<4;i++){offsets[n]=c->input[i];bytes[n++]=c->input_bytes[i];}
    for(uint32_t i=0;i<2;i++){offsets[n]=c->output[i];bytes[n++]=c->output_bytes[i];}
    if(c->image<0x100000u || c->image>=0x1000000u || c->image_bytes<256u ||
       c->image_bytes>0x1000000u-c->image)return 0;
    /* The first seven descriptors are image/pools; I/O occupies the last six. */
    for(uint32_t i=n-6;i<n;i++)if(offsets[i]<0x1000000u)return 0;
    for(uint32_t i=0;i<n;i++) {
        if((offsets[i]&127u) || !bytes[i] || offsets[i]>=0x2000000u || bytes[i]>0x2000000u-offsets[i])return 0;
        for(uint32_t j=0;j<i;j++)if(offsets[i]<offsets[j]+bytes[j] && offsets[j]<offsets[i]+bytes[i])return 0;
    }
    return 1;
}
typedef struct {
    volatile uint32_t generation,op,sequence,error,done[3];
    uint32_t arena,prepared_a;
    QdConv conv;
} QdsWork;
uint32_t qdrc_execute(const QdConv *c,uint8_t *arena,uint32_t tid,uint32_t prepared_a);
uint32_t qdrc_power(uint32_t on);
#if QDS_NATIVE_1X1
void qdrc_invalidate_native(void);
#endif
#endif
