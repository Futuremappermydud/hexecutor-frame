#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
#include "linearclip_u16.h"

static uint8_t ib[8192] __attribute__((aligned(128)));
static uint8_t ob[32768] __attribute__((aligned(128)));
static uint8_t hole[8192] __attribute__((aligned(128)));
static uint8_t inplace_data[8192] __attribute__((aligned(128)));
static uint32_t itab[8],otab[256],htab[8];
static uint8_t *ip[2]={0,ib},*op[2]={0,ob},*hp[2]={0,hole};
static uint8_t *inplace_pool[2]={0,inplace_data};

static uint16_t *at(uint8_t *p,int H,int W,int D) {
    return (uint16_t *)(p+2048+(D/32)*2048+(H%8)*256+((W%4)/2)*128+(W%2)*2+(D%32)*4);
}
static uint16_t *tensor_at(const QdBlockedU16 *t,int b,int h,int w,int d) {
    int B=b+t->origin[0],H=h+t->origin[1],W=w+t->origin[2],D=d+t->origin[3];
    int nbh=(t->meta[1]+7)/8,nbw=(t->meta[2]+3)/4,nbd=(t->meta[3]+31)/32;
    uint32_t index=(((B*nbh+H/8)*nbw+W/4)*nbd)+t->slice+D/32;
    uint32_t e=t->blocks[index];
    return (uint16_t *)(t->pools[e&127]+(e&~127u)+(H%8)*256+((W%4)/2)*128+(W%2)*2+(D%32)*4);
}
int main(void) {
    SIM_ACQUIRE_HVX;SIM_SET_HVX_DOUBLE_MODE;
    QdBlockedU16 in={ip,itab,8,0,{1,8,4,72},{1,5,3,47},{0,1,1,21}};
    QdBlockedU16 out={op,otab,8,0,{1,8,4,72},{1,5,3,47},{0,1,1,21}};
    QdBlockedU16 miss={hp,htab,8,0,{1,8,4,72},{1,5,3,47},{0,1,1,21}};
    for(int i=0;i<8;i++)itab[i]=otab[i]=htab[i]=0;
    for(int i=0;i<3;i++)itab[i]=otab[i]=(2048+i*2048)|1;
    memset(ib,0x6b,sizeof ib);memset(ob,0xa5,sizeof ob);memset(hole,0x3c,sizeof hole);
    for(int h=1;h<6;h++)for(int w=1;w<4;w++)for(int d=21;d<68;d++)*at(ib,h,w,d)=(uint16_t)(h*7901+w*1301+d*997);
    QdLinearclipParams p={-12345,32100,27111,17,0};
    QdBlockedU16Linearclip prepared;
    if(!qd_blocked_linearclip_u16_prepare(&in,&out,&p,&prepared)||
       !qd_blocked_linearclip_u16_apply(&prepared))return 1;
    for(int h=1;h<6;h++)for(int w=1;w<4;w++)for(int d=21;d<68;d++) {
        uint32_t want=qd_linearclip_u16_oracle(*at(ib,h,w,d),&p);
        if(*at(ob,h,w,d)!=want){printf("mismatch h%d w%d d%d got%u want%u input%u\n",h,w,d,(unsigned)*at(ob,h,w,d),(unsigned)want,(unsigned)*at(ib,h,w,d));return 2;}
    }
    QdLinearclipParams p2={-9000,5000,30000,17,0};
    if(!qd_blocked_linearclip_u16_apply_params(&prepared,&p2))return 8;
    for(int h=1;h<6;h++)for(int w=1;w<4;w++)for(int d=21;d<68;d++)
        if(*at(ob,h,w,d)!=qd_linearclip_u16_oracle(*at(ib,h,w,d),&p2))return 9;
    /* Output must preserve every unaddressed byte, including crouton padding. */
    uint8_t touched_bytes[sizeof ob]={0};
    for(int h=1;h<6;h++)for(int w=1;w<4;w++)for(int d=21;d<68;d++) {
        size_t off=(size_t)((uint8_t *)at(ob,h,w,d)-ob);
        touched_bytes[off]=touched_bytes[off+1]=1;
    }
    for(size_t i=0;i<sizeof ob;i++) {
        if(!touched_bytes[i]&&ob[i]!=0xa5){printf("guard/padding changed at %lu\n",(unsigned long)i);return 3;}
    }
    /* Missing source is numeric zero and missing destination is a no-op. */
    htab[0]=0;memset(hole,0x3c,sizeof hole);
    if(!qd_blocked_linearclip_u16(&miss,&out,&p))return 4;
    for(int h=1;h<6;h++)for(int w=1;w<4;w++)for(int d=21;d<68;d++)
        if(*at(ob,h,w,d)!=qd_linearclip_u16_oracle(0,&p))return 5;
    /* Empty logical tensors are accepted no-ops, including through the
     * checked convenience API. */
    uint8_t before[sizeof ob];memcpy(before,ob,sizeof ob);
    in.dims[3]=out.dims[3]=0;
    if(!qd_blocked_linearclip_u16(&in,&out,&p)||memcmp(before,ob,sizeof ob))return 6;
    in.dims[3]=out.dims[3]=47;
    /* Identical injective mapping is safe in-place: every physical vector is
     * loaded before its own store and no later logical vector aliases it. */
    memcpy(inplace_data,ib,sizeof ib);
    QdBlockedU16 in_place=in;in_place.pools=inplace_pool;in_place.blocks=itab;
    if(!qd_blocked_linearclip_u16(&in_place,&in_place,&p))return 10;
    for(int h=1;h<6;h++)for(int w=1;w<4;w++)for(int d=21;d<68;d++)
        if(*at(inplace_data,h,w,d)!=qd_linearclip_u16_oracle(*at(ib,h,w,d),&p))return 11;
    /* Independent metadata, origins, slices, and scattered output blocks.
     * W parity and D%32 match, while H/B origins and W/D coordinates shift. */
    QdBlockedU16 shifted_out={op,otab,40,2,{2,16,8,104},{1,5,3,47},{1,5,3,53}};
    memset(otab,0,sizeof otab);memset(ob,0xa5,sizeof ob);
    int out_nbh=2,out_nbw=2,out_nbd=4,block_no=0;
    for(int hg=0;hg<2;hg++)for(int W=0;W<4;W+=2)for(int db=1;db<=3;db++) {
        int H=hg*8+shifted_out.origin[1],PW=W-in.origin[2]+shifted_out.origin[2];
        int idx=((((1*out_nbh+H/8)*out_nbw+PW/4)*out_nbd)+shifted_out.slice+db);
        otab[idx]=(2048+(11-block_no++)*2048)|1;
    }
    if(!qd_blocked_linearclip_u16_prepare(&in,&shifted_out,&p,&prepared)||
       !qd_blocked_linearclip_u16_apply(&prepared))return 14;
    for(int h=0;h<5;h++)for(int w=0;w<3;w++)for(int d=0;d<47;d++)
        if(*tensor_at(&shifted_out,0,h,w,d)!=qd_linearclip_u16_oracle(*at(ib,h+1,w+1,d+21),&p)) {
            printf("shifted mapping mismatch %d %d %d\n",h,w,d);return 15;
        }
    otab[0]=2048|1;otab[1]=otab[0];otab[2]=6144|1;
    if(qd_blocked_linearclip_u16(&in,&out,&p)){puts("accepted aliased output table");return 7;}
    uint32_t repeated[8];memcpy(repeated,itab,sizeof repeated);repeated[1]=repeated[0];
    in_place.blocks=repeated;
    if(qd_blocked_linearclip_u16_prepare(&in_place,&in_place,&p,&prepared)) {
        puts("accepted repeated-block in-place alias");return 12;
    }
    uint32_t shifted[8];memcpy(shifted,itab,sizeof shifted);
    shifted[0]=itab[0];shifted[1]=1;shifted[2]=itab[1];
    QdBlockedU16 partial=in;partial.blocks=shifted;partial.block_count=3;
    if(qd_blocked_linearclip_u16_prepare(&in,&partial,&p,&prepared)) {
        puts("accepted shifted partial overlap");return 13;
    }
    puts("PASS blocked u16 linearclip: exact oracle, reparameterize, tails, guards, holes, empty no-op, safe in-place, alias rejection");
    SIM_RELEASE_HVX;return 0;
}
