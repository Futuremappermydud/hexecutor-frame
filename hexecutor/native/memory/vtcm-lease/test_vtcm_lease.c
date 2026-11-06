/* Offline contract test: cc -std=c11 -O2 -Wall -Wextra -Werror -o test_vtcm_lease test_vtcm_lease.c && ./test_vtcm_lease */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../qd_vtcm_lease.h"

static uint64_t regs[128];
uint64_t qdvt_test_read_tlb(uint32_t i) { return regs[i]; }
static unsigned char *vtcm, *backup;
static QdVtcmLease lease;
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); return 1; } } while(0)

int main(void) {
    vtcm=aligned_alloc(128,QDVT_BYTES); backup=aligned_alloc(128,QDVT_BACKUP_BYTES);
    CHECK(vtcm && backup);
    regs[8]=QDVT_TLB8_VALUE; regs[9]=QDVT_TLB9_VALUE;
    for (uint32_t i=0;i<QDVT_BYTES;i++) vtcm[i]=(unsigned char)((i*37u+(i>>12)*13u)^0xa7u);
    memset(&lease,0,sizeof lease);
    CHECK(qdvt_save(&lease,vtcm,QDVT_BASE,QDVT_BYTES,QDVT_BACKUP_BASE,backup)==QDVT_OK);
    CHECK(lease.phase==1 && !memcmp(vtcm,backup,QDVT_BYTES));
    memset(vtcm,0x5c,QDVT_BYTES);
    CHECK(qdvt_restore_verify(&lease,vtcm)==QDVT_OK);
    CHECK(qdvt_release_verified(&lease)==QDVT_OK && !memcmp(vtcm,backup,QDVT_BYTES));

    /* Refuse altered mappings and preserve the snapshot for recovery. */
    memset(&lease,0,sizeof lease);
    CHECK(qdvt_save(&lease,vtcm,QDVT_BASE,QDVT_BYTES,QDVT_BACKUP_BASE,backup)==QDVT_OK);
    unsigned char saved_first=backup[0], saved_last=backup[QDVT_BYTES-1];
    memset(vtcm,0x39,QDVT_BYTES); regs[9]^=0x1000;
    CHECK(qdvt_restore_verify(&lease,vtcm)==QDVT_MAPPING_CHANGED);
    CHECK(lease.phase==1 && lease.error==QDVT_MAPPING_CHANGED);
    CHECK(backup[0]==saved_first && backup[QDVT_BYTES-1]==saved_last);
    regs[9]=QDVT_TLB9_VALUE;
    CHECK(qdvt_restore_verify(&lease,vtcm)==QDVT_OK);

    /* Snapshot integrity is checked before restoring; a damaged backup must
     * not be treated as the desired original just because copy-back matches. */
    memset(&lease,0,sizeof lease);
    CHECK(qdvt_save(&lease,vtcm,QDVT_BASE,QDVT_BYTES,QDVT_BACKUP_BASE,backup)==QDVT_OK);
    memset(vtcm,0x66,QDVT_BYTES); backup[4096]^=0x80;
    CHECK(qdvt_restore_verify(&lease,vtcm)==QDVT_RESTORE_MISMATCH);
    CHECK(lease.phase==1 && lease.error==QDVT_RESTORE_MISMATCH && vtcm[4096]==0x66);

    /* A chunk swap preserves unweighted XOR and sum; position weighting must
     * catch it before restore writes into the live VTCM image. */
    memset(&lease,0,sizeof lease);
    for (uint32_t i=0;i<QDVT_BYTES;i++) vtcm[i]=(unsigned char)((i*19u+(i>>7)*7u)^0x53u);
    CHECK(qdvt_save(&lease,vtcm,QDVT_BASE,QDVT_BYTES,QDVT_BACKUP_BASE,backup)==QDVT_OK);
    unsigned char temp[128] __attribute__((aligned(128)));
    memcpy(temp,backup,128); memcpy(backup,backup+128,128); memcpy(backup+128,temp,128);
    memset(vtcm,0xd2,QDVT_BYTES);
    CHECK(qdvt_restore_verify(&lease,vtcm)==QDVT_RESTORE_MISMATCH);
    CHECK(lease.phase==1 && lease.error==QDVT_RESTORE_MISMATCH && vtcm[0]==0xd2 && vtcm[QDVT_BYTES-1]==0xd2);

    memset(&lease,0,sizeof lease);
    CHECK(qdvt_save(&lease,vtcm,QDVT_BASE,QDVT_BYTES,QDVT_BACKUP_BASE,backup+1)==QDVT_BAD_RANGE);
    CHECK(qdvt_save(&lease,vtcm,QDVT_BASE,QDVT_BYTES,QDVT_BACKUP_BASE,vtcm)==QDVT_BAD_RANGE);
    CHECK(qdvt_save(&lease,vtcm,QDVT_BASE,QDVT_BYTES-128,QDVT_BACKUP_BASE,backup)==QDVT_BAD_RANGE);
    CHECK(qdvt_save(&lease,vtcm,QDVT_BASE,QDVT_BYTES,QDVT_BACKUP_BASE+128,backup)==QDVT_BAD_RANGE);
    CHECK(qdvt_save(&lease,vtcm,QDVT_BASE,QDVT_BYTES,QDVT_BACKUP_BASE,backup)==QDVT_OK);
    CHECK(qdvt_save(&lease,vtcm,QDVT_BASE,QDVT_BYTES,QDVT_BACKUP_BASE,backup)==QDVT_BAD_STATE);
    CHECK(qdvt_release_verified(&lease)==QDVT_BAD_STATE);
    puts("VTCM lease offline contract PASS");
    free(vtcm); free(backup); return 0;
}
