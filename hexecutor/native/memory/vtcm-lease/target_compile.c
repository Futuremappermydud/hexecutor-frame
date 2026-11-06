/* Hexagon-only compile smoke test for the target HVX implementation. */
#include "../../qd_vtcm_lease.h"

uint32_t qdvt_target_compile_smoke(QdVtcmLease *lease) {
    uint32_t rc=qdvt_save(lease,(volatile uint8_t *)QDVT_BASE,QDVT_BASE,QDVT_BYTES,
                          QDVT_BACKUP_BASE,(uint8_t *)QDVT_BACKUP_BASE);
    if (rc) return rc;
    rc=qdvt_restore_verify(lease,(volatile uint8_t *)QDVT_BASE);
    if (rc) return rc;
    return qdvt_release_verified(lease);
}
