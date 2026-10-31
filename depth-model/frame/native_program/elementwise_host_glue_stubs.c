/* Preflight-only host link stubs. Numerical behavior is tested on Hexagon. */
#include "../blocked_glue/add_u8.h"
#include "../blocked_glue/linearclip_u8_u16.h"
int qd_blocked_add_u8_apply(const QdBlockedAddU8 *p,const QdAddU8Params *q) {
    (void)p;(void)q;return 1;
}
int qd_blocked_linearclip_u8_u16_apply(const QdBlockedLinearclipU8U16 *p,
        const QdLinearclipU8U16Params *q) {
    (void)p;(void)q;return 1;
}
