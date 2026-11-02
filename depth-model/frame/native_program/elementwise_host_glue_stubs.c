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
#include "../blocked_glue/elementwise_u16.h"
int qd_blocked_ew16_apply(const QdBlockedU16 *a,const QdBlockedU16 *b,const QdBlockedU16 *out,
                          const QdEw16Params *p,int32_t bscalar) {
    (void)a;(void)b;(void)out;(void)p;(void)bscalar;return 1;
}
/* sign_blocked.h / linearclip_u16.h name their descriptor QdBlockedU16, which
 * elementwise_u16.h (above) already typedefs. */
#define QdBlockedU16 QdBlockedU16Glue
#include "../blocked_glue/sign_blocked.h"
#undef QdBlockedU16
int qd_blocked_sign_u16_apply(const QdBlockedSignU16 *prepared,int32_t input_zero,
        const QdSignLevels *levels) {
    (void)prepared;(void)input_zero;(void)levels;return 1;
}
int qd_blocked_linearclip_u16_apply_params(const QdBlockedU16Linearclip *prepared,
        const QdLinearclipParams *p) {
    (void)prepared;(void)p;return 1;
}
int qd_blocked_add_u8_whole(const QdBlockedAddU8 *p) { (void)p; return 0; }
int qd_blocked_add_u8_apply_whole(const QdBlockedAddU8 *p,const QdAddU8Params *q,uint32_t part,uint32_t parts) {
    (void)p; (void)q; (void)part; (void)parts; return 0;
}
