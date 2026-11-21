// LD_PRELOAD shim for QAIRT's x86_64 HTP simulator (libQnnHtp.so 2.50.0.260828).
//
// The runtime executes ops from run lists of 32-byte entries {fn, op, wait, done}
// through three exported dispatchers (continue_execution_fast and the two
// background-thread variants). Each is reached through the PLT, so this shim
// interposes them, rewrites every entry once to call a trampoline, and runs each
// op under a global mutex. After every op it compares VTCM with a shadow copy and
// appends the changed 2 KB blocks to a write log. The VTCM base comes from
// nn_os_vtcm_hmx_acquire (VtcmReq: size at +0x8, base at +0x20).
// The offsets come from the simulator 2.50 disassembly.
//
// Environment:
//   QD_TRACE_LOG  text log: "seq list idx fn_offset thread nblocks"
//   QD_WORKER_CHECKPOINTS  add worker wait/done addresses to the text log
//   QD_WRITES     binary write log (records below); omit to log only
//   QD_CALLS      text log of exported HVX kernel calls made by each op:
//                 "seq name rdi rsi rdx rcx r8 r9 | stack0..3 | xmm0..3 (low 64 bits)"
//                 with VTCM pointers printed as v+offset
//   QD_TENSOR_CONFIG  lines: list idx argument_slot tensor_id storage_bytes
//   QD_TENSORS    direct DDR outputs: u32 seq, tensor_id, bytes, then data
// Write record: u32 seq, u32 list, u32 idx, u32 block, then 2048 data bytes,
// where block * 2048 is the VTCM offset.
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <map>
#include <atomic>
#include <mutex>
#include <vector>

namespace {
using OpFn = int (*)(void *, void *, int);
struct ExecEntry { OpFn fn; void *op; int *wait; int *done; };
struct Wrapped { uint64_t magic; OpFn fn; void *op; uint32_t list, idx; };
constexpr uint64_t MAGIC = 0x7164747261636521ull;
constexpr size_t BLOCK = 2048;

std::mutex mu;
uint8_t *vtcm_base;
uint64_t vtcm_size;
std::vector<uint8_t> shadow;
FILE *trace_log, *writes;
bool worker_checkpoints;
uintptr_t lib_base;
uint32_t seq;
std::map<ExecEntry *, uint32_t> lists;
struct TensorDump { uint32_t slot, tid, bytes; };
std::map<std::pair<uint32_t, uint32_t>, std::vector<TensorDump>> tensor_dumps;
FILE *tensor_log;

void init_locked() {
    static bool done;
    if (done) return;
    done = true;
    worker_checkpoints = getenv("QD_WORKER_CHECKPOINTS") != nullptr;
    if (const char *p = getenv("QD_TRACE_LOG")) trace_log = fopen(p, "w");
    if (const char *p = getenv("QD_WRITES")) writes = fopen(p, "wb");
    if (const char *p = getenv("QD_TENSOR_CONFIG")) {
        FILE *f = fopen(p, "r");
        uint32_t l, i, s, t, n;
        if (f) {
            while (fscanf(f, "%u %u %u %u %u", &l, &i, &s, &t, &n) == 5)
                tensor_dumps[{l, i}].push_back({s, t, n});
            fclose(f);
        }
    }
    if (const char *p = getenv("QD_TENSORS")) tensor_log = fopen(p, "wb");
    Dl_info info;
    void *sym = dlsym(RTLD_NEXT, "continue_execution_fast");
    if (sym && dladdr(sym, &info)) lib_base = (uintptr_t)info.dli_fbase;
}

void log_writes_locked(uint32_t list, uint32_t idx, OpFn fn) {
    uint32_t changed = 0;
    if (writes && vtcm_base) {
        if (shadow.size() != vtcm_size) shadow.assign(vtcm_size, 0);
        for (size_t b = 0; b < vtcm_size / BLOCK; ++b) {
            uint8_t *cur = vtcm_base + b * BLOCK;
            if (memcmp(cur, &shadow[b * BLOCK], BLOCK) == 0) continue;
            uint32_t hdr[4] = {seq, list, idx, (uint32_t)b};
            fwrite(hdr, sizeof hdr, 1, writes);
            fwrite(cur, 1, BLOCK, writes);
            memcpy(&shadow[b * BLOCK], cur, BLOCK);
            ++changed;
        }
    }
    if (trace_log)
        fprintf(trace_log, "%u %u %u %#lx %ld %u\n", seq, list, idx,
                (unsigned long)((uintptr_t)fn - lib_base), (long)syscall(SYS_gettid), changed);
    ++seq;
}

// Global, not thread-local: ops run one at a time under `mu`, but an op may hand
// kernel calls to worker threads.
std::atomic<uint32_t> cur_seq{~0u};

int trampoline(void *arg, void *graph, int flag) {
    Wrapped *w = (Wrapped *)arg;
    std::lock_guard<std::mutex> lock(mu);
    cur_seq = seq;
    int rc = w->fn(w->op, graph, flag);
    // Run-list argument structures hold tensor pointers in configured 8-byte
    // slots. Direct rank-4 tensors store their data pointer at +24 in this SDK.
    // process_vm_readv makes a mismatched SDK layout fail closed.
    if (tensor_log) for (auto d : tensor_dumps[{w->list, w->idx}]) {
        auto read = [](void *dst, const void *src, size_t n) {
            iovec local{dst, n}, remote{const_cast<void *>(src), n};
            return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == (ssize_t)n;
        };
        void *tensor = nullptr, *data = nullptr;
        std::vector<uint8_t> buf(d.bytes);
        bool indirect = (d.slot & 0x80000000u) != 0;
        uint32_t slot = d.slot & 0x7fffffffu;
        bool ok = read(&tensor, (char *)w->op + 8 * slot, 8) &&
                  read(&data, (char *)tensor + 24, 8);
        if (ok && indirect) {
            ok = d.bytes % BLOCK == 0;
            for (size_t b = 0; ok && b < d.bytes / BLOCK; ++b) {
                void *block = nullptr;
                ok = read(&block, (char *)data + b * sizeof(void *), sizeof(void *)) &&
                     read(buf.data() + b * BLOCK, block, BLOCK);
            }
        } else if (ok) {
            ok = read(buf.data(), data, d.bytes);
        }
        if (!ok) {
            fprintf(stderr, "QD tensor dump failed: list=%u idx=%u tid=%u\n", w->list, w->idx, d.tid);
            continue;
        }
        uint32_t header[] = {seq, d.tid, d.bytes};
        fwrite(header, sizeof header, 1, tensor_log);
        fwrite(buf.data(), 1, buf.size(), tensor_log);
    }
    cur_seq = ~0u;
    log_writes_locked(w->list, w->idx, w->fn);
    return rc;
}

void wrap_list(ExecEntry *list, unsigned count) {
    std::lock_guard<std::mutex> lock(mu);
    init_locked();
    auto it = lists.find(list);
    if (it != lists.end()) return;
    uint32_t id = lists.size();
    lists[list] = id;
    unsigned n = 0;
    for (unsigned i = 0; i < count; ++i) {
        if (list[i].fn == trampoline) continue;
        if (trace_log && worker_checkpoints && id == 1)
            fprintf(trace_log, "# worker-checkpoints %u %#lx %#lx\n", i,
                    (unsigned long)(uintptr_t)list[i].wait, (unsigned long)(uintptr_t)list[i].done);
        Wrapped *w = new Wrapped{MAGIC, list[i].fn, list[i].op, id, i};
        list[i].op = w;
        list[i].fn = trampoline;
        ++n;
    }
    if (trace_log) fprintf(trace_log, "# list %u at %p count %u wrapped %u\n", id, (void *)list, count, n);
}
}  // namespace

#define FORWARD(name)                                                                              \
    extern "C" int name(void *a1, void *a2, ExecEntry *list, unsigned count, void *a5, void *a6,   \
                        uint64_t s1, uint64_t s2, uint64_t s3, uint64_t s4, uint64_t s5, uint64_t s6) { \
        using Fn = int (*)(void *, void *, ExecEntry *, unsigned, void *, void *, uint64_t, uint64_t,  \
                           uint64_t, uint64_t, uint64_t, uint64_t);                                \
        static Fn orig = (Fn)dlsym(RTLD_NEXT, #name);                                              \
        wrap_list(list, count);                                                                    \
        return orig(a1, a2, list, count, a5, a6, s1, s2, s3, s4, s5, s6);                          \
    }
FORWARD(continue_execution_fast)
FORWARD(continue_execution_bkgrnd_thread)
FORWARD(continue_execution_bkgrnd_thread_dual)

#define ACQ _Z22nn_os_vtcm_hmx_acquireR7VtcmReqRN4hnnx6McDataERK12HexagonNNEnvRNSt3__110unique_ptrIA_13DmaAlignedBufPFvPS9_EEERNS8_I9OsVxuInfoNS7_14default_deleteISG_EEEEbRKNS7_6vectorINS7_7variantIJNS1_11MasterEntryENS1_10SlaveEntryENS1_13DoorbellEntryENS1_12OpTraceEntryENS1_14SharedBufEntryEEEENS7_9allocatorISS_EEEEPFijPvESY_
#define STR2(x) #x
#define STR(x) STR2(x)
extern "C" int ACQ(void *req, void *mc, void *env, void *buf, void *vxu, bool b, void *vec, void *cb, void *cbarg) {
    using Fn = int (*)(void *, void *, void *, void *, void *, bool, void *, void *, void *);
    static Fn orig = (Fn)dlsym(RTLD_NEXT, STR(ACQ));
    int rc = orig(req, mc, env, buf, vxu, b, vec, cb, cbarg);
    std::lock_guard<std::mutex> lock(mu);
    init_locked();
    vtcm_base = *(uint8_t **)((char *)req + 0x20);
    vtcm_size = *(uint64_t *)((char *)req + 0x8);
    if (trace_log) fprintf(trace_log, "# vtcm base=%p size=%#lx rc=%d\n", vtcm_base, (unsigned long)vtcm_size, rc);
    return rc;
}

// --------------------------------------------------------------------------
// Kernel call logging. Each listed exported function gets an assembly stub that
// saves the argument registers, logs them, restores them and tail-jumps to the
// real function, so the callee sees an untouched frame.
// --------------------------------------------------------------------------
#define QD_KERNELS(X) \
    X(0, hlx_linearclip_u16) X(1, hlx_elementwise_add_sub_u16) X(2, hlx_elementwise_add_u8) \
    X(3, hlx_elementwise_mul_u16) X(4, hlx_elementwise_mul_u8) X(5, hlx_elementwise_sign_u16) \
    X(6, hlx_abs_u16) X(7, hlx_requantize_u16) X(8, hlx_elementwise_mul1111_u16) \
    X(9, hlx_elementwise_mul_bhwd_bhw32_u16) X(10, hlx_elementwise_add1111_crouton_u16) \
    X(11, elementwise_add_crouton_u8) X(12, elementwise_add_crouton_u16) X(13, elementwise_addsub_crouton_u16) \
    X(14, elementwise_mul_crouton_u16) X(15, elementwise_mul_crouton_u8) X(16, elementwise_mul111d_crouton_u16) \
    X(17, elementwise_mul1111_crouton_u16) X(18, elementwise_add1111_crouton_u16) X(19, elementwise_add111d_crouton_u16) \
    X(20, elementwise_add_flat_u16) X(21, elementwise_sub_flat_u16) X(22, elementwise_mul_flat_u16) \
    X(23, elementwise_mul1111_flat_u16) X(24, elementwise_mulbcd1_crouton_u16) X(25, requantize_crouton_u8) \
    X(26, requantize_crouton_u16) X(27, requantize_flat_u8) X(28, requantize_flat_u16) \
    X(29, linearclip_u16tou8_flat) X(30, convert_b_h_same_quant) X(31, convert_h_b_same_quant) \
    X(32, resizenear_b_x2) X(33, resizenear_h_x2) X(34, resizenear_b_x4) X(35, decimate2x2_b_00) \
    X(36, decimate_general_b) X(37, decimate_h_stride2x2_00) X(38, d2s_shortd_flatout_u16) \
    X(39, d2s_crd_shortd_flatout_u16) X(40, transpose_b_hxw) X(41, transpose_b_wxd) X(42, transpose_h_hxw) \
    X(43, transpose_h_wxd) X(44, quantize_sf_to_u16) X(45, quantize_sf_to_u8) X(46, concat_depth_crouton_b) \
    X(47, concat_depth_crouton_h) X(48, hvx_reduce_sum_u16_case_1) X(49, hvx_reduce_sum_u16_case_8) \
    X(50, elementwise_add_flat_u8) X(51, elementwise_mul_flat_u8) X(52, elementwise_add111d_crouton_u8) \
    X(53, elementwise_mul111d_crouton_u8) X(54, decimate2x1_b_00) X(55, decimate1x2_b_00) \
    X(56, elementwise_add_bhw1_111d_crouton_u16) X(57, elementwise_addflat111d_crouton_u8) \
    X(58, weights_signed_trivial_and_shuffled_transpose_128x32b_flat) X(59, hvx_reduce_sum_u16_case_2) \
    X(60, hvx_reduce_sum_u16_case_4) X(61, hvx_reduce_sum_u16_case_6)

#define QD_NAME(i, n) #n,
static const char *qd_names[] = {QD_KERNELS(QD_NAME)};
constexpr int QD_N = sizeof(qd_names) / sizeof(qd_names[0]);
extern "C" __attribute__((visibility("hidden"))) void *qd_real[QD_N];
void *qd_real[QD_N];
static FILE *calls_log;
static std::mutex calls_mu;

extern "C" __attribute__((visibility("hidden"))) void qd_log_call(uint64_t *regs, uint64_t *stack, int idx) {
    std::lock_guard<std::mutex> lock(calls_mu);
    if (!qd_real[idx]) qd_real[idx] = dlsym(RTLD_NEXT, qd_names[idx]);  // backend loads late
    if (!calls_log) return;
    auto show = [](uint64_t v) {
        if (vtcm_base && v >= (uint64_t)vtcm_base && v < (uint64_t)vtcm_base + vtcm_size)
            fprintf(calls_log, " v+%lx", (unsigned long)(v - (uint64_t)vtcm_base));
        else
            fprintf(calls_log, " %lx", (unsigned long)v);
    };
    fprintf(calls_log, "%u %s", cur_seq.load(), qd_names[idx]);
    for (int i = 0; i < 6; ++i) show(regs[i]);
    fprintf(calls_log, " |");
    for (int i = 0; i < 4; ++i) show(stack[i]);
    fprintf(calls_log, " |");
    for (int i = 0; i < 4; ++i) fprintf(calls_log, " %lx", (unsigned long)regs[6 + 2 * i]);
    fprintf(calls_log, "\n");
}

__attribute__((constructor)) static void qd_open_log() {
    if (const char *p = getenv("QD_CALLS")) calls_log = fopen(p, "w");
}

// Frame: 6 GPRs (48 B) + 8 XMM (128 B) + pad = 192 B; stack args at rbp+16.
#define QD_STUB(i, n)                                                    \
    asm(".globl " #n "\n.type " #n ",@function\n" #n ":\n"               \
        "push %rbp\nmov %rsp,%rbp\nsub $192,%rsp\n"                      \
        "mov %rdi,0(%rsp)\nmov %rsi,8(%rsp)\nmov %rdx,16(%rsp)\n"        \
        "mov %rcx,24(%rsp)\nmov %r8,32(%rsp)\nmov %r9,40(%rsp)\n"         \
        "movdqu %xmm0,48(%rsp)\nmovdqu %xmm1,64(%rsp)\nmovdqu %xmm2,80(%rsp)\n" \
        "movdqu %xmm3,96(%rsp)\nmovdqu %xmm4,112(%rsp)\nmovdqu %xmm5,128(%rsp)\n" \
        "movdqu %xmm6,144(%rsp)\nmovdqu %xmm7,160(%rsp)\nmov %rax,176(%rsp)\n" \
        "mov %rsp,%rdi\nlea 16(%rbp),%rsi\nmov $" #i ",%edx\n"           \
        "call qd_log_call\n"                                             \
        "mov 0(%rsp),%rdi\nmov 8(%rsp),%rsi\nmov 16(%rsp),%rdx\n"        \
        "mov 24(%rsp),%rcx\nmov 32(%rsp),%r8\nmov 40(%rsp),%r9\n"         \
        "movdqu 48(%rsp),%xmm0\nmovdqu 64(%rsp),%xmm1\nmovdqu 80(%rsp),%xmm2\n" \
        "movdqu 96(%rsp),%xmm3\nmovdqu 112(%rsp),%xmm4\nmovdqu 128(%rsp),%xmm5\n" \
        "movdqu 144(%rsp),%xmm6\nmovdqu 160(%rsp),%xmm7\nmov 176(%rsp),%rax\n" \
        "leave\njmp *qd_real+8*" #i "(%rip)\n");
QD_KERNELS(QD_STUB)
