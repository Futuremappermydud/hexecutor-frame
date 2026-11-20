/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Isolated client for the Model vendor's signed libhexagon RPC interface.
 * Recovered against libhexagon.so SHA 9750f275... (Oct 30, 2026).
 * Uses the preserved DSP-side ET12 wrapper to avoid the raw-context route's
 * dependency on a DSP libQnnSystem.so that is absent from this firmware.
 * Build with the Android NDK aarch64-linux-android29-clang, -ldl.
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stddef.h>

/* QAIC host size 56, DSP size 48, wire metadata size 44. The low dtype byte
 * is 0 for f32 and 1 for u8; 0x100 denotes the ExecuTorch scalar encoding. */
typedef struct {
    int32_t dims[8], rank, dtype;
    void *data;
    int32_t bytes;
} NNTensorDef;
_Static_assert(sizeof(NNTensorDef) == 56 && offsetof(NNTensorDef, data) == 40, "RPC ABI");

static void tensor(NNTensorDef *t, int channels, int h, int w, int dtype) {
    *t = (NNTensorDef){.dims = {1, channels, h, w}, .rank = 4, .dtype = dtype};
    t->bytes = channels * h * w * (dtype == 0x100 ? 4 : 1);
    t->data = malloc((size_t)t->bytes);
    if (!t->data) exit(2);
}
static void tensor_file(NNTensorDef *t, const char *dir, const char *name, int writing) {
    char path[2048];
    if (snprintf(path, sizeof path, "%s/%s.raw", dir, name) >= (int)sizeof path) exit(2);
    FILE *f = fopen(path, writing ? "wb" : "rb");
    if (!f) { perror(path); exit(2); }
    size_t n = writing ? fwrite(t->data, 1, t->bytes, f) : fread(t->data, 1, t->bytes, f);
    if (n != (size_t)t->bytes || (!writing && fgetc(f) != EOF)) {
        fprintf(stderr, "wrong tensor file size: %s\n", path); exit(2);
    }
    fclose(f);
}
static void profile_file(const char *dir, const void *data, size_t bytes) {
    char path[2048];
    if (snprintf(path, sizeof path, "%s/profile.bin", dir) >= (int)sizeof path) exit(2);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(data, 1, bytes, f) != bytes) { perror(path); exit(2); }
    fclose(f);
}

static double seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static uint64_t checksum(const void *data, size_t n) {
    const unsigned char *p = data;
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}
static void *symbol(void *lib, const char *name) {
    void *p = dlsym(lib, name);
    if (!p) { fprintf(stderr, "%s: %s\n", name, dlerror()); exit(2); }
    return p;
}
static void dsp_log(void *lib, uint64_t handle) {
    int (*get_size)(uint64_t, uint32_t *) = symbol(lib, "hexagon_nnGetLogSize");
    int (*get_log)(uint64_t, char *, int) = symbol(lib, "hexagon_nnGetLog");
    uint32_t size = 0;
    int rc = get_size(handle, &size);
    if (rc || !size || size > 1048576) return;
    char *s = calloc(1, (size_t)size + 1);
    if (s && !get_log(handle, s, (int)size)) printf("dsp_log: %s\n", s);
    free(s);
}
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2 && argc != 4 && argc != 5) {
        fprintf(stderr, "usage: %s model.{bin,pte} [input_dir output_dir [repeat]]\n", argv[0]); return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); rewind(f);
    if (bytes < 4 || bytes > INT32_MAX) return 2;
    void *data = malloc((size_t)bytes);
    if (!data || fread(data, 1, (size_t)bytes, f) != (size_t)bytes) return 2;
    fclose(f);
    void *lib = dlopen("/vendor/lib64/libhexagon.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    int (*open_rpc)(const char *, uint64_t *) = symbol(lib, "hexagon_open");
    int (*close_rpc)(uint64_t) = symbol(lib, "hexagon_close");
    int (*version)(uint64_t, uint32_t *) = symbol(lib, "hexagon_getDspVersion");
    int et = bytes >= 8 && !memcmp((char *)data + 4, "ET12", 4);
    int (*create)(uint64_t, const void *, int, uint32_t *, uint32_t *, uint32_t *, int *) =
        symbol(lib, et ? "hexagon_nnExecutorchCreateModel2" : "hexagon_nnOssQnnCreate");
    int (*destroy)(uint64_t, uint32_t) = symbol(lib, et ? "hexagon_nnExecutorchDestroyModel" : "hexagon_nnOssQnnDestroy");
    uint64_t handle = UINT64_MAX;
    int rc = open_rpc("file:///libhexagon_skel.so?hexagon_skel_handle_invoke&_modver=1.0&_dom=cdsp", &handle);
    printf("open rc=0x%x handle=0x%llx\n", rc, (unsigned long long)handle);
    if (rc) return 1;
    uint32_t arch = 0;
    rc = version(handle, &arch);
    printf("dsp_version rc=0x%x value=0x%x (%u)\n", rc, arch, arch);
    if (getenv("QD_DEBUG")) {
        int (*debug)(uint64_t, int) = symbol(lib, "hexagon_nnSetDebugLevel");
        printf("debug rc=0x%x\n", debug(handle, 5));
    }
    uint32_t model = UINT32_MAX, inputs = 0, outputs = 0; int status = -1;
    double start = seconds();
    rc = create(handle, data, (int)bytes, &model, &inputs, &outputs, &status);
    printf("create kind=%s rc=0x%x status=%d model=%u inputs=%u outputs=%u seconds=%.6f\n",
           et ? "executorch" : "oss", rc, status, model, inputs, outputs, seconds() - start);
    int failed = rc || status;
    int created = !failed;
    if (!failed && argc >= 4 && et) {
        if (inputs != 4 || outputs != 2) { fprintf(stderr, "unexpected model contract\n"); return 2; }
        int (*sizes)(uint64_t, uint32_t, uint32_t *, int, uint32_t *, int) =
            symbol(lib, "hexagon_nnExecutorchGetInputOutputBufferSizes");
        int (*run)(uint64_t, uint32_t, const NNTensorDef *, int, NNTensorDef *, int, int *) =
            symbol(lib, "hexagon_nnExecutorchRun");
        int (*run3)(uint64_t, uint32_t, const NNTensorDef *, int, int, NNTensorDef *, int,
                    void *, int, int *) = symbol(lib, "hexagon_nnExecutorchRun3");
        const char *profile_env = getenv("QD_QNN_PROFILE");
        int profile_level = profile_env ? atoi(profile_env) : 0;
        if (profile_level < 0 || profile_level > 1) {
            fprintf(stderr, "QD_QNN_PROFILE must be 0 (off) or 1 (vendor per-node linting)\n"); return 2;
        }
        /* NNETPerfInfo is 352 bytes on both ends of this RPC. */
        int profile_count = 512;
        size_t profile_bytes = (size_t)profile_count * 352;
        void *profile = profile_level ? calloc(1, profile_bytes) : NULL;
        if (profile_level && !profile) return 2;
        NNTensorDef in[4], out[2];
        tensor(&in[0], 1, 320, 256, 0x101); tensor(&in[1], 1, 320, 256, 0x101);
        tensor(&in[2], 2, 320, 256, 0x100); tensor(&in[3], 2, 4, 4, 0x100);
        tensor(&out[0], 2, 320, 256, 0x101); tensor(&out[1], 2, 320, 256, 0x100);
        uint32_t ib[4] = {0}, ob[2] = {0};
        rc = sizes(handle, model, ib, 4, ob, 2);
        printf("sizes rc=0x%x inputs=%u,%u,%u,%u outputs=%u,%u\n", rc, ib[0], ib[1], ib[2], ib[3], ob[0], ob[1]);
        for (int i = 0; i < 4; i++) if (rc || ib[i] != (uint32_t)in[i].bytes) return 2;
        for (int i = 0; i < 2; i++) if (ob[i] != (uint32_t)out[i].bytes) return 2;
        const char *names[] = {"x", "y", "hidden", "t"};
        for (int i = 0; i < 4; i++) tensor_file(&in[i], argv[2], names[i], 0);
        int repeat = argc == 5 ? atoi(argv[4]) : 1;
        if (repeat < 1 || repeat > 1000) return 2;
        for (int i = 0; i < repeat; i++) {
            for (int j = 0; j < 2; j++) memset(out[j].data, 0xa5, out[j].bytes);
            status = -1; start = seconds();
            if (profile_level) {
                memset(profile, 0, profile_bytes);
                rc = run3(handle, model, in, 4, profile_level, out, 2, profile, profile_count, &status);
            } else rc = run(handle, model, in, 4, out, 2, &status);
            double ms = (seconds() - start) * 1000;
            printf("run index=%d rc=0x%x status=%d ms=%.6f rank=%d,%d dtype=0x%x,0x%x fnv64=%016llx,%016llx\n",
                   i, rc, status, ms, out[0].rank, out[1].rank, out[0].dtype, out[1].dtype,
                   (unsigned long long)checksum(out[0].data, out[0].bytes),
                   (unsigned long long)checksum(out[1].data, out[1].bytes));
            if (rc || status) { failed = 1; dsp_log(lib, handle); break; }
        }
        if (!failed) { tensor_file(&out[0], argv[3], "u8", 1); tensor_file(&out[1], argv[3], "f32", 1); }
        if (profile) profile_file(argv[3], profile, profile_bytes);
        free(profile);
        for (int i = 0; i < 4; i++) free(in[i].data);
        for (int i = 0; i < 2; i++) free(out[i].data);
    }
    if (created) printf("destroy rc=0x%x\n", destroy(handle, model));
    if (getenv("QD_DEBUG")) dsp_log(lib, handle);
    printf("close rc=0x%x\n", close_rpc(handle));
    free(data); dlclose(lib);
    return failed ? 1 : 0;
}
