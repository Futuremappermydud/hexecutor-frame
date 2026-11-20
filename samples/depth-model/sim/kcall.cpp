// Call one exported kernel function of QAIRT's x86_64 HTP simulator library
// directly, for calibrating reference arithmetic (research tool).
//
//   kcall LIB SPEC
//
// SPEC lines (whitespace separated, '#' comments):
//   buf NAME BYTES [FILE]        2 KB-aligned buffer, optionally loaded from FILE
//   ptrs NAME BUF STRIDE COUNT   array of COUNT pointers BUF + i*STRIDE
//   call FUNC A1..A6 | S1..S6 | X0..X3
//                                integer args, stack args and xmm bit patterns;
//                                an argument may be a number (C syntax) or
//                                @NAME for a buffer / pointer array address
//   callrel OFFSET ...           same, for an unexported function at OFFSET
//                                from the library's load base (helpers)
//   save BUF FILE                write a buffer after the call
//   repeat N                     run the call N times (for timing; default 1)
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct Buf { uint8_t *p; size_t n; };
static std::map<std::string, Buf> bufs;

// Direct calls run scalar simulator implementations on this process's main
// thread. The executor's worker-thread guard does not apply to this harness.
// Build with -rdynamic so libQnnHtp's PLT resolves this guard here.
extern "C" void _Z9check_hvxv() {}

static uint64_t value(const std::string &s) {
    if (!s.empty() && s[0] == '@') {
        auto it = bufs.find(s.substr(1));
        if (it == bufs.end()) { fprintf(stderr, "unknown buffer %s\n", s.c_str()); exit(2); }
        return (uint64_t)it->second.p;
    }
    return strtoull(s.c_str(), nullptr, 0);
}

typedef uint64_t (*Fn)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                       double, double, double, double,
                       uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: kcall LIB SPEC\n"); return 2; }
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    std::ifstream spec(argv[2]);
    std::string line;
    int repeat = 1;
    while (std::getline(spec, line)) {
        std::istringstream ss(line);
        std::string op;
        if (!(ss >> op) || op[0] == '#') continue;
        if (op == "buf") {
            std::string name, file; size_t n;
            ss >> name >> n >> file;
            uint8_t *p = (uint8_t *)aligned_alloc(2048, (n + 2047) / 2048 * 2048);
            memset(p, 0, n);
            if (!file.empty()) {
                FILE *f = fopen(file.c_str(), "rb");
                if (!f || fread(p, 1, n, f) != n) { fprintf(stderr, "short read %s\n", file.c_str()); return 1; }
                fclose(f);
            }
            bufs[name] = {p, n};
        } else if (op == "ptrs") {
            std::string name, base; size_t stride, count;
            ss >> name >> base >> stride >> count;
            uint64_t b = value("@" + base);
            uint64_t *arr = (uint64_t *)aligned_alloc(64, (count * 8 + 63) / 64 * 64);
            for (size_t i = 0; i < count; ++i) arr[i] = b + i * stride;
            bufs[name] = {(uint8_t *)arr, count * 8};
        } else if (op == "repeat") {
            ss >> repeat;
        } else if (op == "call" || op == "callrel") {
            std::string fname, tok;
            ss >> fname;
            std::vector<uint64_t> ia, sa; std::vector<uint64_t> xa;
            int section = 0;
            while (ss >> tok) {
                if (tok == "|") { ++section; continue; }
                (section == 0 ? ia : section == 1 ? sa : xa).push_back(value(tok));
            }
            ia.resize(6); sa.resize(6); xa.resize(4);
            double x[4];
            for (int i = 0; i < 4; ++i) memcpy(&x[i], &xa[i], 8);
            Fn fn;
            if (op == "callrel") {
                Dl_info info;
                dladdr(dlsym(lib, "elementwise_mul_crouton_u16"), &info);
                fn = (Fn)((uint8_t *)info.dli_fbase + strtoull(fname.c_str(), nullptr, 0));
            } else {
                fn = (Fn)dlsym(lib, fname.c_str());
            }
            if (!fn) { fprintf(stderr, "no symbol %s\n", fname.c_str()); return 1; }
            uint64_t rc = 0;
            for (int r = 0; r < repeat; ++r)
                rc = fn(ia[0], ia[1], ia[2], ia[3], ia[4], ia[5], x[0], x[1], x[2], x[3],
                        sa[0], sa[1], sa[2], sa[3], sa[4], sa[5]);
            printf("rc %llu\n", (unsigned long long)rc);
        } else if (op == "save") {
            std::string name, file;
            ss >> name >> file;
            Buf b = bufs.at(name);
            FILE *f = fopen(file.c_str(), "wb");
            fwrite(b.p, 1, b.n, f);
            fclose(f);
        }
    }
    return 0;
}
