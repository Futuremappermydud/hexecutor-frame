#include "detail.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#ifdef __linux__
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sched.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace hexecutor::detail {
using Clock = std::chrono::steady_clock;
[[maybe_unused]] static void fence() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
struct Transport::Impl {
    Artifacts artifacts;
    Bytes code, payload, saved_code, saved_payload, saved_work;
    std::vector<unsigned> workers{0, 1, 2};
    uint32_t sequence = 0;
    uint8_t *memory = nullptr;
    int lock = -1, socket = -1;
    std::vector<int> descriptors;
    bool installed = false, registered = false, active = false, persistent = false, closed = false;
    unsigned calls = 0;
    std::string retained;
#ifdef __linux__
    pid_t eye = 0;
    bool eye_stopped = false;
    std::vector<std::string> eye_command;
    std::filesystem::path eye_cwd;
    cpu_set_t eye_affinity{};
#endif
    uint8_t *arena() const { return memory + arena_offset; }
    uint32_t load(size_t offset) const {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        return *reinterpret_cast<volatile uint32_t *>(memory + offset);
    }
    void store(size_t offset, uint32_t value) {
        *reinterpret_cast<volatile uint32_t *>(memory + offset) = value;
    }
    bool idle() const { return load(ctrl) == 1 && load(ctrl + 4) == 0 && load(ctrl + 8) == 1; }
    uint32_t marker(unsigned worker, unsigned field) const {
        return *reinterpret_cast<volatile uint32_t *>(arena() + 128 + worker * 16 + field * 4);
    }
    bool done() const {
        for (auto tid : workers)
            if (marker(tid, 0) != sequence)
                return false;
        return true;
    }
    bool exited_all() const {
        for (auto tid : workers)
            if (marker(tid, 3) != exited)
                return false;
        return true;
    }
    bool errors() const {
        if (!memory || workers.size() != 4)
            return false;
        for (auto tid : workers)
            if (marker(tid, 0) == sequence && marker(tid, 1))
                return true;
        return false;
    }
    bool any_worker_errors() const {
        if (!memory)
            return false;
        for (auto tid : workers)
            if (marker(tid, 1))
                return true;
        return false;
    }
    void retain(const std::string &why) {
        if (retained.empty())
            retained = why;
    }
    template <class F> void wait(F predicate, double timeout = -1, bool closing = false) {
#ifndef __linux__
        (void)closing;
#endif
        const auto deadline = Clock::now() + std::chrono::duration<double>(
                                                 timeout < 0 ? artifacts.timeout_seconds : timeout);
        while (!predicate()) {
#ifdef __linux__
            if (installed && registered && !closing && load(work + 24) != code_base)
                throw std::runtime_error("DSP callback disappeared after reset/suspend");
            if (socket >= 0) {
                pollfd p{socket, POLLIN, 0};
                while (poll(&p, 1, 0) > 0) {
                    char buffer[4096];
                    auto n = recv(socket, buffer, sizeof(buffer), MSG_DONTWAIT);
                    require(n != 0, "dsp_service disconnected");
                    if (n < 0) {
                        require(errno == EAGAIN || errno == EINTR, "dsp_service receive failed");
                        break;
                    }
                    p.revents = 0;
                }
            }
#endif
            require(Clock::now() < deadline,
                    "CDSP request timed out; workers may still own memory");
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
#ifdef __linux__
    static char state(pid_t pid) {
        try {
            auto b = read_file("/proc/" + std::to_string(pid) + "/stat");
            std::string s(b.begin(), b.end());
            auto at = s.rfind(") ");
            return at == std::string::npos ? '\0' : s.at(at + 2);
        } catch (const std::exception &) {
            return '\0';
        }
    }
    static std::vector<pid_t> eyes() {
        std::vector<pid_t> pids;
        for (const auto &entry : std::filesystem::directory_iterator("/proc")) {
            auto name = entry.path().filename().string();
            if (name.empty() ||
                !std::all_of(name.begin(), name.end(), [](char c) { return c >= '0' && c <= '9'; }))
                continue;
            try {
                auto b = read_file(entry.path() / "comm");
                std::string comm(b.begin(), b.end());
                if (comm == "eyetracking\n") {
                    auto p = pid_t(std::stol(name));
                    if (state(p) != 'Z' && state(p) != '\0')
                        pids.push_back(p);
                }
            } catch (const std::exception &) {
            }
        }
        return pids;
    }
    bool eye_paused() const { return !eye || state(eye) == 'T'; }
    static void send_all(int fd, const void *data, size_t size) {
        auto *p = static_cast<const uint8_t *>(data);
        while (size) {
            auto n = send(fd, p, size, MSG_NOSIGNAL);
            if (n < 0 && errno == EINTR)
                continue;
            require(n > 0, "dsp_service send failed");
            p += n;
            size -= size_t(n);
        }
    }
    void restore_eye() {
        if (!eye_stopped)
            return;
        if (state(eye) != '\0' && state(eye) != 'Z') {
            auto existing = eyes();
            require(std::find(existing.begin(), existing.end(), eye) != existing.end(),
                    "paused eye PID no longer identifies the saved client");
            require(kill(eye, SIGTERM) == 0 && kill(eye, SIGCONT) == 0,
                    "cannot stop saved eye client for re-upload");
            wait(
                [&] {
                    auto s = state(eye);
                    return s == '\0' || s == 'Z';
                },
                5, true);
        }
        auto deadline = Clock::now() + std::chrono::milliseconds(350);
        auto live = eyes();
        while (live.empty() && Clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            live = eyes();
        }
        require(live.size() <= 1, "multiple replacement eye clients found");
        if (live.empty()) {
            const char *home = std::getenv("HOME");
            require(home, "HOME is required for the eye restoration log");
            auto logpath = std::filesystem::path(home) / "frame-dsp/depth-model-eye-restored.log";
            std::filesystem::create_directories(logpath.parent_path());
            int log = ::open(logpath.c_str(), O_CREAT | O_WRONLY | O_APPEND | O_CLOEXEC, 0600);
            require(log >= 0, "cannot create eye restoration log");
            int pipefd[2];
            require(pipe(pipefd) == 0, "eye restore pipe failed");
            std::vector<char *> argv;
            for (auto &word : eye_command)
                argv.push_back(word.data());
            argv.push_back(nullptr);
            pid_t child = fork();
            if (child < 0) {
                ::close(log);
                ::close(pipefd[0]);
                ::close(pipefd[1]);
                throw std::runtime_error("eye restore fork failed");
            }
            if (child == 0) {
                ::close(pipefd[0]);
                pid_t grandchild = fork();
                if (grandchild < 0)
                    _exit(126);
                if (grandchild) {
                    (void)!write(pipefd[1], &grandchild, sizeof(grandchild));
                    _exit(0);
                }
                ::close(pipefd[1]);
                setsid();
                if (chdir(eye_cwd.c_str()) ||
                    sched_setaffinity(0, sizeof(eye_affinity), &eye_affinity))
                    _exit(126);
                if (dup2(log, STDOUT_FILENO) < 0 || dup2(log, STDERR_FILENO) < 0)
                    _exit(126);
                // Do not leak the worker's inherited application socket into eye tracking.
                long limit = sysconf(_SC_OPEN_MAX);
                for (int fd = 3; fd < limit; fd++)
                    ::close(fd);
                execv(argv[0], argv.data());
                _exit(127);
            }
            ::close(log);
            ::close(pipefd[1]);
            pid_t restored = 0;
            auto bytes = read(pipefd[0], &restored, sizeof(restored));
            ::close(pipefd[0]);
            int status;
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
            }
            require(bytes == sizeof(restored) && restored > 0, "eye restore child failed");
            std::this_thread::sleep_for(std::chrono::seconds(1));
            deadline = Clock::now() + std::chrono::seconds(2);
            while (eyes().size() != 1 && Clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            require(eyes().size() == 1, "eye client did not stay running; inspect restoration log");
        }
        eye_stopped = false;
    }
    void history(const std::string &digest) {
        const char *home = std::getenv("HOME");
        require(home, "HOME required for code-page history");
        auto path = std::filesystem::path(home) / ".cache/depth-model/code-pages.json";
        std::string text = "{}";
        if (std::filesystem::exists(path)) {
            auto b = read_file(path);
            text.assign(b.begin(), b.end());
        }
        std::string key = artifacts.firmware_sha256 + ":0x87606000";
        auto at = text.find('"' + key + '"');
        if (at != std::string::npos) {
            auto colon = text.find(':', at + key.size() + 2), quote = text.find('"', colon + 1);
            require(colon != std::string::npos && quote != std::string::npos &&
                        text.substr(quote + 1, 64) == digest,
                    "loader page previously ran a different image");
            return;
        }
        auto end = text.rfind('}');
        require(end != std::string::npos, "invalid code-page history");
        bool empty = text.find('"') == std::string::npos;
        text.insert(end,
                    (empty ? "" : ",") + std::string("\n  \"") + key + "\": \"" + digest + "\"\n");
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path);
        require(bool(out << text), "cannot save loader history");
    }
    void open() {
        verify_file(artifacts.firmware, artifacts.firmware_sha256);
        lock = ::open("/tmp/depth-model-cdsp.lock", O_CREAT | O_WRONLY | O_CLOEXEC, 0600);
        require(lock >= 0 && flock(lock, LOCK_EX | LOCK_NB) == 0,
                "another CDSP session holds the lock");
        socket = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        require(socket >= 0, "cannot create CDSP socket");
        timeval timeout{2, 0};
        setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path + 1, "dsp_service", 11);
        require(connect(socket, reinterpret_cast<sockaddr *>(&address),
                        offsetof(sockaddr_un, sun_path) + 12) == 0,
                "cannot connect to dsp_service");
        uint32_t request = 4;
        send_all(socket, &request, 4);
        char response[64];
        alignas(cmsghdr) char control[CMSG_SPACE(16)]{};
        iovec iov{response, sizeof(response)};
        msghdr msg{};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        auto received = recvmsg(socket, &msg, MSG_CMSG_CLOEXEC);
        if (received >= 0)
            for (auto *c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c))
                if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
                    auto count = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
                    auto *fds = reinterpret_cast<int *>(CMSG_DATA(c));
                    for (size_t i = 0; i < count; i++)
                        descriptors.push_back(fds[i]);
                }
        require(received == 4 && std::memcmp(response, "ADSP", 4) == 0 && descriptors.size() == 2 &&
                    !(msg.msg_flags & MSG_CTRUNC),
                "unexpected dsp_service handshake");
        struct stat info{};
        require(fstat(descriptors[1], &info) == 0 && info.st_size == dma_bytes,
                "unexpected shared DMA size");
        void *p = mmap(nullptr, dma_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptors[1],
                       0x80000000);
        require(p != MAP_FAILED, "cannot map CDSP DMA buffer");
        memory = static_cast<uint8_t *>(p);
        require(get<uint64_t>(memory, 0xa90000) == 0x1614df651d03f542 &&
                    get<uint64_t>(memory, 0xa90008) == 9,
                "CDSP protocol header mismatch");
        require(load(work + 24) == 0, "DSP callback already in use");
        wait([&] { return idle(); }, 5);
        auto live = eyes();
        require(live.size() <= 1, "multiple eye clients are running");
        if (!live.empty()) {
            eye = live.front();
            auto dir = std::filesystem::path("/proc") / std::to_string(eye);
            auto raw = read_file(dir / "cmdline");
            size_t start = 0;
            for (size_t i = 0; i < raw.size(); i++)
                if (!raw[i]) {
                    if (i > start)
                        eye_command.emplace_back(reinterpret_cast<char *>(raw.data() + start),
                                                 i - start);
                    start = i + 1;
                }
            require(!eye_command.empty(), "missing eye command line");
            eye_command[0] = std::filesystem::read_symlink(dir / "exe").string();
            eye_cwd = std::filesystem::read_symlink(dir / "cwd");
            require(sched_getaffinity(eye, sizeof(eye_affinity), &eye_affinity) == 0,
                    "cannot preserve eye CPU affinity");
            require(kill(eye, SIGSTOP) == 0, "cannot pause eye tracking");
            eye_stopped = true;
            wait([&] { return state(eye) == 'T'; }, 5);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        wait([&] { return idle(); }, 5);
        require(load(work + 24) == 0, "DSP callback changed during setup");
        saved_code.assign(memory + code_offset, memory + code_offset + code_bytes);
        saved_payload.assign(memory + payload_offset, memory + payload_offset + payload_bytes);
        auto zero = [](const uint8_t *b, size_t n) {
            return std::all_of(b, b + n, [](uint8_t x) { return x == 0; });
        };
        require(zero(saved_code.data(), saved_code.size()) || saved_code == code,
                "callback code page is occupied");
        require(zero(saved_payload.data(), saved_payload.size()), "payload region is occupied");
        require(zero(arena(), arena_bytes), "dedicated arena is occupied");
        history(sha256(read_file(artifacts.loader)));
        saved_work.assign(memory + work, memory + work + 32);
        std::copy(payload.begin(), payload.end(), memory + payload_offset);
        std::copy(code.begin(), code.end(), memory + code_offset);
        installed = true;
    }
    void register_callback() {
        require(installed && memory, "callback pages not installed");
        require(idle(), "DSP not idle before callback dispatch");
        require(eye_paused(), "paused eye client changed state");
        require(load(work + 24) == (registered ? code_base : 0), "callback registration changed");
        require(std::equal(code.begin(), code.end(), memory + code_offset),
                "loader page changed before registration");
        if (!registered) {
            store(work + 24, code_base);
            fence();
            registered = true;
        }
    }
    void wake() {
        sigset_t blocked, previous;
        sigemptyset(&blocked);
        sigaddset(&blocked, SIGTERM);
        sigaddset(&blocked, SIGINT);
        require(sigprocmask(SIG_BLOCK, &blocked, &previous) == 0,
                "cannot guard callback publication");
        try {
            fence();
            register_callback();
            store(ctrl + 4, 1);
            store(ctrl + 8, 4);
            active = true;
            send_all(socket, "work", 4);
        } catch (...) {
            sigprocmask(SIG_SETMASK, &previous, nullptr);
            throw;
        }
        sigprocmask(SIG_SETMASK, &previous, nullptr);
    }
    void release() {
        store(ctrl + 8, 1);
        store(ctrl + 4, 0);
        wait([&] { return load(ctrl) == 1; }, -1, true);
        active = persistent = false;
    }
#endif
    Impl(const Artifacts &a, const std::filesystem::path &image, bool fourth) : artifacts(a) {
        code = load_elf(read_file(a.loader), code_base, code_bytes);
        payload = load_elf(read_file(image), payload_base, payload_bytes);
        if (fourth)
            workers.push_back(5);
    }
    void submit(const std::function<void(uint32_t)> &prepare) {
#ifdef __linux__
        require(eye_paused(), "paused eye client changed state during inference");
        if (persistent && exited_all()) {
            wait([&] { return load(ctrl) == 1 || load(ctrl) == 3; });
            release();
        }
        require(!persistent || registered, "persistent workers without registered callback");
        require(persistent || (!active && load(ctrl) == 1), "previous DSP job is active");
        require(sequence != UINT32_MAX, "DSP command sequence exhausted");
        prepare(++sequence);
        if (!persistent) {
            persistent = true;
            wake();
        }
        wait([&] {
            if (done())
                return true;
            require(!exited_all(), "DSP workers exited before completing the command");
            return false;
        });
        fence();
        ++calls;
#else
        (void)prepare;
        throw std::runtime_error("Frame transport requires Linux");
#endif
    }
    void stop() {
#ifdef __linux__
        if (!persistent)
            return;
        if (errors()) {
            retain("four-worker bootstrap restoration failed");
            throw std::runtime_error(retained);
        }
        if (!exited_all()) {
            require(sequence != UINT32_MAX, "DSP command sequence exhausted during exit");
            ++sequence;
            std::memset(arena() + 128, 0, 48);
            uint32_t fields[20] = {0x32564451, 6, 0, 2};
            fields[9] = ~sequence;
            fields[15] = arena_bytes;
            fields[16] = 20000;
            std::memcpy(arena(), fields, sizeof(fields));
            fence();
            put<uint32_t>(arena(), 8, sequence);
            fence();
            wait([&] { return exited_all(); }, -1, true);
        }
        if (errors()) {
            retain("four-worker bootstrap exit failed");
            throw std::runtime_error(retained);
        }
        wait([&] { return load(ctrl) == 1 || load(ctrl) == 3; }, -1, true);
        release();
#endif
    }
    void close() {
        if (closed)
            return;
#ifdef __linux__
        if (errors())
            retain("four-worker status reports restoration error");
        if (memory && !active && !idle())
            retain("unexpected DSP control state without an owned job");
        if (memory && persistent && retained.empty() && registered &&
            load(work + 24) == code_base) {
            try {
                stop();
            } catch (const std::exception &e) {
                retain(e.what());
            }
        }
        if (memory && registered && load(work + 24) == code_base &&
            (get<uint32_t>(arena(), 256) == 0x41584931 ||
             get<uint32_t>(arena(), 512) == 0x484d5831))
            retain("DSP clock/power restoration incomplete");
        if (memory && !active && !idle())
            retain("DSP is not idle after owned worker shutdown");
        bool reset = false;
        if (memory && registered && load(work + 24) == 0 && retained.empty()) {
            bool clean = std::all_of(memory + code_offset, memory + code_offset + code_bytes,
                                     [](uint8_t x) { return x == 0; });
            reset = get<uint64_t>(memory, 0xa90000) == 0x1614df651d03f542 &&
                    get<uint64_t>(memory, 0xa90008) == 9 && load(ctrl) == 1 &&
                    (clean || std::equal(code.begin(), code.end(), memory + code_offset));
            if (reset) {
                active = registered = installed = false;
            } else
                retain("callback disappeared without a verified reset");
        }
        if (memory && !reset && installed && load(work + 24) != (registered ? code_base : 0))
            retain("unexpected callback pointer during cleanup");
        if (memory && active && retained.empty()) {
            try {
                wait([&] { return (load(ctrl) == 1 || load(ctrl) == 3) && done(); }, -1, true);
                release();
            } catch (const std::exception &e) {
                retain(e.what());
            }
        }
        bool safe = !active && retained.empty();
        if (memory && installed && safe && idle()) {
            if (registered)
                store(work + 24, 0);
            std::copy(saved_work.begin(), saved_work.end(), memory + work);
            std::copy(saved_code.begin(), saved_code.end(), memory + code_offset);
            std::copy(saved_payload.begin(), saved_payload.end(), memory + payload_offset);
            std::memset(arena(), 0, arena_bytes);
            fence();
            registered = installed = false;
        }
        if (memory) {
            munmap(memory, dma_bytes);
            memory = nullptr;
        }
        if (socket >= 0) {
            ::close(socket);
            socket = -1;
        }
        for (int fd : descriptors)
            ::close(fd);
        descriptors.clear();
        try {
            if (safe)
                restore_eye();
        } catch (...) {
            if (lock >= 0) {
                ::close(lock);
                lock = -1;
            }
            closed = true;
            throw;
        }
        if (lock >= 0) {
            ::close(lock);
            lock = -1;
        }
        closed = true;
        require(retained.empty(),
                "DSP session retained for explicit recovery; eye tracking remains paused: " +
                    retained);
#else
        closed = true;
#endif
    }
    ~Impl() {
        try {
            close();
        } catch (const std::exception &e) {
            std::cerr << "[Hexecutor] " << e.what() << '\n';
        }
    }
};
Transport::Transport(const Artifacts &a, const std::filesystem::path &payload, bool fourth)
    : impl_(std::make_unique<Impl>(a, payload, fourth)) {
#ifdef __linux__
    impl_->open();
#else
    throw std::runtime_error("Frame transport requires Linux");
#endif
}
Transport::~Transport() = default;
uint8_t *Transport::arena() const { return impl_->arena(); }
uint32_t Transport::sequence() const { return impl_->sequence; }
void Transport::submit(const std::function<void(uint32_t)> &prepare) { impl_->submit(prepare); }
void Transport::invoke(const std::function<void(uint32_t)> &prepare,
                       const std::function<void()> &collect) {
#ifdef __linux__
    require(!impl_->active && impl_->idle() && impl_->sequence != UINT32_MAX,
            "previous probe job is active");
    prepare(++impl_->sequence);
    impl_->wake();
    impl_->wait(
        [&] { return (impl_->load(ctrl) == 1 || impl_->load(ctrl) == 3) && impl_->done(); });
    fence();
    collect();
    impl_->release();
    ++impl_->calls;
#else
    (void)prepare;
    (void)collect;
    throw std::runtime_error("Frame transport requires Linux");
#endif
}
void Transport::stop() { impl_->stop(); }
void Transport::retain(const std::string &reason) { impl_->retain(reason); }
void Transport::close() { impl_->close(); }
bool Transport::has_run() const { return impl_->active || impl_->persistent || impl_->calls != 0; }
bool Transport::worker_errors() const { return impl_->any_worker_errors(); }

void verify_chunk_map(const Artifacts &artifacts) {
#ifdef __linux__
    auto text = [](const char *p) {
        auto b = read_file(p);
        return std::string(b.begin(), b.end());
    };
    require(std::stoul(text("/sys/module/cdsp_mem/parameters/size_mb")) == 128,
            "requires a 128 MB memory lease");
    require(std::stoull(text("/sys/module/cdsp_mem/parameters/iova")) == 0xa0000000,
            "wrong CDSP IOVA");
    auto probe_options = artifacts;
    probe_options.timeout_seconds = 30;
    Transport probe(probe_options, artifacts.chunk_probe);
    int fd = ::open("/dev/cdsp_mem", O_RDWR | O_CLOEXEC);
    require(fd >= 0, "cannot open memory-map probe window");
    void *p = mmap(nullptr, 128u << 20, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    require(p != MAP_FAILED, "cannot map probe window");
    try {
        uint32_t nonce = std::random_device{}() & ~0x0fe00000u;
        for (unsigned i = 0; i < 64; i++)
            put<uint32_t>(p, size_t(i) << 21, (i << 21) ^ nonce);
        auto *arena = probe.arena();
        probe.invoke(
            [&](uint32_t seq) {
                std::memset(arena, 0, 4096);
                put<uint32_t>(arena, 16, nonce);
                put<uint32_t>(arena, 8, seq);
                fence();
            },
            [&] {
                require(get<uint32_t>(arena, 2048) == 0x31544d51 && !get<uint32_t>(arena, 2052) &&
                            !get<uint32_t>(arena, 2072),
                        "chunk-map probe failed");
                for (unsigned i = 0; i < 64; i++)
                    require(
                        (get<uint32_t>(arena, 2048 + (100 + i) * 4) ^ nonce) == (i << 21),
                        "CDSP resolves private memory to a wrong chunk; reboot before inference");
            });
    } catch (...) {
        munmap(p, 128u << 20);
        throw;
    }
    munmap(p, 128u << 20);
    probe.close();
#else
    (void)artifacts;
    throw std::runtime_error("Frame memory verification requires Linux");
#endif
}
} // namespace hexecutor::detail
