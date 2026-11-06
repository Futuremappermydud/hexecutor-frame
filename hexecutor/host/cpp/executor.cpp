#include "detail.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>
#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
extern "C" void nprg2_sha256(const void *, uint32_t, uint8_t[32]);

namespace hexecutor {
using namespace detail;
size_t TensorSpec::bytes() const {
    require(!name.empty(), "tensor name is empty");
    size_t size = 0;
    switch (type) {
    case ScalarType::u8:
    case ScalarType::i8:
        size = 1;
        break;
    case ScalarType::u16:
    case ScalarType::i16:
        size = 2;
        break;
    case ScalarType::i32:
    case ScalarType::f32:
        size = 4;
        break;
    default:
        throw std::invalid_argument("unsupported tensor scalar type");
    }
    for (auto dimension : shape) {
        require(dimension && dimension <= std::numeric_limits<size_t>::max() / size,
                "tensor size overflow");
        size *= dimension;
    }
    return size;
}
void Program::validate() const {
    require(inputs.size() == 4 && outputs.size() == 2,
            "qdrx-v1 backend requires four inputs and two outputs");
    for (const auto *group : {&inputs, &outputs}) {
        std::vector<std::string> names;
        for (const auto &spec : *group) {
            spec.bytes();
            require(std::find(names.begin(), names.end(), spec.name) == names.end(),
                    "duplicate tensor name");
            names.push_back(spec.name);
        }
    }
}
Bytes read_file(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "cannot open " + path.string());
    Bytes data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    require(!file.bad() && data.size() <= UINT32_MAX,
            "cannot read supported file range: " + path.string());
    return data;
}
std::string sha256(const Bytes &data) {
    require(data.size() <= UINT32_MAX, "hash input exceeds supported range");
    std::array<uint8_t, 32> digest{};
    nprg2_sha256(data.data(), uint32_t(data.size()), digest.data());
    std::ostringstream out;
    for (auto byte : digest)
        out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return out.str();
}
void verify_file(const std::filesystem::path &path, const std::string &expected) {
    require(sha256(read_file(path)) == expected, "artifact hash mismatch: " + path.string());
}
Bytes load_elf(const Bytes &elf, uint32_t base, size_t capacity) {
    require(elf.size() >= 52 && std::memcmp(elf.data(), "\177ELF\1\1\1", 7) == 0 &&
                get<uint16_t>(elf.data(), 18) == 164,
            "expected a little-endian 32-bit Hexagon ELF");
    uint32_t offset = get<uint32_t>(elf.data(), 28);
    uint16_t count = get<uint16_t>(elf.data(), 44);
    require(get<uint32_t>(elf.data(), 24) == base && get<uint16_t>(elf.data(), 42) == 32,
            "ELF entry/program-header ABI mismatch");
    require(offset <= elf.size() && size_t(count) * 32 <= elf.size() - offset,
            "ELF program headers exceed file");
    Bytes code(capacity);
    unsigned loads = 0;
    for (unsigned i = 0; i < count; i++) {
        const auto *p = elf.data() + offset + i * 32;
        if (get<uint32_t>(p, 0) != 1)
            continue;
        uint32_t at = get<uint32_t>(p, 4), va = get<uint32_t>(p, 8), files = get<uint32_t>(p, 16),
                 memory = get<uint32_t>(p, 20);
        require(va >= base && uint64_t(va) - base <= capacity && files <= memory &&
                    memory <= capacity - (va - base) && at <= elf.size() &&
                    files <= elf.size() - at,
                "ELF load segment exceeds owned region");
        std::copy_n(elf.data() + at, files, code.data() + (va - base));
        ++loads;
    }
    require(loads != 0, "ELF has no loadable segment");
    return code;
}

struct Session::Impl {
    Program program;
    Artifacts artifacts;
    Bytes image, constants, engine, prepared;
    std::unique_ptr<Transport> transport;
    uint8_t *memory = nullptr;
    std::array<uint32_t, 7> sizes{}, pools{};
    std::array<uint32_t, 4> inputs{}, input_bytes{};
    std::array<uint32_t, 2> outputs{}, output_bytes{};
    std::vector<std::pair<uint32_t, uint32_t>> guards;
    uint32_t cursor = 1 << 20, commands = 0;
    bool preserving = false, l2 = false, fourth = false, closed = false, initialized = false;
    static bool feature(const Bytes &b, const std::string &s) {
        return std::search(b.begin(), b.end(), s.begin(), s.end()) != b.end();
    }
    uint32_t allocate(size_t n) {
        require(n && n < 32u << 20, "invalid resident allocation");
        uint32_t off = (cursor + 255) & ~127u, end = (off + uint32_t(n) + 127) & ~127u;
        require(end <= ((32u << 20) - 128), "resident fixed data exceeds its region");
        guards.emplace_back(off - 128, 128);
        guards.emplace_back(end, 128);
        cursor = end + 128;
        return off;
    }
    void check_guards() {
        for (auto [at, n] : guards)
            for (uint32_t i = 0; i < n; i++)
                require(memory[at + i] == 0xa5, "resident allocation guard changed");
    }
    Impl(Program p, Artifacts a) : program(std::move(p)), artifacts(std::move(a)) {
        require(std::isfinite(artifacts.timeout_seconds) && artifacts.timeout_seconds > 0 &&
                    artifacts.timeout_seconds <= 3600,
                "invalid DSP timeout");
        program.validate();
        image = read_file(program.image);
        constants = read_file(program.constants);
        require(image.size() >= 256 && get<uint32_t>(image.data(), 0) == 0x58524451,
                "QDRX image required");
        Bytes expected(image.begin() + 64, image.begin() + 96);
        std::array<uint8_t, 32> digest{};
        nprg2_sha256(constants.data(), uint32_t(constants.size()), digest.data());
        require(std::equal(expected.begin(), expected.end(), digest.begin()),
                "constant pool does not match image");
        for (unsigned i = 0; i < 7; i++)
            sizes[i] = get<uint32_t>(image.data(), 96 + 4 * i);
        require(sizes[5] == constants.size(), "constant pool size mismatch");
        commands = get<uint32_t>(image.data(), 168);
        require(commands != 0, "program has no commands");
        auto boot = read_file(artifacts.bootstrap);
        auto raw_engine = read_file(artifacts.engine);
        preserving = feature(boot, std::string("QDBOOT01\1\0\0\0", 12));
        fourth = feature(boot, std::string("QDTHRD51\1\0\0\0", 12));
        l2 = feature(boot, "QDL2BOOT02");
        require(l2 == feature(raw_engine, "QDL2ENGN03"), "L2 engine/bootstrap features differ");
        require(preserving, "preserving bootstrap required");
        engine = load_elf(raw_engine, 0xa0000000, 1 << 20);
        if (!artifacts.prepared_program.empty())
            prepared = read_file(artifacts.prepared_program);
        require(!prepared.empty(), "prepared program artifact is required");
        for (size_t i = 0; i < 4; i++) {
            require(program.inputs[i].bytes() <= UINT32_MAX, "input too large");
            input_bytes[i] = uint32_t(program.inputs[i].bytes());
        }
        for (size_t i = 0; i < 2; i++) {
            require(program.outputs[i].bytes() <= UINT32_MAX, "output too large");
            output_bytes[i] = uint32_t(program.outputs[i].bytes());
        }
    }
    void open() {
#ifdef __linux__
        verify_chunk_map(artifacts);
        transport = std::make_unique<Transport>(artifacts, artifacts.bootstrap, fourth);
        require(std::stoul(std::string(read_text("/sys/module/cdsp_mem/parameters/size_mb"))) ==
                    128,
                "requires a 128 MB cdsp_mem allocation");
        require(std::stoull(read_text("/sys/module/cdsp_mem/parameters/iova")) == 0xa0000000,
                "wrong CDSP IOVA");
        int fd = ::open("/dev/cdsp_mem", O_RDWR | O_CLOEXEC);
        require(fd >= 0, "cannot open cdsp_mem");
        void *mapped = mmap(nullptr, 128u << 20, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        require(mapped != MAP_FAILED, "cannot map cdsp_mem");
        memory = static_cast<uint8_t *>(mapped);
        std::copy(engine.begin(), engine.end(), memory);
        auto image_off = allocate(image.size());
        std::copy(image.begin(), image.end(), memory + image_off);
        pools[5] = allocate(sizes[5]);
        std::copy(constants.begin(), constants.end(), memory + pools[5]);
        auto prepared_off = allocate(prepared.size());
        std::copy(prepared.begin(), prepared.end(), memory + prepared_off);
        require(cursor <= 16u << 20, "immutable data exceeds its region");
        cursor = 16u << 20;
        for (unsigned i = 0; i < 7; i++)
            if (i != 5 && sizes[i])
                pools[i] = allocate(sizes[i]);
        for (unsigned i = 0; i < 4; i++)
            inputs[i] = allocate(input_bytes[i]);
        for (unsigned i = 0; i < 2; i++)
            outputs[i] = allocate(output_bytes[i]);
        guards.insert(guards.end(), {{(80u << 20) - 4096, 4096},
                                     {88u << 20, 4096},
                                     {(64u << 20) - 4096, 4096},
                                     {(128u << 20) - 4096, 4096},
                                     {(96u << 20) - 128, 128},
                                     {104u << 20, 128}});
        if (l2) {
            for (uint32_t offset : {0u, 0x40000u, 0x58000u, 0x68000u, 0x78000u})
                guards.emplace_back((112u << 20) + offset, 128);
            guards.emplace_back((112u << 20) + (fourth ? 0xc3e00 : 0xa0000), 128);
        }
        for (auto [at, n] : guards)
            std::memset(memory + at, 0xa5, n);
        auto *arena = transport->arena();
        std::memset(arena, 0, 32768);
        std::vector<uint32_t> config = {0x31534451, 1, image_off, uint32_t(image.size())};
        config.insert(config.end(), pools.begin(), pools.end());
        config.insert(config.end(), sizes.begin(), sizes.end());
        config.insert(config.end(), inputs.begin(), inputs.end());
        config.insert(config.end(), input_bytes.begin(), input_bytes.end());
        config.insert(config.end(), outputs.begin(), outputs.end());
        config.insert(config.end(), output_bytes.begin(), output_bytes.end());
        config.insert(config.end(), {32u << 20, (32u << 20) - 4096, uint32_t(engine.size())});
        require(config.size() == 33, "resident config ABI mismatch");
        std::memcpy(arena + 4096, config.data(), config.size() * 4);
        uint32_t extension[12] = {0x31504e51, 1, prepared_off, uint32_t(prepared.size())};
        nprg2_sha256(prepared.data(), uint32_t(prepared.size()),
                     reinterpret_cast<uint8_t *>(extension + 4));
        std::memcpy(arena + 4096 + 132, extension, sizeof(extension));
        initialized = true;
#else
        throw std::runtime_error("Frame DSP execution requires Linux");
#endif
    }
    static std::string read_text(const char *path) {
        auto bytes = read_file(path);
        return {bytes.begin(), bytes.end()};
    }
    std::vector<Bytes> run(const std::vector<Buffer> &values) {
        require(!closed && transport && memory, "session is closed");
        require(values.size() == 4, "input count mismatch");
        for (unsigned i = 0; i < 4; i++)
            require(values[i].data && values[i].bytes == input_bytes[i],
                    "input buffer size mismatch");
        for (unsigned i = 0; i < 4; i++)
            std::memcpy(memory + inputs[i], values[i].data, values[i].bytes);
        auto *arena = transport->arena();
        transport->submit([&](uint32_t sequence) {
            std::memset(arena + 128, 0, 48);
            uint32_t packet[7] = {0x32564451, 6, sequence - 1, 3, 0, commands, 0};
            std::memcpy(arena, packet, sizeof(packet));
            put<uint32_t>(arena, 3840, 0);
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            put<uint32_t>(arena, 8, sequence);
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
        });
        require(get<uint32_t>(arena, 2048) == 0x31534451 && get<uint32_t>(arena, 2052) == 0,
                "resident DSP command failed");
        require(get<uint32_t>(arena, 2048 + 60) == 0x52444951 &&
                    get<uint32_t>(arena, 1024) == 0x31525051 &&
                    get<uint32_t>(arena, 1024 + 32) == commands,
                "DSP did not execute the complete prepared program");
        std::vector<Bytes> out;
        for (unsigned i = 0; i < 2; i++)
            out.emplace_back(memory + outputs[i], memory + outputs[i] + output_bytes[i]);
        check_guards();
        return out;
    }
    void close() {
        if (closed)
            return;
        try {
            if (transport) {
                if (transport->worker_errors()) {
                    transport->retain("bootstrap/worker failure");
                    throw std::runtime_error("worker restoration failed");
                }
                bool ran = transport->has_run();
                try {
                    transport->stop();
                } catch (...) {
                    transport->retain("VTCM lease exit did not complete");
                    throw;
                }
                if (transport->worker_errors()) {
                    transport->retain("worker exit/stack verification failed");
                    throw std::runtime_error("worker exit/stack check failed");
                }
                auto *a = transport->arena();
                if (get<uint32_t>(a, 3072)) {
                    transport->retain("resident translations still active");
                    throw std::runtime_error("resident JTLB still active");
                }
                if (ran) {
                    if (get<uint32_t>(a, 3328) != 0x31545651 || get<uint32_t>(a, 3332) != 2 ||
                        get<uint32_t>(a, 3336)) {
                        transport->retain("VTCM restoration not verified");
                        throw std::runtime_error("VTCM lease restoration failed");
                    }
                    uint32_t hx = 0, hs = 0;
                    for (uint32_t row = 0; row < (8u << 20) / 128; row++) {
                        uint32_t sum = 0;
                        for (unsigned i = 0; i < 32; i++) {
                            auto word = get<uint32_t>(memory, (96u << 20) + row * 128 + i * 4);
                            hx ^= word;
                            sum += word;
                        }
                        hs += sum * (row + 1);
                    }
                    if (hx != get<uint32_t>(a, 3328 + 44) || hs != get<uint32_t>(a, 3328 + 48)) {
                        transport->retain("VTCM snapshot fingerprint mismatch");
                        throw std::runtime_error("VTCM fingerprint mismatch");
                    }
                }
                if (memory && initialized) {
                    check_guards();
                    require(std::equal(constants.begin(), constants.end(), memory + pools[5]),
                            "constant pool changed");
                }
            }
        } catch (...) {
            auto failure = std::current_exception();
            if (transport) {
                transport->retain("resident close verification failed");
                try {
                    transport->close();
                } catch (...) {
                }
            }
            release_memory();
            closed = true;
            std::rethrow_exception(failure);
        }
        try {
            if (transport)
                transport->close();
        } catch (...) {
            release_memory();
            closed = true;
            throw;
        }
        release_memory();
        closed = true;
    }
    void release_memory() {
#ifdef __linux__
        if (memory)
            munmap(memory, 128u << 20);
#endif
        memory = nullptr;
    }
    ~Impl() {
        try {
            close();
        } catch (const std::exception &e) {
            std::cerr << "[Hexecutor] " << e.what() << '\n';
        }
    }
};
Session::Session(Program p, Artifacts a)
    : impl_(std::make_unique<Impl>(std::move(p), std::move(a))) {
    impl_->open();
}
Session::~Session() = default;
std::vector<Bytes> Session::run(const std::vector<Buffer> &values) { return impl_->run(values); }
void Session::close() { impl_->close(); }
} // namespace hexecutor
