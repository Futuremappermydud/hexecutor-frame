// SPDX-License-Identifier: LGPL-3.0-or-later
#include "detail.hpp"
#include "hexecutor/cdsp_mem.h"
#include "hexecutor/workload_memory.hpp"
#include "tensor/graph.h"
#include "qnn16/program.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <utility>
#ifdef __linux__
#include <fcntl.h>
#include <sys/ioctl.h>
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
    if (backend == Backend::qdrx_v1)
        require(inputs.size() == 4 && outputs.size() == 2,
                "qdrx-v1 backend requires four inputs and two outputs");
    else {
        require((backend == Backend::tensor_f32 || backend == Backend::qnn16_f16) &&
                    !inputs.empty() && !outputs.empty() && inputs.size() <= HXFG_MAX_IO &&
                    outputs.size() <= HXFG_MAX_IO,
                "invalid tensor backend I/O count");
        for (const auto *group : {&inputs, &outputs})
            for (const auto &spec : *group)
                require(spec.type == ScalarType::f32 || spec.type == ScalarType::i32,
                        "tensor backend requires float32 or int32 I/O");
    }
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

size_t persistent_state_extent(const Bytes &image) {
    HxHeader h{};
    require(image.size() >= sizeof h, "HXFG image shorter than its header");
    std::memcpy(&h, image.data(), sizeof h);
    require(uint64_t(h.tensor_offset) + uint64_t(h.tensors) * sizeof(HxTensor) <= image.size(),
            "HXFG tensor table is outside the image");
    uint64_t end = 0;
    for (uint32_t i = 0; i < h.tensors; i++) {
        HxTensor t;
        std::memcpy(&t, image.data() + h.tensor_offset + size_t(i) * sizeof t, sizeof t);
        if (t.flags & HX_TENSOR_PERSISTENT)
            end = std::max<uint64_t>(end, uint64_t(t.offset) + hx_tensor_bytes(&t));
    }
    require(end <= (256u << 20), "persistent state region is too large");
    return size_t(end);
}

namespace {
/* Decode LINEAR_Q (one activation row) on the DSP is fastest with 32 rows interleaved: rewrite those weights in
 * place into the grouped layout (same size) and the op's format to 2/3 in this session's copies. The files and
 * their hashes are untouched (checked above); the session's own pool-change check compares against these copies. */
static void group_linear_q(HxGraph &graph, Bytes &image, Bytes &constants) {
    auto *ops = reinterpret_cast<HxOp *>(image.data() + graph.header->op_offset);
    const auto qualifies = [&](const HxOp &o) {
        const uint32_t k = o.p[3], n = o.p[4];
        const auto &out = graph.tensors[o.out];
        const size_t bytes = size_t(k / 32) * n * (o.p[0] ? 34 : 18);
        return o.p[0] <= 1 && !(k & 63u) && !(n & 31u) && k <= 4096 && !(o.p[1] & 127u) && out.h == 1 &&
               out.w == 1 && size_t(o.p[1]) + bytes <= constants.size();
    };
    /* Weights may be shared (tied): regroup a block only if every LINEAR_Q that reads it can take the result. */
    std::map<uint32_t, std::vector<HxOp *>> users;
    std::map<uint32_t, bool> ok;
    for (uint32_t i = 0; i < graph.header->ops; i++)
        if (ops[i].kind == HX_LINEAR_Q) {
            auto &o = ops[i];
            auto [it, fresh] = ok.emplace(o.p[1], true);
            const auto &first = users[o.p[1]];
            it->second = it->second && qualifies(o) &&
                         (first.empty() || (first[0]->p[0] == o.p[0] && first[0]->p[3] == o.p[3] &&
                                            first[0]->p[4] == o.p[4]));
            users[o.p[1]].push_back(&o);
        }
    std::vector<uint8_t> scratch;
    for (auto &[offset, list] : users) {
        if (!ok[offset])
            continue;
        const HxOp &o = *list[0];
        scratch.resize(size_t(o.p[3]) * 34);
        hx_lq_group_layout(constants.data() + offset, o.p[0], o.p[3], o.p[4], scratch.data());
        for (HxOp *u : list)
            u->p[0] += 2;
    }
}
HxGraph preflight_tensor(const Bytes &image, const Bytes &constants) {
    HxGraph graph{};
    uint32_t bad = HXFG_NONE;
    /* Persistent tensors are checked against the state region, not the arena: bind a region of the
     * right size first (preflight never touches it; a Session binds the real one afterwards). */
    if (const size_t state = persistent_state_extent(image))
        hx_graph_set_state(&graph, const_cast<uint8_t *>(image.data()), uint32_t(state));
    uint32_t error = hx_graph_open(&graph, image.data(), uint32_t(image.size()), constants.data(),
                                   uint32_t(constants.size()), nullptr, HXFG_MAX_ARENA_BYTES, &bad);
    require(error == HX_OK, "tensor preflight failed: error " + std::to_string(error) + " at op " +
                                std::to_string(bad));
    std::array<uint8_t, 32> digest{};
    nprg2_sha256(constants.data(), uint32_t(constants.size()), digest.data());
    require(std::equal(digest.begin(), digest.end(), graph.header->constant_sha256),
            "tensor constant pool hash mismatch");
    return graph;
}
} // namespace
static Q16Program preflight_qnn16(const Bytes &image, const Bytes &constants) {
    Q16Program program{};
    uint32_t bad = Q16P_NONE;
    uint32_t error = q16_program_open(&program, image.data(), uint32_t(image.size()),
                                      uint32_t(constants.size()), &bad);
    require(error == Q16_OK, "Q16P preflight failed: error " + std::to_string(error) + " at op " +
                                 std::to_string(bad));
    std::array<uint8_t, 32> digest{};
    nprg2_sha256(constants.data(), uint32_t(constants.size()), digest.data());
    require(std::equal(digest.begin(), digest.end(), program.header->constant_sha256),
            "Q16P constant pool hash mismatch");
    return program;
}
// The arena holds only the exchange area; scratch is engine-private (engine.c).
[[maybe_unused]] static uint32_t q16_arena_bytes(const Q16Header *h) {
    require(h->scratch_bytes <= 0x7ff000u, "Q16P scratch exceeds the engine region");
    return (h->exchange_bytes + 4095u) & ~4095u;
}
Program load_qnn16_program(const std::filesystem::path &image,
                           const std::filesystem::path &constants) {
    auto bytes = read_file(image), weights = read_file(constants);
    auto q = preflight_qnn16(bytes, weights);
    Program p;
    p.image = image;
    p.constants = constants;
    p.backend = Backend::qnn16_f16;
    for (uint32_t i = 0; i < q.header->inputs + q.header->outputs; i++) {
        const auto &d = q.io[i].dims;
        bool input = i < q.header->inputs;
        auto &group = input ? p.inputs : p.outputs;
        group.push_back({std::string(input ? "input" : "output") + std::to_string(group.size()),
                         ScalarType::f32,
                         {d.b, d.h, d.w, d.c}});
    }
    p.validate();
    return p;
}
Program load_tensor_program(const std::filesystem::path &image,
                            const std::filesystem::path &constants) {
    auto bytes = read_file(image), weights = read_file(constants);
    auto graph = preflight_tensor(bytes, weights);
    Program p;
    p.image = image;
    p.constants = constants;
    p.backend = Backend::tensor_f32;
    for (uint32_t i = 0; i < graph.header->inputs + graph.header->outputs; i++) {
        const auto &t = graph.tensors[graph.io[i]];
        bool input = i < graph.header->inputs;
        auto &group = input ? p.inputs : p.outputs;
        ScalarType st = ScalarType::f32;
        if (t.dtype == HX_DTYPE_INT32 || t.dtype == HX_DTYPE_BOOL)
            st = ScalarType::i32;
        group.push_back({std::string(input ? "input" : "output") + std::to_string(group.size()),
                         st,
                         {1, t.h, t.w, t.c}});
    }
    p.validate();
    return p;
}

struct Session::Impl {
    Program program;
    std::vector<Program> programs;
    std::vector<Bytes> tensor_images, tensor_constants;
    std::vector<HxGraph> graphs;
    std::vector<Q16Program> q16;
    uint32_t q16_spill_bytes = 0;
    HxSessionConfig tensor_config{};
    ExecutionProfile profile;
    Artifacts artifacts;
    Bytes image, constants, engine, prepared;
    std::unique_ptr<Transport> transport;
    std::array<std::chrono::microseconds, HXFG_MAX_GRAPHS> last_dsp_time{};
    /* A run is warm when the previous command ran the same graph in the same
     * wake: only then are its resident VTCM records still staged. */
    size_t last_graph = SIZE_MAX;
    uint32_t last_generation = 0;
    uint8_t *memory = nullptr;
    WorkloadLease lease;
    size_t mapped_bytes = 128u << 20;
    size_t engine_mapped_bytes = 128u << 20;
    int pin_fd = -1;
    std::array<uint32_t, 7> sizes{}, pools{};
    std::array<uint32_t, 4> inputs{}, input_bytes{};
    std::array<uint32_t, 2> outputs{}, output_bytes{};
    std::vector<std::pair<uint32_t, uint32_t>> guards;
    bool high_constants = false, low_q16_spill = false;
    uint32_t cursor = 1 << 20, allocation_limit = (32u << 20) - 128, commands = 0;
    bool preserving = false, l2 = false, fourth = false, closed = false, initialized = false;
    static bool feature(const Bytes &b, const std::string &s) {
        return std::search(b.begin(), b.end(), s.begin(), s.end()) != b.end();
    }
    uint32_t max_allocation = 32u << 20, guard_floor = 0;
    uint32_t allocate(size_t n) {
        require(n && n < max_allocation, "invalid resident allocation");
        uint32_t off = (cursor + 255) & ~127u, end = (off + uint32_t(n) + 127) & ~127u;
        require(end <= allocation_limit, "resident fixed data exceeds its region");
        guards.emplace_back(off - 128, 128);
        guards.emplace_back(end, 128);
        cursor = end + 128;
        return off;
    }
    void check_guards() {
        for (auto [at, n] : guards)
            require(std::all_of(memory + at, memory + at + n, [](uint8_t v) { return v == 0xa5; }),
                    "resident allocation guard changed");
    }
    Impl(std::vector<Program> ps, Artifacts a) : programs(std::move(ps)), artifacts(std::move(a)) {
        require(!programs.empty() && programs.size() <= HXFG_MAX_GRAPHS, "invalid program count");
        program = programs.front();
        require(std::isfinite(artifacts.timeout_seconds) && artifacts.timeout_seconds > 0 &&
                    artifacts.timeout_seconds <= 3600,
                "invalid DSP timeout");
        require(std::isfinite(artifacts.max_wake_seconds) && artifacts.max_wake_seconds > 0 &&
                    artifacts.max_wake_seconds <= 3600,
                "invalid DSP wake limit");
        program.validate();
        if (program.backend == Backend::qnn16_f16) {
            for (const auto &p : programs) {
                p.validate();
                require(p.backend == Backend::qnn16_f16, "cannot mix executor backends");
                tensor_images.push_back(read_file(p.image));
                tensor_constants.push_back(read_file(p.constants));
                q16.push_back(preflight_qnn16(tensor_images.back(), tensor_constants.back()));
                const auto *h = q16.back().header;
                q16_spill_bytes = std::max(q16_spill_bytes, h->reserved[1]);
                require(p.inputs.size() == h->inputs && p.outputs.size() == h->outputs,
                        "Q16P program I/O count differs from image");
                for (uint32_t i = 0; i < h->inputs + h->outputs; i++) {
                    const auto &spec = i < h->inputs ? p.inputs[i] : p.outputs[i - h->inputs];
                    require(spec.bytes() == q16.back().io[i].bytes,
                            "Q16P program I/O size differs from image");
                }
            }
            auto boot = read_file(artifacts.bootstrap);
            require(feature(boot, "HXTBOOT1") && feature(boot, std::string("QDBOOT01\1\0\0\0", 12)),
                    "Q16P backend requires the preserving tensor bootstrap");
            fourth = feature(boot, std::string("QDTHRD51\1\0\0\0", 12));
            engine = load_elf(read_file(artifacts.engine), 0xa0000000, 1 << 20);
            require(feature(engine, "HXQ16E01"), "Q16P engine feature missing");
            if (q16_spill_bytes) {
                require(feature(engine, "HXQ16SP3"), "Q16P engine lacks DDR spill/fill support");
                require(!artifacts.persistent_state_bytes && !artifacts.persistent_state_offset,
                        "Q16P spill/fill uses the session's state region");
            }
            require(fourth && feature(engine, "HXTPART4"),
                    "Q16P engine and bootstrap must both use four workers");
            preserving = true;
            return;
        }
        if (program.backend == Backend::tensor_f32) {
            for (const auto &p : programs) {
                p.validate();
                require(p.backend == Backend::tensor_f32, "cannot mix executor backends");
                tensor_images.push_back(read_file(p.image));
                tensor_constants.push_back(read_file(p.constants));
                graphs.push_back(preflight_tensor(tensor_images.back(), tensor_constants.back()));
                const auto &g = graphs.back();
                require(p.inputs.size() == g.header->inputs &&
                            p.outputs.size() == g.header->outputs,
                        "tensor program I/O count differs from image");
                for (uint32_t i = 0; i < g.header->inputs + g.header->outputs; i++) {
                    const auto &spec =
                        i < g.header->inputs ? p.inputs[i] : p.outputs[i - g.header->inputs];
                    const auto &t = g.tensors[g.io[i]];
                    std::vector<size_t> shape;
                    for (auto d : spec.shape)
                        if (d != 1)
                            shape.push_back(d);
                    std::vector<size_t> expected;
                    for (auto d : {t.h, t.w, t.c})
                        if (d != 1)
                            expected.push_back(d);
                    require(shape == expected && spec.bytes() == size_t(hx_tensor_elements(&t)) * 4,
                            "tensor program I/O shape differs from image");
                }
            }
            auto boot = read_file(artifacts.bootstrap);
            require(feature(boot, "HXTBOOT1") && feature(boot, std::string("QDBOOT01\1\0\0\0", 12)),
                    "tensor backend requires a preserving tensor bootstrap");
            fourth = feature(boot, std::string("QDTHRD51\1\0\0\0", 12));
            engine = load_elf(read_file(artifacts.engine), 0xa0000000, 1 << 20);
            require(feature(engine, "HXTENG01"), "tensor engine feature missing");
            require(fourth && feature(engine, "HXTPART4"),
                    "tensor engine and bootstrap must both use four workers");
            const char *grouped = std::getenv("HEXECUTOR_LQ_GROUPED");
            if (feature(engine, "HXLQG001") && !(grouped && std::string(grouped) == "0"))
                for (size_t i = 0; i < graphs.size(); i++)
                    group_linear_q(graphs[i], tensor_images[i], tensor_constants[i]);
            preserving = true;
            return;
        }
        require(programs.size() == 1, "QDRX supports one program per session");
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
        const bool qnn16 = program.backend == Backend::qnn16_f16;
        if (!qnn16)
            verify_chunk_map(artifacts);
        size_t engine_window_mb = 128;
        if (const char *env = std::getenv("HEXECUTOR_CHUNK_WINDOW_MB")) {
            char *end = nullptr;
            errno = 0;
            unsigned long v = std::strtoul(env, &end, 10);
            require(errno == 0 && end != env && *end == '\0' && v >= 128 && v <= 4095,
                    "HEXECUTOR_CHUNK_WINDOW_MB must be an integer in [128, 4095]");
            if (qnn16)
                require(v <= CDSP_MEM_MAX_SIZE_MB && v % 16 == 0,
                        "Q16P window must be a multiple of 16 in [128, 1536]");
            engine_window_mb = v;
        }
        if (qnn16) {
            if (engine_window_mb > 128 &&
                std::any_of(q16.begin(), q16.end(), [](const Q16Program &p) {
                    return p.header->ops > Q16P_WIDE_PROFILE_CYCLES;
                }))
                require(feature(engine, "HXQ16PC1"),
                        "Q16P engine profiling overlaps the wide-window JTLB state");
            const std::string marker = "QDWIND01";
            const std::string expected = marker + char(engine_window_mb / 16);
            const auto matches_window = [&](const Bytes &bytes) {
                return feature(bytes, expected) ||
                       (engine_window_mb == 128 && !feature(bytes, marker));
            };
            require(matches_window(engine) && matches_window(read_file(artifacts.bootstrap)) &&
                        matches_window(read_file(artifacts.chunk_probe)),
                    "Q16P engine, bootstrap and chunk probe must match HEXECUTOR_CHUNK_WINDOW_MB");
        }
        low_q16_spill = q16_spill_bytes && engine_window_mb == 128 && feature(engine, "HXQ16SP1");
        if (q16_spill_bytes) {
            uint64_t constant_bytes = 0;
            for (const auto &pool : tensor_constants)
                constant_bytes += ((uint64_t(pool.size()) + 127u) & ~uint64_t(127)) + 512u;
            if (low_q16_spill) {
                require(constant_bytes <= (24u << 20) && q16_spill_bytes <= Q16P_LOW_SPILL_END - Q16P_LOW_SPILL_START - 8192u,
                        "Q16P constants or spill backing exceed their baseline DDR regions");
                require(!artifacts.persistent_state_offset && !artifacts.persistent_state_bytes,
                        "Q16P baseline spill cannot use a caller-specified persistent state region");
            } else {
                const uint64_t required = uint64_t(HXFG_HIGH_CONSTANT_START) + q16_spill_bytes + 8192u + 512u + constant_bytes;
                require(required <= (uint64_t(engine_window_mb) << 20),
                        "Q16P constants and spill backing exceed the mapped high-DDR window");
            }
        }
        size_t workload_mb = calculate_workload_size_mb(programs, artifacts);
        if (qnn16)
            workload_mb = std::max(workload_mb, engine_window_mb);
        lease = WorkloadMemoryManager::instance().acquire_lease(workload_mb);
        size_t lease_mb = WorkloadMemoryManager::instance().current_size_mb();
        mapped_bytes = lease_mb << 20;

        const auto open_transport = [this] {
            transport = std::make_unique<Transport>(artifacts, artifacts.bootstrap, fourth);
            transport->set_exit_check([this] { verify_exit(true); });
        };
        if (!qnn16)
            open_transport();
        require(lease_mb >= 128, "requires a cdsp_mem allocation of at least 128 MB");
        require(lease_mb >= workload_mb, "cdsp_mem reservation smaller than workload requirement");
        require(std::stoull(read_text("/sys/module/cdsp_mem/parameters/iova")) == 0xa0000000,
                "wrong CDSP IOVA");
        int fd = ::open("/dev/cdsp_mem", O_RDWR | O_CLOEXEC);
        require(fd >= 0, "cannot open cdsp_mem");
        uint32_t pin_mb = static_cast<uint32_t>(lease_mb);
        if (::ioctl(fd, CDSP_MEM_IOC_PIN, &pin_mb) < 0) {
            // Unpinned, another process could resize the window under the DSP.
            int e = errno;
            ::close(fd);
            require(false, std::string("cdsp_mem pin failed: ") + std::strerror(e));
        }
        pin_fd = fd;
        if (qnn16) {
            bool restored = true;
            try {
                verify_chunk_map(artifacts, &restored);
            } catch (...) {
                if (!restored)
                    lease.retain_size_on_release();
                throw;
            }
            open_transport();
        }
        void *mapped = mmap(nullptr, mapped_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        require(mapped != MAP_FAILED, "cannot map cdsp_mem");
        memory = static_cast<uint8_t *>(mapped);
        std::copy(engine.begin(), engine.end(), memory);
        engine_mapped_bytes = std::min(mapped_bytes, engine_window_mb << 20);
        if (program.backend == Backend::tensor_f32 || program.backend == Backend::qnn16_f16) {
            const bool q = program.backend == Backend::qnn16_f16;
            tensor_config.magic = HXFG_CONFIG_MAGIC;
            tensor_config.version = 1;
            tensor_config.code_bytes = uint32_t(engine.size());
            tensor_config.graphs = uint32_t(programs.size());
            allocation_limit = (16u << 20) - 4096;
            for (size_t i = 0; i < programs.size(); i++) {
                auto &c = tensor_config.graph[i];
                c.image = allocate(tensor_images[i].size());
                c.image_bytes = uint32_t(tensor_images[i].size());
                std::copy(tensor_images[i].begin(), tensor_images[i].end(), memory + c.image);
            }
            cursor = 16u << 20;
            allocation_limit = (32u << 20) - 4096;
            for (size_t i = 0; i < programs.size(); i++) {
                auto &c = tensor_config.graph[i];
                c.arena_bytes = q ? q16_arena_bytes(q16[i].header) : graphs[i].header->arena_bytes;
                c.arena = allocate(c.arena_bytes);
                if (!q)
                    graphs[i].arena = memory + c.arena;
            }
            cursor = 32u << 20;
            /* The Q16P engine keeps its private, L1-cached scratch in
             * [56 MiB, 64 MiB - 4 KiB) of the same page (engine.c). */
            /* Tensor graphs may use relative [32, 96) MiB for constants (or higher when the loaded engine maps > 128 MiB);
             * the Q16P engine keeps its scratch in [56, 64) MiB. */
            allocation_limit = q ? (56u << 20) - 128 : (96u << 20) - 4096; /* above 96 MiB: VTCM backup, L2, guards */
            max_allocation = HXFG_MAX_CONSTANT_BYTES + 1u;
            /* Pools that do not fit the low region go above the first 128 MiB, inside the engine's mapped
             * window; the persistent state follows them there. */
            if (engine_mapped_bytes > (128u << 20)) {
                uint64_t need = 0;
                for (const auto &c : tensor_constants)
                    need += ((c.size() + 127) & ~size_t(127)) + 512;
                if (q16_spill_bytes || need > allocation_limit - cursor || std::getenv("HEXECUTOR_HIGH_CONSTANTS")) {
                    require(engine_mapped_bytes > size_t(HXFG_HIGH_CONSTANT_START) + 8192,
                            "engine window too small for the high constant region");
                    cursor = HXFG_HIGH_CONSTANT_START;
                    allocation_limit = uint32_t(std::min<size_t>(engine_mapped_bytes - 8192, HXFG_HIGH_CONSTANT_END));
                    high_constants = true;
                }
            }
            for (size_t i = 0; i < programs.size(); i++) {
                auto &c = tensor_config.graph[i];
                c.constant_bytes = uint32_t(tensor_constants[i].size());
                c.constants = allocate(c.constant_bytes);
                std::copy(tensor_constants[i].begin(), tensor_constants[i].end(),
                          memory + c.constants);
            }
            const uint32_t constants_end = cursor;
            // Q16P jobs share temporary spill backing in the session state region.
            size_t max_state_bytes = q16_spill_bytes;
            /* Persistent tensors carry a name hash (reserved[1]); graphs that use the same name share one
             * place, and each distinct name gets its own, so stages of one model never alias. Tensors
             * without a name keep the offset the lowering gave them, below all named ones. */
            std::map<uint32_t, uint32_t> shared_state_offsets;
            const auto align_state = [](size_t v) { return (v + 127) & ~size_t(127); };
            for (size_t i = 0; !q && i < programs.size(); i++) {
                const auto *h = graphs[i].header;
                for (uint32_t j = 0; j < h->tensors; j++) {
                    const auto &t = graphs[i].tensors[j];
                    if ((t.reserved[0] & HX_TENSOR_PERSISTENT) && t.reserved[1] == 0)
                        max_state_bytes = std::max(max_state_bytes, size_t(t.offset + hx_tensor_bytes(&t)));
                }
            }
            max_state_bytes = align_state(max_state_bytes);
            std::map<uint32_t, size_t> shared_state_bytes;
            for (size_t i = 0; !q && i < programs.size(); i++) {
                const auto *h = graphs[i].header;
                for (uint32_t j = 0; j < h->tensors; j++) {
                    const auto &t = graphs[i].tensors[j];
                    if (!(t.reserved[0] & HX_TENSOR_PERSISTENT) || t.reserved[1] == 0)
                        continue;
                    const size_t bytes = hx_tensor_bytes(&t);
                    auto it = shared_state_offsets.find(t.reserved[1]);
                    if (it == shared_state_offsets.end()) {
                        shared_state_offsets[t.reserved[1]] = uint32_t(max_state_bytes);
                        shared_state_bytes[t.reserved[1]] = bytes;
                        max_state_bytes = align_state(max_state_bytes + bytes);
                    } else {
                        require(shared_state_bytes[t.reserved[1]] == bytes, "shared persistent state sizes differ");
                    }
                }
            }
            if (artifacts.persistent_state_bytes > 0)
                max_state_bytes = std::max(max_state_bytes, artifacts.persistent_state_bytes);
            if (max_state_bytes > 0) {
                max_state_bytes = (max_state_bytes + 127) & ~127u;
                if (low_q16_spill) {
                    cursor = Q16P_LOW_SPILL_START;
                    allocation_limit = Q16P_LOW_SPILL_END - 4096u;
                }
                if (artifacts.persistent_state_offset > 0) {
                    tensor_config.state = artifacts.persistent_state_offset;
                } else {
                    tensor_config.state = allocate(max_state_bytes);
                }
                tensor_config.state_bytes = uint32_t(max_state_bytes);
                std::memset(memory + tensor_config.state, 0, tensor_config.state_bytes);
                for (size_t i = 0; !q && i < programs.size(); i++) {
                    graphs[i].state = memory + tensor_config.state;
                    graphs[i].state_bytes = tensor_config.state_bytes;
                    if (!shared_state_offsets.empty()) {
                        auto *img_tensors = reinterpret_cast<HxTensor *>(memory + tensor_config.graph[i].image + graphs[i].header->tensor_offset);
                        for (uint32_t j = 0; j < graphs[i].header->tensors; j++) {
                            if ((img_tensors[j].reserved[0] & HX_TENSOR_PERSISTENT) && img_tensors[j].reserved[1] != 0) {
                                auto it = shared_state_offsets.find(img_tensors[j].reserved[1]);
                                if (it != shared_state_offsets.end())
                                    img_tensors[j].offset = it->second;
                            }
                        }
                    }
                }
            }
            require(hx_session_config_valid(&tensor_config), "invalid tensor memory config");
            guard_floor = high_constants ? 0 : low_q16_spill ? constants_end : cursor;
            install_guards();
            auto *arena = transport->arena();
            std::memset(arena, 0, 32768);
            std::memcpy(arena + 4096, &tensor_config, sizeof(tensor_config));
            initialized = true;
            return;
        }
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
        install_guards();
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
    void install_guards() {
        for (auto g : std::initializer_list<std::pair<uint32_t, uint32_t>>{
                 {(80u << 20) - 4096, 4096},
                 {88u << 20, 4096},
                 {(64u << 20) - 4096, 4096},
                 {uint32_t(engine_mapped_bytes - 4096), 4096},
                 {(96u << 20) - 128, 128},
                 {104u << 20, 128}})
            if (g.first >= guard_floor && (!low_q16_spill ||
                g.first + g.second <= tensor_config.state || g.first >= tensor_config.state + tensor_config.state_bytes))
                guards.push_back(g);
        if (l2) {
            for (uint32_t offset : {0u, 0x40000u, 0x58000u, 0x68000u, 0x78000u})
                guards.emplace_back((112u << 20) + offset, 128);
            guards.emplace_back((112u << 20) + (fourth ? 0xc3e00 : 0xa0000), 128);
        }
        for (auto [at, n] : guards)
            std::memset(memory + at, 0xa5, n);
    }
    static std::string read_text(const char *path) {
        auto bytes = read_file(path);
        return {bytes.begin(), bytes.end()};
    }
    std::vector<Bytes> run(const std::vector<Buffer> &values, size_t program_index) {
        require(!closed && transport && memory, "session is closed");
        require(program_index < programs.size(), "program index out of range");
        auto guard = transport->lock();
        if (program.backend == Backend::qnn16_f16)
            return run_qnn16(values, program_index);
        if (program.backend == Backend::tensor_f32) {
            auto &g = graphs[program_index];
            const auto &p = programs[program_index];
            require(values.size() == p.inputs.size(), "input count mismatch");
            for (size_t i = 0; i < values.size(); i++)
                require(values[i].data && values[i].bytes == p.inputs[i].bytes(),
                        "input buffer size mismatch");
            for (size_t i = 0; i < values.size(); i++) {
                const auto &t = g.tensors[g.io[i]];
                hx_pack(reinterpret_cast<float *>(g.arena + t.offset),
                        static_cast<const float *>(values[i].data), &t);
            }
            auto *arena = transport->arena();
            const bool warm = last_graph == program_index &&
                              last_generation == transport->generation();
            transport->submit([&](uint32_t sequence) {
                std::memset(arena + 128, 0, 48);
                uint32_t packet[7] = {0x32564451,
                                      6,
                                      sequence - 1,
                                      3,
                                      uint32_t(program_index),
                                      uint32_t(programs.size()),
                                      (artifacts.tensor_hmx ? 0u : 1u) | (artifacts.tensor_fp16 ? 0u : 2u)};
                std::memcpy(arena, packet, sizeof(packet));
                put<uint32_t>(arena, 3840, 0);
                __atomic_thread_fence(__ATOMIC_SEQ_CST);
                put<uint32_t>(arena, 8, sequence);
                __atomic_thread_fence(__ATOMIC_SEQ_CST);
            }, warm ? last_dsp_time[program_index] : std::chrono::microseconds(0));
            require(get<uint32_t>(arena, 2048) == HXFG_RESULT_MAGIC &&
                        get<uint32_t>(arena, 2052) == HX_OK &&
                        get<uint32_t>(arena, 2060) == g.header->ops,
                    "tensor DSP command failed: error " +
                        std::to_string(get<uint32_t>(arena, 2052)) + " at op " +
                        std::to_string(get<uint32_t>(arena, 2056)));
            profile.cycles = get<uint64_t>(arena, 2064);
            profile.timer_ticks = get<uint64_t>(arena, 2080);
            /* A prediction delays seeing a completion by its error, so keep only
             * the time of a run that did not stage resident records: a graph
             * switch or a new wake re-stages, and later warm runs are far shorter. */
            const auto dsp_time = std::chrono::microseconds(profile.timer_ticks * 10 / 192);
            const uint32_t generation = transport->generation();
            if (last_graph == program_index && last_generation == generation)
                last_dsp_time[program_index] = dsp_time;
            last_graph = program_index;
            last_generation = generation;
            profile.matrix_ops = get<uint32_t>(arena, 2072);
            profile.blocked_fp16 = get<uint32_t>(arena, 2076) != 0;
            profile.op_cycles.resize(g.header->ops);
            std::memcpy(profile.op_cycles.data(), arena + 16384, profile.op_cycles.size() * 8);
            std::vector<Bytes> out;
            for (size_t i = 0; i < p.outputs.size(); i++) {
                const auto &t = g.tensors[g.io[g.header->inputs + i]];
                out.emplace_back(p.outputs[i].bytes());
                hx_unpack(reinterpret_cast<float *>(out.back().data()),
                          reinterpret_cast<const float *>(g.arena + t.offset), &t);
            }
            check_guards();
            return out;
        }
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
    std::vector<Bytes> run_qnn16(const std::vector<Buffer> &values, size_t index) {
        const auto &q = q16[index];
        const auto &p = programs[index];
        const auto *h = q.header;
        uint8_t *exchange = memory + tensor_config.graph[index].arena;
        require(values.size() == p.inputs.size(), "input count mismatch");
        for (size_t i = 0; i < values.size(); i++)
            require(values[i].data && values[i].bytes == p.inputs[i].bytes(),
                    "input buffer size mismatch");
        for (size_t i = 0; i < values.size(); i++)
            std::memcpy(exchange + q.io[i].offset, values[i].data, values[i].bytes);
        auto *arena = transport->arena();
        transport->submit([&](uint32_t sequence) {
            std::memset(arena + 128, 0, 48);
            uint32_t packet[7] = {0x32564451, 6, sequence - 1, 3, uint32_t(index),
                                  uint32_t(programs.size()), 0};
            std::memcpy(arena, packet, sizeof(packet));
            put<uint32_t>(arena, 3840, 0);
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            put<uint32_t>(arena, 8, sequence);
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
        }, last_dsp_time[index]);
        require(get<uint32_t>(arena, 2048) == 0x52363151 && get<uint32_t>(arena, 2052) == Q16_OK &&
                    get<uint32_t>(arena, 2060) == h->ops,
                "Q16P DSP command failed: error " + std::to_string(get<uint32_t>(arena, 2052)) +
                    " at op " + std::to_string(get<uint32_t>(arena, 2056)));
        profile.cycles = get<uint64_t>(arena, 2064);
        profile.timer_ticks = get<uint64_t>(arena, 2080);
        profile.matrix_ops = get<uint32_t>(arena, 2072);
        profile.blocked_fp16 = true;
        /* Runs re-stage everything, so every run predicts the next one. */
        last_dsp_time[index] = std::chrono::microseconds(profile.timer_ticks * 10 / 192);
        const uint32_t profile_limit = engine_mapped_bytes > (128u << 20)
                                           ? Q16P_WIDE_PROFILE_CYCLES
                                           : Q16P_PROFILE_CYCLES;
        profile.op_cycles.resize(std::min(h->ops, profile_limit));
        std::memcpy(profile.op_cycles.data(), arena + Q16P_PROFILE_OFFSET,
                    profile.op_cycles.size() * sizeof(uint64_t));
        std::vector<Bytes> out;
        for (size_t i = 0; i < p.outputs.size(); i++) {
            const auto &io = q.io[h->inputs + i];
            out.emplace_back(exchange + io.offset, exchange + io.offset + io.bytes);
        }
        check_guards();
        return out;
    }
    /* After every worker exit, not only the last: each wake leases VTCM again
     * and overwrites the snapshot, so a missed check could not be redone later. */
    void verify_exit(bool ran) {
        if (transport->worker_errors()) {
            transport->retain("worker exit/stack verification failed");
            throw std::runtime_error("worker exit/stack check failed");
        }
        auto *a = transport->arena();
        /* The JTLB mapper's state moved from 3072 to 28672 for windows above 128 MiB (qd_owned_memory.h). */
        if (get<uint32_t>(a, engine_mapped_bytes > (128u << 20) ? 28672 : 3072)) {
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
    }
    void close() {
        if (closed)
            return;
        try {
            if (transport) {
                transport->quiesce();
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
                verify_exit(ran);
                if (memory && initialized) {
                    check_guards();
                    if (program.backend == Backend::tensor_f32 ||
                        program.backend == Backend::qnn16_f16) {
                        for (size_t i = 0; i < tensor_constants.size(); i++)
                            require(std::equal(tensor_constants[i].begin(),
                                               tensor_constants[i].end(),
                                               memory + tensor_config.graph[i].constants),
                                    "tensor constant pool changed");
                    } else
                        require(std::equal(constants.begin(), constants.end(), memory + pools[5]),
                                "constant pool changed");
                }
#ifdef __linux__
                if (pin_fd >= 0) {
                    ::ioctl(pin_fd, CDSP_MEM_IOC_UNPIN);
                    ::close(pin_fd);
                    pin_fd = -1;
                }
#endif
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
            if (program.backend == Backend::qnn16_f16)
                lease.retain_size_on_release();
            release_memory();
            closed = true;
            std::rethrow_exception(failure);
        }
        try {
            if (transport)
                transport->close();
        } catch (...) {
            if (program.backend == Backend::qnn16_f16)
                lease.retain_size_on_release();
            release_memory();
            closed = true;
            throw;
        }
        release_memory();
        closed = true;
    }
    void release_memory() {
#ifdef __linux__
        if (pin_fd >= 0) {
            ::close(pin_fd);
            pin_fd = -1;
        }
        if (memory)
            munmap(memory, mapped_bytes);
#endif
        memory = nullptr;
        lease.release();
    }
    ~Impl() {
        try {
            close();
        } catch (const std::exception &e) {
            std::cerr << "[Hexecutor] " << e.what() << '\n';
        }
    }
    void reset_state() {
        if (!q16_spill_bytes && tensor_config.state_bytes) {
            if (memory) {
                std::memset(memory + tensor_config.state, 0, tensor_config.state_bytes);
            }
            for (auto &g : graphs) {
                if (g.state)
                    std::memset(g.state, 0, g.state_bytes);
            }
#ifdef __linux__
            if (transport && initialized) {
                auto guard = transport->lock();
                auto *arena = transport->arena();
                transport->submit([&](uint32_t sequence) {
                    std::memset(arena + 128, 0, 48);
                    uint32_t packet[7] = {0x32564451, 6, sequence - 1, 4, 0, 0, 0};
                    std::memcpy(arena, packet, sizeof(packet));
                    put<uint32_t>(arena, 3840, 0);
                    __atomic_thread_fence(__ATOMIC_SEQ_CST);
                    put<uint32_t>(arena, 8, sequence);
                    __atomic_thread_fence(__ATOMIC_SEQ_CST);
                });
                require(get<uint32_t>(arena, 2048) == HXFG_RESULT_MAGIC &&
                            get<uint32_t>(arena, 2052) == HX_OK,
                        "reset_state command failed");
            }
#endif
        }
    }
    size_t state_bytes() const {
        return q16_spill_bytes ? 0 : tensor_config.state_bytes;
    }
};
Session::Session(Program p, Artifacts a)
    : Session(std::vector<Program>{std::move(p)}, std::move(a)) {}
Session::Session(std::vector<Program> ps, Artifacts a)
    : impl_(std::make_unique<Impl>(std::move(ps), std::move(a))) {
    impl_->open();
}
Session::~Session() = default;
std::vector<Bytes> Session::run(const std::vector<Buffer> &values, size_t index) {
    return impl_->run(values, index);
}
const ExecutionProfile &Session::profile() const { return impl_->profile; }
void Session::reset_state() { impl_->reset_state(); }
size_t Session::state_bytes() const { return impl_->state_bytes(); }
void Session::close() { impl_->close(); }
} // namespace hexecutor
