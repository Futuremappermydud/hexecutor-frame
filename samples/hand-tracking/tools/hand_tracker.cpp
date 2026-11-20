#include "hand_tracking/tracker.hpp"
#include "hexecutor/broker.hpp"
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <limits>
#include <memory>
namespace {
volatile std::sig_atomic_t interrupted = 0;
void interrupt(int signal) { interrupted = signal; }
struct Ppm {
    std::vector<uint8_t> data;
    uint32_t width = 0, height = 0;
};
std::string token(std::istream &s) {
    std::string result;
    while (s >> std::ws && s.peek() == '#')
        s.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    if (!(s >> result))
        throw std::runtime_error("truncated PPM header");
    return result;
}
Ppm read_ppm(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (token(f) != "P6")
        throw std::runtime_error("input must be a binary RGB PPM (P6)");
    auto dimension = [&] {
        auto t = token(f);
        size_t used = 0;
        auto n = std::stoul(t, &used);
        if (used != t.size() || n == 0 || n > 16384)
            throw std::runtime_error("invalid PPM dimensions");
        return uint32_t(n);
    };
    Ppm image;
    image.width = dimension();
    image.height = dimension();
    if (token(f) != "255")
        throw std::runtime_error("PPM must have 8-bit channels");
    int delimiter = f.get();
    if (delimiter == '\r' && f.peek() == '\n')
        f.get();
    if (delimiter != ' ' && delimiter != '\n' && delimiter != '\t' && delimiter != '\r')
        throw std::runtime_error("invalid PPM delimiter");
    image.data.resize(size_t(image.width) * image.height * 3);
    if (!f.read(reinterpret_cast<char *>(image.data.data()), image.data.size()))
        throw std::runtime_error("truncated PPM pixels");
    return image;
}
void points(const std::array<hand_tracking::Point, 21> &p) {
    std::cout << '[';
    for (size_t i = 0; i < p.size(); i++) {
        if (i)
            std::cout << ',';
        std::cout << '[' << p[i].x << ',' << p[i].y << ',' << p[i].z << ']';
    }
    std::cout << ']';
}
} // namespace
int main(int argc, char **argv) {
    try {
        std::signal(SIGINT, interrupt);
        std::signal(SIGTERM, interrupt);
        hexecutor::Artifacts artifacts;
        std::filesystem::path models, dump_inputs, broker_socket;
        std::vector<std::string> paths;
        uint32_t repeat = 1;
        bool check = false;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            auto value = [&]() {
                if (++i >= argc)
                    throw std::runtime_error("missing option value");
                return std::string(argv[i]);
            };
            if (arg == "--models")
                models = value();
            else if (arg == "--loader")
                artifacts.loader = value();
            else if (arg == "--engine")
                artifacts.engine = value();
            else if (arg == "--bootstrap")
                artifacts.bootstrap = value();
            else if (arg == "--chunk-probe")
                artifacts.chunk_probe = value();
            else if (arg == "--broker")
                broker_socket = value();
            else if (arg == "--dump-inputs")
                dump_inputs = value();
            else if (arg == "--precise")
                artifacts.tensor_hmx = false;
            else if (arg == "--check-package")
                check = true;
            else if (arg == "--repeat") {
                auto v = value();
                size_t used = 0;
                auto n = std::stoul(v, &used);
                if (used != v.size() || !n || n > 10000)
                    throw std::runtime_error("invalid repeat count");
                repeat = uint32_t(n);
            } else if (!arg.empty() && arg[0] == '-')
                throw std::runtime_error("unknown option: " + arg);
            else
                paths.push_back(arg);
        }
        if ((models.empty() && broker_socket.empty()) || (!check && paths.empty()) ||
            (check && models.empty())) {
            std::cerr << "usage: frame_hand_tracker --models DIR --loader ELF --engine ELF "
                         "--bootstrap ELF --chunk-probe ELF [--precise] [--repeat N] "
                         "[--dump-inputs DIR] frame.ppm ...\n"
                         "       frame_hand_tracker --broker SOCKET [--repeat N] frame.ppm ...\n"
                         "       frame_hand_tracker --models DIR --check-package\n";
            return 2;
        }
        std::vector<hexecutor::Program> programs;
        if (!models.empty())
            programs = hand_tracking::load_models(models);
        if (check) {
            std::cout << "MediaPipe palm and landmark packages validated\n";
            return 0;
        }
        // Decode before opening the lease so a missing/bad image fails before
        // touching DSP state. This tool processes a bounded replay sequence.
        std::vector<Ppm> images;
        for (const auto &path : paths)
            images.push_back(read_ppm(path));
        // Either own the device session, or submit to the CDSP broker (hexecutord),
        // which must have graphs named "palm" and "landmark" registered.
        std::unique_ptr<hexecutor::Session> session;
        std::unique_ptr<hexecutor::broker::Client> client;
        std::vector<hexecutor::broker::Client::Stream> streams;
        if (broker_socket.empty())
            session = std::make_unique<hexecutor::Session>(std::move(programs), artifacts);
        else {
            client = std::make_unique<hexecutor::broker::Client>(broker_socket);
            for (const char *graph : {"palm", "landmark"}) {
                hexecutor::broker::StreamSpec spec;
                spec.graph = graph;
                spec.cls = hexecutor::broker::Class::interactive;
                spec.latest_only = false; // replay: every frame must run
                spec.deadline_ms = 10000;
                streams.push_back(client->open(spec));
            }
        }
        if (!dump_inputs.empty())
            std::filesystem::create_directories(dump_inputs);
        uint64_t inference = 0;
        hand_tracking::Tracker tracker([&](size_t index, const std::vector<float> &input) {
            if (!dump_inputs.empty()) {
                auto path = dump_inputs / (std::string(index ? "landmark-" : "palm-") +
                                           std::to_string(inference) + "-input.f32");
                std::ofstream f(path, std::ios::binary);
                if (!f.write(reinterpret_cast<const char *>(input.data()), input.size() * 4))
                    throw std::runtime_error("cannot dump inference input");
            }
            inference++;
            if (client)
                return client->run(streams.at(index), {{input.data(), input.size() * 4}});
            return session->run({{input.data(), input.size() * 4}}, index);
        });
        uint64_t frame = 0;
        for (uint32_t r = 0; r < repeat && !interrupted; r++)
            for (const auto &image : images) {
                if (interrupted)
                    break;
                auto start = std::chrono::steady_clock::now();
                auto out = tracker.process(
                    {image.data.data(), image.width, image.height, size_t(image.width) * 3},
                    (frame + 1) * 33333333ull);
                double ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count();
                std::cout << "{\"frame\":" << frame++ << ",\"width\":" << image.width
                          << ",\"height\":" << image.height << ",\"ms\":" << ms
                          << ",\"detector_ran\":" << (out.detector_ran ? "true" : "false")
                          << ",\"hands\":[";
                for (size_t i = 0; i < out.hands.size(); i++) {
                    const auto &h = out.hands[i];
                    if (i)
                        std::cout << ',';
                    std::cout << "{\"id\":" << h.id << ",\"presence\":" << h.presence
                              << ",\"right_probability\":" << h.right_probability
                              << ",\"landmarks\":";
                    points(h.landmarks);
                    std::cout << ",\"model_world_landmarks\":";
                    points(h.model_world_landmarks);
                    std::cout << '}';
                }
                std::cout << "]}\n" << std::flush;
            }
        if (session)
            session->close();
        return interrupted ? 128 + interrupted : 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
