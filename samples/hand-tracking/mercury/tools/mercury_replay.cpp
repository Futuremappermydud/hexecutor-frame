// Offline replay of rectified grayscale stereo pairs through the Mercury port.
//
//   mercury_replay --focal F --cx X --cy Y --baseline B
//                  --exchange "/abs/venv/python /abs/mercury_onnx_exchange.py --models /abs/dir"
//                  left0.pgm right0.pgm [left1.pgm right1.pgm ...]
//
// --exchange starts a development helper that answers network requests over a
// pipe (see samples/hand-tracking/tools/mercury_onnx_exchange.py). It exists to check
// the port against the reference networks; the live viewer uses the DSP. One
// JSON object per frame is written to stdout; frames are 1/fps apart.
#include "mercury/tracker.hpp"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

struct Pgm {
    int width = 0, height = 0;
    std::vector<uint8_t> pixels;
};

Pgm read_pgm(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::string magic;
    int maxval = 0;
    Pgm p;
    f >> magic;
    auto skip = [&] {
        while (f >> std::ws && f.peek() == '#') f.ignore(1 << 20, '\n');
    };
    skip();
    f >> p.width;
    skip();
    f >> p.height;
    skip();
    f >> maxval;
    f.get();
    if (!f || magic != "P5" || maxval != 255 || p.width <= 0 || p.height <= 0)
        throw std::runtime_error(path + ": expected an 8-bit binary PGM (P5)");
    p.pixels.resize(size_t(p.width) * size_t(p.height));
    f.read(reinterpret_cast<char*>(p.pixels.data()), std::streamsize(p.pixels.size()));
    if (!f) throw std::runtime_error(path + ": truncated");
    return p;
}

// Networks served by a child process over pipes; offline development only.
class ExchangeNetworks : public mercury::Networks {
public:
    explicit ExchangeNetworks(const std::string& command) {
        int to_child[2], from_child[2];
        if (pipe(to_child) || pipe(from_child)) throw std::runtime_error("pipe failed");
        pid_ = fork();
        if (pid_ < 0) throw std::runtime_error("fork failed");
        if (pid_ == 0) {
            dup2(to_child[0], 0);
            dup2(from_child[1], 1);
            close(to_child[1]);
            close(from_child[0]);
            execl("/bin/sh", "sh", "-c", command.c_str(), static_cast<char*>(nullptr));
            _exit(127);
        }
        close(to_child[0]);
        close(from_child[1]);
        out_ = to_child[1];
        in_ = from_child[0];
    }
    ~ExchangeNetworks() override {
        close(out_);
        close(in_);
        int status = 0;
        waitpid(pid_, &status, 0);
    }

    void detect(const mercury::DetectionInput* in, mercury::DetectionOutput* out, size_t count) override {
        header("detect", count);
        for (size_t i = 0; i < count; i++) write_all(in[i].image.data(), sizeof(in[i].image));
        for (size_t i = 0; i < count; i++) {
            float v[8];
            read_all(v, sizeof v);
            for (int h = 0; h < 2; h++) {
                out[i].hand_exists[size_t(h)] = v[h];
                out[i].cx[size_t(h)] = v[2 + h];
                out[i].cy[size_t(h)] = v[4 + h];
                out[i].size[size_t(h)] = v[6 + h];
            }
        }
    }

    void keypoints(const mercury::KeypointInput* in, mercury::KeypointOutput* out, size_t count) override {
        header("keypoint", count);
        for (size_t i = 0; i < count; i++) {
            write_all(in[i].image.data(), sizeof(in[i].image));
            write_all(in[i].last_keypoints.data(), sizeof(in[i].last_keypoints));
            write_all(&in[i].use_last_keypoints, sizeof(float));
        }
        for (size_t i = 0; i < count; i++) {
            read_all(out[i].heatmap_xy.data(), sizeof(out[i].heatmap_xy));
            read_all(out[i].heatmap_depth.data(), sizeof(out[i].heatmap_depth));
            read_all(out[i].scalar_extras.data(), sizeof(out[i].scalar_extras));
            read_all(out[i].curls.data(), sizeof(out[i].curls));
        }
    }

private:
    void header(const char* kind, size_t count) {
        const std::string h = std::string(kind) + " " + std::to_string(count) + "\n";
        write_all(h.data(), h.size());
    }
    void write_all(const void* p, size_t n) {
        const char* c = static_cast<const char*>(p);
        while (n) {
            ssize_t w = write(out_, c, n);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) throw std::runtime_error("exchange helper closed its input");
            c += w;
            n -= size_t(w);
        }
    }
    void read_all(void* p, size_t n) {
        char* c = static_cast<char*>(p);
        while (n) {
            ssize_t r = read(in_, c, n);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) throw std::runtime_error("exchange helper exited");
            c += r;
            n -= size_t(r);
        }
    }
    pid_t pid_ = -1;
    int out_ = -1, in_ = -1;
};

void print_hand(std::ostream& o, const mercury::Hand& h) {
    o << "{\"active\":" << (h.active ? "true" : "false") << ",\"tracked\":" << (h.tracked ? "true" : "false")
      << ",\"reprojection_error\":" << h.reprojection_error;
    if (h.tracked) {
        o << ",\"joints\":[";
        for (size_t j = 0; j < h.joints.size(); j++) {
            const auto& p = h.joints[j].pose.position;
            o << (j ? "," : "") << "[" << p.x << "," << p.y << "," << p.z << "]";
        }
        o << "]";
    }
    o << "}";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::signal(SIGPIPE, SIG_IGN);
        float focal = 0, cx = -1, cy = -1, baseline = 0, fps = 30;
        std::string exchange;
        std::vector<std::string> files;
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
                return argv[++i];
            };
            if (a == "--focal") focal = std::stof(value());
            else if (a == "--cx") cx = std::stof(value());
            else if (a == "--cy") cy = std::stof(value());
            else if (a == "--baseline") baseline = std::stof(value());
            else if (a == "--fps") fps = std::stof(value());
            else if (a == "--exchange") exchange = value();
            else if (!a.empty() && a[0] == '-') throw std::runtime_error("unknown option " + a);
            else files.push_back(a);
        }
        if (focal <= 0 || cx < 0 || cy < 0 || baseline <= 0 || exchange.empty() || files.empty() ||
            files.size() % 2 || fps <= 0)
            throw std::runtime_error(
                "usage: mercury_replay --focal F --cx X --cy Y --baseline B --exchange CMD [--fps N] "
                "left.pgm right.pgm ...");
        Pgm first = read_pgm(files[0]);
        mercury::StereoCalibration cal;
        cal.left = {focal, focal, cx, cy, first.width, first.height};
        cal.right = cal.left;
        cal.left_in_right.position = {-baseline, 0, 0};
        ExchangeNetworks nets(exchange);
        mercury::Tracker tracker(cal, nets);
        for (size_t f = 0; f + 1 < files.size(); f += 2) {
            Pgm l = read_pgm(files[f]), r = read_pgm(files[f + 1]);
            mercury::GrayImage li{l.pixels.data(), l.width, l.height, size_t(l.width)};
            mercury::GrayImage ri{r.pixels.data(), r.width, r.height, size_t(r.width)};
            const int64_t t = int64_t((double(f / 2) + 1) * 1e9 / fps);
            mercury::Result res = tracker.process(li, ri, t);
            std::ostringstream o;
            o << "{\"frame\":" << f / 2 << ",\"timestamp_ns\":" << t << ",\"detections\":" << res.detections_run
              << ",\"keypoints\":" << res.keypoints_run << ",\"hand_size\":" << tracker.target_hand_size()
              << ",\"left\":";
            print_hand(o, res.hands[0]);
            o << ",\"right\":";
            print_hand(o, res.hands[1]);
            o << "}";
            std::cout << o.str() << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "mercury_replay: " << e.what() << "\n";
        return 1;
    }
}
