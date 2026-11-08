// Failure handling of the IR-dot bridge and pulse processes, driven by fake processes.
// Run from the repository root; needs a python3 on PATH.
#include "ir_dot_bridge.hpp"
#include "calibration.hpp"
#include "model_depth_geometry.hpp"
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace FrameSpatial;
using Clock = std::chrono::steady_clock;

namespace {
std::string dir;

void writeScript(const std::string& behaviour) {
    std::ofstream f(dir + "/fake_bridge.py");
    f << R"PY(
import os, socket, struct, sys, time
W, H, RW, FOOT = )PY" << MODEL_WIDTH << ", " << MODEL_HEIGHT << ", " << RECT_WIDTH << ", " << kIrFootprintSize << R"PY(
BEHAVIOUR = ")PY" << behaviour << R"PY("
s = socket.socket(fileno=3)
s.sendall(struct.pack("<5I2f", 0x30444F54, W, H, RW, FOOT, )PY" << RECT_FOCAL << R"PY(, 0.29))
def rx(n):
    b = bytearray()
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c: sys.exit(0)
        b += c
    return b
count_path = os.environ["FAKE_COUNT"]
n = int(open(count_path).read()) if os.path.exists(count_path) else 0
open(count_path, "w").write(str(n + 1))
while True:
    head = rx(20)
    seq = struct.unpack("<5I", head)[4]
    rx(W * H * 5 + 2 * 1056 * 1024)
    if BEHAVIOUR == "hang" and n == 0: time.sleep(60)
    if BEHAVIOUR == "truncate" and n == 0:
        s.sendall(b"T" * 40); time.sleep(60)
    if BEHAVIOUR == "stale" and n == 0: seq += 7
    if BEHAVIOUR == "die" and n == 0: sys.exit(1)
    plane = BEHAVIOUR != "badplane" or n > 0
    fl = [1.5, 0, 0, 1, 2, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0.5, 0.01] if BEHAVIOUR != "badplane" or n else [1.5, 3, 0, 1, 2] + [0] * 11
    s.sendall(struct.pack("<5I16f", 0x34444F54, seq, 1234, 1, 99, *fl))
    s.sendall(bytes(FOOT * FOOT) + bytes(RW * RW * 4))
)PY";
}

struct Inputs {
    std::vector<float> disparity = std::vector<float>(size_t(MODEL_WIDTH) * MODEL_HEIGHT);
    std::vector<uint8_t> uncertainty = std::vector<uint8_t>(disparity.size());
    std::vector<uint8_t> image = std::vector<uint8_t>(size_t(1056) * 1024);
};

// The first exchange fails and kills the process, then one respawned exchange works.
void failsThenRecovers(const std::string& behaviour) {
    unlink((dir + "/count").c_str());
    writeScript(behaviour);
    std::atomic<bool> running{true};
    IrBridgeClient client(&running);
    client.transferTimeoutMs = 500;
    assert(client.start());
    Inputs in;
    IrBridgeReply reply;
    assert(!client.exchange(in.disparity.data(), in.uncertainty.data(), in.image.data(), in.image.data(), reply));
    assert(!client.connected());
    assert(!client.exchange(in.disparity.data(), in.uncertainty.data(), in.image.data(), in.image.data(), reply));
    assert(client.lastError().find("backing off") != std::string::npos);
    std::this_thread::sleep_for(std::chrono::milliseconds(1050));
    assert(client.exchange(in.disparity.data(), in.uncertainty.data(), in.image.data(), in.image.data(), reply));
    assert(client.spawnCount() == 2 && reply.kept == 1234 && reply.hasPlane && reply.c == 2.f);
    assert(reply.depth.size() == size_t(RECT_WIDTH) * RECT_HEIGHT);
    std::cout << "[Test] bridge recovers from " << behaviour << ": PASS" << std::endl;
}
}  // namespace

int main() {
    char tmpl[] = "/tmp/irbridgeXXXXXX";
    dir = mkdtemp(tmpl);
    const char* python = std::getenv("TEST_PYTHON");
    setenv("FRAME_DEPTH_PYTHON", python ? python : "/usr/bin/env", 1);
    if (!python) {
        std::cerr << "set TEST_PYTHON to an absolute python3" << std::endl;
        return 2;
    }
    setenv("FRAME_IR_DOT_SCRIPT", (dir + "/fake_bridge.py").c_str(), 1);
    setenv("FAKE_COUNT", (dir + "/count").c_str(), 1);
    setenv("FRAME_DEPTH_SPHERICAL_DEG", "0.29", 1);

    writeScript("ok");
    {
        std::atomic<bool> running{true};
        IrBridgeClient client(&running);
        assert(client.start());
        Inputs in;
        IrBridgeReply reply;
        for (int i = 0; i < 3; ++i)
            assert(client.exchange(in.disparity.data(), in.uncertainty.data(), in.image.data(), in.image.data(), reply));
        assert(client.spawnCount() == 1);
        std::cout << "[Test] repeated exchange on one process: PASS" << std::endl;
    }
    for (const char* behaviour : {"hang", "truncate", "stale", "die", "badplane"}) failsThenRecovers(behaviour);

    // A step mismatch must fail the handshake: the prior would be read at the wrong scale.
    {
        setenv("FRAME_DEPTH_SPHERICAL_DEG", "0.25", 1);
        std::atomic<bool> running{true};
        IrBridgeClient client(&running);
        assert(!client.start() && client.lastError() == "incompatible IR-dot bridge");
        setenv("FRAME_DEPTH_SPHERICAL_DEG", "0.29", 1);
        std::cout << "[Test] handshake rejects a different spherical step: PASS" << std::endl;
    }

    // Clearing the run flag ends a blocked transfer within one poll slice.
    {
        unlink((dir + "/count").c_str());
        writeScript("hang");
        std::atomic<bool> running{true};
        IrBridgeClient client(&running);
        assert(client.start());
        std::thread stopper([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            running.store(false);
        });
        Inputs in;
        IrBridgeReply reply;
        const auto t0 = Clock::now();
        assert(!client.exchange(in.disparity.data(), in.uncertainty.data(), in.image.data(), in.image.data(), reply));
        stopper.join();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
        assert(ms < 1500);  // includes the SIGTERM reap
        std::cout << "[Test] stop interrupts a hung bridge in " << ms << " ms: PASS" << std::endl;
    }

    // Pulse process.
    {
        IrPulseProcess pulse;
        assert(pulse.poll() == IrPulseProcess::State::Idle);
        assert(pulse.start("exit 0"));
        while (pulse.poll() == IrPulseProcess::State::Running) {}
        assert(pulse.start("exit 3"));
        IrPulseProcess::State state;
        while ((state = pulse.poll()) == IrPulseProcess::State::Running) {}
        assert(state == IrPulseProcess::State::Failed);

        // The emitter is only safe to interrupt with SIGINT: the script must get to run its finally.
        const std::string marker = dir + "/off";
        const std::string py(python);
        assert(pulse.start("exec " + py + " -c \"import time\ntry:\n    print('on',flush=True); time.sleep(60)\nfinally:\n    open('" + marker + "','w').write('off')\" >/dev/null"));
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        assert(pulse.poll() == IrPulseProcess::State::Running);
        assert(pulse.abort());
        assert(access(marker.c_str(), F_OK) == 0);
        std::cout << "[Test] abort lets the emitter script run its off path: PASS" << std::endl;

        assert(pulse.start("trap '' INT; sleep 60"));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        assert(!pulse.abort(200));  // reports that SIGKILL was needed
        std::cout << "[Test] abort escalates and reports it: PASS" << std::endl;
    }
    std::cout << "All IR-dot bridge tests PASS" << std::endl;
    return 0;
}
