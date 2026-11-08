// Bounded, streaming capture of XRService's long-exposure OG01A1B stereo pair.
// Uses IrPairCapture's read-only QUERYBUF/pidfd path; never opens /dev/video*.
//
// With --pulse-every N the tool owns the BELICE emitter: every N long exposures it saves
// a reference pair, pulses the emitter inside the next long exposure, saves that lit pair,
// and saves the following long pair as a second reference. A pulse fires only after a fresh frame arrives, so a frozen feed stops the emitter.
// A random share of pulses are shams (no light) to measure false positives per session.
#include "belice_emitter.hpp"
#include "ir_pair_capture.hpp"
#include <time.h>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>

using namespace FrameSpatial;
using Clock = std::chrono::steady_clock;
static volatile sig_atomic_t running = 1;
static void stopSignal(int) { running = 0; }

static int64_t rawNowNs() {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &now);
    return int64_t(now.tv_sec) * 1000000000ll + now.tv_nsec;
}

static void waitUntilRaw(int64_t targetNs) {
    while (true) {
        const int64_t left = targetNs - rawNowNs();
        if (left <= 0) return;
        if (left > 300000) std::this_thread::sleep_for(std::chrono::nanoseconds(left - 200000));
    }
}

static bool writeBytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
    const auto temporary = path.string() + ".part";
    std::ofstream out(temporary, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    out.close();
    if (!out) return false;
    std::filesystem::rename(temporary, path);
    return true;
}

// Disk writes stay off the timing thread. A full queue drops the pair rather than delay a pulse.
class PairWriter {
public:
    PairWriter(std::filesystem::path out) : out_(std::move(out)), index_(out_ / "pairs.jsonl"), thread_([this] { run(); }) {}
    ~PairWriter() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        ready_.notify_one();
        thread_.join();
    }
    bool add(IrPairCapture::Pair&& pair, int id, const std::string& role, int cycle) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (jobs_.size() >= 32) {
                ++dropped_;
                return false;
            }
            jobs_.push_back(Job{std::move(pair), id, role, cycle});
        }
        ready_.notify_one();
        return true;
    }
    bool failed() const { return failed_; }
    int dropped() const { return dropped_; }

private:
    struct Job {
        IrPairCapture::Pair pair;
        int id;
        std::string role;
        int cycle;
    };
    void run() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [this] { return done_ || !jobs_.empty(); });
            if (jobs_.empty()) return;
            Job job = std::move(jobs_.front());
            jobs_.pop_front();
            lock.unlock();
            const std::string prefix = "pair-" + std::to_string(job.id);
            if (!writeBytes(out_ / (prefix + "-left.u8"), job.pair.left) ||
                !writeBytes(out_ / (prefix + "-right.u8"), job.pair.right)) {
                failed_ = true;
                continue;
            }
            index_ << "{\"pair\":" << job.id << ",\"left_seq\":" << job.pair.leftSeq
                   << ",\"right_seq\":" << job.pair.rightSeq << ",\"left_raw_ns\":" << job.pair.leftTsNs
                   << ",\"right_raw_ns\":" << job.pair.rightTsNs << ",\"left_monotonic_ns\":" << job.pair.leftMonotonicNs
                   << ",\"dot_score\":" << IrPairCapture::dotScore(job.pair.left)
                   << ",\"right_dot_score\":" << IrPairCapture::dotScore(job.pair.right);
            if (!job.role.empty()) index_ << ",\"role\":\"" << job.role << "\",\"cycle\":" << job.cycle;
            index_ << ",\"left_file\":\"" << prefix << "-left.u8\",\"right_file\":\"" << prefix << "-right.u8\"}\n";
            index_.flush();
            if (!index_) failed_ = true;
        }
    }
    std::filesystem::path out_;
    std::ofstream index_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Job> jobs_;
    bool done_ = false;
    volatile bool failed_ = false;
    int dropped_ = 0;
    std::thread thread_;
};

struct PulseOptions {
    int every = 0;
    double pulseMs = 10.0;
    std::vector<double> leadsMs{8.0};
    int shamPercent = 10;
    bool noEmit = false;
    std::filesystem::path log;
};

static int recordPulsed(IrPairCapture& capture, const std::filesystem::path& out, int durationMs, int maxPairs,
                        const PulseOptions& options) {
    BeliceEmitter emitter;
    if (!options.noEmit && !emitter.open()) {
        std::cerr << "BELICE emitter not available\n";
        return 3;
    }
    std::ofstream log(options.log);
    if (!log) return 1;
    PairWriter writer(out);
    std::mt19937 random{std::random_device{}()};
    std::uniform_int_distribution<int> percent(0, 99);
    const auto end = Clock::now() + std::chrono::milliseconds(durationMs);
    const int64_t pulseNs = int64_t(options.pulseMs * 1e6);
    double periodNs = 33333333.0;
    int64_t lastLongTs = 0, lastCycleTs = 0;
    int cycle = 0, nextId = 0, fired = 0, emitterFailures = 0;
    bool streamLost = false;
    while (running && (durationMs == 0 || Clock::now() < end) && (maxPairs == 0 || nextId < maxPairs) && !writer.failed()) {
        IrPairCapture::Pair ref;
        if (!capture.grab(ref, 300, false)) {
            if (!streamLost) std::cerr << "IR stream waiting: " << capture.lastError() << std::endl;
            streamLost = true;
            continue;
        }
        streamLost = false;
        const double measured = double(ref.leftTsNs - lastLongTs);
        if (lastLongTs && measured > 30e6 && measured < 37e6) periodNs += 0.05 * (measured - periodNs);
        lastLongTs = ref.leftTsNs;
        if (lastCycleTs && double(ref.leftTsNs - lastCycleTs) < (options.every - 0.5) * periodNs) continue;
        if (!capture.fetchPixels(ref)) continue;

        const int thisCycle = cycle++;
        lastCycleTs = ref.leftTsNs;
        const double leadMs = options.leadsMs[size_t(thisCycle) % options.leadsMs.size()];
        const int64_t predicted = ref.leftTsNs + int64_t(periodNs);
        const int64_t onAt = predicted - int64_t(leadMs * 1e6) - pulseNs / 2;
        const bool sham = options.noEmit || percent(random) < options.shamPercent;
        std::string status = sham ? "sham" : "fired";
        int64_t onCommand = 0, onDone = 0, offCommand = 0, offDone = 0;
        if (onAt < rawNowNs() + 500000) {
            status = "late";
        } else {
            waitUntilRaw(onAt);
            onCommand = rawNowNs();
            if (!sham && !emitter.set(true)) status = "emitter_failed";
            onDone = rawNowNs();
            waitUntilRaw(onCommand + pulseNs);
            offCommand = rawNowNs();
            bool off = sham;
            for (int attempt = 0; attempt < 3 && !off; ++attempt) off = emitter.set(false);
            offDone = rawNowNs();
            if (!off) {
                std::cerr << "EMITTER MAY BE ON: off command failed; unplug the board\n";
                return 1;
            }
            if (status == "fired") ++fired;
            emitterFailures = status == "emitter_failed" ? emitterFailures + 1 : 0;
            if (emitterFailures >= 5) {
                std::cerr << "BELICE stopped responding\n";
                return 1;
            }
        }

        IrPairCapture::Pair lit;
        bool matched = false;
        const auto litDeadline = Clock::now() + std::chrono::milliseconds(100);
        while (status != "late" && running && Clock::now() < litDeadline) {
            if (!capture.grab(lit, 20, false)) continue;
            lastLongTs = lit.leftTsNs;
            if (lit.leftTsNs < predicted - 8000000ll) continue;
            matched = lit.leftTsNs < predicted + 8000000ll && capture.fetchPixels(lit);
            break;
        }
        const int refId = nextId + 1, litId = nextId + 2;
        bool saved = false;
        if (matched && writer.add(std::move(ref), refId, "reference", thisCycle)) {
            saved = writer.add(std::move(lit), litId, "lit", thisCycle);
            nextId = saved ? litId : refId;
        }
        bool afterSaved = false;
        if (saved) {
            IrPairCapture::Pair after;
            const int64_t afterPredicted = predicted + int64_t(periodNs);
            const auto afterDeadline = Clock::now() + std::chrono::milliseconds(100);
            while (running && Clock::now() < afterDeadline) {
                if (!capture.grab(after, 20, false)) continue;
                lastLongTs = after.leftTsNs;
                if (after.leftTsNs < afterPredicted - 8000000ll) continue;
                afterSaved = after.leftTsNs < afterPredicted + 8000000ll && capture.fetchPixels(after) &&
                             writer.add(std::move(after), nextId + 1, "after", thisCycle);
                if (afterSaved) ++nextId;
                break;
            }
        }
        log << "{\"cycle\":" << thisCycle << ",\"status\":\"" << status << "\",\"lead_ms\":" << leadMs
            << ",\"pulse_ms\":" << options.pulseMs << ",\"predicted_raw_ns\":" << predicted
            << ",\"on_command_raw_ns\":" << onCommand << ",\"on_done_raw_ns\":" << onDone
            << ",\"off_command_raw_ns\":" << offCommand << ",\"off_done_raw_ns\":" << offDone
            << ",\"saved\":" << (saved ? "true" : "false") << ",\"after_saved\":" << (afterSaved ? "true" : "false") << "}\n";
        log.flush();
    }
    std::cout << "IR cycles=" << cycle << " fired=" << fired << " pairs=" << nextId
              << " dropped=" << writer.dropped() << "\n";
    return nextId && !writer.failed() ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: record_ir_pairs OUT_DIR DURATION_MS SPACING_MS MAX_PAIRS\n"
                     "         [--pulse-every N --pulse-ms MS --lead-ms MS[,MS...] --sham-percent P\n"
                     "          --pulses-out FILE --no-emit]\n";
        return 2;
    }
    const std::filesystem::path out = argv[1];
    const int duration = std::stoi(argv[2]), spacing = std::stoi(argv[3]), maxPairs = std::stoi(argv[4]);
    PulseOptions options;
    options.log = out / "pulses.jsonl";
    for (int i = 5; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--no-emit") {
            options.noEmit = true;
            continue;
        }
        if (i + 1 >= argc) return 2;
        const std::string value = argv[++i];
        if (flag == "--pulse-every") options.every = std::stoi(value);
        else if (flag == "--pulse-ms") options.pulseMs = std::stod(value);
        else if (flag == "--sham-percent") options.shamPercent = std::stoi(value);
        else if (flag == "--pulses-out") options.log = value;
        else if (flag == "--lead-ms") {
            options.leadsMs.clear();
            std::stringstream list(value);
            for (std::string item; std::getline(list, item, ',');) options.leadsMs.push_back(std::stod(item));
        } else return 2;
    }
    if (duration < 0 || spacing < 20 || maxPairs < 0) return 2;
    if (options.every) {
        const double duty = options.pulseMs / (options.every * 33.33);
        if (options.every < 3 || options.pulseMs <= 0 || options.pulseMs > 20 || duty > 0.1 || options.leadsMs.empty() ||
            options.shamPercent < 0 || options.shamPercent > 50) {
            std::cerr << "pulse limits: every >= 3, 0 < pulse <= 20 ms, duty <= 10%, sham 0-50%\n";
            return 2;
        }
    }
    std::signal(SIGTERM, stopSignal);
    std::signal(SIGINT, stopSignal);
    std::filesystem::create_directories(out);
    IrPairCapture capture;
    if (!capture.initialize()) {
        std::cerr << "IR capture: " << capture.lastError() << '\n';
        return 1;
    }
    if (options.every) return recordPulsed(capture, out, duration, maxPairs, options);

    int count = 0;
    {
        PairWriter writer(out);
        const auto end = Clock::now() + std::chrono::milliseconds(duration);
        bool streamLost = false;
        while (running && (duration == 0 || Clock::now() < end) && (maxPairs == 0 || count < maxPairs) &&
               !writer.failed()) {
            const auto next = Clock::now() + std::chrono::milliseconds(spacing);
            IrPairCapture::Pair pair;
            if (!capture.grab(pair, 300)) {
                if (!streamLost) std::cerr << "IR stream waiting: " << capture.lastError() << std::endl;
                streamLost = true;
                continue;
            }
            streamLost = false;
            if (writer.add(std::move(pair), count + 1, "", 0)) ++count;
            std::this_thread::sleep_until(next);
        }
    }
    std::cout << "IR pairs=" << count << "\n";
    return count ? 0 : 1;
}
