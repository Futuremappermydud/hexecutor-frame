// E02b acceptance probe: V4L2-sequence-paired capture without DSP/OpenXR.
// usage: probe_timed_pairs SECONDS [REQUEST_INTERVAL_MS]
// Requests pairs at the given interval (0 = back to back; 500 mimics depth
// cadence) and prints one CSV row per delivered pair plus a summary on stderr.
#include "dma_capture.hpp"
#include "v4l2_capture_clock.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
using namespace FrameSpatial;
static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(q * (v.size() - 1) + 0.5))];
}
int main(int argc, char** argv) {
    const double seconds = argc > 1 ? std::atof(argv[1]) : 10;
    const int interval = argc > 2 ? std::atoi(argv[2]) : 0;
    setenv("FRAME_CAPTURE_TIMING", "v4l2", 1);
    DMACapture capture;
    const pid_t pid = DMACapture::findCompositorPid();
    if (pid <= 0 || !capture.initialize(pid)) return 1;
    std::printf("sequence,left_seq,right_seq,left_slot,right_slot,eof_skew_ms,left_write_minus_eof_ms,right_write_minus_eof_ms,"
                "age_at_delivery_ms,copy_ms,skipped\n");
    std::vector<double> skew, lw, rw, age, copy;
    uint64_t pairs = 0, mismatched = 0, timeouts = 0, skipped = 0;
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < end) {
        if (!capture.captureFrame(false)) { ++timeouts; continue; }
        const auto* p = capture.timedPair();
        if (!p) { ++timeouts; continue; }
        ++pairs;
        mismatched += p->leftSequence != p->rightSequence;
        skipped += p->skippedSequences;
        const double s = (p->rightEndOfReadoutNs - p->leftEndOfReadoutNs) * 1e-6;
        const double a = (p->copyEndNs - p->leftEndOfReadoutNs) * 1e-6;
        skew.push_back(s); age.push_back(a);
        lw.push_back((p->leftWriteStartNs - p->leftEndOfReadoutNs) * 1e-6);
        rw.push_back((p->rightWriteStartNs - p->rightEndOfReadoutNs) * 1e-6);
        copy.push_back((p->copyEndNs - p->copyBeginNs) * 1e-6);
        std::printf("%u,%u,%u,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%llu\n", p->sequence, p->leftSequence, p->rightSequence,
                    p->leftSlot, p->rightSlot, s, lw.back(), rw.back(), a, copy.back(), (unsigned long long)p->skippedSequences);
        if (interval > 0) std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    }
    const auto st = capture.timedStats();
    std::fprintf(stderr,
                 "pairs %llu (L/R sequence mismatches %llu), timeouts %llu, skipped sequences %llu\n"
                 "ring frames L %llu R %llu, unmatched L %llu R %llu, torn copies %llu, probes %llu (%.2f ms each)\n"
                 "eof skew R-L ms p5/p50/p95 %.3f/%.3f/%.3f | write-EOF ms L p50 %.2f (p5 %.2f p95 %.2f) R p50 %.2f\n"
                 "age at delivery (copy end - left EOF) ms p5/p50/p95 %.1f/%.1f/%.1f | copy ms p50 %.1f\n",
                 (unsigned long long)pairs, (unsigned long long)mismatched, (unsigned long long)timeouts,
                 (unsigned long long)skipped, (unsigned long long)st.leftFrames, (unsigned long long)st.rightFrames,
                 (unsigned long long)st.leftUnmatched, (unsigned long long)st.rightUnmatched,
                 (unsigned long long)st.tornCopies, (unsigned long long)st.probes, st.probeMsMean,
                 pct(skew, .05), pct(skew, .5), pct(skew, .95), pct(lw, .5), pct(lw, .05), pct(lw, .95), pct(rw, .5),
                 pct(age, .05), pct(age, .5), pct(age, .95), pct(copy, .5));
    return pairs ? 0 : 1;
}
