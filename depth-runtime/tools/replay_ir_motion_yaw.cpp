// Offline replay of the tag-free IR motion-stereo yaw estimator (spline lens).
//
//   replay_ir_motion_yaw problem.txt out_samples.csv [trace.txt]
//
// Reads the problem written by export_problem.py (format documented in
// depth-runtime/include/ir_motion_stereo_yaw.hpp), runs the kernel single-threaded, and writes one CSV row
// per accepted sample with REF's column order at full precision. The optional trace file lists, per
// keyframe, the chosen partner indices and the corner list, for parity debugging.

#include "ir_motion_stereo_yaw.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using FrameSpatial::MotionYaw::Problem;
using FrameSpatial::MotionYaw::Reference;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

double parseReal(const std::string& token) {
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0') fail("bad number: " + token);
    return value;
}

long long parseInteger(const std::string& token) {
    char* end = nullptr;
    const long long value = std::strtoll(token.c_str(), &end, 10);
    if (end == token.c_str() || *end != '\0') fail("bad integer: " + token);
    return value;
}

std::vector<std::string> splitWhitespace(const std::string& line) {
    std::istringstream in(line);
    std::vector<std::string> out;
    std::string token;
    while (in >> token) out.push_back(token);
    return out;
}

std::vector<std::string> splitTabs(const std::string& line) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (true) {
        const std::size_t tab = line.find('\t', start);
        out.push_back(line.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
        if (tab == std::string::npos) break;
        start = tab + 1;
    }
    return out;
}

void readIntrinsics(const std::vector<std::string>& f, FrameSpatial::IrKBSplineIntrinsics& intr) {
    // f[0] = "intrinsics", f[1] = side, then 4 + 4 + 2 + 25 + 25 numbers.
    const std::size_t expected = 2 + 4 + 4 + 2 + 25 + 25;
    if (f.size() != expected) fail("intrinsics line has the wrong number of fields");
    std::size_t p = 2;
    intr.fx = parseReal(f[p++]);
    intr.fy = parseReal(f[p++]);
    intr.cx = parseReal(f[p++]);
    intr.cy = parseReal(f[p++]);
    for (double& k : intr.k) k = parseReal(f[p++]);
    for (double& s : intr.nodeSpacing) s = parseReal(f[p++]);
    for (double& v : intr.sU) v = parseReal(f[p++]);
    for (double& v : intr.sV) v = parseReal(f[p++]);
}

void readRotation(const std::vector<std::string>& f, std::array<double, 9>& R) {
    if (f.size() != 2 + 9) fail("rotation line has the wrong number of fields");
    for (std::size_t i = 0; i < 9; ++i) R[i] = parseReal(f[2 + i]);
}

struct ParsedProblem {
    Problem problem;
    std::vector<std::string> leftPaths, rightPaths;
};

ParsedProblem readProblem(const std::string& path) {
    std::ifstream in(path);
    if (!in) fail("cannot open " + path);
    ParsedProblem parsed;
    Problem& prob = parsed.problem;
    std::string line;
    bool sawFormat = false, sawGrid = false, sawLeft = false, sawRight = false, sawBaseline = false;
    bool sawKeyframes = false, sawInverse = false, sawRL = false, sawRR = false, sawR1 = false;
    std::size_t expectedReferences = 0;
    bool sawReferences = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        if (line.compare(0, 4, "ref\t") == 0) {
            const std::vector<std::string> f = splitTabs(line);
            if (f.size() != 1 + 1 + 9 + 3 + 2) fail("ref line has the wrong number of fields");
            Reference ref;
            ref.when = parseInteger(f[1]);
            for (std::size_t i = 0; i < 9; ++i) ref.R[i] = parseReal(f[2 + i]);
            for (std::size_t i = 0; i < 3; ++i) ref.t[i] = parseReal(f[11 + i]);
            prob.references.push_back(ref);
            parsed.leftPaths.push_back(f[14]);
            parsed.rightPaths.push_back(f[15]);
            continue;
        }
        const std::vector<std::string> f = splitWhitespace(line);
        if (f.empty()) continue;
        const std::string& key = f[0];
        if (key == "format") {
            if (f.size() != 2 || f[1] != "frame-motion-yaw-v1") fail("unknown problem format");
            sawFormat = true;
        } else if (key == "grid") {
            if (f.size() != 5) fail("grid line has the wrong number of fields");
            if (parseInteger(f[1]) != FrameSpatial::MotionYaw::kWidth ||
                parseInteger(f[2]) != FrameSpatial::MotionYaw::kHeight ||
                parseInteger(f[4]) != FrameSpatial::MotionYaw::kHalf)
                fail("grid constants do not match the compiled kernel");
            prob.kHi = parseReal(f[3]);
            sawGrid = true;
        } else if (key == "intrinsics") {
            if (f.size() < 2) fail("intrinsics line without side");
            if (f[1] == "left") {
                readIntrinsics(f, prob.leftIntr);
                sawLeft = true;
            } else if (f[1] == "right") {
                readIntrinsics(f, prob.rightIntr);
                sawRight = true;
            } else {
                fail("unknown intrinsics side: " + f[1]);
            }
        } else if (key == "rotation") {
            if (f.size() < 2) fail("rotation line without name");
            if (f[1] == "rect0.RL") {
                readRotation(f, prob.rect0RL);
                sawRL = true;
            } else if (f[1] == "rect0.RR") {
                readRotation(f, prob.rect0RR);
                sawRR = true;
            } else if (f[1] == "rect1.RR") {
                readRotation(f, prob.rect1RR);
                sawR1 = true;
            } else {
                fail("unknown rotation: " + f[1]);
            }
        } else if (key == "baseline") {
            if (f.size() != 2) fail("baseline line has the wrong number of fields");
            prob.baseline = parseReal(f[1]);
            sawBaseline = true;
        } else if (key == "keyframes") {
            if (f.size() != 2) fail("keyframes line has the wrong number of fields");
            prob.keyframes = int(parseInteger(f[1]));
            sawKeyframes = true;
        } else if (key == "inverse") {
            if (f.size() < 2) fail("inverse line without count");
            const std::size_t count = std::size_t(parseInteger(f[1]));
            if (f.size() != 2 + count) fail("inverse line has the wrong number of values");
            prob.inverse.clear();
            for (std::size_t i = 0; i < count; ++i) prob.inverse.push_back(parseReal(f[2 + i]));
            sawInverse = true;
        } else if (key == "references") {
            if (f.size() != 2) fail("references line has the wrong number of fields");
            expectedReferences = std::size_t(parseInteger(f[1]));
            sawReferences = true;
        } else {
            fail("unknown problem line: " + key);
        }
    }
    if (!(sawFormat && sawGrid && sawLeft && sawRight && sawRL && sawRR && sawR1 && sawBaseline &&
          sawKeyframes && sawInverse && sawReferences))
        fail("problem.txt is missing a required record");
    if (prob.references.size() != expectedReferences) fail("reference count does not match the references line");
    return parsed;
}

std::vector<std::uint8_t> readRaw(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open " + path);
    std::vector<std::uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::size_t expected = std::size_t(FrameSpatial::MotionYaw::kRawWidth) * FrameSpatial::MotionYaw::kRawHeight;
    if (data.size() != expected) fail("raw frame has the wrong size: " + path);
    return data;
}

std::string referencePath(const std::filesystem::path& problemDirectory, const std::string& path) {
    const std::filesystem::path reference(path);
    if (!reference.is_relative()) return path;
    const auto relativeToProblem = problemDirectory / reference;
    return std::filesystem::exists(relativeToProblem) ? relativeToProblem.string() : path;
}

void writeSamples(const std::string& path, const std::vector<FrameSpatial::MotionYaw::Sample>& samples) {
    std::FILE* out = std::fopen(path.c_str(), "w");
    if (!out) fail("cannot write " + path);
    std::fprintf(out, "i,t_s,u,v,distance,expected,d0,d1,s0,sweep_score,dy0\n");
    for (const auto& s : samples) {
        std::fprintf(out, "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", s.i, s.tS, s.u, s.v,
                     s.distance, s.expected, s.d0, s.d1, s.s0, s.sweepScore, s.dy0);
    }
    if (std::fclose(out) != 0) fail("error closing " + path);
}

void writeTrace(const std::string& path, const std::vector<FrameSpatial::MotionYaw::KeyframeTrace>& trace) {
    std::FILE* out = std::fopen(path.c_str(), "w");
    if (!out) fail("cannot write " + path);
    for (const auto& kf : trace) {
        std::fprintf(out, "keyframe %d partners", kf.i);
        for (std::size_t j : kf.partners) std::fprintf(out, " %zu", j);
        std::fprintf(out, " corners %zu\n", kf.corners.size());
        for (const auto& c : kf.corners) std::fprintf(out, "%.17g %.17g\n", c[0], c[1]);
    }
    if (std::fclose(out) != 0) fail("error closing " + path);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: " << argv[0] << " problem.txt out_samples.csv [trace.txt]\n";
        return 2;
    }
    try {
        ParsedProblem parsed = readProblem(argv[1]);
        const Problem& problem = parsed.problem;
        const auto directory = std::filesystem::absolute(argv[1]).parent_path();
        const auto loader = [&parsed, &directory](std::size_t index, bool right) {
            const auto& path = right ? parsed.rightPaths[index] : parsed.leftPaths[index];
            return readRaw(referencePath(directory, path));
        };
        const auto result = FrameSpatial::MotionYaw::runMotionYaw(problem, loader);
        writeSamples(argv[2], result.samples);
        if (argc == 4) writeTrace(argv[3], result.trace);
        std::printf("reference frames %zu\n", problem.references.size());
        if (!result.summary.valid) {
            std::printf("no samples\n");
            return 0;
        }
        std::printf("samples %zu from %zu keyframes\n", result.summary.samples, result.summary.keyframes);
        std::printf("per_degree %.17g\n", result.summary.perDegree);
        std::printf("yaw %.17g\n", result.summary.yaw);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
