// Fit a yaw scan from replayed matches, as frame_yaw_viewer does after recording.
//
//   replay_ir_motion_yaw PROBLEM.txt SAMPLES.csv
//   fit_yaw_scan PROBLEM.txt SAMPLES.csv
//
// PROBLEM.txt supplies the grid step and the reference times that split the recording in
// half. Prints one JSON object. Comparing the problem-offset-*.txt files the viewer writes
// shows which exposure offset makes motion and stereo depth agree best.

#include "yaw_scan.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace FrameSpatial;

std::vector<std::string> fields(const std::string& line, char separator) {
    std::vector<std::string> out;
    std::stringstream in(line);
    std::string field;
    while (std::getline(in, field, separator)) out.push_back(field);
    return out;
}

std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    char text[40];
    std::snprintf(text, sizeof text, "%.6f", value);
    return text;
}

std::string number(const std::optional<double>& value) { return value ? number(*value) : "null"; }

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " PROBLEM.txt SAMPLES.csv\n";
        return 2;
    }
    try {
        std::ifstream problem(argv[1]);
        if (!problem) throw std::runtime_error(std::string("cannot read ") + argv[1]);
        double step = 0;
        int64_t first = 0, last = 0;
        bool any = false;
        for (std::string line; std::getline(problem, line);) {
            if (line.rfind("grid ", 0) == 0) {
                const auto f = fields(line, ' ');
                if (f.size() < 4) throw std::runtime_error("bad grid line");
                step = std::stod(f[3]);
            } else if (line.rfind("ref\t", 0) == 0) {
                const int64_t when = std::stoll(fields(line, '\t').at(1));
                if (!any) first = when;
                last = when;
                any = true;
            }
        }
        if (!(step > 0) || !any) throw std::runtime_error("problem has no grid step or references");

        std::ifstream csv(argv[2]);
        if (!csv) throw std::runtime_error(std::string("cannot read ") + argv[2]);
        std::string header;
        std::getline(csv, header);
        const auto names = fields(header, ',');
        auto column = [&names](const char* name) {
            for (size_t i = 0; i < names.size(); ++i)
                if (names[i] == name) return int(i);
            return -1;
        };
        const int dyColumn = column("dy0");
        std::vector<MotionYaw::Sample> samples;
        double sweep = 0;
        for (std::string line; std::getline(csv, line);) {
            const auto f = fields(line, ',');
            if (f.size() < 10) continue;
            MotionYaw::Sample s;
            s.i = std::stoi(f[0]);
            s.tS = std::stod(f[1]); s.u = std::stod(f[2]); s.v = std::stod(f[3]); s.distance = std::stod(f[4]);
            s.expected = std::stod(f[5]); s.d0 = std::stod(f[6]); s.d1 = std::stod(f[7]); s.s0 = std::stod(f[8]);
            s.sweepScore = std::stod(f[9]);
            if (dyColumn >= 0 && size_t(dyColumn) < f.size() && !f[dyColumn].empty()) {
                try { s.dy0 = std::stod(f[dyColumn]); } catch (const std::exception&) {}
            }
            sweep += s.sweepScore;
            samples.push_back(s);
        }
        const auto fit = fitYawScan(samples, step, double(last - first) * .5e-9);
        static const char* states[] = {"more_support", "need_near_and_far", "unstable", "ready"};
        std::cout << "{\"state\": \"" << states[int(fit.state)] << "\", \"samples\": " << fit.samples
                  << ", \"keyframes\": " << fit.keyframes
                  << ", \"mean_sweep_score\": " << number(samples.empty() ? NAN : sweep / double(samples.size()))
                  << ", \"yaw\": " << number(fit.yawDegrees) << ", \"roll\": " << number(fit.rollDegrees)
                  << ", \"roll_fitted\": " << (fit.rollFitted ? "true" : "false")
                  << ", \"yaw_without_roll\": " << number(fit.yawWithoutRollDegrees)
                  << ", \"residual_scale\": " << number(fit.residualScaleDegrees)
                  << ", \"time_split\": " << number(fit.splitDifferenceDegrees)
                  << ", \"far_minus_near\": " << number(fit.depthDifferenceDegrees)
                  << ", \"row_roll\": " << (fit.rows ? number(fit.rows->rollDegrees) : "null")
                  << ", \"row_pitch\": " << (fit.rows ? number(fit.rows->pitchDegrees) : "null")
                  << ", \"row_yaw\": " << (fit.rows ? number(fit.rows->yawDegrees) : "null")
                  << ", \"row_yaw_calibrated\": " << (fit.rows ? number(fit.rows->calibratedYawDegrees) : "null")
                  << ", \"row_residual_px\": " << (fit.rows ? number(fit.rows->residualPx) : "null")
                  << ", \"free_scale_yaw\": " << (fit.freeScale ? number(fit.freeScale->yawDegrees) : "null")
                  << ", \"free_scale_term\": " << (fit.freeScale ? number(fit.freeScale->scaleTerm) : "null")
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
