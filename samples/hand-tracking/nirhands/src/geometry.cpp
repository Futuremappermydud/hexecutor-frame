#include "nirhands/geometry.hpp"

#include <cctype>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace nirhands {

Vec3 KbCamera::unproject(const Vec2& uv) const {
    const double mx = (uv[0] - cx) / fx, my = (uv[1] - cy) / fy;
    const double rd = std::hypot(mx, my);
    double theta = rd;
    for (int i = 0; i < 8; ++i) {
        const double t2 = theta * theta;
        const double f = distort(theta) - rd;
        const double df = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])));
        theta -= f / df;
    }
    const double scale = rd > 1e-12 ? std::sin(theta) / rd : 1.0;
    return {mx * scale, my * scale, std::cos(theta)};
}

namespace {

// Minimal JSON: enough for the factory calibration files (objects, arrays, numbers, strings, literals).
struct Json {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    double number = 0;
    bool boolean = false;
    std::string string;
    std::vector<Json> array;
    std::map<std::string, Json> object;

    const Json& operator[](const std::string& key) const {
        auto it = object.find(key);
        if (type != Object || it == object.end()) throw std::runtime_error("calibration JSON: missing key " + key);
        return it->second;
    }
    bool has(const std::string& key) const { return type == Object && object.count(key); }
    double num() const {
        if (type != Number) throw std::runtime_error("calibration JSON: expected a number");
        return number;
    }
};

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}
    Json parse() {
        Json v = value();
        space();
        if (i_ != s_.size()) fail("trailing characters");
        return v;
    }

private:
    const std::string& s_;
    size_t i_ = 0;
    [[noreturn]] void fail(const char* what) {
        throw std::runtime_error(std::string("calibration JSON: ") + what + " at byte " + std::to_string(i_));
    }
    void space() { while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_; }
    bool eat(char c) { space(); if (i_ < s_.size() && s_[i_] == c) { ++i_; return true; } return false; }
    std::string str() {
        if (!eat('"')) fail("expected string");
        std::string out;
        while (i_ < s_.size() && s_[i_] != '"') {
            char c = s_[i_++];
            if (c == '\\') {
                if (i_ >= s_.size()) fail("bad escape");
                const char e = s_[i_++];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': i_ += 4; out += '?'; break;  // not needed for calibration values
                    default: out += e;
                }
            } else out += c;
        }
        if (i_ >= s_.size()) fail("unterminated string");
        ++i_;
        return out;
    }
    Json value() {
        space();
        if (i_ >= s_.size()) fail("unexpected end");
        Json v;
        const char c = s_[i_];
        if (c == '{') {
            ++i_;
            v.type = Json::Object;
            if (eat('}')) return v;
            do {
                std::string key = str();
                if (!eat(':')) fail("expected ':'");
                v.object[key] = value();
            } while (eat(','));
            if (!eat('}')) fail("expected '}'");
        } else if (c == '[') {
            ++i_;
            v.type = Json::Array;
            if (eat(']')) return v;
            do v.array.push_back(value()); while (eat(','));
            if (!eat(']')) fail("expected ']'");
        } else if (c == '"') {
            v.type = Json::String;
            v.string = str();
        } else if (s_.compare(i_, 4, "true") == 0) { v.type = Json::Bool; v.boolean = true; i_ += 4; }
        else if (s_.compare(i_, 5, "false") == 0) { v.type = Json::Bool; i_ += 5; }
        else if (s_.compare(i_, 4, "null") == 0) { i_ += 4; }
        else {
            const char* begin = s_.c_str() + i_;
            char* end = nullptr;
            v.number = std::strtod(begin, &end);
            if (end == begin) fail("bad value");
            v.type = Json::Number;
            i_ += size_t(end - begin);
        }
        return v;
    }
};

Json load_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    return Parser(text).parse();
}

Vec3 vec3(const Json& a) {
    if (a.type != Json::Array || a.array.size() != 3) throw std::runtime_error("calibration JSON: expected [x,y,z]");
    return {a.array[0].num(), a.array[1].num(), a.array[2].num()};
}

// Parent <- child from {plus_x, plus_z, position}; plus_y = plus_z x plus_x (session._pose).
Pose pose(const Json& d, double position_scale) {
    const Vec3 x = vec3(d["plus_x"]), z = vec3(d["plus_z"]);
    const Vec3 y = cross(z, x);
    Pose p;
    p.R = Mat3::rows(x, y, z).transposed();  // columns are the child axes
    p.t = position_scale * vec3(d["position"]);
    return p;
}

template<size_t N>
std::array<double, N> numbers(const Json& a) {
    if (a.type != Json::Array || a.array.size() != N)
        throw std::runtime_error("calibration JSON: wrong array length");
    std::array<double, N> out{};
    for (size_t i = 0; i < N; ++i) out[i] = a.array[i].num();
    return out;
}

Rig load(const std::string& xrservice_json_path, const std::string& device_config_json_path,
         std::array<KbSplineCalibration, 2>* splines) {
    const Json cal = load_json(xrservice_json_path);
    const Json dev = load_json(device_config_json_path);
    std::map<std::string, const Json*> entries;
    for (const auto& c : cal["cameras"].array) entries[c["sourceCamera"].string] = &c;
    for (const char* name : {"slam_left", "slam_right"})
        if (!entries.count(name)) throw std::runtime_error(std::string("calibration has no ") + name);
    // Extrinsics are in the slam_right (Cam0) frame, millimetres.
    const Pose right_from_left = pose((*entries["slam_left"])["extrinsics"], 1e-3);
    const Pose cad_from_right = pose(dev["cv"]["cad_from_cal"], 1.0);
    const Pose cad_from_head = pose(dev["head"], 1.0);
    const Pose head_from_right = cad_from_head.inverse() * cad_from_right;
    Rig rig;
    for (const char* name : {"slam_left", "slam_right"}) {
        const bool left = std::string(name) == "slam_left";
        const Json& e = *entries[name];
        const Json* kb = nullptr;
        const Json* spline = nullptr;
        for (const auto& intr : e["intrinsics"].array) {
            if (intr["cameraModel"].string == "kb") kb = &intr;
            if (intr["cameraModel"].string == "kb_spline") spline = &intr;
        }
        if (!kb) throw std::runtime_error(std::string("no Kannala-Brandt intrinsics for ") + name);
        KbCamera cam;
        cam.fx = (*kb)["fx"].num(); cam.fy = (*kb)["fy"].num();
        cam.cx = (*kb)["cx"].num(); cam.cy = (*kb)["cy"].num();
        cam.k = {(*kb)["k1"].num(), (*kb)["k2"].num(), (*kb)["k3"].num(), (*kb)["k4"].num()};
        cam.width = int(e["width"].num());
        cam.height = int(e["height"].num());
        const Pose right_from_cam = left ? right_from_left : Pose{};
        const Pose cam_from_head = (head_from_right * right_from_cam).inverse();
        cam.R_cw = cam_from_head.R;
        cam.t_cw = cam_from_head.t;
        (left ? rig.left : rig.right) = cam;
        if (splines) {
            if (!spline || numbers<2>((*spline)["gridSize"]) != std::array<double, 2>{5, 5})
                throw std::runtime_error(std::string("no 5x5 KB spline intrinsics for ") + name);
            auto& out = (*splines)[left ? 0 : 1];
            out.fx = (*spline)["fx"].num(); out.fy = (*spline)["fy"].num();
            out.cx = (*spline)["cx"].num(); out.cy = (*spline)["cy"].num();
            for (size_t i = 0; i < out.k.size(); ++i) {
                const auto key = "k" + std::to_string(i + 1);
                if (spline->has(key)) out.k[i] = (*spline)[key].num();
            }
            out.spacing = numbers<2>((*spline)["nodeSpacing"]);
            out.sU = numbers<25>((*spline)["sU"]);
            out.sV = numbers<25>((*spline)["sV"]);
        }
    }
    return rig;
}

}  // namespace

Rig load_rig(const std::string& xrservice_json_path, const std::string& device_config_json_path) {
    return load(xrservice_json_path, device_config_json_path, nullptr);
}

Rig load_rig(const std::string& xrservice_json_path, const std::string& device_config_json_path,
             std::array<KbSplineCalibration, 2>& splines) {
    return load(xrservice_json_path, device_config_json_path, &splines);
}

}  // namespace nirhands
