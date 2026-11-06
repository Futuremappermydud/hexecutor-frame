#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <vector>

namespace FrameSpatial {
inline bool surfaceDepth(float z) { return std::isfinite(z) && z >= 0.25f && z <= 5.0f; }
inline bool surfaceRange(float lo, float hi) {
    return hi - lo <= std::max(0.04f, lo * 0.04f);
}

// Average inverse depth on the center's side of an edge. Invalid samples stay
// invalid; hole filling below still requires four measured brackets. A compact
// separable kernel reduces stereo corrugation without flattening slanted planes.
inline std::vector<float> smoothOcclusionDepth(const std::vector<float>& depth,
                                             int width, int height) {
    std::vector<float> smoothed(depth.size(), 0.0f);
    if (width <= 0 || height <= 0 || depth.size() != size_t(width) * height) return smoothed;
    constexpr int weights[5]{1, 2, 3, 2, 1};
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const size_t i = size_t(y) * width + x;
        const float center = depth[i];
        if (!surfaceDepth(center)) continue;
        float inverseSum = 0.0f, weightSum = 0.0f;
        int support = 0;
        for (int dy = -2; dy <= 2; ++dy) for (int dx = -2; dx <= 2; ++dx) {
            const int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            const float z = depth[size_t(ny) * width + nx];
            if (!surfaceDepth(z) || !surfaceRange(std::min(center, z), std::max(center, z))) continue;
            const float weight = float(weights[dx + 2] * weights[dy + 2]);
            inverseSum += weight / z;
            weightSum += weight;
            if (dx || dy) ++support;
        }
        if (support >= 3) smoothed[i] = weightSum / inverseSum;
    }
    return smoothed;
}

// Previous rectified optical coordinates -> current rectified optical coordinates.
// Construct this from capture-time head poses AND the calibrated camera-to-head
// transform, rather than averaging the same image pixel during head motion.
struct OcclusionReprojection {
    std::array<std::array<float, 3>, 3> rotation{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    std::array<float, 3> translation{};
};

struct OcclusionObservation {
    float depth = 0;
    float age = 0;
    unsigned char observations = 0;
    unsigned char evidence = 0;
    // Source stereo age is distinct from agreeing prediction age. Inference may
    // renew it only when the caller supplies the age of its measured anchors.
    float supportAge = 0;
    float directMeasurementAge = 0;
};

// Evidence is provenance/local support, not a calibrated probability. Inferred
// depths never inherit measured confidence merely by agreeing with a plane.
enum OcclusionEvidence : uint8_t { UnknownDepth=0, InferredDepth=1, MeasuredDepth=2, StrongDepth=3 };
inline std::vector<uint8_t> occlusionEvidence(const std::vector<float>& surface,
                                            const std::vector<float>& anchors,
                                            int width, int height) {
    std::vector<uint8_t> evidence(surface.size(),UnknownDepth);
    if(width<=0 || height<=0 || surface.size()!=size_t(width)*height ||
       anchors.size()!=surface.size()) return evidence;
    #pragma omp parallel for
    for(int y=0;y<height;++y) for(int x=0;x<width;++x) {
        const size_t i=size_t(y)*width+x;
        const float z=surface[i];
        if(!surfaceDepth(z)) continue;
        evidence[i]=InferredDepth;
        if(!surfaceDepth(anchors[i]) ||
           !surfaceRange(std::min(z,anchors[i]),std::max(z,anchors[i]))) continue;
        evidence[i]=MeasuredDepth;
        int support=0;
        for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
            if(!(dx||dy)) continue;
            const int nx=x+dx,ny=y+dy;
            if(nx<0 || nx>=width || ny<0 || ny>=height) continue;
            const float neighbor=anchors[size_t(ny)*width+nx];
            if(surfaceDepth(neighbor) &&
               surfaceRange(std::min(z,neighbor),std::max(z,neighbor))) ++support;
        }
        if(support>=6) evidence[i]=StrongDepth;
    }
    return evidence;
}

// Diagnostic events are engineering attribution, not calibrated confidence.
// ReprojectionLoss is indexed in the PREVIOUS image; all other flags use the
// current image. A collision marks the destination for either winning source.
enum OcclusionHistoryEvent : uint16_t {
    HistoryPending=1u<<0, HistoryConfirmed=1u<<1, HistoryMeasuredClear=1u<<2,
    HistoryPredictionExpiry=1u<<3, HistorySourceExpiry=1u<<4,
    HistoryReprojectionLoss=1u<<5, HistoryCollision=1u<<6,
    HistoryMeasuredAgreement=1u<<7, HistoryInferredAgreement=1u<<8,
    HistoryEvidenceChange=1u<<9, HistoryHeld=1u<<10
};

// Reproject observations and their age together. Expiry is tied to genuine
// measurements: transporting an old sample never makes it young again.
inline std::vector<OcclusionObservation> reprojectOcclusionObservations(
    const std::vector<OcclusionObservation>& previous, int width, int height,
    float focal, float cx, float cy, const OcclusionReprojection& transform,
    std::vector<uint16_t>* events = nullptr) {
    std::vector<OcclusionObservation> projected(size_t(width) * height);
    if (previous.size() != projected.size()) return projected;
    if(events && events->size()!=projected.size()) events->assign(projected.size(),0);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const auto& observation = previous[size_t(y) * width + x];
        const float z = observation.depth;
        if (!surfaceDepth(z)) continue;
        const std::array<float, 3> p{(x - cx) * z / focal, (y - cy) * z / focal, z};
        auto q = transform.translation;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) q[row] += transform.rotation[row][col] * p[col];
        const auto lost=[&] {
            if(events) (*events)[size_t(y)*width+x]|=HistoryReprojectionLoss;
        };
        if (!surfaceDepth(q[2])) { lost(); continue; }
        const float px = focal * q[0] / q[2] + cx, py = focal * q[1] / q[2] + cy;
        if (!std::isfinite(px) || !std::isfinite(py) || px < 0 || px > width - 1 ||
            py < 0 || py > height - 1) { lost(); continue; }
        const size_t destination=size_t(std::lround(py))*width+std::lround(px);
        auto& target = projected[destination];
        if(events && surfaceDepth(target.depth)) (*events)[destination]|=HistoryCollision;
        if (!surfaceDepth(target.depth) || q[2] < target.depth) {
            target = observation;
            target.depth = q[2];
        }
    }
    return projected;
}

// Replay tuning changes only confidence mode; legacy behavior stays fixed.
struct OcclusionConfidenceConfig {
    int strongObservations = 2;
    int measuredObservations = 3;
    int inferredObservations = 4;
    float inferredMaxAge = 0.15f;
    float measuredMaxAge = 0.20f;
    float pendingMaxAge = 0.20f;
};

// Coverage needs hysteresis, not just averaging: reject transient patches and
// keep confirmed geometry across brief stereo dropouts. A separate candidate
// prevents one erroneous depth jump from displacing an established surface.
class OcclusionSurfaceHistory {
public:
    explicit OcclusionSurfaceHistory(OcclusionConfidenceConfig config = {}, float legacyMaxAge = 0.2f)
        : config_(config), legacyMaxAge_(legacyMaxAge) {
        if (!std::isfinite(legacyMaxAge) || legacyMaxAge <= 0 || legacyMaxAge > 3)
            throw std::invalid_argument("Invalid legacy occlusion maximum age");
        const auto validCount=[](int count) { return count>=1 && count<=255; };
        const auto validAge=[](float age) { return std::isfinite(age) && age>0 && age<=1; };
        if(!validCount(config.strongObservations) || !validCount(config.measuredObservations) ||
           !validCount(config.inferredObservations) || !validAge(config.inferredMaxAge) ||
           !validAge(config.measuredMaxAge) || !validAge(config.pendingMaxAge))
            throw std::invalid_argument("Invalid occlusion confidence configuration");
    }
    void reset() {
        stable_.clear(); pending_.clear(); lifetimes_.clear(); events_.clear(); width_ = height_ = 0;
        confirmed = pending = held = cleared = 0;
    }
    size_t confirmed = 0, pending = 0, held = 0;
    size_t cleared = 0;
    const std::vector<float>& supportLifetimes() const { return lifetimes_; }
    void enableDiagnostics(bool enabled) { diagnostics_=enabled; if(!enabled) events_.clear(); }
    const std::vector<uint16_t>& events() const { return events_; }

    // provenance preserves the legacy policy while describing real vs inferred
    // input. anchorAges gives seconds since the measured anchors supporting each
    // prediction; inferred agreement alone never refreshes source support.
    std::vector<float> update(const std::vector<float>& measured,
                             int width, int height, float focal, float cx, float cy,
                             const OcclusionReprojection* transform, float elapsed,
                             const std::vector<uint8_t>* evidence = nullptr,
                             const std::vector<uint8_t>* provenance = nullptr,
                             const std::vector<float>* anchorAges = nullptr) {
        confirmed = pending = held = cleared = 0;
        if (width <= 0 || height <= 0 || measured.size() != size_t(width) * height ||
            !std::isfinite(focal) || focal <= 0 || !std::isfinite(cx) || !std::isfinite(cy)) {
            reset();
            return std::vector<float>(measured.size(), 0);
        }
        if(evidence && (evidence->size()!=measured.size() ||
            std::any_of(evidence->begin(),evidence->end(),[](uint8_t e){return e>StrongDepth;}))) {
            reset();
            return std::vector<float>(measured.size(),0);
        }
        if ((provenance && (provenance->size()!=measured.size() ||
             std::any_of(provenance->begin(),provenance->end(),[](uint8_t e){return e>StrongDepth;}))) ||
            (anchorAges && (anchorAges->size()!=measured.size() ||
             std::any_of(anchorAges->begin(),anchorAges->end(),[](float a){return !std::isfinite(a) || a<0;})))) {
            reset();
            return std::vector<float>(measured.size(),0);
        }
        if (!transform || width != width_ || height != height_ ||
            !std::isfinite(elapsed) || elapsed <= 0 || elapsed > legacyMaxAge_) {
            reset();
            elapsed = 0;
        }
        width_ = width; height_ = height;
        if(diagnostics_) events_.assign(measured.size(),0);
        if (stable_.empty()) {
            stable_.resize(measured.size()); pending_.resize(measured.size());
        } else {
            stable_ = reprojectOcclusionObservations(stable_, width, height, focal, cx, cy, *transform, diagnostics_?&events_:nullptr);
            pending_ = reprojectOcclusionObservations(pending_, width, height, focal, cx, cy, *transform, diagnostics_?&events_:nullptr);
        }
        std::vector<float> surface(measured.size(), 0);
        lifetimes_.assign(measured.size(),0);
        for (size_t i = 0; i < measured.size(); ++i) {
            auto& stable = stable_[i];
            auto& candidate = pending_[i];
            stable.age += elapsed;
            candidate.age += elapsed;
            stable.supportAge += elapsed; candidate.supportAge += elapsed;
            stable.directMeasurementAge += elapsed; candidate.directMeasurementAge += elapsed;
            const auto maxAge=[this,evidence](const OcclusionObservation& observation) {
                if (!evidence) return legacyMaxAge_;
                return observation.evidence==InferredDepth ? config_.inferredMaxAge : config_.measuredMaxAge;
            };
            const auto event=[&](uint16_t flag) { if(diagnostics_) events_[i]|=flag; };
            if (stable.age > maxAge(stable)) {
                if(surfaceDepth(stable.depth)) {
                    event(HistoryPredictionExpiry);
                    if(stable.supportAge>=maxAge(stable)) event(HistorySourceExpiry);
                }
                stable = {};
            }
            if (candidate.age > (evidence ? config_.pendingMaxAge : legacyMaxAge_)) candidate = {};
            const float z = measured[i];
            const uint8_t quality=evidence?(*evidence)[i]:uint8_t(MeasuredDepth);
            const uint8_t source=provenance?(*provenance)[i]:quality;
            // Completion derived from current measured anchors may renew source
            // support, but must not be mistaken for a direct stereo observation.
            const auto refreshSupport=[&](OcclusionObservation& observation) {
                if(source>=MeasuredDepth) {
                    observation.supportAge=0; observation.directMeasurementAge=0;
                } else if(source==InferredDepth && anchorAges) {
                    observation.supportAge=(*anchorAges)[i];
                }
            };
            const auto agrees = [z](float old) {
                return surfaceDepth(old) && surfaceRange(std::min(z, old), std::max(z, old));
            };
            if (surfaceDepth(z) && quality!=UnknownDepth) {
                if (agrees(stable.depth)) {
                    // Stronger smoothing now has persistent, confirmed geometry
                    // behind it, rather than a fresh single-frame mesh.
                    const float weight=evidence?(quality==StrongDepth?0.45f:
                                                 quality==MeasuredDepth?0.30f:0.10f):0.15f;
                    stable.depth = 1.0f / (weight / z + (1-weight) / stable.depth);
                    event(source>=MeasuredDepth?HistoryMeasuredAgreement:HistoryInferredAgreement);
                    if(stable.evidence!=quality) event(HistoryEvidenceChange);
                    stable.evidence=quality;
                    stable.age = 0;
                    refreshSupport(stable);
                    candidate = {};
                } else {
                    // A real measurement contradicting an old occluder clears
                    // it now. Confirmation may delay its replacement, but the
                    // stale foreground must not continue hiding virtual pixels.
                    if(evidence && quality>=MeasuredDepth && surfaceDepth(stable.depth)) {
                        stable={}; ++cleared; event(HistoryMeasuredClear);
                    }
                    if (agrees(candidate.depth)) {
                        candidate.depth = 2.0f / (1.0f / z + 1.0f / candidate.depth);
                        candidate.observations=std::min(255,int(candidate.observations)+1);
                        candidate.evidence=std::min(candidate.evidence,quality);
                        candidate.age = 0;
                    } else {
                        candidate = {z, 0, 1, quality};
                        candidate.directMeasurementAge=source>=MeasuredDepth?0:std::numeric_limits<float>::infinity();
                    }
                    refreshSupport(candidate);
                    const int required=evidence?(candidate.evidence==StrongDepth?config_.strongObservations:
                                                  candidate.evidence==InferredDepth?config_.inferredObservations:config_.measuredObservations):3;
                    if (candidate.observations >= required) {
                        stable = candidate;
                        candidate = {};
                        event(HistoryConfirmed);
                    }
                }
            }
            if (surfaceDepth(stable.depth)) {
                surface[i] = stable.depth;
                const float sourceMaxAge=maxAge(stable);
                lifetimes_[i]=std::max(0.0f,std::min(maxAge(stable)-stable.age,
                                                  sourceMaxAge-stable.supportAge));
                if(stable.supportAge>=sourceMaxAge) event(HistorySourceExpiry);
                ++confirmed;
                if (stable.age > 0) { ++held; event(HistoryHeld); }
            }
            if (surfaceDepth(candidate.depth)) { ++pending; event(HistoryPending); }
        }
        return surface;
    }

private:
    OcclusionConfidenceConfig config_;
    float legacyMaxAge_ = 0.20f;
    int width_ = 0, height_ = 0;
    std::vector<OcclusionObservation> stable_, pending_;
    std::vector<float> lifetimes_;
    bool diagnostics_=false;
    std::vector<uint16_t> events_;
};

inline std::vector<float> stabilizeOcclusionDepth(const std::vector<float>& current,
                                                const std::vector<float>& previous,
                                                int width, int height, float focal,
                                                float cx, float cy,
                                                const OcclusionReprojection& transform) {
    auto stable = current;
    if (width <= 0 || height <= 0 || current.size() != size_t(width) * height ||
        previous.size() != current.size() || !std::isfinite(focal) || focal <= 0 ||
        !std::isfinite(cx) || !std::isfinite(cy)) return stable;
    std::vector<float> projected(current.size(), 0.0f);
    // A nearest-depth buffer prevents a background observation from winning a
    // reprojection collision at a foreground silhouette. No radius or dilation.
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const float z = previous[size_t(y) * width + x];
        if (!surfaceDepth(z)) continue;
        const std::array<float, 3> p{(x - cx) * z / focal, (y - cy) * z / focal, z};
        auto q = transform.translation;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) q[row] += transform.rotation[row][col] * p[col];
        if (!surfaceDepth(q[2])) continue;
        const float px = focal * q[0] / q[2] + cx, py = focal * q[1] / q[2] + cy;
        if (!std::isfinite(px) || !std::isfinite(py) || px < 0 || px > width - 1 ||
            py < 0 || py > height - 1) continue;
        const size_t i = size_t(std::lround(py)) * width + std::lround(px);
        if (!surfaceDepth(projected[i]) || q[2] < projected[i]) projected[i] = q[2];
    }
    #pragma omp parallel for
    for (size_t i = 0; i < current.size(); ++i) {
        const float z = current[i], old = projected[i];
        if (!surfaceDepth(z) || !surfaceDepth(old) ||
            !surfaceRange(std::min(z, old), std::max(z, old))) continue;
        // 30% new observation: suppress jitter, but accept changed surfaces
        // immediately when they disagree with history. Never resurrect holes.
        stable[i] = 1.0f / (0.30f / z + 0.70f / old);
    }
    return stable;
}

// Explicit rejection reasons survive plane completion and the enclosed-hole pass.
// Provenance is spatial inference provenance, separate from temporal evidence.
enum OcclusionCompletionVeto : uint8_t {
    CompletionOverlapConflict = 1,
    CompletionContradictorySupport = 2,
    CompletionProtectedBoundary = 4
};
enum OcclusionCompletionProvenance : uint8_t {
    CompletionUnknown = 0, CompletionMeasured = 1,
    CompletionPlane = 2, CompletionHole = 3
};
struct OcclusionCompletionContract {
    std::vector<uint8_t> veto, provenance;
};

// Preserve plane-completed output, but use only smoothed measured brackets.
// Stop bracket searches at an explicit veto or a guide/depth boundary. Neither
// completed pixels nor newly filled holes seed another fill.
inline std::vector<float> fillOcclusionHoles(
    const std::vector<float>& depth, const std::vector<float>& measured,
    int width, int height, OcclusionCompletionContract& contract,
    const std::vector<uint8_t>& guide, int radius = 8) {
    auto filled = depth;
    if (width <= 0 || height <= 0 || depth.size() != size_t(width)*height ||
        measured.size() != depth.size() || guide.size() != depth.size() ||
        contract.veto.size() != depth.size() || contract.provenance.size() != depth.size())
        return filled;
    const int dx[4]{-1, 1, 0, 0}, dy[4]{0, 0, -1, 1};
    for (int y=0; y<height; ++y) for (int x=0; x<width; ++x) {
        const size_t i=size_t(y)*width+x;
        if (surfaceDepth(depth[i]) || contract.veto[i]) continue;
        float z[4]{}; int distance[4]{};
        bool enclosed=true;
        for (int direction=0; direction<4; ++direction) {
            size_t previous=i;
            for (int step=1; step<=radius; ++step) {
                const int nx=x+dx[direction]*step, ny=y+dy[direction]*step;
                if (nx<0 || nx>=width || ny<0 || ny>=height) break;
                const size_t next=size_t(ny)*width+nx;
                if (contract.veto[next] ||
                    std::abs(int(guide[next])-int(guide[previous]))>16 ||
                    (surfaceDepth(measured[previous]) && surfaceDepth(measured[next]) &&
                     !surfaceRange(std::min(measured[previous],measured[next]),
                                   std::max(measured[previous],measured[next])))) break;
                if (surfaceDepth(measured[next])) {
                    z[direction]=measured[next]; distance[direction]=step; break;
                }
                previous=next;
            }
            if (!distance[direction]) { enclosed=false; break; }
        }
        if (!enclosed) continue;
        const auto limits=std::minmax_element(z,z+4);
        if (!surfaceRange(*limits.first,*limits.second)) continue;
        const float horizontal=(distance[1]/z[0]+distance[0]/z[1])/(distance[0]+distance[1]);
        const float vertical=(distance[3]/z[2]+distance[2]/z[3])/(distance[2]+distance[3]);
        filled[i]=2.0f/(horizontal+vertical);
        contract.provenance[i]=CompletionHole;
    }
    return filled;
}

// Legacy/measurement-only helper: callers must supply original measurements.
// The completion worker uses the explicit output/bracket-source overload above.
inline std::vector<float> fillOcclusionHoles(const std::vector<float>& depth,
                                          int width, int height, int radius = 8) {
    auto filled = depth;
    if (width <= 0 || height <= 0 || depth.size() != size_t(width) * height) return filled;
    const int dx[4]{-1, 1, 0, 0}, dy[4]{0, 0, -1, 1};
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const size_t i = size_t(y) * width + x;
        if (surfaceDepth(depth[i])) continue;
        float z[4]{}; int distance[4]{};
        bool enclosed = true;
        for (int direction = 0; direction < 4; ++direction) {
            for (int step = 1; step <= radius; ++step) {
                const int nx = x + dx[direction] * step, ny = y + dy[direction] * step;
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) break;
                const float value = depth[size_t(ny) * width + nx];
                if (surfaceDepth(value)) { z[direction] = value; distance[direction] = step; break; }
            }
            if (!distance[direction]) { enclosed = false; break; }
        }
        if (!enclosed) continue;
        const auto limits = std::minmax_element(z, z + 4);
        if (!surfaceRange(*limits.first, *limits.second)) continue;
        // Inverse depth interpolates a planar surface under perspective.
        const float horizontal = (distance[1] / z[0] + distance[0] / z[1]) /
                                 (distance[0] + distance[1]);
        const float vertical = (distance[3] / z[2] + distance[2] / z[3]) /
                               (distance[2] + distance[3]);
        filled[i] = 2.0f / (horizontal + vertical);
    }
    return filled;
}

// Two-pixel cells reduce GPU work. Every covered sample must agree. Rejected
// coarse cells are subdivided into independently validated one-pixel cells;
// valid portions survive without bridging an unknown sample or a depth edge.
struct OcclusionMeshStats {
    size_t coarseAccepted = 0;
    size_t missingRejected = 0;
    size_t spreadRejected = 0;
    size_t recoveredSubcells = 0;
};
inline std::vector<size_t> occlusionSurfaceIndices(const std::vector<float>& depth,
                                                 int width, int height,
                                                 OcclusionMeshStats* stats = nullptr) {
    std::vector<size_t> indices;
    if (stats) *stats = {};
    if (width <= 0 || height <= 0 || depth.size() != size_t(width) * height) return indices;
    indices.reserve(size_t(width / 2) * (height / 2) * 6);
    const auto validate = [&](int x, int y, int size, bool& missing) {
        float lo = 5.0f, hi = 0.0f;
        missing = false;
        for (int dy = 0; dy <= size; ++dy) for (int dx = 0; dx <= size; ++dx) {
            const float z = depth[size_t(y + dy) * width + x + dx];
            if (!surfaceDepth(z)) { missing = true; continue; }
            lo = std::min(lo, z); hi = std::max(hi, z);
        }
        return !missing && surfaceRange(lo, hi);
    };
    const auto emit = [&](int x, int y, int size) {
        const size_t a = size_t(y) * width + x, b = a + size;
        const size_t c = a + size_t(size) * width, d = c + size;
        indices.insert(indices.end(), {a, b, d, a, d, c});
    };
    for (int y = 0; y + 2 < height; y += 2) for (int x = 0; x + 2 < width; x += 2) {
        bool missing = false;
        if (validate(x, y, 2, missing)) {
            emit(x, y, 2);
            if (stats) ++stats->coarseAccepted;
        } else {
            if (stats) { if (missing) ++stats->missingRejected; else ++stats->spreadRejected; }
            for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx)
                if (validate(x + dx, y + dy, 1, missing)) {
                    emit(x + dx, y + dy, 1);
                    if (stats) ++stats->recoveredSubcells;
                }
        }
    }
    return indices;
}

// The emitted six-index rectangles can have side length one or two. Counting
// each as four pixels would overstate support after conservative subdivision.
inline size_t occlusionSurfaceArea(const std::vector<size_t>& indices, int width) {
    if (width <= 0 || indices.size() % 6) return 0;
    size_t area = 0;
    for (size_t i = 0; i < indices.size(); i += 6) {
        const size_t a = indices[i], b = indices[i + 1], c = indices[i + 5];
        area += (b % width - a % width) * (c / width - a / width);
    }
    return area;
}

inline std::vector<uint8_t> occlusionSurfaceMask(const std::vector<size_t>& indices,
                                               int width, int height) {
    if (width <= 0 || height <= 0) return {};
    std::vector<uint8_t> mask(size_t(width) * height, 0);
    if (indices.size() % 6) return mask;
    for (size_t i = 0; i < indices.size(); i += 6) {
        const size_t a = indices[i], b = indices[i + 1], c = indices[i + 5];
        if (a >= mask.size() || b >= mask.size() || c >= mask.size()) continue;
        for (size_t y = a / width; y < c / width; ++y)
            for (size_t x = a % width; x < b % width; ++x) mask[y * width + x] = 1;
    }
    return mask;
}

// Each emitted rectangle's triangles share the oldest covered sample deadline.
// Interior/edge samples constrain coarse triangles even when not GPU vertices.
inline std::vector<float> occlusionSurfaceCellLifetimes(const std::vector<float>& depth,
                                                       const std::vector<float>& lifetimes,
                                                       int width, int height,
                                                       const std::vector<size_t>& indices) {
    if (width <= 0 || height <= 0 || depth.size() != size_t(width) * height ||
        lifetimes.size() != depth.size() || indices.size() % 6) return {};
    std::vector<float> cells(indices.size()/6,0);
    for (size_t i = 0; i < indices.size(); i += 6) {
        const size_t a = indices[i], b = indices[i + 1], c = indices[i + 5];
        if (a >= depth.size() || b >= depth.size() || c >= depth.size() ||
            a % width >= b % width || a / width >= c / width) continue;
        float minimum=std::numeric_limits<float>::infinity();
        for (size_t y = a / width; y <= c / width; ++y)
            for (size_t x = a % width; x <= b % width; ++x) {
                const float remaining = lifetimes[y * width + x];
                if (!surfaceDepth(depth[y * width + x]) || !std::isfinite(remaining) || remaining <= 0)
                    minimum=0;
                else minimum=std::min(minimum,remaining);
            }
        cells[i/6]=minimum;
    }
    return cells;
}

inline float occlusionSurfaceMinLifetime(const std::vector<float>& depth,
                                        const std::vector<float>& lifetimes,
                                        int width, int height,
                                        const std::vector<size_t>* mesh = nullptr) {
    const auto generated=mesh?std::vector<size_t>{}:occlusionSurfaceIndices(depth,width,height);
    const auto cells=occlusionSurfaceCellLifetimes(depth,lifetimes,width,height,mesh?*mesh:generated);
    return cells.empty()?0:*std::min_element(cells.begin(),cells.end());
}
} // namespace FrameSpatial

