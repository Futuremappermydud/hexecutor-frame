#include "plane_patches.hpp"

#include <algorithm>
#include <cmath>
#include <deque>

namespace FrameSpatial {
namespace {

constexpr double kPi = 3.14159265358979323846;

double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Plane from raw sums: n points, s = sum p, q = sum p p^T. Returns smallest eigenvalue / n.
double planeFromSums(double n, const double s[3], const double q[3][3], double normal[3], double mean[3]) {
    for (int i = 0; i < 3; ++i) mean[i] = s[i] / n;
    double cov[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) cov[i][j] = q[i][j] / n - mean[i] * mean[j];
    return std::max(0.0, smallestEigen(cov, normal));
}

void orientTowards(double normal[3], const double from[3], const double to[3]) {
    const double d[3]{to[0] - from[0], to[1] - from[1], to[2] - from[2]};
    if (dot3(normal, d) < 0)
        for (int i = 0; i < 3; ++i) normal[i] = -normal[i];
}

}  // namespace

double smallestEigen(const double m[3][3], double vector[3]) {
    double a[3][3], v[3][3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) a[i][j] = m[i][j];
    for (int sweep = 0; sweep < 32; ++sweep) {
        const double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        if (off < 1e-30) break;
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) {
                if (std::fabs(a[p][q]) < 1e-300) continue;
                const double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
                const double t = (theta >= 0 ? 1 : -1) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
                const double c = 1 / std::sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < 3; ++k) {
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {
                    const double vkp = v[k][p], vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
    }
    int best = 0;
    for (int i = 1; i < 3; ++i)
        if (a[i][i] < a[best][best]) best = i;
    for (int i = 0; i < 3; ++i) vector[i] = v[i][best];
    const double norm = std::sqrt(dot3(vector, vector));
    for (int i = 0; i < 3; ++i) vector[i] /= norm;
    return a[best][best];
}

PlanePatchMap::PlanePatchMap(const PlanePatchConfig& config) : config_(config) {}

double PlanePatchMap::sigmaZ(double range) const {
    return range * range * config_.disparitySigmaPx * config_.sphericalStepRad / config_.baselineM;
}

double PlanePatchMap::surfaceSigma(double range) const {
    return range * range * config_.surfaceSigmaPx * config_.sphericalStepRad / config_.baselineM;
}

int64_t PlanePatchMap::voxelKey(const double p[3]) const {
    const auto cell = [&](double x) { return (int64_t(std::floor(x / config_.footprintVoxelM)) + (1 << 20)) & 0x1fffff; };
    return (cell(p[0]) << 42) | (cell(p[1]) << 21) | cell(p[2]);
}

std::vector<FramePatch> PlanePatchMap::detect(const float* disparity, const uint8_t* uncertainty,
                                              const PlanePose& pose, int* planarCells,
                                              PlaneFrameStats* rejections) const {
    const int width = config_.width, height = config_.height, cp = config_.cellPixels;
    const double k = config_.sphericalStepRad, b = config_.baselineM;
    const double camera[3]{pose.translation[0], pose.translation[1], pose.translation[2]};
    std::vector<std::array<double, 3>> world(size_t(width) * height);
    std::vector<uint8_t> valid(world.size(), 0);
    for (int v = 0; v < height; ++v)
        for (int u = 0; u < width; ++u) {
            const int i = v * width + u;
            const double d = disparity[i];
            if (!(d > 0) || (uncertainty && uncertainty[i] >= config_.maxUncertainty)) continue;
            const double theta = (u - config_.centreU) * k, phi = (v - config_.centreV) * k;
            const double denominator = std::tan(theta) - std::tan(theta - d * k);
            if (!(denominator > 0)) continue;
            const double rho = b / denominator;
            const double c[3]{rho * std::tan(theta), rho * std::sin(phi), rho * std::cos(phi)};
            const double range = std::sqrt(dot3(c, c));
            if (range < config_.minDepthM || range > config_.maxDepthM) continue;
            for (int r = 0; r < 3; ++r)
                world[i][r] = pose.rotation[r][0] * c[0] + pose.rotation[r][1] * c[1] + pose.rotation[r][2] * c[2] +
                              pose.translation[r];
            valid[i] = 1;
        }

    struct Cell {
        bool planar = false;
        double n = 0, s[3]{}, q[3][3]{}, normal[3]{}, mean[3]{}, range = 0, rms = 0;
    };
    const int cw = width / cp, ch = height / cp;
    std::vector<Cell> cells(size_t(cw) * ch);
    int planar = 0;
    for (int cy = 0; cy < ch; ++cy)
        for (int cx = 0; cx < cw; ++cx) {
            Cell& cell = cells[size_t(cy) * cw + cx];
            for (int v = cy * cp; v < (cy + 1) * cp; ++v)
                for (int u = cx * cp; u < (cx + 1) * cp; ++u) {
                    const int i = v * width + u;
                    if (!valid[i]) continue;
                    const auto& p = world[i];
                    cell.n += 1;
                    for (int r = 0; r < 3; ++r) {
                        cell.s[r] += p[r];
                        for (int c = 0; c < 3; ++c) cell.q[r][c] += p[r] * p[c];
                    }
                }
            if (cell.n < config_.minCellPoints) continue;
            const double lambda = planeFromSums(cell.n, cell.s, cell.q, cell.normal, cell.mean);
            const double d[3]{cell.mean[0] - camera[0], cell.mean[1] - camera[1], cell.mean[2] - camera[2]};
            cell.range = std::sqrt(dot3(d, d));
            const double rms = std::sqrt(lambda);
            cell.rms = rms;
            cell.planar = rms <= std::max<double>(config_.minCellRmsM, config_.cellRmsSigmas * sigmaZ(cell.range));
            if (cell.planar) {
                orientTowards(cell.normal, cell.mean, camera);
                ++planar;
            }
        }
    if (planarCells) *planarCells = planar;

    std::vector<int> order;
    for (int i = 0; i < int(cells.size()); ++i)
        if (cells[i].planar) order.push_back(i);
    std::vector<int> region(cells.size(), -1);
    std::vector<FramePatch> result;
    const double mergeCos = std::cos(config_.mergeAngleDeg * kPi / 180);
    for (int seed : order) {
        if (region[seed] >= 0) continue;
        double n = 0, s[3]{}, q[3][3]{}, normal[3], mean[3];
        std::vector<int> members;
        std::deque<int> queue{seed};
        region[seed] = int(result.size());
        auto add = [&](int index) {
            const Cell& c = cells[index];
            n += c.n;
            for (int r = 0; r < 3; ++r) {
                s[r] += c.s[r];
                for (int cc = 0; cc < 3; ++cc) q[r][cc] += c.q[r][cc];
            }
            members.push_back(index);
            planeFromSums(n, s, q, normal, mean);
            orientTowards(normal, mean, camera);
        };
        add(seed);
        while (!queue.empty()) {
            const int current = queue.front();
            queue.pop_front();
            const int cx = current % cw, cy = current / cw;
            const int neighbours[4][2]{{cx - 1, cy}, {cx + 1, cy}, {cx, cy - 1}, {cx, cy + 1}};
            for (const auto& nb : neighbours) {
                if (nb[0] < 0 || nb[0] >= cw || nb[1] < 0 || nb[1] >= ch) continue;
                const int index = nb[1] * cw + nb[0];
                if (region[index] >= 0 || !cells[index].planar) continue;
                const Cell& c = cells[index];
                if (dot3(c.normal, normal) < mergeCos) continue;
                const double distance = std::fabs(dot3(normal, c.mean) - dot3(normal, mean));
                if (distance > config_.mergeSigmas * std::max<double>(surfaceSigma(c.range), config_.minCellRmsM)) continue;
                region[index] = int(result.size());
                add(index);
                queue.push_back(index);
            }
        }
        if (int(members.size()) < config_.minPatchCells) {
            if (rejections) ++rejections->regionsSmall;
            for (int index : members) region[index] = -2;  // too small: leave unlabelled
            continue;
        }
        FramePatch patch;
        const double lambda = planeFromSums(n, s, q, normal, mean);
        orientTowards(normal, mean, camera);
        double meanRange = 0, cellRms = 0;
        for (int index : members) {
            meanRange += cells[index].range * cells[index].n;
            cellRms += cells[index].rms * cells[index].n;
        }
        meanRange /= n;
        cellRms /= n;
        patch.rms = std::sqrt(lambda);
        // Curvature or steps show as region RMS above the cells' own noise.
        // Curvature beyond the cells' noise and the measured surface ripple.
        const bool curved = patch.rms > std::max<double>({config_.minCellRmsM, config_.regionGrowth * cellRms,
                                                         surfaceSigma(meanRange)});
        const bool noisy = patch.rms > 1.5 * std::max<double>(config_.minCellRmsM, config_.cellRmsSigmas * surfaceSigma(meanRange));
        if (rejections) {
            rejections->regionsCurved += curved;
            rejections->regionsNoisy += noisy && !curved;
        }
        if (curved || noisy) {
            for (int index : members) region[index] = -2;  // curved or stepped: not a plane
            continue;
        }
        for (int r = 0; r < 3; ++r) {
            patch.normal[r] = normal[r];
            patch.centroid[r] = mean[r];
            patch.sum[r] = s[r];
            for (int c = 0; c < 3; ++c) patch.scatter[r][c] = q[r][c];
            patch.viewPosition[r] = camera[r];
            patch.viewDirection[r] = pose.rotation[r][2];
        }
        patch.offset = dot3(normal, mean);
        patch.cells = int(members.size());
        patch.points = int(n);
        patch.offsetSigma = patch.rms / std::sqrt(double(patch.cells));
        std::unordered_set<int64_t> keys;
        for (int index : members) {
            const int cx = index % cw, cy = index / cw;
            for (int v = cy * cp; v < (cy + 1) * cp; ++v)
                for (int u = cx * cp; u < (cx + 1) * cp; ++u) {
                    const int i = v * width + u;
                    if (!valid[i]) continue;
                    patch.pixels.push_back(i);
                    keys.insert(voxelKey(world[i].data()));
                }
        }
        patch.footprint.assign(keys.begin(), keys.end());
        result.push_back(std::move(patch));
    }
    return result;
}

void PlanePatchMap::refit(MapPatch& patch) const {
    double normal[3], mean[3];
    const double previous[3]{patch.normal[0], patch.normal[1], patch.normal[2]};
    planeFromSums(patch.weight, patch.sum, patch.scatter, normal, mean);
    if (dot3(normal, previous) < 0)
        for (double& x : normal) x = -x;
    for (int r = 0; r < 3; ++r) {
        patch.normal[r] = normal[r];
        patch.centroid[r] = mean[r];
    }
    patch.offset = dot3(normal, mean);
    patch.rms = patch.rmsSum / patch.weight;
    patch.offsetSigma = std::max(0.005, patch.rms / std::sqrt(patch.weight));
}

bool PlanePatchMap::newView(const std::vector<std::array<double, 6>>& views, const FramePatch& patch) const {
    const double angleCos = std::cos(config_.viewAngleDeg * kPi / 180);
    for (const auto& view : views) {
        const double d[3]{view[0] - patch.viewPosition[0], view[1] - patch.viewPosition[1], view[2] - patch.viewPosition[2]};
        if (std::sqrt(dot3(d, d)) < config_.viewDistanceM && dot3(view.data() + 3, patch.viewDirection.data()) > angleCos)
            return false;
    }
    return true;
}

PlaneFrameStats PlanePatchMap::integrate(const float* disparity, const uint8_t* uncertainty, const PlanePose& pose,
                                         uint16_t* labels) {
    ++frame_;
    PlaneFrameStats stats;
    std::vector<FramePatch> found = detect(disparity, uncertainty, pose, &stats.planarCells, &stats);
    stats.framePatches = int(found.size());
    if (labels) std::fill(labels, labels + size_t(config_.width) * config_.height, uint16_t(0));
    std::sort(found.begin(), found.end(), [](const FramePatch& a, const FramePatch& b) { return a.cells > b.cells; });
    const double associateCos = std::cos(config_.associateAngleDeg * kPi / 180);
    const double contestCos = std::cos(config_.contestAngleDeg * kPi / 180);
    std::vector<int> supported;
    for (FramePatch& fp : found) {
        int best = -1;
        double bestScore = 1e30;
        std::vector<int> compatible;
        const double fpRange = std::sqrt((fp.centroid[0] - fp.viewPosition[0]) * (fp.centroid[0] - fp.viewPosition[0]) +
                                         (fp.centroid[1] - fp.viewPosition[1]) * (fp.centroid[1] - fp.viewPosition[1]) +
                                         (fp.centroid[2] - fp.viewPosition[2]) * (fp.centroid[2] - fp.viewPosition[2]));
        // Absolute floor only; curvature shows as growth of the fused RMS over each part's own.
        const double fusedLimit = std::max<double>(config_.minCellRmsM, config_.fusedRmsSigmas * surfaceSigma(fpRange));
        for (int m = 0; m < int(patches_.size()); ++m) {
            const MapPatch& mp = patches_[m];
            if (mp.state == PatchState::Retired) continue;
            const double cosine = dot3(fp.normal.data(), mp.normal.data());
            if (std::fabs(cosine) < associateCos) continue;
            const double sign = cosine < 0 ? -1 : 1;
            const double difference = std::fabs(sign * (dot3(mp.normal.data(), fp.centroid.data()) - mp.offset));
            const double sigma = std::sqrt(fp.offsetSigma * fp.offsetSigma + mp.offsetSigma * mp.offsetSigma);
            if (difference > std::max<double>(config_.associateOffsetMinM, config_.associateOffsetSigmas * sigma)) continue;
            bool touching = false;
            for (int64_t key : fp.footprint) {
                const int64_t x = key >> 42, y = (key >> 21) & 0x1fffff, z = key & 0x1fffff;
                for (int a = -1; a <= 1 && !touching; ++a)
                    for (int b = -1; b <= 1 && !touching; ++b)
                        for (int c = -1; c <= 1 && !touching; ++c)
                            touching = mp.footprint.count(((x + a) << 42) | ((y + b) << 21) | (z + c)) > 0;
                if (touching) break;
            }
            if (!touching) continue;
            // The union must still be a plane: refuse when the fused fit's RMS would exceed the
            // noise limit at this range (stops a patch creeping around a curved surface).
            {
                const double w = fp.cells, perPoint = w / fp.points;
                double n = mp.weight + w, sm[3], q[3][3], normal[3], mean[3];
                for (int r = 0; r < 3; ++r) {
                    sm[r] = mp.sum[r] + perPoint * fp.sum[r];
                    for (int c = 0; c < 3; ++c) q[r][c] = mp.scatter[r][c] + perPoint * fp.scatter[r][c];
                }
                const double fused = std::sqrt(planeFromSums(n, sm, q, normal, mean));
                const double own = std::max(fp.rms, mp.rms);
                if (fused > std::max(fusedLimit, config_.fusedGrowth * own)) {
                    ++stats.refusedFused;
                    continue;
                }
            }
            compatible.push_back(m);
            const double score = difference / std::max(sigma, 1e-4);
            if (score < bestScore) {
                bestScore = score;
                best = m;
            }
        }
        if (best < 0) {
            MapPatch mp;
            mp.id = uint16_t(patches_.size() + 1);
            mp.normal = fp.normal;
            mp.firstFrame = frame_;
            patches_.push_back(std::move(mp));
            best = int(patches_.size()) - 1;
            ++stats.created;
        } else {
            ++stats.associated;
        }
        // The observation bridges every other compatible patch it touches: merge them into best.
        for (int other : compatible) {
            if (other == best) continue;
            MapPatch& source = patches_[other];
            MapPatch& target = patches_[best];
            {   // the union must also pass the fused-plane test
                double n = target.weight + source.weight, sm[3], q[3][3], normal[3], mean[3];
                for (int r = 0; r < 3; ++r) {
                    sm[r] = target.sum[r] + source.sum[r];
                    for (int c = 0; c < 3; ++c) q[r][c] = target.scatter[r][c] + source.scatter[r][c];
                }
                const double fused = std::sqrt(planeFromSums(n, sm, q, normal, mean));
                if (fused > std::max(fusedLimit, config_.fusedGrowth * std::max(target.rms, source.rms))) continue;
            }
            target.weight += source.weight;
            target.rmsSum += source.rmsSum;
            for (int r = 0; r < 3; ++r) {
                target.sum[r] += source.sum[r];
                for (int c = 0; c < 3; ++c) target.scatter[r][c] += source.scatter[r][c];
            }
            target.footprint.insert(source.footprint.begin(), source.footprint.end());
            for (const auto& view : source.views) target.views.push_back(view);
            target.supportFrames += source.supportFrames;
            target.firstFrame = std::min(target.firstFrame, source.firstFrame);
            if (source.state == PatchState::Confirmed) target.state = PatchState::Confirmed;
            source.state = PatchState::Retired;
            source.mergedInto = target.id;
            ++stats.merged;
            refit(target);
        }
        MapPatch& mp = patches_[best];
        const double w = fp.cells, perPoint = w / fp.points;
        mp.weight += w;
        mp.rmsSum += fp.rms * w;
        for (int r = 0; r < 3; ++r) {
            mp.sum[r] += perPoint * fp.sum[r];
            for (int c = 0; c < 3; ++c) mp.scatter[r][c] += perPoint * fp.scatter[r][c];
        }
        refit(mp);
        mp.footprint.insert(fp.footprint.begin(), fp.footprint.end());
        if (newView(mp.views, fp)) {
            const auto& p = fp.viewPosition;
            const auto& d = fp.viewDirection;
            mp.views.push_back({p[0], p[1], p[2], d[0], d[1], d[2]});
            if (mp.state == PatchState::Contested) ++mp.recoveryViews;
        }
        mp.contestViews.clear();
        mp.lastSupport = frame_;
        ++mp.supportFrames;
        if (mp.state == PatchState::Tentative && int(mp.views.size()) >= config_.confirmViews)
            mp.state = PatchState::Confirmed;
        if (mp.state == PatchState::Contested && mp.recoveryViews >= config_.confirmViews) {
            mp.state = PatchState::Confirmed;
            mp.recoveryViews = 0;
        }
        supported.push_back(best);
        if (labels)
            for (int i : fp.pixels) labels[i] = mp.id;
    }
    // A confirmed patch that this frame sees replaced over its footprint by a different
    // surface, and does not support anywhere, collects contest views.
    for (int m = 0; m < int(patches_.size()); ++m) {
        MapPatch& mp = patches_[m];
        if (mp.state != PatchState::Confirmed && mp.state != PatchState::Contested) continue;
        if (std::find(supported.begin(), supported.end(), m) != supported.end()) continue;
        for (const FramePatch& fp : found) {
            if (fp.footprint.empty()) continue;
            int overlap = 0;
            for (int64_t key : fp.footprint) overlap += mp.footprint.count(key) > 0;
            if (overlap < config_.contestOverlap * fp.footprint.size()) continue;
            const double cosine = std::fabs(dot3(fp.normal.data(), mp.normal.data()));
            const double difference = std::fabs(dot3(mp.normal.data(), fp.centroid.data()) - mp.offset);
            if (cosine > contestCos && difference < std::max<double>(config_.contestOffsetMinM, 3 * mp.offsetSigma)) continue;
            ++stats.contests;
            mp.lastContest = frame_;
            if (newView(mp.contestViews, fp)) {
                const auto& p = fp.viewPosition;
                const auto& d = fp.viewDirection;
                mp.contestViews.push_back({p[0], p[1], p[2], d[0], d[1], d[2]});
            }
            if (int(mp.contestViews.size()) >= config_.contestViews && mp.state == PatchState::Confirmed) {
                mp.state = PatchState::Contested;
                mp.recoveryViews = 0;
            }
            break;
        }
    }
    for (MapPatch& mp : patches_) {
        if (mp.state == PatchState::Tentative && frame_ - mp.lastSupport > config_.retireTentativeFrames)
            mp.state = PatchState::Retired;
        switch (mp.state) {
            case PatchState::Tentative: ++stats.tentative; break;
            case PatchState::Confirmed: ++stats.confirmed; break;
            case PatchState::Contested: ++stats.contested; break;
            case PatchState::Retired: ++stats.retired; break;
        }
    }
    return stats;
}

}  // namespace FrameSpatial
