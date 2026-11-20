#pragma once
// E09: bounded plane patches detected per frame and associated into a persistent map.
// Detection and association only; the TSDF and mesh are not modified.
//
// Per frame (model grid, spherical left disparity): each cellPixels^2 cell gets a PCA
// plane; a cell is planar when its RMS is within cellRmsSigmas x the depth noise at its
// range (sigma_z = z^2 sigma_d k / B). Planar cells grow into regions while their normal
// stays within mergeAngleDeg of the region plane and their centroid within mergeSigmas
// sigma_z of it. Regions of at least minPatchCells become frame patches.
//
// Map: a frame patch joins a map patch only if the normals agree, the offsets agree within
// max(associateOffsetMinM, associateOffsetSigmas x combined sigma), and the footprints touch
// (shared or neighbouring footprintVoxelM voxels). Disconnected coplanar regions and
// regions separated by an opening therefore stay separate. Support counts independent
// viewpoints (viewDistanceM or viewAngleDeg apart), not frames.
// States: tentative -> confirmed (confirmViews views) -> contested (contestViews views of a
// conflicting surface over its footprint since its last support); tentative patches that
// go retireTentativeFrames frames without support are retired.
#include <array>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace FrameSpatial {

struct PlanePatchConfig {
    int width = 256, height = 320;
    float sphericalStepRad = 0.29f * 3.14159265358979f / 180.f;
    float baselineM = 24.3048f / 380.f;
    float centreU = 127.5f, centreV = 159.5f;
    float minDepthM = 0.25f, maxDepthM = 4.0f;
    int maxUncertainty = 89;
    // Two noise scales of the model output, measured on development data (2026-11-08):
    // within an 8x8 cell, median affine-fit RMS 0.037 px (desk-tags-full-001); across a real
    // table top, 5-10 cm ripple with median |height| deviation 9.2 mm at ~1.3 m (about 0.1 px)
    // and neighbour-cell normal differences of median 6.5, p90 18 degrees (e06 event capture).
    float disparitySigmaPx = 0.04f;   // cell planarity
    float surfaceSigmaPx = 0.1f;      // merge distance and region / fused curvature floors
    int cellPixels = 8;
    int minCellPoints = 40;
    float cellRmsSigmas = 2.f;
    float minCellRmsM = 0.002f;  // below measured real noise beyond about 0.8 m (2.9 mm at 1 m)
    float mergeAngleDeg = 20.f;     // above the measured p90 neighbour-cell normal difference
    float mergeSigmas = 3.f;
    int minPatchCells = 6;
    float regionGrowth = 1.15f; // region RMS may exceed its cells' mean RMS by at most this factor
    float footprintVoxelM = 0.05f;
    float associateAngleDeg = 6.f;
    float associateOffsetMinM = 0.03f;
    float associateOffsetSigmas = 3.f;
    float fusedRmsSigmas = 1.5f;  // fused-plane RMS floor, in surface sigma at the observation range
    float fusedGrowth = 1.15f;    // fused RMS may exceed the parts' own RMS by at most this factor
    float contestAngleDeg = 15.f;
    float contestOffsetMinM = 0.05f;
    float contestOverlap = 0.2f;
    float viewDistanceM = 0.15f, viewAngleDeg = 10.f;
    int confirmViews = 3, contestViews = 3;
    int retireTentativeFrames = 150;
};

struct PlanePose {  // camera-to-world, rectified left camera (x right, y down, z forward)
    float rotation[3][3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    float translation[3]{};
};

enum class PatchState : uint8_t { Tentative = 0, Confirmed = 1, Contested = 2, Retired = 3 };

struct FramePatch {
    std::array<double, 3> normal{}, centroid{};  // world; normal faces the camera
    double offset = 0;                           // normal . x = offset
    double rms = 0, offsetSigma = 0;
    int cells = 0, points = 0;
    double sum[3]{}, scatter[3][3]{};            // world, unweighted
    std::vector<int64_t> footprint;              // unique world voxel keys
    std::vector<int> pixels;                     // model-grid pixel indices
    std::array<double, 3> viewPosition{}, viewDirection{};
};

struct MapPatch {
    uint16_t id = 0;
    PatchState state = PatchState::Tentative;
    std::array<double, 3> normal{}, centroid{};
    double offset = 0, rms = 0, offsetSigma = 0;
    double weight = 0, sum[3]{}, scatter[3][3]{};  // weighted by frame cells / points
    std::unordered_set<int64_t> footprint;
    std::vector<std::array<double, 6>> views;      // position, direction
    std::vector<std::array<double, 6>> contestViews;
    int supportFrames = 0, firstFrame = 0, lastSupport = 0, lastContest = -1;
    int recoveryViews = 0;  // new support views since it became contested
    uint16_t mergedInto = 0;  // set when retired by a merge
    double rmsSum = 0;
};

struct PlaneFrameStats {
    int planarCells = 0, framePatches = 0, associated = 0, created = 0, contests = 0, merged = 0;
    int regionsSmall = 0, regionsCurved = 0, regionsNoisy = 0;  // detection rejections
    int refusedFused = 0;  // associations refused by the fused-plane test
    int confirmed = 0, tentative = 0, contested = 0, retired = 0;
};

class PlanePatchMap {
public:
    explicit PlanePatchMap(const PlanePatchConfig& config = {});
    // disparity: left spherical disparity (model px), uncertainty: uint8 score; both width x
    // height. labels (optional, width x height): map patch id per pixel, 0 for none.
    PlaneFrameStats integrate(const float* disparity, const uint8_t* uncertainty, const PlanePose& pose,
                              uint16_t* labels = nullptr);
    std::vector<FramePatch> detect(const float* disparity, const uint8_t* uncertainty, const PlanePose& pose,
                                   int* planarCells = nullptr, PlaneFrameStats* rejections = nullptr) const;
    const std::vector<MapPatch>& patches() const { return patches_; }
    const PlanePatchConfig& config() const { return config_; }
    int64_t voxelKey(const double p[3]) const;

private:
    double sigmaZ(double range) const;         // cell noise
    double surfaceSigma(double range) const;   // ripple across cells
    void refit(MapPatch& patch) const;
    bool newView(const std::vector<std::array<double, 6>>& views, const FramePatch& patch) const;

    PlanePatchConfig config_;
    std::vector<MapPatch> patches_;
    int frame_ = 0;
};

// Smallest-eigenvalue eigenvector of a symmetric 3x3 matrix (Jacobi); returns the eigenvalue.
double smallestEigen(const double m[3][3], double vector[3]);

}  // namespace FrameSpatial
