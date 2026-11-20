#pragma once
#include "occlusion_surface.hpp"
#include <cstdint>
#include <limits>
#include <utility>

namespace FrameSpatial {
struct OcclusionCompletionStats {
    size_t measured = 0, filled = 0, planes = 0, conflicts = 0;
};

namespace OcclusionCompletionDetail {
using Pixel = std::pair<int, int>;
inline long long cross(const Pixel& a, const Pixel& b, const Pixel& c) {
    return static_cast<long long>(b.first-a.first)*(c.second-a.second) -
           static_cast<long long>(b.second-a.second)*(c.first-a.first);
}
inline std::vector<Pixel> hull(std::vector<Pixel> points) {
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    if (points.size() < 3) return {};
    std::vector<Pixel> result(points.size()*2);
    size_t n = 0;
    for (const auto& p : points) {
        while (n >= 2 && cross(result[n-2], result[n-1], p) <= 0) --n;
        result[n++] = p;
    }
    const size_t lower = n+1;
    for (size_t i = points.size()-1; i-- > 0;) {
        const auto& p = points[i];
        while (n >= lower && cross(result[n-2], result[n-1], p) <= 0) --n;
        result[n++] = p;
    }
    result.resize(n-1);
    return result;
}

// Inverse depth is affine in pixel coordinates on a 3D plane. Coordinates are
// normalized to the tile so nearly collinear support fails a conditioned solve.
inline bool fit(const std::vector<int>& samples, const std::vector<float>& depth,
                int width, int x0, int y0, double plane[3]) {
    double a[3][4]{};
    for (int i : samples) {
        const double p[3]{1, (i%width-x0)/32.0, (i/width-y0)/32.0};
        const double inverse = 1.0/depth[i];
        for (int r=0; r<3; ++r) {
            for (int c=0; c<3; ++c) a[r][c] += p[r]*p[c];
            a[r][3] += p[r]*inverse;
        }
    }
    const double scale = double(samples.size());
    for (int c=0; c<3; ++c) {
        int pivot = c;
        for (int r=c+1; r<3; ++r)
            if (std::abs(a[r][c]) > std::abs(a[pivot][c])) pivot=r;
        if (std::abs(a[pivot][c]) < scale*1e-4) return false;
        for (int k=c; k<4; ++k) std::swap(a[c][k], a[pivot][k]);
        const double divisor = a[c][c];
        for (int k=c; k<4; ++k) a[c][k] /= divisor;
        for (int r=0; r<3; ++r) if (r != c) {
            const double factor=a[r][c];
            for (int k=c; k<4; ++k) a[r][k] -= factor*a[c][k];
        }
    }
    for (int r=0; r<3; ++r) plane[r]=a[r][3];
    return true;
}
inline float predict(const double plane[3], int x, int y, int x0, int y0) {
    const double inverse=plane[0]+plane[1]*(x-x0)/32.0+plane[2]*(y-y0)/32.0;
    return inverse > 0 ? float(1.0/inverse) : 0;
}
} // namespace OcclusionCompletionDetail

// Bounded, image-guided surface completion, not a source of measured depth.
// Fit overlapping 32/64px tiles, split at luma edges, require distributed measured
// support and >=90% depth agreement. Fill only inside the support's convex hull
// and within 24 connected pixels of a measurement. Filled values never seed a
// second fit. Conflicting overlapping planes leave pixels unknown.
inline std::vector<float> completeOcclusionPlanes(
    const std::vector<float>& measured, const std::vector<uint8_t>& guide,
    int width, int height, OcclusionCompletionStats* stats = nullptr,
    OcclusionCompletionContract* contract = nullptr) {
    if (stats) *stats={};
    auto completed=measured;
    if (contract) *contract={};
    if (width <= 0 || height <= 0 || measured.size() != size_t(width)*height ||
        guide.size() != measured.size()) return completed;
    std::vector<uint8_t> veto(measured.size(),0);
    // Connectivity depends only on measured input, so compute it once rather
    // than repeating depth/luma checks for every overlapping tile traversal.
    std::vector<uint8_t> valid(measured.size()), horizontal(measured.size()), vertical(measured.size());
    #pragma omp parallel for
    for (int y=0; y<height; ++y) for (int x=0; x<width; ++x) {
        const int i=y*width+x;
        valid[i]=surfaceDepth(measured[i]);
        const auto connected=[&](int j) {
            return std::abs(int(guide[i])-int(guide[j]))<=16 &&
                !(surfaceDepth(measured[i]) && surfaceDepth(measured[j]) &&
                  !surfaceRange(std::min(measured[i],measured[j]),
                                std::max(measured[i],measured[j])));
        };
        if(x+1<width) horizontal[i]=connected(i+1);
        if(y+1<height) vertical[i]=connected(i+width);
    }
    // Unknown samples bordering a measured/luma discontinuity are protected.
    // Measured samples themselves always remain untouched.
    for (int y=0; y<height; ++y) for (int x=0; x<width; ++x) {
        const int i=y*width+x;
        const auto protect=[&](int j, bool connected) {
            if (connected) return;
            if (!valid[i]) veto[i]|=CompletionProtectedBoundary;
            if (!valid[j]) veto[j]|=CompletionProtectedBoundary;
        };
        if (x+1<width) protect(i+1,horizontal[i]);
        if (y+1<height) protect(i+width,vertical[i]);
    }
    if (stats) for (uint8_t v : valid) stats->measured += v;
    constexpr int reach=24;
    struct Tile { int x, y, size; };
    std::vector<Tile> layout;
    for(int size : {64,32}) for(int y=0;y<height;y+=size/2)
        for(int x=0;x<width;x+=size/2) layout.push_back({x,y,size});
    struct Proposal { int index; float depth; };
    struct TileResult { std::vector<Proposal> proposals; std::vector<int> contradictions; size_t planes=0; };
    std::vector<TileResult> tiles(layout.size());
    // Independent tile outputs avoid races at overlap pixels. Reduction below
    // is deterministic regardless of OpenMP scheduling or number of threads.
    #pragma omp parallel for schedule(dynamic, 4)
    for (int t=0; t<int(layout.size()); ++t) {
        const int x0=layout[t].x, y0=layout[t].y;
        const int tw=std::min(layout[t].size,width-x0), th=std::min(layout[t].size,height-y0);
        auto& result=tiles[t];
        std::vector<int> label(size_t(tw)*th,-1), component, seeds, inliers;
        component.reserve(tw*th); seeds.reserve(tw*th);
        std::vector<int> indices(size_t(tw)*th);
        std::vector<uint8_t> neighbors(size_t(tw)*th), connections(size_t(tw)*th);
        for(int y=0;y<th;++y) for(int x=0;x<tw;++x) {
            const int local=y*tw+x, i=(y0+y)*width+x0+x;
            indices[local]=i;
            neighbors[local]=(x?1:0)|(x+1<tw?2:0)|(y?4:0)|(y+1<th?8:0);
            connections[local]=(x && horizontal[i-1]?1:0)|
                (x+1<tw && horizontal[i]?2:0)|
                (y && vertical[i-width]?4:0)|(y+1<th && vertical[i]?8:0);
        }
        int region=0;
        const auto global=[&](int local) { return indices[local]; };
        const auto visitNeighbors=[&](int local, uint8_t mask, auto visit) {
            if(mask&1) visit(local-1);
            if(mask&2) visit(local+1);
            if(mask&4) visit(local-tw);
            if(mask&8) visit(local+tw);
        };
        for(int start=0; start<tw*th; ++start) {
            if(label[start]>=0) continue;
            component.clear(); seeds.clear();
            component.push_back(start); label[start]=region++;
            for(size_t next=0; next<component.size(); ++next) {
                const int local=component[next], i=global(local);
                if(valid[i]) seeds.push_back(i);
                visitNeighbors(local,connections[local],[&](int neighbor) {
                    if(label[neighbor]>=0) return;
                    label[neighbor]=label[start]; component.push_back(neighbor);
                });
            }
            if(component.size()<64 || seeds.size()<12 || seeds.size()==component.size()) continue;
            double plane[3]{};
            if(!OcclusionCompletionDetail::fit(seeds,measured,width,x0,y0,plane)) continue;
            inliers.clear();
            for(int i:seeds) {
                const float z=OcclusionCompletionDetail::predict(plane,i%width,i/width,x0,y0);
                if(surfaceDepth(z) && surfaceRange(std::min(z,measured[i]),std::max(z,measured[i])))
                    inliers.push_back(i);
            }
            if(inliers.size()<12 || inliers.size()*10<seeds.size()*9 ||
               !OcclusionCompletionDetail::fit(inliers,measured,width,x0,y0,plane)) continue;
            // Recheck every anchor after refitting; disagreements are obstacles.
            std::vector<OcclusionCompletionDetail::Pixel> support;
            std::vector<uint8_t> distance(size_t(tw)*th,255), conflictDistance(distance);
            std::vector<int> queue;
            queue.reserve(component.size());
            std::vector<int> conflicts;
            double squaredError=0;
            for(int i:seeds) {
                const int local=(i/width-y0)*tw+i%width-x0;
                const float z=OcclusionCompletionDetail::predict(plane,i%width,i/width,x0,y0);
                if(surfaceDepth(z) && surfaceRange(std::min(z,measured[i]),std::max(z,measured[i]))) {
                    support.emplace_back(i%width,i/width);
                    squaredError+=std::pow((z-measured[i])/std::max(0.04f,measured[i]*0.04f),2);
                    distance[local]=0; queue.push_back(local);
                } else { conflictDistance[local]=0; conflicts.push_back(local); }
            }
            if(support.size()<12 || support.size()*10<seeds.size()*9 ||
               squaredError/support.size()>0.25) continue;
            const auto polygon=OcclusionCompletionDetail::hull(support);
            if(polygon.size()<3) continue;
            int minX=width,maxX=0,minY=height,maxY=0;
            for(const auto& p:polygon) {
                minX=std::min(minX,p.first); maxX=std::max(maxX,p.first);
                minY=std::min(minY,p.second); maxY=std::max(maxY,p.second);
            }
            if(maxX-minX<8 || maxY-minY<8) continue;
            const auto spread=[&](std::vector<uint8_t>& distances,std::vector<int>& pending,int limit) {
                for(size_t n=0;n<pending.size();++n) {
                    const int local=pending[n];
                    if(distances[local]>=limit) continue;
                    visitNeighbors(local,connections[local],[&](int neighbor) {
                        if(label[neighbor]!=label[start] || distances[neighbor]<=distances[local]+1) return;
                        distances[neighbor]=distances[local]+1; pending.push_back(neighbor);
                    });
                }
            };
            spread(distance,queue,reach); spread(conflictDistance,conflicts,8);
            // Rasterize the hull, retaining the image-connected region mask.
            std::vector<float> rowLo(th,float(width)),rowHi(th,-1);
            for(int y=minY;y<=maxY;++y) for(size_t e=0;e<polygon.size();++e) {
                const auto& a=polygon[e]; const auto& b=polygon[(e+1)%polygon.size()];
                if(y<std::min(a.second,b.second) || y>std::max(a.second,b.second)) continue;
                if(a.second==b.second) {
                    rowLo[y-y0]=std::min(rowLo[y-y0],float(std::min(a.first,b.first)));
                    rowHi[y-y0]=std::max(rowHi[y-y0],float(std::max(a.first,b.first)));
                } else {
                    const float x=a.first+float(y-a.second)*(b.first-a.first)/(b.second-a.second);
                    rowLo[y-y0]=std::min(rowLo[y-y0],x); rowHi[y-y0]=std::max(rowHi[y-y0],x);
                }
            }
            ++result.planes;
            for(int local:component) {
                const int i=global(local),x=i%width,y=i/width;
                if(valid[i] || x<rowLo[y-y0] || x>rowHi[y-y0]) continue;
                if(conflictDistance[local]<=8) {
                    result.contradictions.push_back(i); continue;
                }
                if(distance[local]>reach) continue;
                const float z=OcclusionCompletionDetail::predict(plane,x,y,x0,y0);
                if(surfaceDepth(z)) result.proposals.push_back({i,z});
            }
        }
    }
    std::vector<float> inverseSum(measured.size(),0),lo(measured.size(),5),hi(measured.size(),0);
    std::vector<unsigned char> votes(measured.size(),0);
    for(const auto& tileResult:tiles) {
        if(stats) stats->planes+=tileResult.planes;
        for(int i:tileResult.contradictions) veto[i]|=CompletionContradictorySupport;
        for(const auto& p:tileResult.proposals) {
            inverseSum[p.index]+=1.0f/p.depth; ++votes[p.index];
            lo[p.index]=std::min(lo[p.index],p.depth); hi[p.index]=std::max(hi[p.index],p.depth);
        }
    }
    for(size_t i=0;i<completed.size();++i) if(votes[i]) {
        if(!surfaceRange(lo[i],hi[i])) {
            veto[i]|=CompletionOverlapConflict;
            if(stats) ++stats->conflicts;
        } else if(!veto[i]) {
            completed[i]=votes[i]/inverseSum[i];
            if(stats) ++stats->filled;
        }
    }
    if (contract) {
        contract->veto=std::move(veto);
        contract->provenance.resize(measured.size(),CompletionUnknown);
        for (size_t i=0; i<measured.size(); ++i) {
            if (valid[i]) contract->provenance[i]=CompletionMeasured;
            else if (surfaceDepth(completed[i])) contract->provenance[i]=CompletionPlane;
        }
    }
    return completed;
}
} // namespace FrameSpatial
