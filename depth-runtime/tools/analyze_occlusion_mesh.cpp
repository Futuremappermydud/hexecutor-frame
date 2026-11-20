// Same-input mesh representation replay. Does not establish scene correctness.
#include "occlusion_benchmark.hpp"
#include "cloud_worker.hpp"
#include <chrono>
#include <iomanip>
using namespace FrameSpatial;
using namespace FrameSpatial::OcclusionBenchmark;
struct Stats {
    size_t frames=0,coarse=0,recovered=0,missing=0,spread=0,indices=0;
    std::vector<double> timings;
};
int main(int argc,char** argv) {
    if(argc!=2) { std::cerr<<"Usage: analyze_occlusion_mesh recording.bin\n"; return 2; }
    try {
        Reader reader(argv[1]); Frame frame;
        const auto& h=reader.header;
        // Reproduce the documented candidate confirmation config, not live defaults.
        OcclusionSurfaceHistory history({2,2,3,.20f,.25f,.25f});
        std::array<Stats,5> phases{};
        uint64_t lastFrame=0,lastNs=0; size_t gaps=0,fallbacks=0;
        while(reader.next(frame)) {
            const auto& m=frame.meta;
            if(lastFrame && (m.frame<=lastFrame || m.captureBeginNs<=lastNs))
                throw std::runtime_error("Non-increasing frame IDs/timestamps");
            if(lastFrame && m.frame!=lastFrame+1) ++gaps;
            if(!(m.flags&CapturePose)) ++fallbacks;
            const bool aligned=(m.flags&Aligned) && (!lastFrame || m.frame==lastFrame+1);
            lastFrame=m.frame; lastNs=m.captureBeginNs;
            auto depth=history.update(frame.input,h.width,h.height,h.focal,h.cx,h.cy,
                aligned?&m.transform:nullptr,m.elapsed,&frame.evidence);
            OcclusionMeshStats mesh;
            const auto begin=std::chrono::steady_clock::now();
            auto indices=occlusionSurfaceIndices(depth,h.width,h.height,&mesh);
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            auto& s=phases[m.phase]; ++s.frames; s.coarse+=mesh.coarseAccepted;
            s.recovered+=mesh.recoveredSubcells; s.missing+=mesh.missingRejected;
            s.spread+=mesh.spreadRejected; s.indices+=indices.size(); s.timings.push_back(ms);
        }
        if(!reader.complete || reader.dropped || gaps || fallbacks || !reader.frames)
            throw std::runtime_error("Requires a complete, gap-free, capture-pose recording");
        std::cout<<"Frames="<<reader.frames<<"; same-input candidate config 2,2,3,200,250,250; no ground truth\n";
        std::cout<<"phase,frames,coarse_area_pct,subdivided_area_pct,missing_cells_pct,spread_cells_pct,mean_vertices,mean_upload_MiB,mesh_p50_ms,mesh_p95_ms\n";
        const double pixels=size_t(h.width)*h.height,cells=((h.width-1)/2)*((h.height-1)/2);
        std::cout<<std::fixed<<std::setprecision(3);
        for(size_t p=0;p<phases.size();++p) {
            auto& s=phases[p]; if(!s.frames) continue;
            std::sort(s.timings.begin(),s.timings.end());
            std::cout<<phaseName(p)<<','<<s.frames<<','<<100*s.coarse*4/(pixels*s.frames)<<','
                <<100*(s.coarse*4+s.recovered)/(pixels*s.frames)<<','<<100*s.missing/(cells*s.frames)<<','
                <<100*s.spread/(cells*s.frames)<<','<<double(s.indices)/s.frames<<','
                <<double(s.indices)*sizeof(FrameSpatialCloud::PointVertex)/(s.frames*1048576)<<','<<s.timings[size_t(.5*(s.frames-1))]<<','
                <<s.timings[size_t(.95*(s.frames-1))]<<'\n';
        }
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
