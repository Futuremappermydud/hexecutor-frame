// Replay FRAME_OCCLUSION_RECORD output through both filters. Compare mesh
// coverage changes after capture-pose alignment, excluding startup confirmation.
#include "occlusion_surface.hpp"
#include "calibration.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace FrameSpatial;

std::vector<OcclusionObservation> meshCoverage(const std::vector<float>& depth, int w, int h) {
    std::vector<OcclusionObservation> covered(depth.size());
    const auto indices=occlusionSurfaceIndices(depth,w,h);
    const auto mask=occlusionSurfaceMask(indices,w,h);
    for(size_t i=0;i<mask.size();++i) if(mask[i]) covered[i].depth=depth[i];
    return covered;
}

struct CoverageStats {
    double changes=0, total=0, pixels=0;
    std::vector<OcclusionObservation> previous;
    void observe(const std::vector<float>& depth,int w,int h,
                 const OcclusionReprojection& transform,bool compare) {
        auto current=meshCoverage(depth,w,h);
        if(compare && !previous.empty()) {
            auto aligned=reprojectOcclusionObservations(previous,w,h,RECT_FOCAL,RECT_CX,RECT_CY,transform);
            for(size_t i=0;i<current.size();++i) {
                const bool a=surfaceDepth(current[i].depth),b=surfaceDepth(aligned[i].depth);
                changes+=a!=b;
                total+=a||b;
                pixels+=a;
            }
        }
        previous.swap(current);
    }
};

int main(int argc,char** argv) {
    if(argc!=2) { std::cerr<<"Usage: "<<argv[0]<<" sequence.bin\n"; return 2; }
    std::ifstream input(argv[1],std::ios::binary);
    uint32_t dimensions[2]{};
    input.read(reinterpret_cast<char*>(dimensions),sizeof(dimensions));
    if(!input || dimensions[0]!=RECT_WIDTH || dimensions[1]!=RECT_HEIGHT) return 2;
    const int w=dimensions[0],h=dimensions[1];
    OcclusionSurfaceHistory history;
    std::vector<float> oldHistory;
    CoverageStats oldStats,newStats;
    size_t frames=0,compared=0;
    double held=0;
    for(;;) {
        float elapsed=0; uint8_t aligned=0; OcclusionReprojection transform;
        input.read(reinterpret_cast<char*>(&elapsed),sizeof(elapsed));
        if(input.eof() && input.gcount()==0) break;
        input.read(reinterpret_cast<char*>(&aligned),sizeof(aligned));
        for(auto& row:transform.rotation) input.read(reinterpret_cast<char*>(row.data()),sizeof(float)*3);
        input.read(reinterpret_cast<char*>(transform.translation.data()),sizeof(float)*3);
        std::vector<float> depth(size_t(w)*h);
        input.read(reinterpret_cast<char*>(depth.data()),depth.size()*sizeof(float));
        if(!input) { std::cerr<<"Incomplete frame\n"; return 1; }
        auto oldSurface=aligned && elapsed>0 && elapsed<=0.15f ?
            stabilizeOcclusionDepth(depth,oldHistory,w,h,RECT_FOCAL,RECT_CX,RECT_CY,transform):depth;
        auto newSurface=history.update(depth,w,h,RECT_FOCAL,RECT_CX,RECT_CY,aligned?&transform:nullptr,elapsed);
        const bool compare=frames>=8 && aligned && elapsed>0 && elapsed<=0.20f;
        oldStats.observe(oldSurface,w,h,transform,compare);
        newStats.observe(newSurface,w,h,transform,compare);
        if(compare) { ++compared; held+=history.held; }
        oldHistory.swap(oldSurface);
        ++frames;
    }
    if(!compared || !oldStats.total || !newStats.total) { std::cerr<<"Insufficient covered frames\n"; return 1; }
    std::cout<<"Frames="<<frames<<", compared="<<compared
        <<"\nOld coverage changes="<<100*oldStats.changes/oldStats.total<<"% of union/frame; mean pixels="<<oldStats.pixels/compared
        <<"\nNew coverage changes="<<100*newStats.changes/newStats.total<<"% of union/frame; mean pixels="<<newStats.pixels/compared
        <<"\nMean confirmed samples retained through dropouts="<<held/compared<<'\n';
}
