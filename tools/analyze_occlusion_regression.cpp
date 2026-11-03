// Compare completion contracts on extracted, recorded measured depth and luma.
// Archive headers under namespace BaselineSpatial, as documented in regression artifacts.
#include "occlusion_completion.hpp"
#include "baseline_occlusion_completion.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
using namespace FrameSpatial;
constexpr int W=616,H=616;
template<class T> std::vector<T> load(const std::string& path) {
    std::ifstream file(path,std::ios::binary);std::vector<T> out(W*H);
    file.read(reinterpret_cast<char*>(out.data()),out.size()*sizeof(T));
    if(!file || file.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("Bad dimensions: "+path);
    return out;
}
size_t count(const std::vector<float>& d) {return std::count_if(d.begin(),d.end(),surfaceDepth);}
int main(int argc,char** argv) try {
    if(argc!=2) {std::cerr<<"Usage: analyze_occlusion_regression extracted-pairs-directory\n";return 2;}
    std::cout<<"frame,variant,valid_pixels,valid_pct,coarse_mesh_pct,subdivided_mesh_pct,depth_support_change_pixels_vs_original\n";
    for(const auto frame:{70,210}) {
        const std::string prefix=std::string(argv[1])+"/"+std::to_string(frame);
        const auto measured=load<float>(prefix+"-measured-f32.raw");
        const auto guide=load<uint8_t>(prefix+"-left-u8.raw");
        const auto recorded=load<float>(prefix+"-recorded-input-f32.raw");
        const auto anchors=smoothOcclusionDepth(measured,W,H);
        const auto oldPlanes=BaselineSpatial::completeOcclusionPlanes(anchors,guide,W,H);
        const auto oldInput=BaselineSpatial::fillOcclusionHoles(oldPlanes,W,H);
        OcclusionCompletionContract contract;
        const auto newPlanes=completeOcclusionPlanes(anchors,guide,W,H,nullptr,&contract);
        const auto newInput=fillOcclusionHoles(newPlanes,anchors,W,H,contract,guide);
        const auto newPlanesOldHole=BaselineSpatial::fillOcclusionHoles(newPlanes,W,H);
        OcclusionCompletionContract bracketOnly;
        bracketOnly.veto.assign(W*H,0);bracketOnly.provenance.assign(W*H,CompletionUnknown);
        for(size_t i=0;i<oldPlanes.size();++i) if(surfaceDepth(oldPlanes[i]))
            bracketOnly.provenance[i]=surfaceDepth(anchors[i])?CompletionMeasured:CompletionPlane;
        const auto oldPlanesMeasuredHole=fillOcclusionHoles(oldPlanes,anchors,W,H,bracketOnly,guide);
        const std::pair<const char*,const std::vector<float>*> variants[]={{"recorded-original",&recorded},{"original-recomputed",&oldInput},
            {"changed-plane-only",&newPlanesOldHole},{"changed-hole-only",&oldPlanesMeasuredHole},{"changed-plane-and-hole",&newInput}};
        for(const auto& variant:variants) {
            const auto& d=*variant.second;
            const auto coarse=BaselineSpatial::occlusionSurfaceIndices(d,W,H);
            const auto sub=occlusionSurfaceIndices(d,W,H);
            size_t changed=0;for(size_t i=0;i<d.size();++i) changed+=surfaceDepth(d[i])!=surfaceDepth(oldInput[i]);
            std::cout<<frame<<','<<variant.first<<','<<count(d)<<','<<100.0*count(d)/d.size()<<','
                <<100.0*(coarse.size()/6)*4/d.size()<<','<<100.0*occlusionSurfaceArea(sub,W)/d.size()<<','<<changed<<'\n';
        }
        size_t supportMismatch=0;for(size_t i=0;i<recorded.size();++i)
            supportMismatch+=surfaceDepth(recorded[i])!=surfaceDepth(oldInput[i]);
        if(supportMismatch) throw std::runtime_error("Archived original support does not reproduce recording");
        for(bool candidate:{false,true}) {
            std::vector<double> times;
            for(int run=0;run<6;++run) {
                const auto begin=std::chrono::steady_clock::now();std::vector<float> out;
                if(candidate) {OcclusionCompletionContract c;auto p=completeOcclusionPlanes(anchors,guide,W,H,nullptr,&c);
                    out=fillOcclusionHoles(p,anchors,W,H,c,guide);}
                else {auto p=BaselineSpatial::completeOcclusionPlanes(anchors,guide,W,H);out=BaselineSpatial::fillOcclusionHoles(p,W,H);}
                if(run) times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());
                if(out.empty()) throw std::runtime_error("Empty completion result");
            }
            std::sort(times.begin(),times.end());
            std::cerr<<"Frame="<<frame<<", completion+holes="<<(candidate?"candidate":"original")
                <<", same-host warmed p50_ms="<<times[times.size()/2]<<" (not Frame production timing)\n";
        }
    }
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
