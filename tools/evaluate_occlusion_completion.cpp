// Run old/new shaders on identical rectified camera inputs, then compare the
// former enclosed-hole surface with image-guided plane completion. Neither
// coverage nor plane support is an accuracy measurement without ground truth.
#include "vulkan_stereo_pipeline.hpp"
#include "occlusion_completion.hpp"
#include "depth_speckle_filter.hpp"
#include "calibration.hpp"
#include "occlusion_stage_diagnostics.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace FrameSpatial;
template<class T> std::vector<T> read(const char* path) {
    std::vector<T> data(size_t(RECT_WIDTH)*RECT_HEIGHT);
    std::ifstream file(path,std::ios::binary);
    file.read(reinterpret_cast<char*>(data.data()),data.size()*sizeof(T));
    if(!file || file.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error(std::string("Wrong size or unreadable: ")+path);
    return data;
}
template<class T> void write(const std::string& path,const std::vector<T>& data) {
    std::ofstream file(path,std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
    if(!file) throw std::runtime_error("Cannot write "+path);
}
size_t valid(const std::vector<float>& depth) {
    return std::count_if(depth.begin(),depth.end(),surfaceDepth);
}
int main(int argc,char** argv) try {
    if(argc==5 && std::string(argv[1])=="--completion-only") {
        // Input is the measured stereo dump from this tool. Smooth once outside
        // the timed section, then benchmark only the plane-completion stage.
        const auto measured=smoothOcclusionDepth(read<float>(argv[2]),RECT_WIDTH,RECT_HEIGHT);
        const auto guide=read<uint8_t>(argv[3]);
        std::vector<double> times;
        std::vector<float> completed;
        OcclusionCompletionStats stats;
        for(int n=0;n<31;++n) {
            const auto start=std::chrono::steady_clock::now();
            completed=completeOcclusionPlanes(measured,guide,RECT_WIDTH,RECT_HEIGHT,&stats);
            if(n) times.push_back(std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-start).count());
        }
        std::sort(times.begin(),times.end());
        write(std::string(argv[4])+"-planes.raw",completed);
        std::cout<<"completion_median_ms="<<times[14]<<", completion_p95_ms="<<times[27]
                 <<", measured="<<stats.measured<<", plane_filled="<<stats.filled
                 <<", planes="<<stats.planes<<", overlap_conflicts="<<stats.conflicts<<'\n';
        return 0;
    }
    if(argc!=5) {
        std::cerr<<"Usage: "<<argv[0]<<" left-u8.raw right-u8.raw shader-directory output-prefix\n";
        std::cerr<<"   or: "<<argv[0]<<" --completion-only measured-f32.raw guide-u8.raw output-prefix\n";
        return 2;
    }
    auto left=read<uint8_t>(argv[1]),right=read<uint8_t>(argv[2]);
    VulkanStereoPipeline stereo;
    if(!stereo.initialize(argv[3],true)) throw std::runtime_error(stereo.lastError());
    std::vector<float> depth;
    StereoDiagnostics diagnostics;
    double stereoMs=0;
    for(int n=0;n<6;++n) {
        if(!stereo.process(left,right,depth,n==5?&diagnostics:nullptr)) throw std::runtime_error(stereo.lastError());
        if(n && n<5) stereoMs+=stereo.lastComputeMs();
    }
    const std::string prefix=argv[4];
    writeStereoDiagnostics(prefix,diagnostics,RECT_WIDTH,RECT_HEIGHT);
    writeDiagnosticLuma(prefix+"-left.pgm",left,RECT_WIDTH,RECT_HEIGHT);
    writeDiagnosticLuma(prefix+"-right.pgm",right,RECT_WIDTH,RECT_HEIGHT);
    writeDiagnosticMask(prefix,"supported",depth,RECT_WIDTH,RECT_HEIGHT);
    const size_t rawValid=valid(depth);
    removeDepthSpeckles(depth,RECT_WIDTH,RECT_HEIGHT);
    auto smoothed=smoothOcclusionDepth(depth,RECT_WIDTH,RECT_HEIGHT);
    writeDiagnosticMask(prefix,"speckle",depth,RECT_WIDTH,RECT_HEIGHT);
    writeDiagnosticMask(prefix,"smoothed",smoothed,RECT_WIDTH,RECT_HEIGHT);
    auto old=fillOcclusionHoles(smoothed,RECT_WIDTH,RECT_HEIGHT);
    OcclusionCompletionStats stats;
    std::vector<float> completed;
    OcclusionCompletionContract contract;
    double completionMs=0;
    for(int n=0;n<6;++n) {
        const auto start=std::chrono::steady_clock::now();
        completed=completeOcclusionPlanes(smoothed,left,RECT_WIDTH,RECT_HEIGHT,&stats,&contract);
        if(n) completionMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    }
    writeDiagnosticMask(prefix,"planes",completed,RECT_WIDTH,RECT_HEIGHT);
    completed=fillOcclusionHoles(completed,smoothed,RECT_WIDTH,RECT_HEIGHT,contract,left);
    writeDiagnosticRaw(prefix+"-completion-veto-u8.raw",contract.veto);
    writeDiagnosticRaw(prefix+"-completion-provenance-u8.raw",contract.provenance);
    writeDiagnosticMask(prefix,"pre-history",completed,RECT_WIDTH,RECT_HEIGHT);
    writeDiagnosticMesh(prefix,completed,RECT_WIDTH,RECT_HEIGHT);
    write(std::string(argv[4])+"-measured.raw",depth);
    write(std::string(argv[4])+"-old-surface.raw",old);
    write(std::string(argv[4])+"-completed.raw",completed);
    std::cout<<"stereo_ms="<<stereoMs/4<<", stereo_valid="<<rawValid
             <<", cleaned="<<valid(depth)<<", smoothed="<<stats.measured
             <<", old_surface="<<valid(old)<<", completed_surface="<<valid(completed)
             <<", old_triangles="<<occlusionSurfaceIndices(old,RECT_WIDTH,RECT_HEIGHT).size()/3
             <<", completed_triangles="<<occlusionSurfaceIndices(completed,RECT_WIDTH,RECT_HEIGHT).size()/3
             <<", plane_filled="<<stats.filled<<", planes="<<stats.planes
             <<", overlap_conflicts="<<stats.conflicts<<", completion_ms="<<completionMs/5<<'\n';
} catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
