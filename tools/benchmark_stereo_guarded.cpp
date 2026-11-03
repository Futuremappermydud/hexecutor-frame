#include "vulkan_guarded_stereo_pipeline.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace FrameSpatial;
static std::vector<uint8_t> pgm(const char* path) {
    std::ifstream in(path,std::ios::binary);std::string magic;int w=0,h=0,max=0;
    in>>magic>>w>>h>>max;in.get();
    if(magic!="P5"||w!=616||h!=616||max!=255)throw std::runtime_error("Expected 616x616 P5 PGM");
    std::vector<uint8_t> out(w*h);in.read(reinterpret_cast<char*>(out.data()),out.size());
    if(!in)throw std::runtime_error("Truncated PGM");
    return out;
}
int main(int argc,char** argv) {
    try {
        if(argc!=6)throw std::runtime_error("Usage: benchmark_stereo_guarded shaders left.pgm right.pgm pairs output.csv (environment selects variant)");
        const int pairs=std::stoi(argv[4]);if(pairs<1)throw std::runtime_error("pairs must be positive");
        auto options=GuardedStereoOptions::fromEnvironment(); options.profile=true;
        VulkanGuardedStereoPipeline pipeline,control;
        GuardedStereoOptions reference;
        if(!pipeline.initialize(argv[1],false,false,true,options)||!control.initialize(argv[1],false,false,false,reference))
            throw std::runtime_error(pipeline.lastError()+control.lastError());
        const auto left=pgm(argv[2]),right=pgm(argv[3]);std::vector<float> expected,output;
        if(!control.process(left,right,expected))throw std::runtime_error(control.lastError());
        // Warm the selected variant; exclude it from the CSV.
        if(!pipeline.process(left,right,output)||output!=expected)throw std::runtime_error("Warmup output mismatch");
        std::ofstream csv(argv[5]);if(!csv)throw std::runtime_error("Cannot open output");
        csv<<"pair,success,pair_ms,upload_host_ms,histogram_ms,record_submit_ms,fence_ms,yield_ms,readback_ms,allocated_bytes,valid_pixels,timestamp_bits,timestamp_period_ns,stage,gpu_ms\n";
        for(int i=0;i<pairs;++i) {
            const bool ok=pipeline.process(left,right,output);const auto& p=pipeline.lastProfile();
            const bool equal=ok&&output==expected;
            const size_t valid=std::count_if(output.begin(),output.end(),[](float x){return x>0;});
            for(size_t j=0;j<std::max(size_t(1),p.gpuMs.size());++j) {
                csv<<i<<','<<equal<<','<<p.pairMs<<','<<p.uploadHostMs<<','<<p.histogramMs<<','<<p.recordSubmitMs<<','
                   <<p.fenceWaitMs<<','<<p.yieldMs<<','<<p.readbackMs<<','<<p.allocatedBytes<<','<<valid<<','
                   <<p.timestampValidBits<<','<<p.timestampPeriodNs<<',';
                if(j<p.gpuMs.size())csv<<p.stages[j]<<','<<p.gpuMs[j];else csv<<"unavailable,";
                csv<<'\n';
            }
            csv.flush();
            if(!equal)throw std::runtime_error("Failed frame or output mismatch (recorded in CSV)");
        }
        return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
