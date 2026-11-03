#include "model_depth_pipeline.hpp"
#include "calibration.hpp"
#include "vulkan_stereo_pipeline.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
using namespace FrameSpatial;
int main(int argc,char** argv) {
    if(argc!=4){std::cerr<<"Usage: "<<argv[0]<<" left.raw right.raw output-depth.raw\n";return 2;}
    std::vector<uint8_t> l(RECT_WIDTH*RECT_HEIGHT),r(l.size());
    for(int eye=0;eye<2;eye++) {
        auto& image=eye?r:l;std::ifstream f(argv[1+eye],std::ios::binary);
        if(!f.read(reinterpret_cast<char*>(image.data()),image.size()) || f.peek()!=EOF){std::cerr<<"bad rectified input\n";return 2;}
    }
    std::vector<float> depth;
    {
        ModelDepthPipeline model;
        if(!model.initialize()){std::cerr<<model.lastError()<<'\n';return 1;}
        // With FRAME_DEPTH_TEMPORAL=1 the same pair is a static camera: identity motion.
        const char* framesSetting=std::getenv("MODEL_TEST_FRAMES");
        const int frames=framesSetting?std::max(1,std::atoi(framesSetting)):3;
        std::vector<float> previous;
        for(int i=0;i<frames;i++) {
            const auto begin=std::chrono::steady_clock::now();
            ModelMotion motion;motion.aligned=i>0;
            if(!model.process(l,r,depth,motion)){std::cerr<<model.lastError()<<'\n';return 1;}
            std::vector<float> change;
            for(size_t k=0;k<previous.size();k++)if(depth[k]>0 && previous[k]>0)change.push_back(std::abs(depth[k]-previous[k])/previous[k]);
            std::sort(change.begin(),change.end());
            previous=depth;
            std::vector<float> good;for(float z:depth)if(z>0)good.push_back(z);
            std::sort(good.begin(),good.end());
            std::cout<<"Model frame="<<i<<" inference_ms="<<model.lastComputeMs()
                     <<" total_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()
                     <<" valid="<<good.size()<<" median_m="<<(good.empty()?0:good[good.size()/2])
                     <<" median_rel_change="<<(change.empty()?0:change[change.size()/2])<<std::endl;
        }
    }
    std::ofstream out(argv[3],std::ios::binary);out.write(reinterpret_cast<char*>(depth.data()),depth.size()*sizeof(float));
    if(!out)return 1;
    VulkanStereoPipeline baseline;std::vector<float> reference;
    if(!baseline.initialize() || !baseline.process(l,r,reference)){std::cerr<<baseline.lastError()<<'\n';return 1;}
    std::vector<float> errors;size_t baselineValid=0;
    for(size_t i=0;i<depth.size();i++) {
        baselineValid+=reference[i]>0;
        if(depth[i]>0 && reference[i]>0)errors.push_back(std::abs(depth[i]-reference[i])/reference[i]);
    }
    std::sort(errors.begin(),errors.end());
    std::cout<<"Baseline valid="<<baselineValid<<" overlap="<<errors.size()<<" median_relative_difference="
             <<(errors.empty()?0:errors[errors.size()/2])<<" (comparison, not ground truth)\n";
}
