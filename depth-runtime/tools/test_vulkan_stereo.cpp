#include "vulkan_stereo_pipeline.hpp"
#include "calibration.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
using namespace FrameSpatial;
int main(int argc,char** argv) {
    VulkanStereoPipeline stereo;
    if(!stereo.initialize(argc>1?argv[1]:"shaders")) {
        std::cerr<<stereo.lastError()<<'\n'; return 1;
    }
    VulkanStereoPipeline diagnosticStereo;
    if(!diagnosticStereo.initialize(argc>1?argv[1]:"shaders",true)) {
        std::cerr<<diagnosticStereo.lastError()<<'\n'; return 1;
    }
    StereoDiagnostics diagnostics;
    constexpr int w=RECT_WIDTH,h=RECT_HEIGHT;
    std::vector<uint8_t> left(w*h,100),right(left);
    std::vector<float> depth;
    assert(stereo.process(left,right,depth));
    std::vector<float> diagnosticDepth;
    assert(diagnosticStereo.process(left,right,diagnosticDepth,&diagnostics));
    assert(depth==diagnosticDepth);
    assert(diagnostics.leftMatch.size()==size_t(w)*h*4);
    for(size_t i=0;i<depth.size();++i) {
        assert(diagnostics.supportRejection[i]==uint32_t(diagnostics.consistencyDepth[i]>0 && depth[i]==0));
        assert(!(diagnostics.filterRejection[i] && diagnostics.consistencyDepth[i]>0));
    }
    for(float z:depth) assert(z==0); // A blank surface supplies no stereo evidence.
    // Repeated patterns have equally good, separated hypotheses and must stay
    // invalid even after adjacent disparities stop counting as competitors.
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
        left[y*w+x]=uint8_t(30+(x%8)*25);
        right[y*w+x]=left[y*w+x];
    }
    assert(stereo.process(left,right,depth));
    assert(diagnosticStereo.process(left,right,diagnosticDepth,&diagnostics));
    assert(depth==diagnosticDepth);
    for(int y=16;y<h-16;++y) for(int x=110;x<w-110;++x) assert(depth[y*w+x]==0);
    // A textured plane at fractional disparity exercises subpixel refinement,
    // both reference directions, consistency and local support on the real GPU.
    std::mt19937 random(42);
    std::vector<uint8_t> noise(w*h);
    for(auto& p:noise) p=uint8_t(20+random()%216);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
        int sum=0;
        for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx)
            sum+=noise[std::clamp(y+dy,0,h-1)*w+std::clamp(x+dx,0,w-1)];
        left[y*w+x]=uint8_t(sum/9);
    }
    constexpr float fb=24.282608f;
    for(float disparity:{3.35f,5.2f,8.1f,18.35f}) {
        const int integer=int(disparity); const float fraction=disparity-integer;
        for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
            const int base=std::min(x+integer,w-1),next=std::min(base+1,w-1);
            right[y*w+x]=uint8_t(std::lround((1-fraction)*left[y*w+base]+fraction*left[y*w+next]));
        }
        assert(stereo.process(left,right,depth));
        assert(diagnosticStereo.process(left,right,diagnosticDepth,&diagnostics));
        assert(depth==diagnosticDepth);
        size_t valid=0,bad=0,total=0; double error=0;
        for(int y=16;y<h-16;++y) for(int x=110;x<w-32;++x) {
            ++total; const size_t i=size_t(y)*w+x;
            assert(!diagnostics.filterRejection[i] || diagnostics.consistencyDepth[i]==0);
            assert(diagnostics.supportRejection[i]==uint32_t(diagnostics.consistencyDepth[i]>0 && depth[i]==0));
            const float z=depth[i]; if(z<=0) continue;
            ++valid; const float delta=std::abs(fb/z-disparity);
            error+=delta; bad+=delta>1;
        }
        if(disparity<fb/5) {
            assert(valid==0);
            assert(size_t(std::count(diagnostics.filterRejection.begin(),diagnostics.filterRejection.end(),uint32_t(StereoFilterRange)))>total/2);
        } else assert(valid>total/2 && bad<valid/100);
        std::cout<<"Fractional disparity="<<disparity<<", coverage="<<100.0*valid/total
                 <<"%, accepted mean disparity error="<<(valid?error/valid:0)<<" px\n";
    }
    std::cout<<"Vulkan stereo tests passed: blank/repeated texture, small/fractional disparity, range rejection, diagnostic parity\n";
}
