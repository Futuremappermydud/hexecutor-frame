#include "model_depth_geometry.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <limits>
#include <random>
using namespace FrameSpatial;
int main() {
    std::vector<uint8_t> left(MODEL_WIDTH*MODEL_HEIGHT),right(left.size());
    std::mt19937 rng(29);
    for(auto& v:left)v=uint8_t(rng());
    constexpr int shift=8;
    for(int y=0;y<MODEL_HEIGHT;y++)for(int x=0;x<MODEL_WIDTH-shift;x++)
        right[size_t(y)*MODEL_WIDTH+x]=left[size_t(y)*MODEL_WIDTH+x+shift];
    auto same=modelResize(left,MODEL_WIDTH,MODEL_HEIGHT);assert(same==left);
    const float fb=24.282608f;
    std::vector<float> disparity(left.size(),float(shift));
    auto depth=modelMetricDepth(disparity,left,right,616,616,fb);
    // A horizontal resize changes disparity and focal length by the SAME factor.
    const float expected=fb*(256.f/616.f)/shift;
    size_t valid=0;
    for(float z:depth)if(z>0){assert(std::abs(z-expected)<1e-6f);++valid;}
    assert(valid>300000);
    disparity.assign(disparity.size(),0);depth=modelMetricDepth(disparity,left,right,616,616,fb);
    for(float z:depth)assert(z==0);
    disparity.assign(disparity.size(),std::numeric_limits<float>::quiet_NaN());
    depth=modelMetricDepth(disparity,left,right,616,616,fb);for(float z:depth)assert(z==0);
    disparity.assign(disparity.size(),1e-5f);depth=modelMetricDepth(disparity,left,right,616,616,fb);
    for(float z:depth)assert(z==0);
    disparity.assign(disparity.size(),8);left.assign(left.size(),128);right=left;
    depth=modelMetricDepth(disparity,left,right,616,616,fb);for(float z:depth)assert(z==0);
    // Unsupported correspondence (opposite local ordering) cannot enter fusion.
    for(size_t i=0;i<left.size();i++){left[i]=uint8_t(rng());right[i]=255-left[i];}
    disparity.assign(disparity.size(),.5f);
    depth=modelMetricDepth(disparity,left,right,616,616,1.f,.25f,5.f);
    size_t rejected=0;for(float z:depth)rejected+=z==0;assert(rejected>depth.size()*95/100);
    std::cout<<"Model geometry PASS: resize, scaled metric disparity, range/NaN, texture and census rejection\n";
}
