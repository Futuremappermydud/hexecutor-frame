#include "meshing_pipeline.hpp"
#include <cassert>
#include <iostream>
#include <limits>
#include <random>

using namespace FrameSpatial;
namespace { size_t projections = 0; }
namespace FrameSpatial {
DepthPipeline::DepthPipeline() : m_initialized(false) {}
DepthPipeline::~DepthPipeline() = default;
std::array<float,3> DepthPipeline::rectifiedPointInHead(int x,int y,float z) const {
    ++projections;
    return {x*z/100.f, y*z/100.f, z};
}
}

static void checkFilter() {
    constexpr int width = 31, height = 23;
    constexpr float kernel[]{.06136f,.24477f,.38774f,.24477f,.06136f};
    std::mt19937 rng(581);
    DepthFilterWorkspace workspace;
    // Independent 2D weighted oracle, including holes and stale workspace reuse.
    for (int p : {1,2,4}) for (int radius : {0,1,2}) {
        std::vector<float> depth(width*height);
        for (float& z : depth) z = rng()%5 ? .25f + float(rng()%475)/100.f : 0;
        depth[37] = std::numeric_limits<float>::quiet_NaN();
        depth[90] = std::numeric_limits<float>::infinity();
        const auto& filtered = powerMeanFilterDepth(depth,width,height,workspace,p,radius);
        for (int y=0;y<height;++y) for(int x=0;x<width;++x) {
            float sum=0, weight=0;
            for(int dy=-radius;dy<=radius;++dy) for(int dx=-radius;dx<=radius;++dx) {
                const int nx=x+dx, ny=y+dy;
                if(nx<0 || ny<0 || nx>=width || ny>=height) continue;
                const float z=depth[size_t(ny)*width+nx];
                if(!std::isfinite(z) || z<.25f || z>5.f) continue;
                const float w=kernel[dy+radius]*kernel[dx+radius];
                const float inv=1/z, square=inv*inv;
                sum += (p==1 ? inv : p==2 ? square : square*square)*w;
                weight += w;
            }
            float expected=0;
            if(weight>=.12f) {
                const float avg=sum/weight;
                expected=1.f/(p==1 ? avg : p==2 ? std::sqrt(avg) : std::sqrt(std::sqrt(avg)));
            }
            const float actual=filtered[size_t(y)*width+x];
            assert(std::isfinite(actual) && std::abs(actual-expected)<2e-5f);
        }
        std::fill(depth.begin(),depth.end(),0);
        for(float z:powerMeanFilterDepth(depth,width,height,workspace,p,radius)) assert(z==0);
    }
    std::vector<float> resized(7*5,2.f);
    const auto& result=powerMeanFilterDepth(resized,7,5,workspace);
    assert(result.size()==resized.size());
    for(float z:result) assert(std::abs(z-2.f)<1e-6f);
}

static void checkMesh() {
    DepthPipeline rectifier;
    const XrPosef pose{{0,0,0,1},{.3f,-.2f,.1f}};
    // Odd image dimensions and both even/odd strides exercise ring-row reuse.
    for(int step : {1,2,3}) {
        const int width=19, height=17;
        std::vector<float> depth(width*height,2.f);
        projections=0;
        const auto mesh=triangulateDepthMap(depth,{},rectifier,pose,width,height,step);
        const size_t cols=(width-1)/step+1, rows=(height-1)/step+1;
        assert(mesh.size()==(cols-1)*(rows-1)*6);
        assert(projections==cols*rows);
        for(const auto& v:mesh) {
            assert(std::abs(v.z-2.1f)<1e-6f);
            const float x=(v.x-.3f)*50, y=(v.y+.2f)*50;
            assert(std::abs(x/step-std::round(x/step))<1e-5f);
            assert(std::abs(y/step-std::round(y/step))<1e-5f);
        }
    }
    // Each missing corner still leaves one supported triangle in a single cell.
    for(int missing=0;missing<4;++missing) {
        std::vector<float> depth(4,2.f); depth[missing]=0;
        const auto mesh=triangulateDepthMap(depth,{},rectifier,pose,2,2,1);
        assert(mesh.size()==3);
    }
    const std::vector<float> tear{1,3,1,3};
    projections=0;
    assert(triangulateDepthMap(tear,{},rectifier,pose,2,2,1).empty());
    assert(projections==0);
    assert(triangulateDepthMap({2}, {},rectifier,pose,1,1).empty());
    const float half = std::sqrt(.5f);
    const XrPosef yaw{{0,half,0,half},{.3f,-.2f,.1f}};
    const std::vector<float> plane(19*17,2.f);
    std::vector<FrameSpatialCloud::PointVertex> reused;
    triangulateDepthMap(plane,{},rectifier,yaw,19,17,reused,3);
    for(const auto& v:reused) {
        assert(std::abs(v.x-2.3f)<1e-5f);
        const float x=(.1f-v.z)*50, y=(v.y+.2f)*50;
        assert(std::abs(x/3-std::round(x/3))<1e-5f);
        assert(std::abs(y/3-std::round(y/3))<1e-5f);
    }
    triangulateDepthMap({}, {},rectifier,yaw,0,0,reused);
    assert(reused.empty());
}
int main() {
    checkFilter(); checkMesh();
    std::cout << "Meshing PASS: filter oracle, stale buffers, projection reuse, holes, strides and tear gates\n";
}
