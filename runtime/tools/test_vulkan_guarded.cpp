#include "vulkan_guarded_stereo_pipeline.hpp"
#include "stereo_matcher_experiment.hpp"
#include "depth_speckle_filter.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
using namespace FrameSpatial;
namespace SE=Spatial::StereoExperiment;
void require(bool value,const std::string& message) {if(!value)throw std::runtime_error(message);}
std::vector<uint8_t> pgm(const std::string& path) {
    std::ifstream in(path,std::ios::binary);std::string magic;int w,h,max;
    in>>magic>>w>>h>>max;in.get();require(magic=="P5"&&w==616&&h==616&&max==255,"Expected 616x616 PGM");
    std::vector<uint8_t> result(616*616);in.read(reinterpret_cast<char*>(result.data()),result.size());require(bool(in),"Truncated PGM");return result;
}
void maps(const std::vector<float>& cpu,const std::vector<float>& gpu,float tolerance,const std::string& label,bool allowOneUlp=false) {
    require(cpu.size()==gpu.size(),label+" size");size_t masks=0,values=0;float maximum=0;
    for(size_t i=0;i<cpu.size();++i) {
        masks+=(cpu[i]>0)!=(gpu[i]>0);float delta=std::abs(cpu[i]-gpu[i]);
        maximum=std::max(maximum,delta);
        const float ulp=allowOneUlp?std::max(std::abs(std::nextafter(cpu[i],std::numeric_limits<float>::infinity())-cpu[i]),
                                          std::abs(std::nextafter(gpu[i],std::numeric_limits<float>::infinity())-gpu[i])):0.f;
        values+=delta>std::max(tolerance,ulp);
    }
    std::cout<<label<<" mask_mismatch="<<masks<<" value_mismatch="<<values<<" max_error="<<maximum<<'\n';
    require(masks==0&&values==0,label+" parity");
}
void supported(const SE::Result& cpu,const GuardedStereoDiagnostics& gpu,const std::vector<float>& depth) {
    size_t boundary=0,unexplained=0;
    for(int y=0;y<616;++y)for(int x=0;x<616;++x) {
        const size_t i=size_t(y)*616+x;
        require(gpu.supportRejection[i]==uint32_t(gpu.consistencyDepth[i]>0&&depth[i]==0),"GPU support partition");
        if((cpu.depth[i]>0)==(depth[i]>0)) {
            require(std::abs(cpu.depth[i]-depth[i])<=2e-5f,"Supported depth values");continue;
        }
        const float cc=cpu.preSupportDepth[i],gc=gpu.consistencyDepth[i];
        if(cc<=0||gc<=0||x==0||y==0||x==615||y==615) {++unexplained;continue;}
        int certain=0,uncertain=0;bool onlyThresholds=true;
        for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
            if(dx==0&&dy==0)continue;
            const size_t j=size_t(y+dy)*616+x+dx;
            const float cn=cpu.preSupportDepth[j],gn=gpu.consistencyDepth[j];
            if((cn>0)!=(gn>0)){onlyThresholds=false;continue;}if(cn<=0)continue;
            const float cd=std::abs(cn-cc)-std::max(.04f,.04f*cc),gd=std::abs(gn-gc)-std::max(.04f,.04f*gc);
            const float bound=std::abs(cc-gc)*1.04f+std::abs(cn-gn)+4*std::numeric_limits<float>::epsilon()*std::max({cc,gc,cn,gn,.04f});
            const bool near=std::abs(cd)<=bound&&std::abs(gd)<=bound;
            if((cd<=0)!=(gd<=0)&&!near)onlyThresholds=false;
            if(near)++uncertain;else if(cd<=0&&gd<=0)++certain;
        }
        if(onlyThresholds&&certain<3&&certain+uncertain>=3)++boundary;else ++unexplained;
    }
    std::cout<<"support threshold_only="<<boundary<<" unexplained="<<unexplained<<'\n';
    require(unexplained==0,"Unexplained support-mask difference");
}
void compare(const SE::Result& cpu,const GuardedStereoDiagnostics& gpu,const std::vector<float>& depth,const SE::Config& c) {
    maps(cpu.leftDisparity,gpu.leftDisparity,2e-6f,"left disparity",true);
    maps(cpu.rightDisparity,gpu.rightDisparity,2e-6f,"right disparity",true);
    maps(cpu.preSupportDepth,gpu.consistencyDepth,2e-5f,"LR/range depth");
    maps(cpu.leftAnchors,gpu.leftAnchors,0,"left anchors");maps(cpu.rightAnchors,gpu.rightAnchors,0,"right anchors");
    const auto lm=SE::detail::boundaryMasks(cpu.leftAnchors,c),rm=SE::detail::boundaryMasks(cpu.rightAnchors,c);
    for(size_t i=0;i<depth.size();++i) {
        for(int side=0;side<2;++side) {
            const auto& match=side?gpu.rightMatch:gpu.leftMatch;
            const auto& best=side?cpu.rightBestCost:cpu.leftBestCost;
            const auto& gap=side?cpu.rightUniquenessGap:cpu.leftUniquenessGap;
            const auto& integer=side?cpu.rightBestIntegerDisparity:cpu.leftBestIntegerDisparity;
            const auto& reject=side?cpu.rightMatchRejection:cpu.leftMatchRejection;
            const auto& evidence=side?gpu.rightEvidence:gpu.leftEvidence;
            const auto& samples=side?cpu.rightInformation:cpu.leftInformation;
            const auto& bits=side?cpu.rightSupportBits:cpu.leftSupportBits;
            const auto& stops=side?cpu.rightAnchorStops:cpu.leftAnchorStops;
            const auto& mask=side?gpu.rightMasks:gpu.leftMasks;
            const uint64_t expected=side?rm[i]:lm[i];
            require(match[i*4]==best[i]&&match[i*4+1]==gap[i]&&match[i*4+2]==integer[i]&&match[i*4+3]==reject[i],"Exact winner/confidence diagnostic mismatch at "+std::to_string(i));
            require(evidence[i*4]==samples[i]&&evidence[i*4+1]==bits[i]&&evidence[i*4+2]==stops[i],"Exact selected evidence mismatch");
            require(mask[i*2]==uint32_t(expected)&&mask[i*2+1]==uint32_t(expected>>32),"Exact boundary mask mismatch");
        }
        require(gpu.filterRejection[i]==cpu.lrRejection[i],"LR/range rejection mismatch");
    }
    supported(cpu,gpu,depth);
}
uint32_t hash(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
int main(int argc,char** argv) {
    try {
        require(argc==2||argc==4,"Usage: test_vulkan_guarded shaders [left.pgm right.pgm]");
        SE::Config c;c.supportRadius=8;c.edgeThreshold=32;c.supportGuard=3;c.supportExposure=true;
        for(bool integer:{true,false}) {
            c.refinement=integer?SE::Refinement::Integer:SE::Refinement::Parabola;
            VulkanGuardedStereoPipeline pipeline,plain,control;
            GuardedStereoOptions reference;
            require(control.initialize(argv[1],true,integer,false,reference),control.lastError());
            require(pipeline.initialize(argv[1],true,integer),pipeline.lastError());
            require(plain.initialize(argv[1],false,integer,true),plain.lastError());
            for(const std::string scene:{"blank","periodic","fractional","noise","clipped","silhouette","recorded"}) {
                if(scene=="recorded"&&argc!=4)continue;
                std::vector<uint8_t> left(616*616,128),right=left;
                if(scene=="recorded") {left=pgm(argv[2]);right=pgm(argv[3]);}
                else if(scene=="periodic") {
                    for(int y=0;y<616;++y)for(int x=0;x<616;++x)left[y*616+x]=right[y*616+x]=uint8_t(30+(x%8)*25);
                } else if(scene!="blank") {
                    auto intensity=[&](float x,int y,int layer) {
                        const int ix=int(std::floor(x));const float f=x-ix;
                        const auto value=[&](int xx) {return 80+int(hash(uint32_t(xx/3)*92837111u^uint32_t(y/3)*689287499u^uint32_t(layer*3571+1709))%96);};
                        return uint8_t(std::lround((1-f)*value(ix)+f*value(ix+1)));
                    };
                    const auto fg=[&](float x,int y){return scene=="silhouette"&&x>=240&&x<380&&y>=160&&y<440;};
                    for(int y=0;y<616;++y)for(int x=0;x<616;++x) {
                        const bool lf=fg(x,y),rf=fg(x+18.5f,y);
                        left[y*616+x]=intensity(x,y,lf?1:0);
                        right[y*616+x]=intensity(x+(rf?18.5f:8.25f),y,rf?1:0);
                        if(scene=="noise") {
                            left[y*616+x]=uint8_t(std::clamp(int(left[y*616+x])+int(hash(x+y*616+16381)%13)-6,0,255));
                            right[y*616+x]=uint8_t(std::clamp(int(right[y*616+x])+int(hash(x+y*616+32771)%13)-6,0,255));
                        }
                        if(scene=="clipped")right[y*616+x]=uint8_t(std::clamp(int(right[y*616+x])*2+10,0,255));
                    }
                }
                const auto start=std::chrono::steady_clock::now();
                const auto cpu=SE::match(left,right,c,SE::Method::GuardedAdaptive);
                const double cpuMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                GuardedStereoDiagnostics diagnostic;std::vector<float> output,plainOutput;
                require(pipeline.process(left,right,output,&diagnostic),pipeline.lastError());
                GuardedStereoDiagnostics original;std::vector<float> controlOutput;
                require(control.process(left,right,controlOutput,&original),control.lastError());
                require(controlOutput==output,"Optimized/control depth mismatch");
                require(original.leftDisparity==diagnostic.leftDisparity && original.rightDisparity==diagnostic.rightDisparity &&
                        original.consistencyDepth==diagnostic.consistencyDepth && original.leftMatch==diagnostic.leftMatch &&
                        original.rightMatch==diagnostic.rightMatch && original.leftEvidence==diagnostic.leftEvidence &&
                        original.rightEvidence==diagnostic.rightEvidence && original.leftMasks==diagnostic.leftMasks &&
                        original.rightMasks==diagnostic.rightMasks && original.leftAnchors==diagnostic.leftAnchors &&
                        original.rightAnchors==diagnostic.rightAnchors && original.filterRejection==diagnostic.filterRejection &&
                        original.supportRejection==diagnostic.supportRejection,"Optimized/control exact diagnostic mismatch");
                require(plain.process(left,right,plainOutput),plain.lastError());
                require(output==plainOutput,"Diagnostic/regular interleaved output mismatch");
                std::cout<<"CASE "<<scene<<" mode="<<(integer?"integer":"parabola")<<" cpu_ms="<<cpuMs<<" gpu_ms="<<plain.lastComputeMs()<<std::endl;
                compare(cpu,diagnostic,output,c);
                if(scene=="blank")for(float z:output)require(z==0,"Blank false support");
                for(int run=0;run<2;++run) {std::vector<float> repeat;require(plain.process(left,right,repeat),plain.lastError());require(repeat==output,"Repeat GPU determinism");}
            }
        }
        std::cout<<"Guarded Vulkan CPU-stage parity and repeat/diagnostic controls passed\n";return 0;
    } catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
