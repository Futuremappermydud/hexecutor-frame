#pragma once
#include "vulkan_stereo_pipeline.hpp"
#include "occlusion_surface.hpp"
#include <fstream>
#include <stdexcept>
#include <string>

namespace FrameSpatial {
template<class T> inline void writeDiagnosticRaw(const std::string& path, const std::vector<T>& data) {
    std::ofstream file(path,std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
    if(!file) throw std::runtime_error("Cannot write "+path);
}
inline void writeDiagnosticMask(const std::string& prefix,const std::string& stage,
                                const std::vector<float>& depth,int w,int h) {
    std::ofstream file(prefix+"-"+stage+".pgm",std::ios::binary);
    file<<"P5\n"<<w<<' '<<h<<"\n255\n";
    for(float z:depth) file.put(surfaceDepth(z)?char(255):char(0));
    if(!file) throw std::runtime_error("Cannot write stage mask");
    writeDiagnosticRaw(prefix+"-"+stage+"-f32.raw",depth);
}
inline void writeDiagnosticLuma(const std::string& path,const std::vector<uint8_t>& luma,int w,int h) {
    std::ofstream file(path,std::ios::binary);
    file<<"P5\n"<<w<<' '<<h<<"\n255\n";
    file.write(reinterpret_cast<const char*>(luma.data()),luma.size());
    if(!file) throw std::runtime_error("Cannot write luma");
}
inline void writeStereoDiagnostics(const std::string& prefix,const StereoDiagnostics& d,int w,int h) {
    writeDiagnosticRaw(prefix+"-left-disparity-f32.raw",d.leftDisparity);
    writeDiagnosticRaw(prefix+"-right-disparity-f32.raw",d.rightDisparity);
    writeDiagnosticRaw(prefix+"-left-match-rgba-f32.raw",d.leftMatch);
    writeDiagnosticRaw(prefix+"-right-match-rgba-f32.raw",d.rightMatch);
    writeDiagnosticRaw(prefix+"-filter-rejection-u32.raw",d.filterRejection);
    writeDiagnosticRaw(prefix+"-support-rejection-u32.raw",d.supportRejection);
    writeDiagnosticMask(prefix,"consistency",d.consistencyDepth,w,h);
    const auto mask=[&](const std::string& name,const auto& predicate) {
        std::vector<uint8_t> values(size_t(w)*h);
        for(size_t i=0;i<values.size();++i) values[i]=predicate(i)?255:0;
        writeDiagnosticLuma(prefix+"-"+name+".pgm",values,w,h);
    };
    for(const auto& side:{std::string("left"),std::string("right")}) {
        const auto& match=side=="left"?d.leftMatch:d.rightMatch;
        mask(side+"-match-rejection",[&](size_t i){return match[4*i+3]!=0;});
        mask(side+"-ambiguity",[&](size_t i){return uint32_t(match[4*i+3])&StereoMatchAmbiguous;});
        mask(side+"-cost-rejection",[&](size_t i){return uint32_t(match[4*i+3])&StereoMatchCost;});
    }
    mask("lr-rejection",[&](size_t i){return d.filterRejection[i]&StereoFilterLR;});
    mask("low-disparity-rejection",[&](size_t i){return d.filterRejection[i]&StereoFilterLowDisparity;});
    mask("range-rejection",[&](size_t i){return d.filterRejection[i]&StereoFilterRange;});
    mask("support-rejection",[&](size_t i){return d.supportRejection[i]!=0;});
}
inline void writeDiagnosticMeshIndices(const std::string& prefix,const std::vector<size_t>& indices,int w,int h) {
    auto coverage=occlusionSurfaceMask(indices,w,h);
    for(auto& value:coverage) value=value?255:0;
    writeDiagnosticLuma(prefix+"-mesh.pgm",coverage,w,h);
    const std::vector<uint32_t> indices32(indices.begin(),indices.end());
    writeDiagnosticRaw(prefix+"-mesh-indices-u32.raw",indices32);
}
inline void writeDiagnosticMesh(const std::string& prefix,const std::vector<float>& depth,int w,int h) {
    writeDiagnosticMeshIndices(prefix,occlusionSurfaceIndices(depth,w,h),w,h);
}
inline std::vector<size_t> diagnosticLiveMeshIndices(const std::vector<size_t>& indices,
                                                    const std::vector<float>& lifetimes,
                                                    float displayAge) {
    std::vector<size_t> output;
    if(indices.size()/6!=lifetimes.size() || indices.size()%6) return output;
    for(size_t i=0;i<lifetimes.size();++i)
        if(lifetimes[i]>displayAge) output.insert(output.end(),indices.begin()+i*6,indices.begin()+(i+1)*6);
    return output;
}
} // namespace FrameSpatial
