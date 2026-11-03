#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace FrameSpatial {
constexpr int MODEL_WIDTH = 256;
constexpr int MODEL_HEIGHT = 320;

// Pixel-center resize of already rectified luma. Raw DMA frames never leave
// the capture process. Horizontal and vertical scales are independent.
inline std::vector<uint8_t> modelResize(const std::vector<uint8_t>& image, int width, int height) {
    if(width<2 || height<2 || image.size()!=size_t(width)*height)
        throw std::invalid_argument("invalid rectified luma dimensions");
    std::vector<uint8_t> out(MODEL_WIDTH*MODEL_HEIGHT);
    for(int y=0;y<MODEL_HEIGHT;y++)for(int x=0;x<MODEL_WIDTH;x++) {
        const float sx=std::clamp((x+.5f)*width/MODEL_WIDTH-.5f,0.f,float(width-1));
        const float sy=std::clamp((y+.5f)*height/MODEL_HEIGHT-.5f,0.f,float(height-1));
        const int x0=int(sx),y0=int(sy),x1=std::min(x0+1,width-1),y1=std::min(y0+1,height-1);
        const float a=sx-x0,b=sy-y0;
        const float upper=image[size_t(y0)*width+x0]*(1-a)+image[size_t(y0)*width+x1]*a;
        const float lower=image[size_t(y1)*width+x0]*(1-a)+image[size_t(y1)*width+x1]*a;
        out[size_t(y)*MODEL_WIDTH+x]=uint8_t(std::lround(upper*(1-b)+lower*b));
    }
    return out;
}

inline uint32_t modelCensus(const std::vector<uint8_t>& image,int x,int y) {
    uint32_t code=0;const uint8_t center=image[size_t(y)*MODEL_WIDTH+x];
    for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++) {
        if(!dx && !dy)continue;
        code=(code<<1)|(image[size_t(y+dy)*MODEL_WIDTH+x+dx]<center);
    }
    return code;
}

// Plane 0 is treated as a provisional left-image disparity in model pixels.
// Reject range, borders, low texture and unsupported matches independently of
// the model's auxiliary head (whose confidence semantics are unknown).
inline std::vector<float> modelMetricDepth(const std::vector<float>& disparity,
    const std::vector<uint8_t>& left,const std::vector<uint8_t>& right,
    int width,int height,float focalBaseline,float minDepth=.25f,float maxDepth=5.f,
    const std::vector<uint8_t>* uncertainty=nullptr,int maxUncertainty=89) {
    const size_t pixels=MODEL_WIDTH*MODEL_HEIGHT;
    if(disparity.size()!=pixels || left.size()!=pixels || right.size()!=pixels ||
       width<=0 || height<=0 || !(focalBaseline>0))throw std::invalid_argument("invalid Model depth geometry");
    const float modelFB=focalBaseline*MODEL_WIDTH/width;
    std::vector<float> measured(pixels,0),supported(pixels,0);
    for(int y=2;y<MODEL_HEIGHT-2;y++)for(int x=2;x<MODEL_WIDTH-2;x++) {
        const size_t i=size_t(y)*MODEL_WIDTH+x;const float d=disparity[i];
        if(!std::isfinite(d) || d<=0)continue;
        const float z=modelFB/d;
        if(z<minDepth || z>maxDepth)continue;
        const float rx=x-d;
        if(rx<2 || rx>MODEL_WIDTH-3)continue;
        const int xr=int(std::lround(rx));
        // Model uncertainty gate (Model: "Confidence maps ... threshold: 89"). It keeps
        // low-texture walls/floors that the contrast and census checks discard.
        if(uncertainty) {
            if((*uncertainty)[i]>=maxUncertainty)continue;
            measured[i]=z;continue;
        }
        uint8_t lo=255,hi=0;
        for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++) {
            const auto v=left[size_t(y+dy)*MODEL_WIDTH+x+dx];lo=std::min(lo,v);hi=std::max(hi,v);
        }
        if(int(hi)-lo<12)continue;
        if(__builtin_popcount(modelCensus(left,x,y)^modelCensus(right,xr,y))>8)continue;
        measured[i]=z;
    }
    for(int y=2;y<MODEL_HEIGHT-2;y++)for(int x=2;x<MODEL_WIDTH-2;x++) {
        const size_t i=size_t(y)*MODEL_WIDTH+x;const float z=measured[i];if(z<=0)continue;
        int support=0;
        for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++) {
            if(!dx && !dy)continue;
            const float neighbor=measured[size_t(y+dy)*MODEL_WIDTH+x+dx];
            if(neighbor>0 && std::abs(neighbor-z)<=std::max(.04f,.04f*z))++support;
        }
        if(support>=3)supported[i]=z;
    }
    // Nearest-neighbor expansion preserves invalid holes and depth steps.
    std::vector<float> out(size_t(width)*height,0);
    for(int y=0;y<height;y++)for(int x=0;x<width;x++) {
        const int mx=std::min(MODEL_WIDTH-1,int((x+.5f)*MODEL_WIDTH/width));
        const int my=std::min(MODEL_HEIGHT-1,int((y+.5f)*MODEL_HEIGHT/height));
        out[size_t(y)*width+x]=supported[size_t(my)*MODEL_WIDTH+mx];
    }
    return out;
}
} // namespace FrameSpatial
