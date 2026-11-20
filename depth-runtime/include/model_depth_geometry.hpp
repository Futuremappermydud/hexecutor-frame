#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "frame_omp.hpp"

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
// Model-style rectified spherical model images (see model_spherical.py):
// column u -> azimuth (u-127.5)*k, row v -> elevation (v-159.5)*k, baseline B
// along +x. rho = B/(tan aL - tan aR) is the distance from the baseline axis;
// a pinhole ray (rx, ry, 1) at elevation el sees z = rho * cos(el).
// Gate: uncertainty < maxUncertainty, depth range, right column inside, and
// 3 of 8 neighbours within max(4 cm, 4%) of rho. Output: pinhole depth map.
struct ModelSphericalMap {
    float step=0,baseline=0;
    int width=0,height=0;
    std::vector<int32_t> modelIndex;   // per pinhole pixel, -1 outside
    std::vector<float> cosElevation;   // per pinhole pixel
};
inline ModelSphericalMap modelSphericalMap(float step,float baseline,int width,int height,
                                           float focal,float cx,float cy) {
    ModelSphericalMap m;m.step=step;m.baseline=baseline;m.width=width;m.height=height;
    m.modelIndex.assign(size_t(width)*height,-1);m.cosElevation.assign(size_t(width)*height,0);
    for(int y=0;y<height;y++)for(int x=0;x<width;x++) {
        const float rx=(x-cx)/focal,ry=(y-cy)/focal;
        const float az=std::atan2(rx,std::hypot(ry,1.f)),el=std::atan2(ry,1.f);
        const int u=int(std::lround(az/step+(MODEL_WIDTH-1)*.5f)),v=int(std::lround(el/step+(MODEL_HEIGHT-1)*.5f));
        const size_t i=size_t(y)*width+x;
        m.cosElevation[i]=std::cos(el);
        if(u>=0 && v>=0 && u<MODEL_WIDTH && v<MODEL_HEIGHT)m.modelIndex[i]=v*MODEL_WIDTH+u;
    }
    return m;
}
// Persistent scratch for one depth worker: rho on the model grid and, per
// model pixel, the supported rho (0 if rejected) next to its confidence.
struct ModelSphericalWorkspace {
    struct Cell {float rho,confidence;};
    std::vector<float> rho;
    std::vector<Cell> cells;
};
// Depth and confidence (cell-rate confidence of the uncertainty value; .05 outside
// the model grid) in one pass; both outputs are fully overwritten.
inline void modelSphericalPostprocess(const std::vector<float>& disparity,
    const std::vector<uint8_t>& uncertainty,const ModelSphericalMap& map,int maxUncertainty,
    ModelSphericalWorkspace& workspace,std::vector<float>& depth,std::vector<float>& confidence,
    float minDepth=.25f,float maxDepth=5.f) {
    const size_t pixels=MODEL_WIDTH*MODEL_HEIGHT;
    if(disparity.size()!=pixels || uncertainty.size()!=pixels || !(map.step>0))
        throw std::invalid_argument("invalid Model spherical depth input");
    // Border rows/columns of rho are never written and stay zero.
    if(workspace.rho.size()!=pixels)workspace.rho.assign(pixels,0);
    workspace.cells.resize(pixels);
    float* const rho=workspace.rho.data();
    ModelSphericalWorkspace::Cell* const cells=workspace.cells.data();
    std::array<float,256> confidenceOf;
    for(size_t u=0;u<confidenceOf.size();++u)
        confidenceOf[u]=std::clamp((maxUncertainty-float(u))/float(maxUncertainty-50),.05f,1.f);
    // These depend only on the column/row, not on the predicted disparity.
    std::array<float,MODEL_WIDTH> azimuth{},tanAzimuth{};
    std::array<float,MODEL_HEIGHT> cosElevation{};
    for(int u=0;u<MODEL_WIDTH;++u) {
        azimuth[u]=(u-(MODEL_WIDTH-1)*.5f)*map.step;
        tanAzimuth[u]=std::tan(azimuth[u]);
    }
    for(int v=0;v<MODEL_HEIGHT;++v)
        cosElevation[v]=std::cos((v-(MODEL_HEIGHT-1)*.5f)*map.step);
    FRAME_OMP_FOR
    for(int v=2;v<MODEL_HEIGHT-2;v++)for(int u=2;u<MODEL_WIDTH-2;u++) {
        const size_t i=size_t(v)*MODEL_WIDTH+u;const float d=disparity[i];
        float out=0;
        if(std::isfinite(d) && d>0 && uncertainty[i]<maxUncertainty && u-d>=2) {
            const float al=azimuth[u],ar=al-d*map.step;
            if(!(al>1.5f || ar<-1.5f)) {
                const float gap=tanAzimuth[u]-std::tan(ar);
                if(gap>0) {
                    const float r=map.baseline/gap;
                    const float z=r*cosElevation[v];
                    if(!(z<minDepth || z>maxDepth))out=r;
                }
            }
        }
        rho[i]=out;
    }
    FRAME_OMP_FOR
    for(int v=0;v<MODEL_HEIGHT;v++)for(int u=0;u<MODEL_WIDTH;u++) {
        const size_t i=size_t(v)*MODEL_WIDTH+u;
        float supported=0;
        const float r=rho[i];
        if(r>0) {
            int support=0;
            for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++) {
                if(!dx && !dy)continue;
                const float n=rho[size_t(v+dy)*MODEL_WIDTH+u+dx];
                if(n>0 && std::abs(n-r)<=std::max(.04f,.04f*r))++support;
            }
            if(support>=3)supported=r;
        }
        cells[i]={supported,confidenceOf[uncertainty[i]]};
    }
    const size_t count=size_t(map.width)*map.height;
    depth.resize(count);confidence.resize(count);
    const int32_t* const index=map.modelIndex.data();
    const float* const cosine=map.cosElevation.data();
    float* const depthOut=depth.data();float* const confidenceOut=confidence.data();
    FRAME_OMP_FOR
    for(int y=0;y<map.height;y++)for(size_t i=size_t(y)*map.width,e=i+map.width;i<e;i++) {
        const int m=index[i];
        if(m>=0) {
            const auto c=cells[m];
            depthOut[i]=c.rho>0?c.rho*cosine[i]:0.f;
            confidenceOut[i]=c.confidence;
        } else {depthOut[i]=0;confidenceOut[i]=.05f;}
    }
}
inline std::vector<float> modelSphericalDepth(const std::vector<float>& disparity,
    const std::vector<uint8_t>& uncertainty,const ModelSphericalMap& map,int maxUncertainty=89,
    float minDepth=.25f,float maxDepth=5.f) {
    ModelSphericalWorkspace workspace;
    std::vector<float> depth,confidence;
    modelSphericalPostprocess(disparity,uncertainty,map,maxUncertainty,workspace,depth,confidence,minDepth,maxDepth);
    return depth;
}
} // namespace FrameSpatial
