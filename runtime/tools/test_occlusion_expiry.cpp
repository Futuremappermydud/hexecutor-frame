#include "occlusion_surface.hpp"
#include "occlusion_capture_gate.hpp"
#include <cassert>
#include <iostream>
using namespace FrameSpatial;
int main() {
    OcclusionCaptureGate gate;
    assert(!gate.update(false,true,1));
    assert(gate.update(true,false,2));
    assert(gate.update(false,true,2.014));
    assert(gate.update(false,true,2.24));
    assert(!gate.update(false,true,2.26)); // Gaps do not renew the grace period.
    assert(!gate.update(false,true,2.27));
    assert(gate.update(true,false,3));
    assert(!gate.update(false,false,3.01)); // Explicit incompatible submission.
    assert(!gate.update(false,true,3.02));
    assert(gate.update(true,false,4)); gate.reset();
    assert(!gate.update(false,true,4.01)); // Session/tracking reset is immediate.
    constexpr int w=5,h=5;
    OcclusionReprojection identity;
    OcclusionSurfaceHistory history({1,1,1,.15f,.25f,.25f});
    std::vector<float> plane(w*h,1),missing(w*h,0),anchorAges(w*h,0);
    std::vector<uint8_t> measured(w*h,MeasuredDepth),inferred(w*h,InferredDepth),unknown(w*h,UnknownDepth);
    auto update=[&](const auto& depth,const auto& evidence,float elapsed,const std::vector<float>* anchors=nullptr) {
        return history.update(depth,w,h,10,2,2,&identity,elapsed,&evidence,&evidence,anchors);
    };
    update(plane,measured,0);
    assert(std::abs(history.supportLifetimes()[0]-.25f)<1e-6f);
    update(missing,unknown,.10f);
    const auto mixed=history.supportLifetimes();
    assert(std::abs(mixed[0]-.15f)<1e-6f);
    measured[1*w+1]=UnknownDepth; plane[1*w+1]=0;
    const auto depth=update(plane,measured,.04f);
    assert(std::abs(history.supportLifetimes()[0]-.25f)<1e-6f);
    assert(std::abs(history.supportLifetimes()[1*w+1]-.11f)<1e-6f);
    // An interior sample constrains its coarse cell even when it is not a vertex.
    assert(std::abs(occlusionSurfaceMinLifetime(depth,history.supportLifetimes(),w,h)-.11f)<1e-6f);
    // Cells carry independent deadlines; an old interior sample cannot discard
    // fresh cells elsewhere in the same publication.
    auto cellLifetimes=occlusionSurfaceCellLifetimes(depth,history.supportLifetimes(),w,h,
                                                    occlusionSurfaceIndices(depth,w,h));
    assert(cellLifetimes.size()==4 && std::abs(cellLifetimes[0]-.11f)<1e-6f);
    assert(std::abs(cellLifetimes[1]-.25f)<1e-6f);
    auto expiredLifetimes=history.supportLifetimes(); expiredLifetimes[1*w+1]=0;
    cellLifetimes=occlusionSurfaceCellLifetimes(depth,expiredLifetimes,w,h,
                                              occlusionSurfaceIndices(depth,w,h));
    assert(cellLifetimes[0]==0 && cellLifetimes[1]>.2f);
    // Transport never resets source support age or extends its absolute deadline.
    update(missing,unknown,.04f);
    assert(std::abs(history.supportLifetimes()[1*w+1]-.07f)<1e-6f);
    // Predictions with no declared measured-anchor support cannot keep refreshing
    // source stereo age, even though their prediction age is continuously fresh.
    history.reset(); update(std::vector<float>(w*h,1),inferred,0);
    for(int i=0;i<6;++i) update(std::vector<float>(w*h,1),inferred,.05f);
    assert(history.supportLifetimes()[0]==0);
    // Explicit current measured anchors legitimately renew inferred support with
    // the shorter inferred deadline. Old anchor ages remain bounded independently.
    history.reset();
    for(int i=0;i<10;++i) update(std::vector<float>(w*h,1),inferred,.05f,&anchorAges);
    assert(std::abs(history.supportLifetimes()[0]-.15f)<1e-6f);
    anchorAges.assign(w*h,.14f);
    update(std::vector<float>(w*h,1),inferred,.02f,&anchorAges);
    assert(std::abs(history.supportLifetimes()[0]-.01f)<1e-6f);
    history.reset(); assert(history.supportLifetimes().empty());
    // Legacy confirmation counts remain three, independently of confidence config.
    plane.assign(w*h,1); measured.assign(w*h,MeasuredDepth);
    assert(history.update(plane,w,h,10,2,2,nullptr,0,nullptr,&measured,&anchorAges)[0]==0);
    assert(history.update(plane,w,h,10,2,2,&identity,.02f,nullptr,&measured,&anchorAges)[0]==0);
    assert(history.update(plane,w,h,10,2,2,&identity,.02f,nullptr,&measured,&anchorAges)[0]==1);
    // Guarded GPU pairs can arrive 800 ms apart; confirmation must survive
    // that gap, and missing data must still expire at the capture deadline.
    OcclusionSurfaceHistory guarded({}, 2.5f);
    const std::vector<float> fresh(w*h,1), absent(w*h,0), zeroAges(w*h,0);
    assert(guarded.update(fresh,w,h,10,2,2,nullptr,0,nullptr,&measured,&zeroAges)[0]==0);
    assert(guarded.update(fresh,w,h,10,2,2,&identity,.8f,nullptr,&measured,&zeroAges)[0]==0);
    assert(guarded.update(fresh,w,h,10,2,2,&identity,.8f,nullptr,&measured,&zeroAges)[0]==1);
    assert(std::abs(guarded.supportLifetimes()[0]-2.5f)<1e-6f);
    for(int i=0;i<3;++i)
        assert(guarded.update(absent,w,h,10,2,2,&identity,.8f)[0]==1);
    assert(guarded.supportLifetimes()[0]<.11f);
    assert(guarded.update(absent,w,h,10,2,2,&identity,.2f)[0]==0);
    std::cout<<"Expiry tests passed: mixed support ages, independent cell and covered interior deadlines, reprojection, unanchored inference, measured-anchor provenance, legacy confirmation, reset\n";
}
