#include "occlusion_surface.hpp"
#include <cassert>
#include <iostream>
using namespace FrameSpatial;
int main() {
    constexpr int w=5,h=5; const size_t n=w*h;
    OcclusionReprojection identity;
    OcclusionSurfaceHistory history({2,2,1,.1f,.2f,.2f});
    std::vector<float> plane(n,1),missing(n,0),farther(n,2),anchors(n,0);
    std::vector<uint8_t> measured(n,MeasuredDepth),inferred(n,InferredDepth),unknown(n,UnknownDepth);
    auto update=[&](const auto& depth,const auto& evidence,float elapsed,const OcclusionReprojection* transform=nullptr,const std::vector<float>* ages=nullptr) {
        return history.update(depth,w,h,10,2,2,transform?transform:(elapsed>0?&identity:nullptr),elapsed,&evidence,&evidence,ages);
    };
    update(plane,measured,0,nullptr);
    assert(history.events().empty()); // Ordinary production/replay allocates no diagnostic image.
    history.enableDiagnostics(true); history.reset();
    update(plane,measured,0,nullptr);
    assert(history.events()[0]==HistoryPending);
    update(plane,measured,.04f);
    assert(history.events()[0]==HistoryConfirmed);
    update(plane,measured,.04f);
    assert(history.events()[0]==HistoryMeasuredAgreement);
    update(farther,measured,.04f);
    assert(history.events()[0]==(HistoryMeasuredClear|HistoryPending));
    update(farther,measured,.04f);
    assert(history.events()[0]==HistoryConfirmed);
    update(missing,unknown,.15f);
    assert(history.events()[0]==HistoryHeld);
    update(missing,unknown,.1f);
    assert(history.events()[0]==(HistoryPredictionExpiry|HistorySourceExpiry));
    // Repeated predictions renew prediction age, not unsupported source age.
    history.reset(); update(plane,inferred,0,nullptr);
    assert(history.events()[0]==HistoryConfirmed);
    for(int i=0;i<3;++i) update(plane,inferred,.04f);
    assert((history.events()[0]&(HistoryInferredAgreement|HistorySourceExpiry))==
           (HistoryInferredAgreement|HistorySourceExpiry));
    assert(history.supportLifetimes()[0]==0);
    update(plane,inferred,.04f,&identity,&anchors);
    assert(!(history.events()[0]&HistorySourceExpiry));
    assert(history.supportLifetimes()[0]>.09f);
    update(plane,measured,.04f);
    assert(history.events()[0]&(HistoryMeasuredAgreement|HistoryEvidenceChange));
    // Out-of-view samples are marked at the old source coordinates.
    history.reset(); update(plane,measured,0,nullptr); update(plane,measured,.04f);
    OcclusionReprojection shifted; shifted.translation[0]=.4f;
    update(missing,unknown,.04f,&shifted);
    assert(history.events()[4]&HistoryReprojectionLoss);
    assert(!(history.events()[0]&HistoryReprojectionLoss));
    // Multiple old samples rounding onto one pixel mark the current destination.
    history.reset(); update(plane,measured,0,nullptr); update(plane,measured,.04f);
    OcclusionReprojection away; away.translation[2]=1;
    update(missing,unknown,.04f,&away);
    assert(history.events()[2*w+2]&HistoryCollision);
    history.enableDiagnostics(false); update(plane,measured,.04f);
    assert(history.events().empty());
    std::cout<<"History event tests passed: opt-in allocation, pending/confirmation, clear, prediction/source expiry, inferred provenance, evidence change, transport loss and collisions\n";
}
