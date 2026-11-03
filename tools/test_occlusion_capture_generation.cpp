#include "cloud_worker.hpp"
#include <cassert>
#include <iostream>
using namespace FrameSpatialCloud;
using namespace FrameSpatial;
int main() {
    gRunning.store(true); setOcclusionCaptureEnabled(true);
    auto triangle=[] { return std::vector<PointVertex>{{0,0,-1,0xffffffff},{1,0,-1,0xffffffff},{0,1,-1,0xffffffff}}; };
    uint64_t historyGeneration=gCaptureGeneration.load();
    const uint64_t computationGeneration=historyGeneration;
    OcclusionSurfaceHistory history;
    OcclusionReprojection identity;
    std::vector<float> plane(9,1);
    history.update(plane,3,3,10,1,1,nullptr,0);
    history.update(plane,3,3,10,1,1,&identity,.04f);
    assert(history.update(plane,3,3,10,1,1,&identity,.04f)[0]==1);
    auto initial=triangle(); assert(publishOcclusionCloud(initial,10,10.2,computationGeneration));
    // Simulate disable/re-enable entirely while an old computation is in flight.
    setOcclusionCaptureEnabled(false);
    assert(gCloud.empty() && gCloudExpiry.load()==0);
    const auto disabledGeneration=gCaptureGeneration.load();
    assert(disabledGeneration!=computationGeneration);
    setOcclusionCaptureEnabled(false);
    assert(gCaptureGeneration.load()==disabledGeneration);
    setOcclusionCaptureEnabled(true);
    auto oldResult=triangle();
    assert(!publishOcclusionCloud(oldResult,10.05,10.25,computationGeneration));
    assert(gCloud.empty() && gCloudTime.load()==0 && gCloudExpiry.load()==0);
    // The worker discards confirmation state when it next consumes this generation.
    assert(resetOcclusionHistoryOnCaptureChange(history,historyGeneration));
    assert(!resetOcclusionHistoryOnCaptureChange(history,historyGeneration));
    assert(history.supportLifetimes().empty());
    assert(history.update(plane,3,3,10,1,1,nullptr,0)[0]==0);
    assert(history.update(plane,3,3,10,1,1,&identity,.04f)[0]==0);
    assert(history.update(plane,3,3,10,1,1,&identity,.04f)[0]==1);
    auto restarted=triangle(); assert(publishOcclusionCloud(restarted,10.1,10.3,historyGeneration));
    assert(!gCloud.empty() && gCloudTime.load()==10.1 && gCloudExpiry.load()==10.3);
    // A valid render pose without a capture-time pose must never publish a mesh.
    auto fallback=triangle();
    assert(!publishOcclusionCloud(fallback,10.15,10.35,historyGeneration,false));
    assert(gCloud.empty() && gCloudTime.load()==0 && gCloudExpiry.load()==0);
    auto aligned=triangle(); assert(publishOcclusionCloud(aligned,10.16,10.36,historyGeneration,true));
    // Worker failure rejects publication independently of capture enable state.
    gRunning.store(false); auto failed=triangle();
    assert(!publishOcclusionCloud(failed,10.2,10.4,historyGeneration));
    assert(gCloud.empty() && gCloudExpiry.load()==0);
    std::cout<<"Capture generation tests passed: in-flight disable/re-enable, idempotent pause, history reset, confirmation restart, new publication, capture-pose fallback suppression, worker failure\n";
}
