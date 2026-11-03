#include "occlusion_benchmark.hpp"
#include <cassert>
#include <cstdio>
using namespace FrameSpatial;
using namespace FrameSpatial::OcclusionBenchmark;
int main() {
    OcclusionReprojection identity;
    const int w=9,h=9; const size_t n=w*h;
    std::vector<float> plane(n,1),missing(n,0);
    std::vector<uint8_t> strong(n,StrongDepth),inferred(n,InferredDepth),unknown(n,UnknownDepth);
    assert(occlusionEvidence(plane,plane,w,h)[4*w+4]==StrongDepth);
    assert(occlusionEvidence(plane,missing,w,h)[4*w+4]==InferredDepth);
    OcclusionSurfaceHistory fast,legacy;
    auto update=[&](OcclusionSurfaceHistory& history,const std::vector<float>& d,const std::vector<uint8_t>* e) {
        return history.update(d,w,h,10,4,4,&identity,.02f,e);
    };
    assert(update(fast,plane,&strong)[0]==0);
    assert(update(legacy,plane,nullptr)[0]==0);
    assert(update(fast,plane,&strong)[0]==1);
    assert(update(legacy,plane,nullptr)[0]==0);
    assert(update(legacy,plane,nullptr)[0]==1);
    std::vector<float> farther(n,2);
    assert(update(fast,farther,&strong)[0]==0 && fast.cleared==n);
    assert(update(legacy,farther,nullptr)[0]==1);
    assert(update(fast,farther,&strong)[0]==2);
    fast.reset();
    for(int i=0;i<3;++i) assert(update(fast,plane,&inferred)[0]==0);
    assert(update(fast,plane,&inferred)[0]==1);
    for(int i=0;i<8;++i) update(fast,missing,&unknown);
    assert(update(fast,missing,&unknown)[0]==0);
    fast.reset();
    for(int i=0;i<3;++i) assert(fast.update(plane,w,h,10,4,4,&identity,.10f,&inferred)[0]==0);
    assert(fast.update(plane,w,h,10,4,4,&identity,.10f,&inferred)[0]==1);
    fast.reset(); update(fast,plane,&inferred);
    assert(update(fast,plane,&strong)[0]==0);
    assert(update(fast,plane,&strong)[0]==0);
    assert(update(fast,plane,&strong)[0]==1);
    fast.reset(); update(fast,plane,&strong); update(fast,plane,&strong);
    assert(update(fast,farther,&inferred)[0]==1);
    // Replay variants alter confidence confirmation, while measured disagreement
    // still clears immediately and legacy ignores the confidence configuration.
    OcclusionConfidenceConfig tunedConfig{2,2,3,.15f,.20f,.20f};
    bool invalidConfigRejected=false;
    try { auto invalid=tunedConfig; invalid.measuredObservations=0; OcclusionSurfaceHistory bad(invalid); }
    catch(const std::invalid_argument&) { invalidConfigRejected=true; }
    assert(invalidConfigRejected);
    OcclusionSurfaceHistory tuned(tunedConfig),configuredLegacy(tunedConfig);
    std::vector<uint8_t> measured(n,MeasuredDepth);
    assert(update(tuned,plane,&measured)[0]==0);
    assert(update(tuned,plane,&measured)[0]==1);
    assert(update(tuned,farther,&measured)[0]==0 && tuned.cleared==n);
    assert(update(tuned,farther,&measured)[0]==2);
    assert(update(configuredLegacy,plane,nullptr)[0]==0);
    assert(update(configuredLegacy,plane,nullptr)[0]==0);
    assert(update(configuredLegacy,plane,nullptr)[0]==1);
    tuned.reset();
    for(int i=0;i<2;++i) assert(update(tuned,plane,&inferred)[0]==0);
    assert(update(tuned,plane,&inferred)[0]==1);
    assert(update(tuned,farther,&inferred)[0]==1); // inferred contradictions never clear
    for(int i=0;i<8;++i) update(tuned,missing,&unknown);
    assert(update(tuned,missing,&unknown)[0]==0); // confirmation tuning cannot extend hold
    tuned.reset(); update(tuned,plane,&measured);
    tuned.update(missing,w,h,10,4,4,&identity,.15f,&unknown);
    tuned.update(missing,w,h,10,4,4,&identity,.10f,&unknown);
    assert(update(tuned,plane,&measured)[0]==0); // expired pending observations cannot confirm
    // Longer hold is independent of the fixed 200 ms capture-gap reset.
    OcclusionSurfaceHistory longerHold({2,2,3,.20f,.25f,.25f});
    for(int i=0;i<3;++i) update(longerHold,plane,&inferred);
    assert(longerHold.update(missing,w,h,10,4,4,&identity,.18f,&unknown)[0]==1);
    assert(longerHold.update(missing,w,h,10,4,4,&identity,.04f,&unknown)[0]==0);
    longerHold.reset();
    update(longerHold,plane,&measured); update(longerHold,plane,&measured);
    assert(longerHold.update(missing,w,h,10,4,4,&identity,.21f,&unknown)[0]==0);
    const std::string path="/tmp/frame-benchmark-codec-test.bin";
    Header header; header.width=w; header.height=h; header.warmup=0; header.segment=.001f; header.recovery=0;
    Metadata meta; meta.flags=Guarded; meta.frame=1; meta.captureBeginNs=1000000; meta.captureEndNs=1000100; meta.publishNs=1000200;
    const std::vector<uint8_t> pixels(n,123);
    { Recorder recorder(path,header); recorder.append(meta,pixels,pixels,plane,plane,strong);
      meta.captureBeginNs+=10000000; meta.captureEndNs+=10000000; meta.publishNs+=10000000;
      recorder.append(meta,pixels,pixels,plane,plane,strong); }
    Reader reader(path); Frame frame;
    assert(reader.next(frame)); assert(frame.meta.flags==Guarded); assert(frame.left==pixels && frame.input==plane && frame.evidence==strong);
    assert(!reader.next(frame)); assert(reader.frames==1 && reader.complete && !reader.dropped);
    { std::ofstream bad(path,std::ios::binary|std::ios::trunc); writeHeader(bad,header); }
    bool rejected=false; try { Reader bad(path); bad.next(frame); } catch(const std::exception&) { rejected=true; }
    assert(rejected); std::remove(path.c_str());
    std::cout<<"Confidence transitions and benchmark codec tests passed\n";
}
