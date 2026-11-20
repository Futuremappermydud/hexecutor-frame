#include "occlusion_completion.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace FrameSpatial;
int main() {
    constexpr int w=96,h=80;
    std::vector<uint8_t> image(w*h,100);
    std::vector<float> measured(w*h,0),truth(w*h);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
        const int i=y*w+x;
        truth[i]=1.0f/(0.5f+0.001f*x+0.0005f*y);
        if(x%12<3 && y%12<3) measured[i]=truth[i];
    }
    OcclusionCompletionStats stats;
    auto completed=completeOcclusionPlanes(measured,image,w,h,&stats);
    assert(stats.filled>size_t(w*h/2));
    assert(completed[30*w+30]>0);
    assert(occlusionSurfaceIndices(completed,w,h).size()>occlusionSurfaceIndices(measured,w,h).size());
    for(size_t i=0;i<measured.size();++i) {
        if(surfaceDepth(measured[i])) assert(completed[i]==measured[i]);
        if(surfaceDepth(completed[i])) assert(std::abs(completed[i]-truth[i])<1e-5f);
    }
    // Guide edges separate an unsupported strip even when both sides have the
    // same depths. No completion may jump the strip or use it as new support.
    for(int y=0;y<h;++y) for(int x=40;x<=44;++x) {
        image[y*w+x]=200; measured[y*w+x]=0;
    }
    completed=completeOcclusionPlanes(measured,image,w,h);
    for(int y=0;y<h;++y) for(int x=40;x<=44;++x) assert(completed[y*w+x]==0);
    // The cached undirected connections retain the exact luma threshold:
    // a 16-level change connects in both directions, a 17-level change stops.
    for(int y=0;y<h;++y) for(int x=40;x<=44;++x) image[y*w+x]=116;
    completed=completeOcclusionPlanes(measured,image,w,h);
    assert(completed[30*w+42]>0);
    for(int y=0;y<h;++y) for(int x=40;x<=44;++x) image[y*w+x]=117;
    completed=completeOcclusionPlanes(measured,image,w,h);
    for(int y=0;y<h;++y) for(int x=40;x<=44;++x) assert(completed[y*w+x]==0);
    // No extrapolation outside measured support, including same-color objects.
    image.assign(w*h,100);
    for(int y=0;y<h;++y) for(int x=48;x<w;++x) measured[y*w+x]=0;
    completed=completeOcclusionPlanes(measured,image,w,h);
    for(int y=0;y<h;++y) for(int x=48;x<w;++x) assert(completed[y*w+x]==0);
    // Two different depths without an image boundary must not produce a ramp
    // across their unmeasured dividing line.
    for(int y=0;y<h;++y) for(int x=0;x<w;++x)
        measured[y*w+x]=(x%12<3 && y%12<3)?(x<48?1.0f:2.0f):0;
    completed=completeOcclusionPlanes(measured,image,w,h);
    assert(completed[30*w+46]==0 && completed[30*w+47]==0);
    assert(completed[30*w+30]==1 && completed[30*w+66]==2);
    // Collinear anchors cannot determine a plane, and a blank image with no
    // measured depth cannot conjure a room surface.
    measured.assign(w*h,0);
    for(int x=0;x<w;++x) measured[30*w+x]=1;
    assert(completeOcclusionPlanes(measured,image,w,h)==measured);
    measured.assign(w*h,0);
    assert(completeOcclusionPlanes(measured,image,w,h)==measured);
    assert(completeOcclusionPlanes(measured,{},w,h)==measured);
    // A large enclosed hole stays unknown beyond the bounded support distance.
    measured.assign(w*h,1);
    for(int y=8;y<72;++y) for(int x=8;x<88;++x) measured[y*w+x]=0;
    assert(completeOcclusionPlanes(measured,image,w,h)[40*w+48]==0);
    // A lone invalid floating-point sample is not a fit anchor.
    measured.assign(w*h,0); measured[0]=std::numeric_limits<float>::infinity();
    completed=completeOcclusionPlanes(measured,image,w,h,&stats);
    assert(stats.measured==0 && stats.filled==0);
    // Inferred brackets alone cannot enclose a hole. Preserve previously
    // completed values, and retain their explicit plane provenance.
    constexpr int hw=17, hh=17;
    std::vector<float> anchors(hw*hh,0), inferred(hw*hh,0);
    std::vector<uint8_t> guide(hw*hh,100);
    OcclusionCompletionContract contract;
    contract.veto.assign(hw*hh,0);
    contract.provenance.assign(hw*hh,CompletionUnknown);
    const int center=8*hw+8;
    for(int i : {center-1,center+1,center-hw,center+hw}) {
        inferred[i]=1; contract.provenance[i]=CompletionPlane;
    }
    assert(fillOcclusionHoles(inferred,hw,hh)[center]==1); // Previous chaining.
    auto corrected=fillOcclusionHoles(inferred,anchors,hw,hh,contract,guide);
    assert(corrected==inferred && contract.provenance[center]==CompletionUnknown);
    // Actual measured enclosure fills once with hole provenance. Veto at the
    // target or along a bracket search prevents filling for every veto reason.
    anchors=inferred;
    for(int i : {center-1,center+1,center-hw,center+hw})
        contract.provenance[i]=CompletionMeasured;
    corrected=fillOcclusionHoles(inferred,anchors,hw,hh,contract,guide);
    assert(corrected[center]==1 && contract.provenance[center]==CompletionHole);
    for(uint8_t reason : {CompletionOverlapConflict,CompletionContradictorySupport,
                          CompletionProtectedBoundary}) {
        contract.veto[center]=reason;
        assert(fillOcclusionHoles(inferred,anchors,hw,hh,contract,guide)[center]==0);
        contract.veto[center]=0;
        contract.veto[center+1]=reason;
        assert(fillOcclusionHoles(inferred,anchors,hw,hh,contract,guide)[center]==0);
        contract.veto[center+1]=0;
    }
    // Guide and depth barriers cannot be erased by four agreeing brackets.
    guide[center+1]=117;
    assert(fillOcclusionHoles(inferred,anchors,hw,hh,contract,guide)[center]==0);
    guide[center+1]=116;
    assert(fillOcclusionHoles(inferred,anchors,hw,hh,contract,guide)[center]==1);
    guide[center+1]=100;
    anchors[center+1]=2;
    assert(fillOcclusionHoles(inferred,anchors,hw,hh,contract,guide)[center]==0);
    assert(fillOcclusionHoles(inferred,anchors,hw,hh,contract,{})==inferred);
    // Locally acceptable planes over a curved surface disagree in overlaps.
    // The old pass actually reopens these conflicts; the corrected one cannot.
    measured.assign(w*h,0); image.assign(w*h,100);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x)
        if(x%6==0 && y%6==0) measured[y*w+x]=1.f+0.00006f*(x-48)*(x-48);
    completed=completeOcclusionPlanes(measured,image,w,h,&stats,&contract);
    assert(stats.conflicts>0);
    const auto oldHoles=fillOcclusionHoles(completed,w,h);
    corrected=fillOcclusionHoles(completed,measured,w,h,contract,image);
    size_t previouslyReopened=0;
    for(size_t i=0;i<measured.size();++i) {
        if(contract.veto[i]&CompletionOverlapConflict) {
            assert(completed[i]==0 && corrected[i]==0);
            previouslyReopened+=surfaceDepth(oldHoles[i]);
        }
        if(surfaceDepth(measured[i])) {
            assert(corrected[i]==measured[i]);
            assert(contract.provenance[i]==CompletionMeasured);
        } else if(surfaceDepth(completed[i])) {
            assert(corrected[i]==completed[i]);
            assert(contract.provenance[i]==CompletionPlane);
        }
    }
    assert(previouslyReopened>0);
    // A contradictory anchor protects its nearby unknown support region, even
    // if another overlapping tile offers an otherwise acceptable prediction.
    for(int y=0;y<h;++y) for(int x=0;x<w;++x)
        measured[y*w+x]=(x%6==0 && y%6==0)?1.f:0;
    measured[36*w+36]=2;
    completed=completeOcclusionPlanes(measured,image,w,h,&stats,&contract);
    assert(contract.veto[36*w+37]&CompletionContradictorySupport);
    corrected=fillOcclusionHoles(completed,measured,w,h,contract,image);
    assert(corrected[36*w+36]==2 && corrected[36*w+37]==0);
    for(int y=0;y<h;++y) image[y*w+40]=117;
    completed=completeOcclusionPlanes(measured,image,w,h,&stats,&contract);
    assert(contract.veto[30*w+40]&CompletionProtectedBoundary);
    assert(fillOcclusionHoles(completed,measured,w,h,contract,image)[30*w+40]==0);
    // Reduction masks/provenance as well as depths remain deterministic.
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif
    const auto eightDepth=completeOcclusionPlanes(measured,image,w,h,&stats,&contract);
    auto eightContract=contract;
    const auto eightHoles=fillOcclusionHoles(eightDepth,measured,w,h,eightContract,image);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    const auto oneDepth=completeOcclusionPlanes(measured,image,w,h,&stats,&contract);
    assert(oneDepth==eightDepth && contract.veto==eightContract.veto);
    const auto oneHoles=fillOcclusionHoles(oneDepth,measured,w,h,contract,image);
    assert(oneHoles==eightHoles && contract.provenance==eightContract.provenance);
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif
    // Deterministic tile reduction on the actual frame size; time the worker
    // operation independently of Vulkan and render-loop submission.
    constexpr int size=616;
    image.assign(size*size,100); measured.assign(size*size,0);
    for(int y=0;y<size;++y) for(int x=0;x<size;++x)
        if(x%12<3 && y%12<3) measured[y*size+x]=1.0f/(0.5f+0.0001f*x);
    const auto start=std::chrono::steady_clock::now();
    completed=completeOcclusionPlanes(measured,image,size,size,&stats);
    const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    assert(completed==completeOcclusionPlanes(measured,image,size,size));
    std::cout<<"Completion tests passed: slanted sparse planes, guide edges/thresholds, silhouettes, "
             <<"depth steps, collinear/no support, bounded holes, invalid input, inferred brackets, "
             <<"conflict/boundary vetoes, measured preservation, deterministic masks/provenance; "
             <<stats.measured<<" measured, "<<stats.filled<<" filled, "<<ms<<" ms\n";
}
