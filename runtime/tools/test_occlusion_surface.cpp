#include "occlusion_surface.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
using namespace FrameSpatial;
int main() {
    constexpr int w=25, h=25;
    std::vector<float> plane(w*h, 1.0f);
    for (int y=9; y<=15; ++y) for (int x=9; x<=15; ++x) plane[y*w+x]=0;
    auto filled=fillOcclusionHoles(plane,w,h);
    assert(filled[12*w+12]==1.0f); // Seven-pixel hole filled in one frame.
    assert(occlusionSurfaceIndices(filled,w,h).size()==12*12*6);
    // A foreground/background discontinuity must stay open.
    for (int y=0; y<h; ++y) for (int x=13; x<w; ++x) plane[y*w+x]=2.0f;
    filled=fillOcclusionHoles(plane,w,h);
    assert(filled[12*w+12]==0);
    auto indices=occlusionSurfaceIndices(filled,w,h);
    for (size_t i=0; i<indices.size(); i+=3) {
        float lo=5,hi=0;
        for (int j=0;j<3;++j) { lo=std::min(lo,filled[indices[i+j]]); hi=std::max(hi,filled[indices[i+j]]); }
        assert(surfaceRange(lo,hi));
    }
    // No extrapolation past a table silhouette, including its open notch.
    plane.assign(w*h,0);
    for (int y=0;y<h;++y) for (int x=0;x<12;++x) plane[y*w+x]=1;
    for (int x=9;x<12;++x) plane[12*w+x]=0;
    filled=fillOcclusionHoles(plane,w,h);
    assert(filled==plane);
    indices=occlusionSurfaceIndices(filled,w,h);
    for (size_t i:indices) assert(i%w<12 && plane[i]==1);
    // Check the interior of a coarse cell; corner-only tests bridge this hole.
    plane.assign(w*h,1); plane[1*w+1]=0;
    indices=occlusionSurfaceIndices(plane,w,h);
    assert(indices.size()==(12*12-1)*6);
    plane[1*w+1]=std::numeric_limits<float>::quiet_NaN();
    assert(occlusionSurfaceIndices(plane,w,h).size()==indices.size());
    // A missing coarse corner leaves three fully supported one-pixel cells.
    std::vector<float> patch(9,1); patch[0]=0;
    OcclusionMeshStats stats;
    auto partial=occlusionSurfaceIndices(patch,3,3,&stats);
    assert(partial.size()==18 && occlusionSurfaceArea(partial,3)==3);
    assert(stats.missingRejected==1 && stats.recoveredSubcells==3);
    assert(occlusionSurfaceMask(partial,3,3)==std::vector<uint8_t>({0,1,0,1,1,0,0,0,0}));
    // An interior hole participates in all four subcells, so none can bridge it.
    patch.assign(9,1); patch[4]=0;
    assert(occlusionSurfaceIndices(patch,3,3).empty());
    // A shared-edge missing sample suppresses just its touching subcells.
    patch.assign(9,1); patch[1]=0;
    partial=occlusionSurfaceIndices(patch,3,3);
    assert(occlusionSurfaceArea(partial,3)==2);
    // A one-pixel depth step keeps both sides independently supported.
    patch.assign(9,1);
    for(int y=0;y<3;++y) patch[y*3+2]=2;
    partial=occlusionSurfaceIndices(patch,3,3,&stats);
    assert(occlusionSurfaceArea(partial,3)==2 && stats.spreadRejected==1);
    // Thin objects do not gain a surface wider than their actual support.
    patch.assign(15,0);
    for(int y=0;y<3;++y) patch[y*5+1]=patch[y*5+2]=1;
    partial=occlusionSurfaceIndices(patch,5,3);
    assert(occlusionSurfaceArea(partial,5)==2);
    for(size_t i:partial) assert(i%5==1 || i%5==2);
    // Lifetime includes a coarse interior sample that is absent from vertices.
    patch.assign(9,1); std::vector<float> lifetime(9,.2f); lifetime[4]=.01f;
    assert(occlusionSurfaceMinLifetime(patch,lifetime,3,3)==.01f);
    patch[0]=0; lifetime[0]=0;
    partial=occlusionSurfaceIndices(patch,3,3);
    assert(occlusionSurfaceMinLifetime(patch,lifetime,3,3,&partial)==.01f);
    lifetime[4]=0;
    assert(occlusionSurfaceMinLifetime(patch,lifetime,3,3,&partial)==0);
    plane.assign(w*h,0); plane[0]=1;
    assert(fillOcclusionHoles(plane,w,h)==plane);
    assert(occlusionSurfaceIndices(plane,w,h).empty());
    // A large missing patch remains unknown, rather than growing a surface.
    plane.assign(w*h,1);
    for(int y=3;y<22;++y) for(int x=3;x<22;++x) plane[y*w+x]=0;
    assert(fillOcclusionHoles(plane,w,h)[12*w+12]==0);
    // Spatial jitter falls without blending foreground/background or inventing
    // measurements at silhouettes. Inverse-depth planes retain their slope.
    plane.assign(w*h,1);
    for (int y=0;y<h;++y) for(int x=0;x<w;++x)
        plane[y*w+x] += ((x+y)%2 ? 0.015f : -0.015f);
    auto smooth=smoothOcclusionDepth(plane,w,h);
    double before=0,after=0;
    for (int y=2;y<h-2;++y) for(int x=2;x<w-2;++x) {
        before+=std::pow(plane[y*w+x]-1,2);
        after+=std::pow(smooth[y*w+x]-1,2);
    }
    assert(after<before*0.05);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) plane[y*w+x]=x<12?1:2;
    smooth=smoothOcclusionDepth(plane,w,h);
    assert(smooth==plane);
    plane[12*w+11]=0;
    assert(smoothOcclusionDepth(plane,w,h)[12*w+11]==0);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) plane[y*w+x]=1.0f/(0.5f+0.001f*x+0.002f*y);
    smooth=smoothOcclusionDepth(plane,w,h);
    for(int y=2;y<h-2;++y) for(int x=2;x<w-2;++x)
        assert(std::abs(smooth[y*w+x]-plane[y*w+x])<1e-5f);

    // Common-mode frame jitter needs temporal filtering as well as spatial
    // filtering. Compare RMS after convergence, not just one blend operation.
    OcclusionReprojection identity;
    std::vector<float> history(w*h,1);
    before=after=0;
    for(int frame=0;frame<60;++frame) {
        plane.assign(w*h,frame%2?1.015f:0.985f);
        auto stable=stabilizeOcclusionDepth(plane,history,w,h,10,12,12,identity);
        if(frame>10) {
            before+=std::pow(plane[12*w+12]-1,2);
            after+=std::pow(stable[12*w+12]-1,2);
        }
        history.swap(stable);
    }
    assert(after<before*0.05);
    // New foreground and newly exposed background update on the first frame.
    history.assign(w*h,2); plane.assign(w*h,1);
    assert(stabilizeOcclusionDepth(plane,history,w,h,10,12,12,identity)==plane);
    history.assign(w*h,1); plane.assign(w*h,2);
    assert(stabilizeOcclusionDepth(plane,history,w,h,10,12,12,identity)==plane);
    plane.assign(w*h,0); plane[0]=std::numeric_limits<float>::infinity();
    assert(stabilizeOcclusionDepth(plane,history,w,h,10,12,12,identity)==plane);
    assert(stabilizeOcclusionDepth(plane,{},w,h,10,12,12,identity)==plane);

    // Align a translated camera in optical space, rather than smearing the old
    // image into the current pixel. Also check nearest-depth reprojection ties.
    OcclusionReprojection translated;
    translated.translation={0.2f,0,0};
    history.assign(w*h,0); history[12*w+6]=1; history[12*w+7]=2;
    plane.assign(w*h,0); plane[12*w+8]=1.02f;
    auto stable=stabilizeOcclusionDepth(plane,history,w,h,10,12,12,translated);
    assert(stable[12*w+8]>1 && stable[12*w+8]<1.01f);
    assert(stable[12*w+6]==0 && stable[12*w+7]==0);
    translated.translation={0,0,-0.5f};
    history.assign(w*h,2); plane.assign(w*h,1.5f);
    stable=stabilizeOcclusionDepth(plane,history,w,h,10,12,12,translated);
    assert(std::abs(stable[12*w+12]-1.5f)<1e-6f);
    // A rotation moves the optical center's old ray to x=14. It must blend at
    // that location only. No history is extrapolated into unmeasured pixels.
    OcclusionReprojection rotated;
    const float angle=std::atan(0.2f), c=std::cos(angle), s=std::sin(angle);
    rotated.rotation={{{c,0,s},{0,1,0},{-s,0,c}}};
    history.assign(w*h,0); history[12*w+12]=1;
    plane.assign(w*h,0); plane[12*w+14]=c+0.02f;
    stable=stabilizeOcclusionDepth(plane,history,w,h,10,12,12,rotated);
    assert(stable[12*w+14]>c && stable[12*w+14]<c+0.01f);
    assert(stable[12*w+12]==0);

    // Mesh coverage, not just depth RMS: every frame loses a different quarter
    // of its measurements. Confirmed triangles must survive these dropouts.
    OcclusionSurfaceHistory coverage;
    plane.assign(w*h,1);
    assert(occlusionSurfaceIndices(coverage.update(plane,w,h,10,12,12,nullptr,0),w,h).empty());
    assert(occlusionSurfaceIndices(coverage.update(plane,w,h,10,12,12,&identity,0.04f),w,h).empty());
    stable=coverage.update(plane,w,h,10,12,12,&identity,0.04f);
    const auto fullMesh=occlusionSurfaceIndices(stable,w,h);
    assert(fullMesh.size()==12*12*6);
    size_t rawArea=0, retainedArea=0;
    for(int frame=0;frame<20;++frame) {
        plane.assign(w*h,1);
        for(int y=0;y<h;++y) for(int x=0;x<w;++x)
            if((x+3*y+frame)%4==0) plane[y*w+x]=0;
        rawArea+=occlusionSurfaceArea(occlusionSurfaceIndices(plane,w,h),w);
        stable=coverage.update(plane,w,h,10,12,12,&identity,0.04f);
        assert(occlusionSurfaceIndices(stable,w,h)==fullMesh);
        retainedArea+=occlusionSurfaceArea(occlusionSurfaceIndices(stable,w,h),w);
        assert(coverage.held>0);
    }
    assert(rawArea<retainedArea && retainedArea==20*occlusionSurfaceArea(fullMesh,w));
    // Transient depths cannot replace the established plane. A sustained new
    // surface replaces it after three measured frames, without averaging edges.
    plane.assign(w*h,0.5f);
    assert(coverage.update(plane,w,h,10,12,12,&identity,0.04f)[12*w+12]==1);
    plane.assign(w*h,1);
    assert(coverage.update(plane,w,h,10,12,12,&identity,0.04f)[12*w+12]==1);
    plane.assign(w*h,2);
    coverage.update(plane,w,h,10,12,12,&identity,0.04f);
    coverage.update(plane,w,h,10,12,12,&identity,0.04f);
    assert(coverage.update(plane,w,h,10,12,12,&identity,0.04f)[12*w+12]==2);
    // Holding does not refresh age. After 200 ms even a repeatedly reprojected
    // surface disappears. An explicit tracking/capture gap clears it immediately.
    plane.assign(w*h,0);
    for(int frame=0;frame<6;++frame) stable=coverage.update(plane,w,h,10,12,12,&identity,0.04f);
    assert(occlusionSurfaceIndices(stable,w,h).empty());
    plane.assign(w*h,1);
    coverage.reset();
    coverage.update(plane,w,h,10,12,12,nullptr,0);
    coverage.update(plane,w,h,10,12,12,&identity,0.04f);
    coverage.update(plane,w,h,10,12,12,&identity,0.04f);
    assert(occlusionSurfaceIndices(coverage.update(plane,w,h,10,12,12,&identity,0.3f),w,h).empty());
    assert(occlusionSurfaceIndices(coverage.update(plane,w,h,10,12,12,nullptr,0.04f),w,h).empty());
    // A moving, one-frame false patch is never confirmed in empty space.
    coverage.reset();
    for(int frame=0;frame<10;++frame) {
        plane.assign(w*h,0);
        for(int y=5;y<10;++y) for(int x=frame;x<frame+4;++x) plane[y*w+x]=0.5f+frame*0.2f;
        stable=coverage.update(plane,w,h,10,12,12,frame?&identity:nullptr,0.04f);
        assert(occlusionSurfaceIndices(stable,w,h).empty());
    }
    // Confirmed age/confidence follow camera translation as well as geometry.
    coverage.reset(); plane.assign(w*h,1);
    coverage.update(plane,w,h,10,12,12,nullptr,0);
    coverage.update(plane,w,h,10,12,12,&identity,0.04f);
    coverage.update(plane,w,h,10,12,12,&identity,0.04f);
    plane.assign(w*h,0); translated.translation={0.2f,0,0};
    stable=coverage.update(plane,w,h,10,12,12,&translated,0.04f);
    assert(stable[12*w+14]==1 && stable[12*w+0]==0);
    // Confirmation follows a patch across pixels while the camera moves.
    coverage.reset(); plane.assign(w*h,0); plane[12*w+10]=1;
    coverage.update(plane,w,h,10,12,12,nullptr,0);
    plane[12*w+10]=0; plane[12*w+12]=1;
    assert(coverage.update(plane,w,h,10,12,12,&translated,0.04f)[12*w+12]==0);
    plane[12*w+12]=0; plane[12*w+14]=1;
    assert(coverage.update(plane,w,h,10,12,12,&translated,0.04f)[12*w+14]==1);
    // Expired candidates cannot count old observations toward confirmation.
    coverage.reset(); plane.assign(w*h,1);
    coverage.update(plane,w,h,10,12,12,nullptr,0);
    plane.assign(w*h,0);
    coverage.update(plane,w,h,10,12,12,&identity,0.11f);
    coverage.update(plane,w,h,10,12,12,&identity,0.11f);
    plane.assign(w*h,1);
    assert(occlusionSurfaceIndices(coverage.update(plane,w,h,10,12,12,&identity,0.04f),w,h).empty());
    std::vector<float> full(616*616,1);
    const auto start=std::chrono::steady_clock::now();
    auto filtered=fillOcclusionHoles(smoothOcclusionDepth(full,616,616),616,616);
    coverage.reset();
    coverage.update(filtered,616,616,380,308,308,nullptr,0);
    coverage.update(filtered,616,616,380,308,308,&identity,0.04f);
    filtered=coverage.update(filtered,616,616,380,308,308,&identity,0.04f);
    auto mesh=occlusionSurfaceIndices(filtered,616,616);
    assert(mesh.size()==307*307*6 && mesh.size()<600000);
    // A shallow corrugated surface can reject every coarse cell while all four
    // independently supported subcells survive. Exercise the actual worst bound.
    for(int y=0;y<616;++y) for(int x=0;x<616;++x)
        full[size_t(y)*616+x]=1+.03f*float(x%4<=2?x%4:4-x%4);
    const auto maximumMesh=occlusionSurfaceIndices(full,616,616,&stats);
    assert(stats.coarseAccepted==0 && stats.recoveredSubcells==307*307*4);
    assert(maximumMesh.size()==307*307*24);
    assert(occlusionSurfaceArea(maximumMesh,616)==614*614);
    const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::cout<<"Surface tests passed: coverage dropout hysteresis, transient patch rejection, bounded history age, spatial/temporal noise reduction, slanted planes, moving surfaces, pose reprojection, enclosed holes, depth steps, open edges, cell interiors, invalid depth, capacity; full frame "<<ms<<" ms\n";
}
