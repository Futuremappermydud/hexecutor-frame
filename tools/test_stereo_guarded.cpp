#include "stereo_matcher_experiment.hpp"
#include <iostream>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Spatial::StereoExperiment;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void parity(const Result& a, const Result& b) {
    require(a.leftDisparity == b.leftDisparity && a.rightDisparity == b.rightDisparity &&
        a.leftBestCost == b.leftBestCost && a.rightBestCost == b.rightBestCost &&
        a.leftUniquenessGap == b.leftUniquenessGap && a.rightUniquenessGap == b.rightUniquenessGap &&
        a.leftBestIntegerDisparity == b.leftBestIntegerDisparity && a.rightBestIntegerDisparity == b.rightBestIntegerDisparity &&
        a.leftParabolaDisparity == b.leftParabolaDisparity && a.rightParabolaDisparity == b.rightParabolaDisparity &&
        a.leftMatchRejection == b.leftMatchRejection && a.rightMatchRejection == b.rightMatchRejection &&
        a.lrRejection == b.lrRejection && a.supportRejection == b.supportRejection &&
        a.leftInformation == b.leftInformation && a.rightInformation == b.rightInformation &&
        a.preSupportDepth == b.preSupportDepth && a.depth == b.depth, "Full-stage parity failed");
}
int main() {
    try {
        Config c; c.width = 64; c.height = 40; c.maxDisparity = 24;
        c.supportRadius = 8; c.edgeThreshold = 32;
        std::vector<uint8_t> l(2560), r(2560);
        std::mt19937 gen(1709);
        for (auto& value : l) value = uint8_t(80 + gen() % 96);
        for (int y = 0; y < c.height; ++y) for (int x = 0; x < c.width; ++x)
            r[size_t(y)*c.width+x] = l[size_t(y)*c.width+std::min(x+8,c.width-1)];
        for (auto refinement : {Refinement::Integer, Refinement::Parabola}) {
            c.refinement = refinement;
            parity(match(l,r,c,Method::Adaptive),match(l,r,c,Method::GuardedAdaptive));
        }
        // LR anchors require both independently accepted integer matches.
        std::vector<float> ld(2560,0), rd(2560,0);
        ld[16*64+30] = 8; rd[16*64+22] = 8;
        auto la = detail::consistentAnchors(ld,rd,c,false);
        auto ra = detail::consistentAnchors(rd,ld,c,true);
        require(la[16*64+30] == 8 && ra[16*64+22] == 8,"Bilateral anchor addressing");
        rd[16*64+22] = 10;
        require(detail::consistentAnchors(ld,rd,c,false)[16*64+30] == 0,"Inconsistent anchor retained");
        rd[16*64+22] = 0;
        require(detail::consistentAnchors(ld,rd,c,false)[16*64+30] == 0,"Missing anchor fabricated");
        // Census anchor boundaries censor only independently known conflicts.
        std::vector<float> anchors(2560,8);
        for (int y = 0; y < 40; ++y) for (int x = 32; x < 64; ++x) anchors[y*64+x] = 18;
        auto masks = detail::boundaryMasks(anchors,c);
        require(__builtin_popcountll(masks[16*64+31]) == 27,"Descriptor bit geometry at boundary");
        require(__builtin_popcountll(masks[16*64+28]) == 48,"Interior descriptor changed");
        anchors[16*64+31] = 0;
        require(__builtin_popcountll(detail::boundaryMasks(anchors,c)[16*64+31]) == 48,"Unknown label inferred");
        // An anchor edge in just the target stops an arm even without a luma edge.
        std::vector<uint64_t> lc(2560,0), rc(2560,0), full(2560,(uint64_t(1)<<48)-1);
        std::vector<float> a(2560,8), b=a;
        std::fill(l.begin(),l.end(),100); r=l;
        b[16*64+23] = 18;
        c.supportRadius = 4; c.supportGuard = 1;
        auto v = detail::guardedCost(lc,rc,l,r,a,b,full,full,1,1,30,16,8,false,c);
        require(v.samples == 13 && v.stops == 1 && v.cost == 240,"Target-only anchor arm stop");
        c.supportGuard = 2;
        auto rm = full; rm[16*64+22] = (uint64_t(1)<<23)-1;
        v = detail::guardedCost(lc,rc,l,r,a,b,full,rm,1,1,30,16,8,false,c);
        require(v.cost == 240 && v.samples == 0,"Insufficient center bits accepted");
        // Controlled affine exposure spans scale by the exact gain when unclipped.
        std::vector<uint8_t> ramp(100), gain(100);
        for (int i=0;i<100;++i) { ramp[i]=uint8_t(40+i); gain[i]=uint8_t(2*(40+i)-40); }
        require(std::abs(detail::supportContrastScale(gain)-2*detail::supportContrastScale(ramp))<1e-12,"Contrast gain scaling");
        require(detail::supportContrastScale(std::vector<uint8_t>(100,128)) == .5,"Constant contrast floor");
        // All modes reject uniform evidence; OMP must preserve every diagnostic.
        c.supportRadius=8;
        for (int guard=0;guard<=3;++guard) for (bool exposure:{false,true}) {
            c.supportGuard=guard;c.supportExposure=exposure;
            auto blank=match(l,r,c,Method::GuardedAdaptive);
            for (float z:blank.depth) require(z==0,"Uniform evidence accepted");
        }
        for (auto& value:l) value=uint8_t(80+gen()%96);
        for (int y=0;y<c.height;++y) for(int x=0;x<c.width;++x)
            r[size_t(y)*c.width+x]=l[size_t(y)*c.width+std::min(x+8,c.width-1)];
        c.supportGuard=3;c.supportExposure=true;
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        const auto one=match(l,r,c,Method::GuardedAdaptive);
#ifdef _OPENMP
        omp_set_num_threads(8);
#endif
        const auto eight=match(l,r,c,Method::GuardedAdaptive);
        parity(one,eight);
        require(one.leftAnchors==eight.leftAnchors && one.rightAnchors==eight.rightAnchors &&
            one.leftSupportBits==eight.leftSupportBits && one.rightSupportBits==eight.rightSupportBits &&
            one.leftAnchorStops==eight.leftAnchorStops && one.rightAnchorStops==eight.rightAnchorStops,"OMP diagnostic parity");
        for(size_t i=0;i<l.size();++i) {
            require(eight.leftSupportBits[i]>=24*eight.leftInformation[i] &&
                eight.leftSupportBits[i]<=48*eight.leftInformation[i] &&
                eight.leftAnchorStops[i]<=4,"Information bounds");
        }
        c.supportGuard=4;
        try { match(l,r,c,Method::GuardedAdaptive); require(false,"Invalid guard accepted"); }
        catch(const std::invalid_argument&) {}
        std::cout<<"Guarded support parity, anchor/mask bounds, exposure and OMP controls passed\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
