#include "occlusion_benchmark.hpp"
#include <cassert>
#include <cstdio>
using namespace FrameSpatial::OcclusionBenchmark;
int main() {
    Header h; h.width=h.height=1;
    Metadata m; m.phase=1; m.flags=CapturePose; m.frame=1;
    m.captureBeginNs=1; m.captureEndNs=2; m.publishNs=3;
    std::vector<uint8_t> pixels{100},evidence{FrameSpatial::MeasuredDepth};
    std::vector<float> depth{1};
    const std::string file="/tmp/frame-occlusion-budget-test.bin";
    // Keep v2 reading while permitting more than the old 512-frame cap.
    {
        std::ofstream out(file,std::ios::binary); writeHeader(out,h);
        for(uint32_t i=0;i<513;++i) {
            m.frame=i+1; write(out,FrameTag); write(out,uint32_t(MetadataBytes+11));
            writeMetadata(out,m); writePixels(out,pixels); writePixels(out,pixels);
            writePixels(out,depth); writePixels(out,depth); writePixels(out,evidence);
        }
        write(out,EndTag); write(out,uint32_t(513)); write(out,uint32_t(0)); write(out,uint32_t(1));
    }
    Reader reader(file); Frame frame; while(reader.next(frame)) {}
    assert(reader.frames==513 && reader.complete && !reader.dropped);
    // Exhausting a declared budget never silently makes a short capture complete.
    {
        Recorder recorder(file,h,2);
        for(uint32_t i=0;i<3;++i) { m.frame=i+1; recorder.append(m,pixels,pixels,depth,depth,evidence); }
        assert(!recorder.active());
    }
    Reader shortReader(file); while(shortReader.next(frame)) {}
    assert(shortReader.frames==2 && !shortReader.complete && !shortReader.dropped);
    for(uint32_t invalid: {0u,MaxFrameLimit+1}) {
        bool rejected=false;
        try { Recorder recorder(file,h,invalid); }
        catch(const std::invalid_argument&) { rejected=true; }
        assert(rejected);
    }
    std::remove(file.c_str());
    std::cout<<"Budget tests passed: extended v2 reading, explicit incomplete exhaustion, invalid budget rejection\n";
}
