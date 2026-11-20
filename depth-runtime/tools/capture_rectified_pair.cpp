// Capture processed luma through the production DMA discovery/UBWC/KB4 path.
// No video devices, hardcoded compositor descriptors, or raw DMA exports.
#include "dma_capture.hpp"
#include "depth_pipeline.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
using namespace FrameSpatial;
int main(int argc,char** argv) {
    if(argc!=2) { std::cerr<<"Usage: "<<argv[0]<<" output-prefix\n"; return 2; }
    DMACapture capture;
    const auto pid=DMACapture::findCompositorPid();
    if(pid<=0 || !capture.initialize(pid)) return 1;
    DepthPipeline rectifier;
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(!capture.captureFrame()) {
        if(std::chrono::steady_clock::now()>deadline) return 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    rectifier.rectify(capture.getLeftBuffer(),capture.getRightBuffer());
    for(int eye=0;eye<2;++eye) {
        const auto& pixels=eye?rectifier.getRectifiedRight():rectifier.getRectifiedLeft();
        const std::string path=std::string(argv[1])+(eye?"-right.raw":"-left.raw");
        std::ofstream output(path,std::ios::binary);
        output.write(reinterpret_cast<const char*>(pixels.data()),pixels.size());
        if(!output) return 1;
        std::cout<<path<<": "<<pixels.size()<<" rectified luma bytes\n";
    }
}
