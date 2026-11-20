#include "tsdf_volume.hpp"
#include "calibration.hpp"
#include "depth_speckle_filter.hpp"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace FrameSpatial;

int main(int argc,char** argv) {
    if(argc!=3){std::cerr<<"usage: replay_room_tsdf frames.bin mesh.ply\n";return 2;}
    std::ifstream file;
    std::istream* input=&std::cin;
    if(std::string(argv[1])!="-") {file.open(argv[1],std::ios::binary);input=&file;}
    uint32_t count=0;
    if(!input->read(reinterpret_cast<char*>(&count),sizeof(count)) || count==0 || count>10000)return 2;
    TsdfVolume volume;
    std::vector<float> depth(size_t(RECT_WIDTH)*RECT_HEIGHT);
    std::vector<uint8_t> luma(depth.size());
    std::vector<float> confidence(depth.size());
    for(uint32_t frame=0;frame<count;frame++) {
        std::array<float,12> transform{};
        if(!input->read(reinterpret_cast<char*>(transform.data()),sizeof(transform)) ||
           !input->read(reinterpret_cast<char*>(depth.data()),depth.size()*sizeof(float)) ||
           !input->read(reinterpret_cast<char*>(luma.data()),luma.size()) ||
           !input->read(reinterpret_cast<char*>(confidence.data()),confidence.size()*sizeof(float)))return 2;
        TsdfPose pose;
        for(int row=0;row<3;row++)for(int col=0;col<3;col++)pose.rotation[row][col]=transform[row*3+col];
        for(int row=0;row<3;row++)pose.translation[row]=transform[9+row];
        removeDepthSpeckles(depth,RECT_WIDTH,RECT_HEIGHT);
        volume.integrate(depth.data(),luma.data(),RECT_WIDTH,RECT_HEIGHT,RECT_FOCAL,RECT_CX,RECT_CY,pose,confidence.data());
    }
    std::vector<TsdfVertex> mesh;
    volume.extractMesh(mesh,480000);
    std::ofstream output(argv[2]);
    output<<"ply\nformat ascii 1.0\nelement vertex "<<mesh.size()
          <<"\nproperty float x\nproperty float y\nproperty float z\nelement face "<<mesh.size()/3
          <<"\nproperty list uchar int vertex_indices\nend_header\n";
    for(const auto& vertex:mesh)output<<vertex.x<<' '<<vertex.y<<' '<<vertex.z<<'\n';
    for(size_t index=0;index<mesh.size();index+=3)output<<"3 "<<index<<' '<<index+1<<' '<<index+2<<'\n';
    if(!output)return 1;
    std::cout<<"frames="<<count<<" leaves="<<volume.leafCount()<<" triangles="<<mesh.size()/3<<'\n';
}
