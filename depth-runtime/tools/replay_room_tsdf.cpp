#include "tsdf_volume.hpp"
#include "calibration.hpp"
#include "depth_speckle_filter.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace FrameSpatial;

// E07 options select TsdfIntegrateOptions; the defaults reproduce the live integration.
static int usage(){
    std::cerr<<"usage: replay_room_tsdf [--robust] [--frame-cap W] [--carve-after N] [--voxel M] [--truncation M]\n"
               "                        [--max-depth M] [--incremental] [--mesh-every N] [--snapshot-every N DIR]\n"
               "                        frames.bin|- mesh.ply\n"
               "  --snapshot-every writes DIR/snap-FRAME.f32 (triangle-list xyz float32) every N frames (E09)\n";
    return 2;
}

int main(int argc,char** argv) {
    TsdfIntegrateOptions options;
    TsdfConfig config;
    std::vector<std::string> positional;
    int snapshotEvery=0,meshEvery=0;
    bool incremental=false;
    std::string snapshotDir;
    for(int i=1;i<argc;i++){
        const std::string arg=argv[i];
        if(arg=="--robust")options.robust=true;
        else if(arg=="--frame-cap"&&i+1<argc){options.frameWeightCap=true;options.frameWeightCapValue=std::strtof(argv[++i],nullptr);}
        else if(arg=="--carve-after"&&i+1<argc)options.carveAfter=std::atoi(argv[++i]);
        else if(arg=="--voxel"&&i+1<argc)config.voxel=std::strtof(argv[++i],nullptr);
        else if(arg=="--truncation"&&i+1<argc)config.truncation=std::strtof(argv[++i],nullptr);
        else if(arg=="--max-depth"&&i+1<argc)config.maxDepth=std::strtof(argv[++i],nullptr);
        else if(arg=="--incremental")incremental=true;
        else if(arg=="--mesh-every"&&i+1<argc)meshEvery=std::atoi(argv[++i]);
        else if(arg=="--snapshot-every"&&i+2<argc){snapshotEvery=std::atoi(argv[++i]);snapshotDir=argv[++i];}
        else if(arg.rfind("--",0)==0)return usage();
        else positional.push_back(arg);
    }
    if(positional.size()!=2||options.carveAfter<1||(options.frameWeightCap&&!(options.frameWeightCapValue>0))||
       !(config.voxel>0)||!(config.truncation>=config.voxel)||!(config.maxDepth>config.minDepth)||
       snapshotEvery<0||meshEvery<0)return usage();
    std::ifstream file;
    std::istream* input=&std::cin;
    if(positional[0]!="-") {file.open(positional[0],std::ios::binary);input=&file;}
    uint32_t count=0;
    if(!input->read(reinterpret_cast<char*>(&count),sizeof(count)) || count==0 || count>10000)return 2;
    TsdfVolume volume(config);
    std::vector<double> integrateMs,meshMs;
    std::vector<size_t> rebuiltLeaves;
    size_t peakLeaves=0,evicted=0;
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
        const auto began=std::chrono::steady_clock::now();
        const TsdfStats stats=volume.integrate(depth.data(),luma.data(),RECT_WIDTH,RECT_HEIGHT,RECT_FOCAL,RECT_CX,RECT_CY,
                                               pose,confidence.data(),options);
        integrateMs.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count());
        peakLeaves=std::max(peakLeaves,volume.leafCount());
        evicted+=stats.evictedLeaves;
        const bool snapshot=snapshotEvery>0&&((frame+1)%snapshotEvery==0||frame+1==count);
        const bool periodic=meshEvery>0&&((frame+1)%meshEvery==0||frame+1==count);
        if(snapshot||periodic){
            std::vector<TsdfVertex> current;
            const auto meshBegan=std::chrono::steady_clock::now();
            if(incremental)volume.extractMeshIncremental(current,480000);
            else volume.extractMesh(current,480000);
            meshMs.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-meshBegan).count());
            if(incremental)rebuiltLeaves.push_back(volume.lastMeshRebuiltLeaves());
            if(snapshot){
                std::ofstream out(snapshotDir+"/snap-"+std::to_string(frame+1)+".f32",std::ios::binary);
                for(const auto& vertex:current){const float xyz[3]{vertex.x,vertex.y,vertex.z};out.write(reinterpret_cast<const char*>(xyz),12);}
                if(!out)return 1;
            }
        }
    }
    std::vector<TsdfVertex> mesh;
    const auto meshBegan=std::chrono::steady_clock::now();
    if(incremental)volume.extractMeshIncremental(mesh,480000);
    else volume.extractMesh(mesh,480000);
    meshMs.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-meshBegan).count());
    if(incremental)rebuiltLeaves.push_back(volume.lastMeshRebuiltLeaves());
    std::ofstream output(positional[1]);
    output<<"ply\nformat ascii 1.0\nelement vertex "<<mesh.size()
          <<"\nproperty float x\nproperty float y\nproperty float z\nelement face "<<mesh.size()/3
          <<"\nproperty list uchar int vertex_indices\nend_header\n";
    for(const auto& vertex:mesh)output<<vertex.x<<' '<<vertex.y<<' '<<vertex.z<<'\n';
    for(size_t index=0;index<mesh.size();index+=3)output<<"3 "<<index<<' '<<index+1<<' '<<index+2<<'\n';
    if(!output)return 1;
    std::cout<<"frames="<<count<<" leaves="<<volume.leafCount()<<" triangles="<<mesh.size()/3
             <<" peak_leaves="<<peakLeaves<<" evicted="<<evicted<<" robust="<<options.robust
             <<" frame_cap="<<(options.frameWeightCap?options.frameWeightCapValue:0.f)
             <<" carve_after="<<options.carveAfter<<" voxel="<<config.voxel<<" truncation="<<config.truncation
             <<" max_depth="<<config.maxDepth<<" incremental="<<incremental<<" mesh_every="<<meshEvery;
    std::sort(integrateMs.begin(),integrateMs.end());
    const auto pct=[&](double q){return integrateMs[size_t(q*(integrateMs.size()-1))];};
    std::sort(meshMs.begin(),meshMs.end());
    const auto meshPct=[&](double q){return meshMs[size_t(q*(meshMs.size()-1))];};
    std::cout<<" integrate_ms_p50="<<pct(.5)<<" integrate_ms_p95="<<pct(.95)
             <<" mesh_ms_p50="<<meshPct(.5)<<" mesh_ms_p95="<<meshPct(.95);
    if(incremental){
        std::sort(rebuiltLeaves.begin(),rebuiltLeaves.end());
        const auto rebuiltPct=[&](double q){return rebuiltLeaves[size_t(q*(rebuiltLeaves.size()-1))];};
        std::cout<<" rebuilt_leaves_p50="<<rebuiltPct(.5)<<" rebuilt_leaves_p95="<<rebuiltPct(.95)
                 <<" cached_leaves="<<volume.cachedMeshLeafCount()
                 <<" cache_vertex_bytes="<<volume.cachedMeshVertexBytes();
    }
    std::cout<<" max_triangles=480000\n";
}
