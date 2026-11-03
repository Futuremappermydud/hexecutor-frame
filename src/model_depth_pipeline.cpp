#include "model_depth_pipeline.hpp"
#include "model_depth_geometry.hpp"
#include "calibration.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <fcntl.h>
extern char** environ;

namespace FrameSpatial {
namespace {
std::string setting(const char* name,const std::string& fallback) {
    const char* value=std::getenv(name);return value && *value?value:fallback;
}
}
ModelDepthPipeline::~ModelDepthPipeline(){stop();}
bool ModelDepthPipeline::transfer(void* data,size_t bytes,bool sending) {
    auto* ptr=static_cast<uint8_t*>(data);
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(timeoutSeconds_);
    while(bytes) {
        const auto remaining=std::chrono::duration_cast<std::chrono::milliseconds>(deadline-std::chrono::steady_clock::now()).count();
        if(remaining<=0){error_="Model worker timed out";return false;}
        pollfd descriptor{fd_,short(sending?POLLOUT:POLLIN),0};
        const int ready=poll(&descriptor,1,int(remaining));
        if(ready<0 && errno==EINTR)continue;
        if(ready<=0){error_=ready==0?"Model worker timed out":std::strerror(errno);return false;}
        const ssize_t n=sending?send(fd_,ptr,bytes,MSG_NOSIGNAL|MSG_DONTWAIT):recv(fd_,ptr,bytes,MSG_DONTWAIT);
        if(n<0 && (errno==EINTR || errno==EAGAIN))continue;
        if(n<=0){error_="Model worker disconnected; see its startup/error log";return false;}
        ptr+=n;bytes-=size_t(n);
    }
    return true;
}
bool ModelDepthPipeline::initialize(const std::string& script) {
    if(fd_>=0)return true;
    const std::string home=setting("HOME","/home/steamos");
    const std::string python=setting("FRAME_DEPTH_PYTHON",home+"/frame-dsp/depth-model/.venv/bin/python");
    const std::string backend=setting("FRAME_DEPTH_BACKEND","hybrid");
    if(backend!="hybrid" && backend!="prepared" && backend!="prepared-qnn"){error_="FRAME_DEPTH_BACKEND must be hybrid, prepared or prepared-qnn";return false;}
    const bool qnn=backend=="prepared-qnn";
    const bool prepared=backend=="prepared" || qnn;
    // The prepared engine needs cdsp_mem loaded (128 MB) and takes ~15.5 s per pair.
    const std::string package=setting("FRAME_DEPTH_PACKAGE",home+(qnn?"/frame-dsp/depth-model/wave1-20261101":prepared?"/frame-dsp/depth-model/perf-20261029"
                                                                          :"/frame-dsp/depth-model/depth-model-direct-v5"));
    const std::string model=setting("FRAME_DEPTH_MODEL","primary");
    timeoutSeconds_=std::atoi(setting("FRAME_DEPTH_TIMEOUT_S",qnn?"30":prepared?"60":"30").c_str());
    if(timeoutSeconds_<=0){error_="FRAME_DEPTH_TIMEOUT_S must be a positive number of seconds";return false;}
    int sockets[2];
    if(socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,sockets)) {error_=std::strerror(errno);return false;}
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions,sockets[0]);
    posix_spawn_file_actions_adddup2(&actions,sockets[1],3);
    if(sockets[1]!=3)posix_spawn_file_actions_addclose(&actions,sockets[1]);
    // Keep model and transport diagnostics in the application's log.
    posix_spawn_file_actions_adddup2(&actions,STDERR_FILENO,STDOUT_FILENO);
    temporal_=setting("FRAME_DEPTH_TEMPORAL","0")==std::string("1");
    censusGate_=setting("FRAME_DEPTH_GATE","uncertainty")==std::string("census");
    maxUncertainty_=std::clamp(std::atoi(setting("FRAME_DEPTH_UNCERTAINTY_MAX","89").c_str()),1,256);
    // Optional replay capture: every request (header, motion block, resized pair).
    const std::string record=setting("FRAME_DEPTH_RECORD","");
    if(!record.empty()) {
        recordFd_=open(record.c_str(),O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC,0644);
        recordLimit_=uint32_t(std::max(1,std::atoi(setting("FRAME_DEPTH_RECORD_FRAMES","200").c_str())));
        if(recordFd_<0)std::cerr<<"[ModelDepth] cannot record to "<<record<<": "<<std::strerror(errno)<<std::endl;
    }
    const char* args[]{python.c_str(),"-u",script.c_str(),"--fd","3","--package",package.c_str(),"--model",model.c_str(),
                       "--backend",backend.c_str(),temporal_?"--temporal":nullptr,nullptr};
    pid_t child=-1;
    const int result=posix_spawn(&child,python.c_str(),&actions,nullptr,const_cast<char* const*>(args),environ);
    posix_spawn_file_actions_destroy(&actions);close(sockets[1]);
    if(result){close(sockets[0]);error_=std::strerror(result);return false;}
    fd_=sockets[0];pid_=child;
    uint32_t hello[4]{};
    if(!transfer(hello,sizeof(hello),false) || hello[0]!=0x36524451 || hello[1]!=MODEL_WIDTH || hello[2]!=MODEL_HEIGHT || hello[3]!=3) {
        if(error_.empty())error_="incompatible Model depth bridge";
        stop();return false;
    }
    std::cout<<"[ModelDepth] Persistent "<<(qnn?"QNN run-list executor (~23 ms)":prepared?"prepared-executor (exact)":"HMX hybrid")
             <<" worker ready; experimental plane-0 disparity, "
             <<(temporal_?"Model-style warped temporal state":"zero temporal state (t=0)")
             <<", gate="<<(censusGate_?"contrast+census":"uncertainty<"+std::to_string(maxUncertainty_))<<std::endl;
    return true;
}
bool ModelDepthPipeline::process(const std::vector<uint8_t>& left,const std::vector<uint8_t>& right,std::vector<float>& depth,
                                 const ModelMotion& motion) {
    depth.clear();
    if(fd_<0){error_="Model depth worker is not initialized";return false;}
    try {
        auto l=modelResize(left,RECT_WIDTH,RECT_HEIGHT),r=modelResize(right,RECT_WIDTH,RECT_HEIGHT);
        uint32_t header[]{0x36504451,++sequence_,MODEL_WIDTH,MODEL_HEIGHT};
        // flags (bit0 aligned, bit1 reset), R row-major, t, three pad words.
        uint32_t block[16]{};
        block[0]=(motion.aligned?1u:0u)|(motion.reset?2u:0u);
        float values[12];
        for(int i=0;i<3;i++)for(int j=0;j<3;j++)values[i*3+j]=motion.rotation[i][j];
        for(int i=0;i<3;i++)values[9+i]=motion.translation[i];
        std::memcpy(block+1,values,sizeof(values));
        if(recordFd_>=0 && recorded_<recordLimit_) {
            bool ok=write(recordFd_,header,sizeof(header))==ssize_t(sizeof(header)) &&
                    write(recordFd_,block,sizeof(block))==ssize_t(sizeof(block)) &&
                    write(recordFd_,l.data(),l.size())==ssize_t(l.size()) &&
                    write(recordFd_,r.data(),r.size())==ssize_t(r.size());
            if(!ok || ++recorded_==recordLimit_) {
                std::cout<<"[ModelDepth] recording "<<(ok?"complete":"failed")<<": "<<recorded_<<" pairs"<<std::endl;
                close(recordFd_);recordFd_=-1;
            }
        }
        if(!transfer(header,sizeof(header),true) || !transfer(block,sizeof(block),true) ||
           !transfer(l.data(),l.size(),true) || !transfer(r.data(),r.size(),true))return false;
        // Reply: magic, sequence, pixel count, inference milliseconds (f32).
        uint32_t reply[4]{};
        if(!transfer(reply,sizeof(reply),false))return false;
        if(reply[0]!=0x364f4451 || reply[1]!=sequence_ || reply[2]!=MODEL_WIDTH*MODEL_HEIGHT){error_="invalid Model depth reply";return false;}
        std::memcpy(&milliseconds_,reply+3,sizeof(float));
        std::vector<float> disparity(MODEL_WIDTH*MODEL_HEIGHT);
        if(!transfer(disparity.data(),disparity.size()*sizeof(float),false))return false;
        std::vector<uint8_t> uncertainty(MODEL_WIDTH*MODEL_HEIGHT);
        if(!transfer(uncertainty.data(),uncertainty.size(),false))return false;
        depth=modelMetricDepth(disparity,l,r,RECT_WIDTH,RECT_HEIGHT,FOCAL_BASELINE,.25f,5.f,
                               censusGate_?nullptr:&uncertainty,maxUncertainty_);
        confidence_.assign(size_t(RECT_WIDTH)*RECT_HEIGHT,1.f);
        for(int y=0;y<RECT_HEIGHT;y++)for(int x=0;x<RECT_WIDTH;x++) {
            const int mx=std::min(MODEL_WIDTH-1,int((x+.5f)*MODEL_WIDTH/RECT_WIDTH));
            const int my=std::min(MODEL_HEIGHT-1,int((y+.5f)*MODEL_HEIGHT/RECT_HEIGHT));
            const float u=uncertainty[size_t(my)*MODEL_WIDTH+mx];
            confidence_[size_t(y)*RECT_WIDTH+x]=std::clamp((maxUncertainty_-u)/float(maxUncertainty_-50),.05f,1.f);
        }
        return true;
    } catch(const std::exception& e){error_=e.what();return false;}
}
void ModelDepthPipeline::stop() {
    if(recordFd_>=0){close(recordFd_);recordFd_=-1;}
    if(fd_>=0){shutdown(fd_,SHUT_RDWR);close(fd_);fd_=-1;}
    if(pid_<0)return;
    // EOF normally exits the Python context, restoring clocks, workers and gaze.
    // SIGTERM also goes through FrameTransport's bounded cleanup handler; the
    // first wait covers one in-flight request so it is not interrupted mid-run.
    for(int phase=0;phase<2;phase++) {
        const int polls=phase==0?std::max(160,timeoutSeconds_*20):160;
        for(int i=0;i<polls;i++) {
            const auto result=waitpid(pid_,nullptr,WNOHANG);
            if(result==pid_ || (result<0 && errno==ECHILD)){pid_=-1;return;}
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if(phase==0)kill(pid_,SIGTERM);
    }
    std::cerr<<"[ModelDepth] Worker "<<pid_<<" has not exited; retained for safe DSP cleanup"<<std::endl;
    pid_=-1;
}
}
