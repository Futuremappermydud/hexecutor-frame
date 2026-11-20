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
#include <filesystem>
#include <fstream>
#include <iomanip>
extern char** environ;

namespace FrameSpatial {
namespace {
std::string setting(const char* name,const std::string& fallback) {
    const char* value=std::getenv(name);return value && *value?value:fallback;
}
template<class Value> bool writeArray(const std::filesystem::path& path,const std::vector<Value>& values) {
    std::ofstream file(path,std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(Value));
    return bool(file);
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
bool ModelDepthPipeline::initialize(const std::string& executable) {
    if(fd_>=0)return true;
    const std::string home=setting("HOME","/home/steamos");
    const std::string worker=setting("FRAME_DEPTH_WORKER",executable);
    const std::string backend=setting("FRAME_DEPTH_BACKEND","prepared-qnn");
    if(backend!="prepared" && backend!="prepared-qnn"){error_="FRAME_DEPTH_BACKEND must be prepared or prepared-qnn";return false;}
    const bool qnn=backend=="prepared-qnn";
    // Both native engines need cdsp_mem loaded (128 MB).
    const std::string package=setting("FRAME_DEPTH_PACKAGE",home+(qnn?"/frame-dsp/depth-model/wave1-20261101":"/frame-dsp/depth-model/perf-20261029"));
    const std::string model=setting("FRAME_DEPTH_MODEL","primary");
    if(qnn && model!="primary"){error_="prepared-qnn supports primary only; use FRAME_DEPTH_BACKEND=prepared for alternate";return false;}
    backend_=backend;model_=model;package_=package;
    timeoutSeconds_=std::atoi(setting("FRAME_DEPTH_TIMEOUT_S",qnn?"30":"60").c_str());
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
    spherical_=setting("FRAME_DEPTH_SPHERICAL","0")==std::string("1");
    sphericalDegrees_=setting("FRAME_DEPTH_SPHERICAL_DEG","0.29");
    maxUncertainty_=std::clamp(std::atoi(setting("FRAME_DEPTH_UNCERTAINTY_MAX","89").c_str()),1,256);
    // Optional replay capture: every request (header, motion block, resized pair).
    const std::string record=setting("FRAME_DEPTH_RECORD","");
    recordV2Directory_=setting("FRAME_DEPTH_RECORD_V2","");
    secondDisparity_=!recordV2Directory_.empty();
    if(temporal_) {
        const double degrees=spherical_?std::atof(sphericalDegrees_.c_str()):0.0;
        temporalState_=std::make_unique<ModelTemporalState>(degrees*3.14159265358979323846/180.0);
        hidden_.resize(ModelTemporalState::HiddenValues);
        transforms_.resize(ModelTemporalState::TransformValues);
        // Starting values, not measured limits: at live rates a step is tens of ms.
        temporalMaxGapNs_=int64_t(std::atof(setting("FRAME_DEPTH_TEMPORAL_MAX_GAP_MS","250").c_str())*1e6);
        temporalMaxDegrees_=std::atof(setting("FRAME_DEPTH_TEMPORAL_MAX_DEG","30").c_str());
    }
    const bool rightPlane=secondDisparity_||temporal_;
    const int requestedFrames=std::atoi(setting("FRAME_DEPTH_RECORD_FRAMES","200").c_str());
    recordLimit_=requestedFrames==0?UINT32_MAX:uint32_t(std::max(1,requestedFrames));
    if(secondDisparity_) {
        std::filesystem::create_directories(recordV2Directory_);
        std::ofstream info(std::filesystem::path(recordV2Directory_)/"capture.json");
        info<<"{\"version\":2,\"model\":\""<<model<<"\",\"backend\":\""<<backend
            <<"\",\"projection\":\""<<(spherical_?"spherical":"pinhole")
            <<"\",\"temporal\":"<<(temporal_?"true":"false")
            <<",\"model_shape\":[320,256],\"rectified_shape\":[616,616],\"right_disparity\":true,\"capture_clock\":\"CLOCK_MONOTONIC\",\"exposure_timestamp\":false,\"config_sha256\":\""
            <<setting("FRAME_DEPTH_CONFIG_SHA256","")<<"\"}\n";
        if(!info){error_="cannot create Model v2 capture metadata";return false;}
    }
    if(!record.empty()) {
        recordFd_=open(record.c_str(),O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC,0644);
        if(recordFd_<0)std::cerr<<"[ModelDepth] cannot record to "<<record<<": "<<std::strerror(errno)<<std::endl;
        // Spherical runs also keep the spherical inputs, for an offline A/B on identical captures.
        if(spherical_)recordSphericalFd_=open((record+".sph").c_str(),O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC,0644);
    }
    std::vector<const char*> args{worker.c_str(),"--fd","3","--package",package.c_str(),"--model",model.c_str(),
                                  "--backend",backend.c_str(),"--spherical-deg",spherical_?sphericalDegrees_.c_str():"0"};
    if(temporal_)args.push_back("--host-temporal");
    if(rightPlane)args.push_back("--second-disparity");
    if(secondDisparity_) {
        if(setting("FRAME_DEPTH_RECORD_CLOSE_EACH_REPLY","1")==std::string("1"))
            args.push_back("--close-after-reply");
    }
    args.push_back(nullptr);
    pid_t child=-1;
    const int result=posix_spawn(&child,worker.c_str(),&actions,nullptr,const_cast<char* const*>(args.data()),environ);
    posix_spawn_file_actions_destroy(&actions);close(sockets[1]);
    if(result){close(sockets[0]);error_=std::strerror(result);return false;}
    fd_=sockets[0];pid_=child;
    uint32_t hello[4]{};
    if(!transfer(hello,sizeof(hello),false) || hello[0]!=0x36524451 || hello[1]!=MODEL_WIDTH || hello[2]!=MODEL_HEIGHT || hello[3]!=(rightPlane?4u:3u)) {
        if(error_.empty())error_="incompatible Model depth bridge";
        stop();return false;
    }
    std::cout<<"[ModelDepth] Persistent "<<(qnn?"QNN run-list executor (~23 ms)":"prepared-executor (exact)")
             <<" worker ready; experimental plane-0 disparity, "
             <<(temporal_?"Model-style warped temporal state":"zero temporal state (t=0)")
             <<", gate="<<(censusGate_ && !spherical_?"contrast+census":"uncertainty<"+std::to_string(maxUncertainty_))
             <<", input="<<(spherical_?"spherical "+sphericalDegrees_+" deg/px":"resized pinhole")<<std::endl;
    return true;
}
bool ModelDepthPipeline::process(const std::vector<uint8_t>& left,const std::vector<uint8_t>& right,std::vector<float>& depth,
                                 const ModelMotion& motion,const std::vector<uint8_t>* sphericalLeft,
                                 const std::vector<uint8_t>* sphericalRight,float sphericalStep,
                                 const ModelCaptureMetadata* captureMetadata) {
    depth.clear();
    if(fd_<0){error_="Model depth worker is not initialized";return false;}
    try {
        auto l=modelResize(left,RECT_WIDTH,RECT_HEIGHT),r=modelResize(right,RECT_WIDTH,RECT_HEIGHT);
        const size_t modelPixels=size_t(MODEL_WIDTH)*MODEL_HEIGHT;
        const bool useSpherical=spherical_ && sphericalLeft && sphericalRight && sphericalStep>0 &&
                                sphericalLeft->size()==modelPixels && sphericalRight->size()==modelPixels;
        if(spherical_ && !useSpherical){error_="spherical Model input requested but not provided";return false;}
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
            if(recordSphericalFd_>=0) {
                const bool sok=write(recordSphericalFd_,header,sizeof(header))==ssize_t(sizeof(header)) &&
                    write(recordSphericalFd_,block,sizeof(block))==ssize_t(sizeof(block)) &&
                    write(recordSphericalFd_,sphericalLeft->data(),modelPixels)==ssize_t(modelPixels) &&
                    write(recordSphericalFd_,sphericalRight->data(),modelPixels)==ssize_t(modelPixels);
                if(!sok || recordFd_<0){close(recordSphericalFd_);recordSphericalFd_=-1;}
            }
        }
        if(!transfer(header,sizeof(header),true) || !transfer(block,sizeof(block),true) ||
           !transfer(const_cast<uint8_t*>(useSpherical?sphericalLeft->data():l.data()),modelPixels,true) ||
           !transfer(const_cast<uint8_t*>(useSpherical?sphericalRight->data():r.data()),modelPixels,true))return false;
        if(temporalState_) {
            const auto warpBegin=std::chrono::steady_clock::now();
            double rotation[3][3],translation[3];
            for(int i=0;i<3;i++){translation[i]=motion.translation[i];for(int j=0;j<3;j++)rotation[i][j]=motion.rotation[i][j];}
            const int64_t poseNs=captureMetadata?captureMetadata->poseTimeNs:0;
            const auto history=temporalState_->admit(motion.aligned,motion.reset,rotation,
                                                     lastPoseNs_&&poseNs?poseNs-lastPoseNs_:0,temporalMaxGapNs_,temporalMaxDegrees_);
            lastPoseNs_=poseNs;
            temporalDropped_[int(history)]++;
            if(history==ModelTemporalState::History::Usable)temporalApplied_++;
            else temporalState_->reset();
            temporalState_->inputs(rotation,translation,hidden_.data(),transforms_.data());
            temporalWarpMs_=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-warpBegin).count();
            if(!transfer(hidden_.data(),hidden_.size()*sizeof(float),true) ||
               !transfer(transforms_.data(),transforms_.size()*sizeof(float),true))return false;
        }
        // Reply: magic, sequence, pixel count, inference milliseconds (f32).
        uint32_t reply[4]{};
        if(!transfer(reply,sizeof(reply),false))return false;
        if(reply[0]!=0x364f4451 || reply[1]!=sequence_ || reply[2]!=MODEL_WIDTH*MODEL_HEIGHT){error_="invalid Model depth reply";return false;}
        std::memcpy(&milliseconds_,reply+3,sizeof(float));
        disparity_.assign(MODEL_WIDTH*MODEL_HEIGHT,0.f);
        if(!transfer(disparity_.data(),disparity_.size()*sizeof(float),false))return false;
        uncertainty_.assign(MODEL_WIDTH*MODEL_HEIGHT,255);
        if(!transfer(uncertainty_.data(),uncertainty_.size(),false))return false;
        std::vector<float> rightDisparity;
        if(secondDisparity_||temporalState_) {
            rightDisparity.resize(modelPixels);
            if(!transfer(rightDisparity.data(),rightDisparity.size()*sizeof(float),false))return false;
        }
        if(temporalState_) {
            const auto saveBegin=std::chrono::steady_clock::now();
            temporalState_->save(disparity_.data(),rightDisparity.data());
            temporalWarpMs_+=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-saveBegin).count();
            if(sequence_%100==1)
                std::cout<<"[ModelDepth] temporal frame "<<sequence_<<": applied "<<temporalApplied_
                         <<", none "<<temporalDropped_[int(ModelTemporalState::History::None)]
                         <<", reset "<<temporalDropped_[int(ModelTemporalState::History::Reset)]
                         <<", unaligned "<<temporalDropped_[int(ModelTemporalState::History::Unaligned)]
                         <<", gap "<<temporalDropped_[int(ModelTemporalState::History::TimeGap)]
                         <<", rotation "<<temporalDropped_[int(ModelTemporalState::History::Rotation)]
                         <<"; splat "<<temporalState_->splatFraction()<<", warp "<<temporalWarpMs_<<" ms, inference "
                         <<milliseconds_<<" ms"<<std::endl;
        }
        const auto postprocessBegin=std::chrono::steady_clock::now();
        std::array<float,256> confidenceOf;
        for(size_t u=0;u<confidenceOf.size();++u)
            confidenceOf[u]=std::clamp((maxUncertainty_-float(u))/float(maxUncertainty_-50),.05f,1.f);
        confidence_.resize(size_t(RECT_WIDTH)*RECT_HEIGHT);
        if(useSpherical) {
            if(sphericalMap_.step!=sphericalStep)
                sphericalMap_=modelSphericalMap(sphericalStep,FOCAL_BASELINE/RECT_FOCAL,RECT_WIDTH,RECT_HEIGHT,
                                                RECT_FOCAL,RECT_CX,RECT_CY);
            depth=modelSphericalDepth(disparity_,uncertainty_,sphericalMap_,maxUncertainty_);
            for(size_t i=0;i<confidence_.size();i++)
                confidence_[i]=sphericalMap_.modelIndex[i]>=0 ?
                    confidenceOf[uncertainty_[sphericalMap_.modelIndex[i]]] : .05f;
        } else {
            depth=modelMetricDepth(disparity_,l,r,RECT_WIDTH,RECT_HEIGHT,FOCAL_BASELINE,.25f,5.f,
                                   censusGate_?nullptr:&uncertainty_,maxUncertainty_);
            for(int y=0;y<RECT_HEIGHT;y++)for(int x=0;x<RECT_WIDTH;x++) {
                const int mx=std::min(MODEL_WIDTH-1,int((x+.5f)*MODEL_WIDTH/RECT_WIDTH));
                const int my=std::min(MODEL_HEIGHT-1,int((y+.5f)*MODEL_HEIGHT/RECT_HEIGHT));
                confidence_[size_t(y)*RECT_WIDTH+x]=confidenceOf[uncertainty_[size_t(my)*MODEL_WIDTH+mx]];
            }
        }
        postprocessMs_=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-postprocessBegin).count();
        if(secondDisparity_ && recordedV2_<recordLimit_) {
            const auto directory=std::filesystem::path(recordV2Directory_);
            const std::string prefix="frame-"+std::to_string(sequence_);
            const auto& modelLeft=useSpherical?*sphericalLeft:l;
            const auto& modelRight=useSpherical?*sphericalRight:r;
            std::vector<uint8_t> validity(depth.size());
            for(size_t index=0;index<depth.size();index++)validity[index]=depth[index]>0;
            const bool written=writeArray(directory/(prefix+"-left.u8"),modelLeft) &&
                writeArray(directory/(prefix+"-right.u8"),modelRight) &&
                writeArray(directory/(prefix+"-rect-left.u8"),left) &&
                writeArray(directory/(prefix+"-rect-right.u8"),right) &&
                writeArray(directory/(prefix+"-disparity.f32"),disparity_) &&
                writeArray(directory/(prefix+"-right-disparity.f32"),rightDisparity) &&
                writeArray(directory/(prefix+"-uncertainty.u8"),uncertainty_) &&
                writeArray(directory/(prefix+"-depth.f32"),depth) &&
                writeArray(directory/(prefix+"-confidence.f32"),confidence_) &&
                writeArray(directory/(prefix+"-validity.u8"),validity);
            std::ofstream frame(directory/(prefix+".json"));
            frame<<std::setprecision(9)<<"{\"version\":2,\"sequence\":"<<sequence_
                <<",\"inference_ms\":"<<milliseconds_<<",\"motion_aligned\":"<<(motion.aligned?"true":"false")
                <<",\"motion_reset\":"<<(motion.reset?"true":"false")
                <<",\"spherical_step_rad\":"<<(useSpherical?sphericalStep:0)
                <<",\"capture_begin_ns\":"<<(captureMetadata?captureMetadata->captureBeginNs:0)
                <<",\"capture_end_ns\":"<<(captureMetadata?captureMetadata->captureEndNs:0)
                <<",\"pose_time_ns\":"<<(captureMetadata?captureMetadata->poseTimeNs:0)
                <<",\"capture_time_pose\":"<<(captureMetadata&&captureMetadata->captureTimePose?"true":"false")
                <<",\"pose_valid\":"<<(captureMetadata&&captureMetadata->poseValid?"true":"false")
                <<",\"position_m\":[";
            for(int i=0;i<3;i++)frame<<(i?",":"")<<(captureMetadata?captureMetadata->position[i]:0);
            frame<<"],\"orientation_xyzw\":[";
            for(int i=0;i<4;i++)frame<<(i?",":"")<<(captureMetadata?captureMetadata->orientation[i]:(i==3?1:0));
            frame<<"],\"camera_to_world_rotation\":[";
            for(int i=0;i<9;i++)frame<<(i?",":"")<<(captureMetadata?captureMetadata->cameraToWorldRotation[i]:(i%4==0?1:0));
            frame<<"],\"camera_to_world_translation_m\":[";
            for(int i=0;i<3;i++)frame<<(i?",":"")<<(captureMetadata?captureMetadata->cameraToWorldTranslation[i]:0);
            frame<<"],\"camera_sequence\":"<<(captureMetadata?captureMetadata->cameraSequence:0)
                 <<",\"left_end_of_readout_ns\":"<<(captureMetadata?captureMetadata->leftEndOfReadoutNs:0)
                 <<",\"right_end_of_readout_ns\":"<<(captureMetadata?captureMetadata->rightEndOfReadoutNs:0)
                 <<",\"left_write_start_ns\":"<<(captureMetadata?captureMetadata->leftWriteStartNs:0)
                 <<",\"right_write_start_ns\":"<<(captureMetadata?captureMetadata->rightWriteStartNs:0)
                 <<",\"skipped_sequences\":"<<(captureMetadata?captureMetadata->skippedSequences:0)
                 <<",\"pose_timing\":\""<<(captureMetadata&&captureMetadata->cameraSequence?"v4l2_end_of_readout_minus_offset":"dma_read_midpoint")<<"\"";
            frame<<",\"left_slot\":"<<(captureMetadata?captureMetadata->leftSlot:-1)
                 <<",\"right_slot\":"<<(captureMetadata?captureMetadata->rightSlot:-1)<<",\"pose_samples\":[";
            if(captureMetadata) {
                bool first=true;
                for(int s=0;s<ModelCaptureMetadata::POSE_SAMPLES;s++) {
                    if(!captureMetadata->poseSampleValid[s])continue;
                    frame<<(first?"":",")<<"{\"offset_ms\":"<<ModelCaptureMetadata::poseSampleOffsetMs[s]<<",\"position_m\":[";
                    for(int i=0;i<3;i++)frame<<(i?",":"")<<captureMetadata->poseSamplePosition[s][i];
                    frame<<"],\"orientation_xyzw\":[";
                    for(int i=0;i<4;i++)frame<<(i?",":"")<<captureMetadata->poseSampleOrientation[s][i];
                    frame<<"]}";first=false;
                }
            }
            frame<<"],\"files_prefix\":\""<<prefix<<"\"}\n";
            if(!written || !frame){error_="Model v2 recording failed";return false;}
            ++recordedV2_;
        }
        return true;
    } catch(const std::exception& e){error_=e.what();return false;}
}
void ModelDepthPipeline::stop() {
    if(recordFd_>=0){close(recordFd_);recordFd_=-1;}
    if(recordSphericalFd_>=0){close(recordSphericalFd_);recordSphericalFd_=-1;}
    if(fd_>=0){shutdown(fd_,SHUT_RDWR);close(fd_);fd_=-1;}
    if(pid_<0)return;
    // EOF normally closes the native session, restoring clocks, workers and gaze.
    // SIGTERM also lets the native worker finish bounded session cleanup; the
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
