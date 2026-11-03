#pragma once
#include "occlusion_surface.hpp"
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace FrameSpatial {
namespace OcclusionBenchmark {
// Version 2 is explicitly serialized little-endian uint32/uint64/IEEE float32;
// no struct padding is written. These are derived luma/depth, never DMA slots.
enum Timing { Capture, Rectify, Stereo, Speckles, Smooth, Completion, Holes,
              Evidence, History, Mesh, PublishAge, TimingCount };
inline const char* timingName(int i) {
    static const char* names[]{"capture_ms","rectify_ms","stereo_ms","speckles_ms",
        "smooth_ms","completion_ms","holes_ms","evidence_ms","history_ms","mesh_ms","publish_age_ms"};
    return names[i];
}
enum Flag : uint32_t { Aligned=1, CapturePose=2, Completed=4, Confidence=8, Guarded=16 };
struct Header {
    uint32_t width=616,height=616;
    float focal=380,cx=308,cy=308;
    // Row-major optical-to-head affine transform, x/y optical pixel rays, +Z.
    std::array<float,12> headFromOptical{};
    float warmup=3,segment=8,recovery=3;
};
struct Metadata {
    uint64_t frame=0,captureBeginNs=0,captureEndNs=0,publishNs=0;
    int64_t poseTime=0;
    uint32_t flags=0,phase=0;
    float elapsed=0;
    std::array<float,TimingCount> timings{};
    // Position xyz, quaternion xyzw in the standing/stage reference space.
    std::array<float,7> headPose{};
    OcclusionReprojection transform;
};
struct Frame {
    Metadata meta;
    std::vector<uint8_t> left,right,evidence;
    std::vector<float> measured,input;
};
inline const char* phaseName(uint32_t phase) {
    static const char* names[]{"warmup","still","head_turn","hand_motion","recovery","finished"};
    return phase<6?names[phase]:"invalid";
}
inline uint32_t phaseAt(double seconds,const Header& h) {
    if(seconds<h.warmup) return 0;
    seconds-=h.warmup;
    if(seconds<3*h.segment) return 1+uint32_t(seconds/h.segment);
    return seconds<3*h.segment+h.recovery?4:5;
}
template<class T> inline void write(std::ostream& out,const T& value) {
    out.write(reinterpret_cast<const char*>(&value),sizeof(T));
}
template<class T> inline void read(std::istream& in,T& value) {
    in.read(reinterpret_cast<char*>(&value),sizeof(T));
    if(!in) throw std::runtime_error("Truncated benchmark recording");
}
inline void writeHeader(std::ostream& out,const Header& h) {
    out.write("FROCCB02",8);
    write(out,h.width); write(out,h.height); write(out,h.focal); write(out,h.cx); write(out,h.cy);
    for(float f:h.headFromOptical) write(out,f);
    write(out,h.warmup); write(out,h.segment); write(out,h.recovery);
}
inline void writeMetadata(std::ostream& out,const Metadata& m) {
    write(out,m.frame); write(out,m.captureBeginNs); write(out,m.captureEndNs);
    write(out,m.publishNs); write(out,m.poseTime); write(out,m.flags); write(out,m.phase); write(out,m.elapsed);
    for(float f:m.timings) write(out,f);
    for(float f:m.headPose) write(out,f);
    for(const auto& row:m.transform.rotation) for(float f:row) write(out,f);
    for(float f:m.transform.translation) write(out,f);
}
inline void readMetadata(std::istream& in,Metadata& m) {
    read(in,m.frame); read(in,m.captureBeginNs); read(in,m.captureEndNs);
    read(in,m.publishNs); read(in,m.poseTime); read(in,m.flags); read(in,m.phase); read(in,m.elapsed);
    for(float& f:m.timings) read(in,f);
    for(float& f:m.headPose) read(in,f);
    for(auto& row:m.transform.rotation) for(float& f:row) read(in,f);
    for(float& f:m.transform.translation) read(in,f);
}
template<class T> inline void writePixels(std::ostream& out,const std::vector<T>& pixels) {
    out.write(reinterpret_cast<const char*>(pixels.data()),pixels.size()*sizeof(T));
}
template<class T> inline void readPixels(std::istream& in,std::vector<T>& pixels,size_t count) {
    pixels.resize(count);
    in.read(reinterpret_cast<char*>(pixels.data()),count*sizeof(T));
    if(!in) throw std::runtime_error("Truncated benchmark pixels");
}
constexpr uint32_t FrameTag=0x324d5246,EndTag=0x32444e45;
constexpr uint32_t MetadataBytes=172;
constexpr uint32_t FrameLimit=512; // Default recording budget; v2 files remain readable.
constexpr uint32_t MaxFrameLimit=65536;

// The capture worker never writes multi-megabyte frame data to disk. A two-frame
// queue bounds RAM and never waits for I/O; overflow is explicit in the footer.
class Recorder {
public:
    Recorder(const std::string& path,const Header& header,uint32_t frameLimit=FrameLimit)
        :header_(header),frameLimit_(frameLimit) {
        if(!frameLimit_ || frameLimit_>MaxFrameLimit)
            throw std::invalid_argument("Recording frame budget must be in [1,65536]");
        file_.open(path,std::ios::binary|std::ios::trunc);
        if(!file_) throw std::runtime_error("Cannot open benchmark "+path);
        writeHeader(file_,header_);
        if(!file_) throw std::runtime_error("Cannot write benchmark header");
        thread_=std::thread([this]{run();});
        std::cout<<"[Benchmark] recording="<<path<<", bounded_frames="<<frameLimit_<<std::endl;
    }
    ~Recorder() { finish(false); if(thread_.joinable()) thread_.join(); }
    Recorder(const Recorder&)=delete;
    Recorder& operator=(const Recorder&)=delete;
    bool active() { std::lock_guard<std::mutex> lock(mutex_); return !closing_; }
    void append(Metadata meta,const std::vector<uint8_t>& left,const std::vector<uint8_t>& right,
                const std::vector<float>& measured,const std::vector<float>& input,
                const std::vector<uint8_t>& evidence) {
        std::unique_lock<std::mutex> lock(mutex_);
        if(closing_) return;
        if(!firstNs_) firstNs_=meta.captureBeginNs;
        meta.phase=phaseAt((meta.captureBeginNs-firstNs_)*1e-9,header_);
        if(meta.phase!=phase_) {
            phase_=meta.phase;
            std::ostringstream message;
            message<<"[Benchmark] phase="<<phaseName(phase_)<<"\n";
            std::cout<<message.str()<<std::flush;
        }
        if(meta.phase==5 || enqueued_>=frameLimit_) {
            complete_=meta.phase==5;
            closing_=true; condition_.notify_one(); return;
        }
        const size_t count=size_t(header_.width)*header_.height;
        if(left.size()!=count || right.size()!=count || measured.size()!=count ||
           input.size()!=count || evidence.size()!=count) {
            closing_=true; condition_.notify_one(); return;
        }
        if(queue_.size()>=2) { ++dropped_; return; }
        queue_.push_back({meta,left,right,evidence,measured,input});
        ++enqueued_; condition_.notify_one();
    }
private:
    void finish(bool complete) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!closing_) { closing_=true; complete_=complete; condition_.notify_one(); }
    }
    void run() {
        uint32_t written=0;
        for(;;) {
            Frame frame;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock,[this]{return closing_||!queue_.empty();});
                if(queue_.empty()) break;
                frame=std::move(queue_.front()); queue_.pop_front();
            }
            write(file_,FrameTag);
            const uint32_t bytes=MetadataBytes+header_.width*header_.height*11;
            write(file_,bytes); writeMetadata(file_,frame.meta);
            writePixels(file_,frame.left); writePixels(file_,frame.right);
            writePixels(file_,frame.measured); writePixels(file_,frame.input); writePixels(file_,frame.evidence);
            if(!file_) {
                std::lock_guard<std::mutex> lock(mutex_);
                closing_=true; complete_=false; queue_.clear(); break;
            }
            ++written;
        }
        uint32_t dropped=0,complete=0;
        { std::lock_guard<std::mutex> lock(mutex_); dropped=dropped_; complete=complete_; }
        write(file_,EndTag); write(file_,written); write(file_,dropped); write(file_,complete);
        file_.flush(); const bool okay=bool(file_); file_.close();
        std::ostringstream message;
        message<<"[Benchmark] done frames="<<written<<", dropped="<<dropped
               <<", complete="<<complete<<", io_ok="<<okay<<"\n";
        std::cout<<message.str()<<std::flush;
    }
    Header header_;
    std::ofstream file_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<Frame> queue_;
    uint64_t firstNs_=0;
    uint32_t phase_=6,enqueued_=0,dropped_=0;
    uint32_t frameLimit_=FrameLimit;
    bool closing_=false,complete_=false;
};

class Reader {
public:
    Header header;
    uint32_t frames=0,dropped=0,complete=0;
    explicit Reader(const std::string& path):file_(path,std::ios::binary) {
        char magic[8]{}; file_.read(magic,8);
        if(!file_ || std::memcmp(magic,"FROCCB02",8)) throw std::runtime_error("Not a v2 benchmark recording");
        read(file_,header.width); read(file_,header.height); read(file_,header.focal);
        read(file_,header.cx); read(file_,header.cy);
        for(float& f:header.headFromOptical) read(file_,f);
        read(file_,header.warmup); read(file_,header.segment); read(file_,header.recovery);
        if(!header.width || !header.height || header.width>1024 || header.height>1024 ||
           !std::isfinite(header.focal) || header.focal<=0 || !std::isfinite(header.cx) ||
           !std::isfinite(header.cy) || !std::isfinite(header.warmup) || header.warmup<0 ||
           !std::isfinite(header.segment) || header.segment<=0 ||
           !std::isfinite(header.recovery) || header.recovery<0 ||
           std::any_of(header.headFromOptical.begin(),header.headFromOptical.end(),
                       [](float f){return !std::isfinite(f);}))
            throw std::runtime_error("Invalid benchmark header");
    }
    bool next(Frame& frame) {
        if(ended_) return false;
        uint32_t tag=0; read(file_,tag);
        if(tag==EndTag) {
            uint32_t count=0; read(file_,count); read(file_,dropped); read(file_,complete);
            if(count!=frames || complete>1 || file_.peek()!=std::char_traits<char>::eof())
                throw std::runtime_error("Invalid benchmark footer");
            ended_=true; return false;
        }
        uint32_t bytes=0; read(file_,bytes);
        const size_t count=size_t(header.width)*header.height;
        if(tag!=FrameTag || bytes!=MetadataBytes+count*11 || frames>=MaxFrameLimit)
            throw std::runtime_error("Invalid benchmark frame size/tag/count");
        readMetadata(file_,frame.meta);
        const auto& m=frame.meta;
        if(m.phase>4 || (m.flags&~31u) || m.captureEndNs<m.captureBeginNs ||
           m.publishNs<m.captureEndNs || !std::isfinite(m.elapsed) || m.elapsed<0 ||
           std::any_of(m.timings.begin(),m.timings.end(),[](float f){return !std::isfinite(f)||f<0;}) ||
           std::any_of(m.headPose.begin(),m.headPose.end(),[](float f){return !std::isfinite(f);}))
            throw std::runtime_error("Invalid benchmark metadata");
        for(const auto& row:m.transform.rotation) for(float f:row)
            if(!std::isfinite(f)) throw std::runtime_error("Invalid benchmark rotation");
        for(float f:m.transform.translation)
            if(!std::isfinite(f)) throw std::runtime_error("Invalid benchmark translation");
        readPixels(file_,frame.left,count); readPixels(file_,frame.right,count);
        readPixels(file_,frame.measured,count); readPixels(file_,frame.input,count); readPixels(file_,frame.evidence,count);
        for(size_t i=0;i<count;++i) {
            if(!std::isfinite(frame.measured[i]) || !std::isfinite(frame.input[i]) ||
               (frame.measured[i]!=0 && !surfaceDepth(frame.measured[i])) ||
               (frame.input[i]!=0 && !surfaceDepth(frame.input[i])) || frame.evidence[i]>StrongDepth ||
               bool(frame.evidence[i])!=surfaceDepth(frame.input[i]))
                throw std::runtime_error("Invalid benchmark depth/evidence");
        }
        ++frames; return true;
    }
private:
    std::ifstream file_;
    bool ended_=false;
};
} // namespace OcclusionBenchmark
} // namespace FrameSpatial
