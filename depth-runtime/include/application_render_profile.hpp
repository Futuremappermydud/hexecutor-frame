#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace FrameSpatialCloud {
// CPU wall times, including blocking calls. GPU execution requires a separate
// timestamp-query measurement; a fence wait is not a GPU duration.
class ApplicationRenderProfile {
public:
    using Clock=std::chrono::steady_clock;
    enum Stage { WaitFrame, Locate, Upload, Acquire, Eye0, Eye1, EndFrame, Total, Period, Count };
    using Sample=std::array<double,Count>;
    ApplicationRenderProfile() {
        const char* setting=std::getenv("FRAME_OPENXR_RENDER_PROFILE");
        enabled_=setting && std::string(setting)=="1";
        if(enabled_) samples_.reserve(120);
    }
    Clock::time_point now() const { return enabled_?Clock::now():Clock::time_point{}; }
    void mark(Sample& sample,Stage stage,Clock::time_point start) const {
        if(enabled_) sample[stage]=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    }
    void finish(Sample sample,Clock::time_point start,bool rendered) {
        if(!enabled_) return;
        mark(sample,Total,start);
        sample[Period]=previous_==Clock::time_point{}?sample[Total]:
            std::chrono::duration<double,std::milli>(start-previous_).count();
        previous_=start;
        samples_.push_back(sample); rendered_+=rendered;
        if(samples_.size()<120) return;
        static const char* names[]{"xrWaitFrame","locate","mesh_upload","swapchain_acquire_wait",
            "app_eye0_draw_wait","app_eye1_draw_wait","xrEndFrame_with_layer","frame","period"};
        std::cout<<"[ApplicationRenderProfile] frames="<<samples_.size()<<", rendered="<<rendered_;
        double period=0; for(const auto& s:samples_) period+=s[Period];
        std::cout<<", Hz="<<samples_.size()*1000/period;
        for(int stage=0;stage<Count;++stage) {
            std::array<double,120> values{};
            for(size_t i=0;i<samples_.size();++i) values[i]=samples_[i][stage];
            std::sort(values.begin(),values.end());
            std::cout<<", "<<names[stage]<<"_p50_ms="<<values[59]
                <<", "<<names[stage]<<"_p95_ms="<<values[113]
                <<", "<<names[stage]<<"_p99_ms="<<values[117];
        }
        std::cout<<std::endl; samples_.clear(); rendered_=0;
    }
private:
    bool enabled_=false;
    size_t rendered_=0;
    Clock::time_point previous_{};
    std::vector<Sample> samples_;
};
}
