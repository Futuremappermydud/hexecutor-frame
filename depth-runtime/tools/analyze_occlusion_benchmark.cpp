#include "occlusion_benchmark.hpp"
#include <iomanip>
using namespace FrameSpatial;
using namespace FrameSpatial::OcclusionBenchmark;
float percentile(std::vector<float> values,float q) {
    if(values.empty()) return 0;
    std::sort(values.begin(),values.end());
    return values[size_t(q*(values.size()-1))];
}
struct Metrics {
    double measured=0,input=0,coverage[2]{},contradictions[2]{},changes[2]{};
    size_t frames=0,comparisons=0,reliable=0;
    std::array<std::vector<float>,TimingCount> timings;
};
int main(int argc,char** argv) {
    if(argc!=3 && argc!=10) {
        std::cerr<<"Usage: analyze_occlusion_benchmark recording.bin frames.csv [--confidence strong-count measured-count inferred-count inferred-ms measured-ms pending-ms]\n";
        return 2;
    }
    try {
        OcclusionConfidenceConfig config;
        if (argc == 10) {
            if (std::string(argv[3]) != "--confidence") throw std::runtime_error("Expected --confidence");
            const auto count=[](const char* value) {
                size_t used=0; const int result=std::stoi(value,&used);
                if(used!=std::string(value).size() || result<1 || result>255)
                    throw std::runtime_error("Observation counts must be integers in [1,255]");
                return result;
            };
            const auto age=[](const char* value) {
                size_t used=0; const float result=std::stof(value,&used)/1000;
                if(used!=std::string(value).size() || !std::isfinite(result) || result<=0 || result>1)
                    throw std::runtime_error("Expiry must be milliseconds in (0,1000]");
                return result;
            };
            config={count(argv[4]),count(argv[5]),count(argv[6]),age(argv[7]),age(argv[8]),age(argv[9])};
        }
        std::cout<<"Confidence config: strong="<<config.strongObservations
            <<", measured="<<config.measuredObservations<<", inferred="<<config.inferredObservations
            <<", inferred_ms="<<config.inferredMaxAge*1000<<", measured_ms="<<config.measuredMaxAge*1000
            <<", pending_ms="<<config.pendingMaxAge*1000<<'\n';
        Reader reader(argv[1]); Frame frame;
        const auto& h=reader.header; const size_t n=size_t(h.width)*h.height;
        std::ofstream csv(argv[2]); if(!csv) throw std::runtime_error("Cannot open CSV output");
        csv<<"frame,phase,aligned,measured_pct,input_pct,legacy_mesh_pct,confidence_mesh_pct,legacy_contradictions,confidence_contradictions,legacy_changed_pct,confidence_changed_pct";
        for(int t=0;t<TimingCount;++t) csv<<','<<timingName(t);
        csv<<'\n';
        OcclusionSurfaceHistory histories[2]{OcclusionSurfaceHistory{},OcclusionSurfaceHistory{config}};
        std::vector<OcclusionObservation> previous[2];
        std::array<Metrics,5> phases{};
        bool guarded=false;
        uint64_t lastFrame=0,lastNs=0,firstNs=0; size_t gaps=0,fallbacks=0;
        while(reader.next(frame)) {
            const auto& m=frame.meta;
            const bool frameGuarded=(m.flags&Guarded)!=0;
            if(!lastFrame) {
                guarded=frameGuarded;
                histories[0]=OcclusionSurfaceHistory({},guarded?2.5f:0.2f);
                std::cout<<"Recorded matcher="<<(guarded?"guarded":"baseline")
                         <<", legacy_max_age_ms="<<(guarded?2500:200)
                         <<"; confidence column is a counterfactual, not guarded live parity\n";
            }
            if(frameGuarded!=guarded || (guarded&&(m.flags&Confidence)))
                throw std::runtime_error("Mixed matcher/invalid guarded retention policy");
            if(lastFrame && (m.frame<=lastFrame || m.captureBeginNs<=lastNs))
                throw std::runtime_error("Non-increasing frame IDs/timestamps");
            const bool gap=lastFrame && m.frame!=lastFrame+1;
            if(gap) ++gaps;
            const bool aligned=(m.flags&Aligned) && !gap;
            const bool pose=m.flags&CapturePose;
            if(!pose) ++fallbacks;
            if(!firstNs) firstNs=m.captureBeginNs;
            lastNs=m.captureBeginNs; lastFrame=m.frame;
            auto& stats=phases[m.phase]; ++stats.frames;
            const size_t measured=std::count_if(frame.measured.begin(),frame.measured.end(),surfaceDepth);
            const size_t input=std::count_if(frame.input.begin(),frame.input.end(),surfaceDepth);
            stats.measured+=measured; stats.input+=input;
            size_t reliable=0;
            for(auto e:frame.evidence) if(e>=MeasuredDepth) ++reliable;
            stats.reliable+=reliable;
            csv<<m.frame<<','<<phaseName(m.phase)<<','<<aligned<<','<<100.0*measured/n<<','<<100.0*input/n;
            double coverage[2]{},changes[2]{}; size_t contradictions[2]{};
            for(int mode=0;mode<2;++mode) {
                std::vector<float> out;
                if(pose) out=histories[mode].update(frame.input,h.width,h.height,h.focal,h.cx,h.cy,
                    aligned?&m.transform:nullptr,m.elapsed,mode?&frame.evidence:nullptr);
                else { histories[mode].reset(); out.assign(n,0.0f); }
                // Mixed one/two-pixel cells use their actual covered image area.
                coverage[mode]=100.0*occlusionSurfaceArea(occlusionSurfaceIndices(out,h.width,h.height),h.width)/n;
                stats.coverage[mode]+=coverage[mode];
                size_t changed=0,united=0;
                const auto projected=aligned?reprojectOcclusionObservations(previous[mode],h.width,h.height,
                    h.focal,h.cx,h.cy,m.transform):std::vector<OcclusionObservation>{};
                for(size_t i=0;i<n;++i) {
                    if(frame.evidence[i]>=MeasuredDepth && surfaceDepth(out[i]) &&
                        !surfaceRange(std::min(out[i],frame.measured[i]),std::max(out[i],frame.measured[i])))
                        ++contradictions[mode];
                    if(!projected.empty()) {
                        const bool a=surfaceDepth(out[i]),b=surfaceDepth(projected[i].depth);
                        united+=a||b; changed+=a!=b;
                    }
                }
                changes[mode]=united?100.0*changed/united:0;
                stats.changes[mode]+=changes[mode]; stats.contradictions[mode]+=contradictions[mode];
                previous[mode].resize(n);
                for(size_t i=0;i<n;++i) previous[mode][i]={out[i],0,1};
            }
            if(aligned) ++stats.comparisons;
            csv<<','<<coverage[0]<<','<<coverage[1]<<','<<contradictions[0]<<','<<contradictions[1]
               <<','<<changes[0]<<','<<changes[1];
            for(int t=0;t<TimingCount;++t) { stats.timings[t].push_back(m.timings[t]); csv<<','<<m.timings[t]; }
            csv<<'\n';
        }
        std::cout<<std::fixed<<std::setprecision(2);
        std::cout<<"Recording: frames="<<reader.frames<<", dropped="<<reader.dropped
                 <<", complete="<<reader.complete<<", sequence_gaps="<<gaps<<", pose_fallbacks="<<fallbacks<<'\n';
        std::cout<<"Depth throughput="<<(lastNs>firstNs?(reader.frames-1)/((lastNs-firstNs)*1e-9):0)<<" Hz\n";
        for(size_t p=0;p<phases.size();++p) {
            const auto& s=phases[p]; if(!s.frames) continue;
            std::cout<<phaseName(p)<<": frames="<<s.frames<<", measured="<<100*s.measured/(n*s.frames)
                <<"%, input="<<100*s.input/(n*s.frames)<<"%\n";
            for(int mode=0;mode<2;++mode) std::cout<<"  "<<(mode?"confidence":"legacy")
                <<": mesh coverage="<<s.coverage[mode]/s.frames<<"%, contradicted retained depth="
                <<(s.reliable?100*s.contradictions[mode]/s.reliable:0)<<"% of reliable measurements"
                <<", coverage change="<<(s.comparisons?s.changes[mode]/s.comparisons:0)<<"%\n";
            for(int t=0;t<TimingCount;++t) std::cout<<"  "<<timingName(t)<<" p50="
                <<percentile(s.timings[t],.5f)<<", p95="<<percentile(s.timings[t],.95f)
                <<", p99="<<percentile(s.timings[t],.99f)<<'\n';
        }
        std::cout<<"Same-input replay; camera-view coverage and consistency proxies, without ground truth.\n"
                 <<"Publish age starts at DMA-read midpoint, not sensor exposure. Timings include recording overhead.\n";
        if(!reader.complete || reader.dropped || gaps)
            std::cout<<"INCOMPLETE/LOSSY: repeat before using this recording for a filter decision.\n";
        if(!reader.frames) throw std::runtime_error("Empty benchmark recording");
        return (!reader.complete || reader.dropped || gaps)?3:0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
