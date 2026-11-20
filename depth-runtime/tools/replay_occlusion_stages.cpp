// Replay recorded rectified luma through current shaders and export selected
// frame diagnostics. Validation consumes the complete v2 recording first.
#include "occlusion_benchmark.hpp"
#include "occlusion_completion.hpp"
#include "occlusion_stage_diagnostics.hpp"
#include "depth_speckle_filter.hpp"
#include "calibration.hpp"
#include <filesystem>
#include <sstream>
#include <set>
#include <iomanip>
using namespace FrameSpatial;
namespace B=FrameSpatial::OcclusionBenchmark;
struct Region {std::string name;int x0,y0,x1,y1;float z;};
std::vector<Region> readRegions(const std::string& path) {
    std::ifstream file(path); if(!file) throw std::runtime_error("Cannot read regions");
    std::vector<Region> regions; std::string line;
    while(std::getline(file,line)) {
        if(line.empty()||line[0]=='#') continue;
        std::replace(line.begin(),line.end(),',',' '); std::istringstream in(line); Region r;
        if(!(in>>r.name>>r.x0>>r.y0>>r.x1>>r.y1>>r.z) || r.x0<0 || r.y0<0 ||
           r.x1>RECT_WIDTH || r.y1>RECT_HEIGHT || r.x0>=r.x1 || r.y0>=r.y1 ||
           !surfaceDepth(r.z)) throw std::runtime_error("Invalid region row: "+line);
        std::string extra; if(in>>extra) throw std::runtime_error("Extra region fields");
        regions.push_back(r);
    }
    if(regions.empty()) throw std::runtime_error("No marked regions");
    return regions;
}
void report(std::ostream& csv,uint64_t frame,const std::string& stage,
            const std::vector<float>& depth,const std::vector<Region>& regions) {
    for(const auto& r:regions) {
        size_t accepted=0; double absError=0,signedError=0;
        for(int y=r.y0;y<r.y1;++y) for(int x=r.x0;x<r.x1;++x) {
            const float z=depth[size_t(y)*RECT_WIDTH+x]; if(!surfaceDepth(z)) continue;
            ++accepted; absError+=std::abs(z-r.z); signedError+=z-r.z;
        }
        const size_t total=size_t(r.x1-r.x0)*(r.y1-r.y0);
        csv<<frame<<','<<stage<<','<<r.name<<','<<total<<','<<accepted<<','
           <<double(total-accepted)/total<<',';
        if(accepted) csv<<absError/accepted<<','<<signedError/accepted;
        else csv<<',';
        csv<<'\n';
    }
}
int main(int argc,char** argv) try {
    if(argc<5 || argc>6) {
        std::cerr<<"Usage: replay_occlusion_stages recording.bin shader-dir output-dir export-frame-ids [regions.csv]\n"
                 <<"Frame IDs are comma separated; all preceding frames are replayed for history.\n";return 2;
    }
    std::set<uint64_t> selected; std::istringstream ids(argv[4]);std::string token;
    while(std::getline(ids,token,',')) {
        size_t used=0;const auto id=std::stoull(token,&used);
        if(used!=token.size() || !id) throw std::runtime_error("Invalid export frame ID");
        selected.insert(id);
    }
    if(selected.empty()) throw std::runtime_error("No export frames");
    B::Reader validation(argv[1]); B::Frame f; uint64_t last=0,lastNs=0; size_t poseFallbacks=0;
    std::set<uint64_t> available;
    while(validation.next(f)) {
        if(f.meta.flags&B::Guarded)
            throw std::runtime_error("Baseline stage replayer cannot model guarded matching/history; use analyze_occlusion_benchmark for recorded guarded legacy retention, or benchmark_stereo_guarded for matcher replay");
        if(last && (f.meta.frame!=last+1 || f.meta.captureBeginNs<=lastNs))
            throw std::runtime_error("Recording has frame gaps or unordered timestamps");
        last=f.meta.frame;lastNs=f.meta.captureBeginNs;
        if(!(f.meta.flags&B::CapturePose)) ++poseFallbacks;
        if(selected.count(last)) available.insert(last);
    }
    if(!validation.complete || validation.dropped) throw std::runtime_error("Incomplete/dropped recording");
    if(available!=selected) throw std::runtime_error("Requested frame missing");
    if(validation.header.width!=RECT_WIDTH||validation.header.height!=RECT_HEIGHT)
        throw std::runtime_error("Shader requires 616x616 recording");
    const auto regions=argc==6?readRegions(argv[5]):std::vector<Region>{};
    std::filesystem::create_directories(argv[3]);
    const std::string output=argv[3];
    std::ofstream csv(output+"/regions.csv");
    csv<<"frame,stage,region,total_marked_pixels,accepted_pixels,missing_fraction,accepted_mae_m,accepted_bias_m\n";
    std::ofstream meshRegions(output+"/mesh-regions.csv");
    meshRegions<<"frame,variant,region,total_marked_pixels,covered_pixels,missing_fraction\n";
    std::ofstream counts(output+"/stages.csv");
    counts<<"frame,phase,supported,speckle,smoothed,planes,legacy_pre_history,corrected_pre_history,completion_off,confirmed,recorded_pre_history,replay_measured_mismatch,lr_rejected,range_rejected,support_rejected\n";
    VulkanStereoPipeline stereo; if(!stereo.initialize(argv[2],true)) throw std::runtime_error(stereo.lastError());
    const OcclusionConfidenceConfig config{2,2,3,.200f,.250f,.250f};
    OcclusionSurfaceHistory correctedHistory{config};
    correctedHistory.enableDiagnostics(true);
    const uint16_t eventBits[]={HistoryPending,HistoryConfirmed,HistoryMeasuredClear,HistoryPredictionExpiry,
        HistorySourceExpiry,HistoryReprojectionLoss,HistoryCollision,HistoryMeasuredAgreement,
        HistoryInferredAgreement,HistoryEvidenceChange,HistoryHeld};
    std::ofstream events(output+"/history-events.csv");
    events<<"frame,capture_end_ns,elapsed_s,pending,confirmed,measured_clear,prediction_expiry,source_expiry,reprojection_loss,collision,measured_agreement,inferred_agreement,evidence_change,held,whole_mesh_min_lifetime_s,whole_mesh_min_deadline_ns,mesh_latest_lifetime_s,mesh_latest_deadline_ns,geometric_area_px,snapshot_live_area_px,simulated_display_75ms_area_px,simulated_display_100ms_area_px\n";
    B::Reader reader(argv[1]); const auto& h=reader.header;
    const auto count=[](const auto& depths){return std::count_if(depths.begin(),depths.end(),surfaceDepth);};
    while(reader.next(f)) {
        const bool save=selected.count(f.meta.frame); StereoDiagnostics diagnostic;std::vector<float> depth;
        if(!stereo.process(f.left,f.right,depth,save?&diagnostic:nullptr)) throw std::runtime_error(stereo.lastError());
        const size_t supported=count(depth);
        const std::string prefix=output+"/frame-"+std::to_string(f.meta.frame);
        if(save) {
            writeStereoDiagnostics(prefix,diagnostic,h.width,h.height);
            writeDiagnosticLuma(prefix+"-left.pgm",f.left,h.width,h.height);
            writeDiagnosticLuma(prefix+"-right.pgm",f.right,h.width,h.height);
            writeDiagnosticMask(prefix,"supported",depth,h.width,h.height);
            report(csv,f.meta.frame,"supported",depth,regions);
        }
        removeDepthSpeckles(depth,h.width,h.height); const auto smoothed=smoothOcclusionDepth(depth,h.width,h.height);
        OcclusionCompletionContract contract;
        const auto planes=completeOcclusionPlanes(smoothed,f.left,h.width,h.height,nullptr,&contract);
        const auto legacy=fillOcclusionHoles(planes,h.width,h.height);
        OcclusionCompletionContract offContract;
        offContract.veto.assign(smoothed.size(),0);
        offContract.provenance.resize(smoothed.size());
        for(size_t i=0;i<smoothed.size();++i) offContract.provenance[i]=surfaceDepth(smoothed[i])?CompletionMeasured:CompletionUnknown;
        const auto off=fillOcclusionHoles(smoothed,smoothed,h.width,h.height,offContract,f.left);
        const auto input=fillOcclusionHoles(planes,smoothed,h.width,h.height,contract,f.left);
        const auto evidence=occlusionEvidence(input,depth,h.width,h.height);
        std::vector<float> confirmed;
        const std::vector<float> anchorAges(input.size(),0);
        if(f.meta.flags&B::CapturePose) confirmed=correctedHistory.update(input,h.width,h.height,h.focal,h.cx,h.cy,
            f.meta.flags&B::Aligned?&f.meta.transform:nullptr,f.meta.elapsed,&evidence,&evidence,&anchorAges);
        else {correctedHistory.reset();confirmed.assign(input.size(),0);}
        const auto meshIndices=occlusionSurfaceIndices(confirmed,h.width,h.height);
        const auto cellLifetimes=occlusionSurfaceCellLifetimes(confirmed,correctedHistory.supportLifetimes(),h.width,h.height,meshIndices);
        const float minLifetime=cellLifetimes.empty()?0:*std::min_element(cellLifetimes.begin(),cellLifetimes.end());
        const float maxLifetime=cellLifetimes.empty()?0:*std::max_element(cellLifetimes.begin(),cellLifetimes.end());
        const auto snapshotMesh=diagnosticLiveMeshIndices(meshIndices,cellLifetimes,0);
        const auto mesh75=diagnosticLiveMeshIndices(meshIndices,cellLifetimes,.075f);
        const auto mesh100=diagnosticLiveMeshIndices(meshIndices,cellLifetimes,.100f);
        if(save) {
            const std::pair<const char*,const std::vector<float>*> stages[]={{"speckle",&depth},{"smoothed",&smoothed},
                {"planes",&planes},{"legacy-pre-history",&legacy},{"completion-off",&off},{"pre-history",&input},{"confirmed",&confirmed}};
            for(const auto& stage:stages) {
                writeDiagnosticMask(prefix,stage.first,*stage.second,h.width,h.height);
                report(csv,f.meta.frame,stage.first,*stage.second,regions);
            }
            writeDiagnosticMeshIndices(prefix+"-support",meshIndices,h.width,h.height);
            writeDiagnosticMeshIndices(prefix,snapshotMesh,h.width,h.height);
            writeDiagnosticMeshIndices(prefix+"-display-age75ms",mesh75,h.width,h.height);
            writeDiagnosticMeshIndices(prefix+"-display-age100ms",mesh100,h.width,h.height);
            writeDiagnosticRaw(prefix+"-support-mesh-cell-lifetimes-f32.raw",cellLifetimes);
            const std::pair<const char*,const std::vector<size_t>*> variants[]={{"geometric",&meshIndices},
                {"snapshot_eligible",&snapshotMesh},{"simulated_display_75ms",&mesh75},{"simulated_display_100ms",&mesh100}};
            for(const auto& variant:variants) {
                const auto mask=occlusionSurfaceMask(*variant.second,h.width,h.height);
                for(const auto& r:regions) {
                    size_t covered=0;
                    for(int y=r.y0;y<r.y1;++y) for(int x=r.x0;x<r.x1;++x) covered+=mask[size_t(y)*h.width+x]!=0;
                    const size_t total=size_t(r.x1-r.x0)*(r.y1-r.y0);
                    meshRegions<<f.meta.frame<<','<<variant.first<<','<<r.name<<','<<total<<','
                        <<covered<<','<<double(total-covered)/total<<'\n';
                }
            }
            writeDiagnosticRaw(prefix+"-history-events-u16.raw",correctedHistory.events());
            writeDiagnosticRaw(prefix+"-support-lifetimes-f32.raw",correctedHistory.supportLifetimes());
            writeDiagnosticRaw(prefix+"-completion-veto-u8.raw",contract.veto);
            writeDiagnosticRaw(prefix+"-completion-provenance-u8.raw",contract.provenance);
            writeDiagnosticMask(prefix,"recorded-measured",f.measured,h.width,h.height);
            writeDiagnosticMask(prefix,"recorded-pre-history",f.input,h.width,h.height);
        }
        events<<f.meta.frame<<','<<f.meta.captureEndNs<<','<<f.meta.elapsed;
        for(uint16_t bit:eventBits) events<<','<<std::count_if(correctedHistory.events().begin(),correctedHistory.events().end(),[&](uint16_t v){return v&bit;});
        events<<','<<minLifetime<<','<<(minLifetime>0?f.meta.captureEndNs+uint64_t(double(minLifetime)*1e9):0)
            <<','<<maxLifetime<<','<<(maxLifetime>0?f.meta.captureEndNs+uint64_t(double(maxLifetime)*1e9):0)
            <<','<<occlusionSurfaceArea(meshIndices,h.width)<<','<<occlusionSurfaceArea(snapshotMesh,h.width)
            <<','<<occlusionSurfaceArea(mesh75,h.width)<<','<<occlusionSurfaceArea(mesh100,h.width)<<'\n';
        size_t mismatches=0;
        for(size_t i=0;i<depth.size();++i) mismatches+=depth[i]!=f.measured[i];
        counts<<f.meta.frame<<','<<B::phaseName(f.meta.phase)<<','<<supported<<','<<count(depth)<<','<<count(smoothed)
            <<','<<count(planes)<<','<<count(legacy)<<','<<count(input)<<','<<count(off)<<','<<count(confirmed)<<','<<count(f.input)<<','<<mismatches;
        if(save) {
            const auto reasons=[&](uint32_t bit){return std::count_if(diagnostic.filterRejection.begin(),diagnostic.filterRejection.end(),[&](uint32_t v){return v&bit;});};
            counts<<','<<reasons(StereoFilterLR)<<','<<reasons(StereoFilterRange)<<','<<std::count(diagnostic.supportRejection.begin(),diagnostic.supportRejection.end(),1u);
        } else counts<<",,,";
        counts<<'\n';
    }
    if(!csv||!counts||!events||!meshRegions) throw std::runtime_error("Output write failed");
    std::ofstream manifest(output+"/replay.txt");
    manifest<<"recording="<<argv[1]<<"\nshader_directory="<<argv[2]<<"\nframes="<<reader.frames
        <<"\npose_fallbacks="<<poseFallbacks<<"\nconfidence_config=2,2,3,200,250,250\n"
        <<"mesh=corrected confidence history, camera coverage with per-cell lifetime eligibility (not display-eye mask)\n"
        <<"export_frame_ids="<<argv[4]<<"\nregions_file="<<(argc==6?argv[5]:"none")<<"\n"
        <<"ground_truth="<<(regions.empty()?"none":"user supplied constant optical-Z rectangle regions")<<'\n';
    if(!manifest) throw std::runtime_error("Manifest write failed");
    std::cout<<"Replayed "<<reader.frames<<" frames; exported "<<selected.size()<<"; pose fallbacks="<<poseFallbacks<<'\n';
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
