// OpenXR viewer: 3D hand joints and bones over native passthrough. Capture/inference stay off the XR loop.
#include "calibration.hpp"
#include "depth_pipeline.hpp"
#include "dma_capture.hpp"
#include "hand_viewer_xr.hpp"
#include "hand_tracking/tracker.hpp"
#include "hand_tracking/viewer_geometry.hpp"
#include "hexecutor/broker.hpp"
#include "mercury/executor_networks.hpp"
#include <atomic>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

namespace {
using Clock=std::chrono::steady_clock;
volatile std::sig_atomic_t interrupted=0;
void interrupt(int) { interrupted=1; }
uint64_t nowNs() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }
struct Options {
    std::filesystem::path models, broker, mercury_dir;
    hexecutor::Artifacts artifacts;
    uint32_t fps=60, seconds=0, max_frames=0;
};
uint32_t number(const std::string& value,uint32_t maximum) {
    size_t count=0; const auto n=std::stoul(value,&count);
    if (count!=value.size() || n<1 || n>maximum) throw std::runtime_error("invalid numeric option: "+value);
    return uint32_t(n);
}
Options parse(int argc,char** argv) {
    Options result;
    for (int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        auto value=[&]() { if (++i>=argc) throw std::runtime_error("missing value for "+arg); return std::string(argv[i]); };
        if (arg=="--models") result.models=value();
        else if (arg=="--mercury-dir") result.mercury_dir=value();
        else if (arg=="--loader") result.artifacts.loader=value();
        else if (arg=="--engine") result.artifacts.engine=value();
        else if (arg=="--bootstrap") result.artifacts.bootstrap=value();
        else if (arg=="--chunk-probe") result.artifacts.chunk_probe=value();
        else if (arg=="--broker") result.broker=value();
        else if (arg=="--fps") result.fps=number(value(),60);
        else if (arg=="--seconds") result.seconds=number(value(),86400);
        else if (arg=="--max-frames") result.max_frames=number(value(),1000000);
        else if (arg=="--precise") result.artifacts.tensor_hmx=false;
        else if (arg=="--help" || arg=="-h") {
            std::cout << "frame_hand_viewer --models DIR --loader ELF --engine ELF --bootstrap ELF --chunk-probe ELF\n"
                         "frame_hand_viewer --mercury-dir DIR --loader ELF --engine ELF --bootstrap ELF --chunk-probe ELF\n"
                         "frame_hand_viewer --broker SOCKET   (hexecutord with graphs palm and landmark)\n"
                         "  [--fps 60] [--precise] [--seconds N] [--max-frames N]\n"
                         "Standalone OpenXR app: 3D hand joints and bones over alpha-blended passthrough.\n";
            std::exit(0);
        } else throw std::runtime_error("unknown option: "+arg);
    }
    if (!result.mercury_dir.empty() && (!result.models.empty() || !result.broker.empty()))
        throw std::runtime_error("Mercury requires a direct DSP session and cannot use MediaPipe models");
    if (result.broker.empty() && ((result.models.empty() && result.mercury_dir.empty()) || result.artifacts.loader.empty() ||
        result.artifacts.engine.empty() || result.artifacts.bootstrap.empty() ||
        result.artifacts.chunk_probe.empty()))
        throw std::runtime_error("explicit model and DSP paths required; see --help");
    return result;
}
struct Shared {
    std::atomic<bool> quit{false}, active{false}, failed{false};
    std::mutex mutex;
    std::shared_ptr<const HandScene> scene;
    uint64_t published_ns=0;
};
std::array<float,3> rotate(const std::array<float,4>& q,const std::array<float,3>& v) {
    const std::array<float,3> t{2*(q[1]*v[2]-q[2]*v[1]),2*(q[2]*v[0]-q[0]*v[2]),2*(q[0]*v[1]-q[1]*v[0])};
    return {v[0]+q[3]*t[0]+q[1]*t[2]-q[2]*t[1],v[1]+q[3]*t[1]+q[2]*t[0]-q[0]*t[2],v[2]+q[3]*t[2]+q[0]*t[1]-q[1]*t[0]};
}
void track(const Options& options,const std::vector<hexecutor::Program>& programs,const HandViewerXr& viewer,Shared& shared) {
    std::unique_ptr<hexecutor::Session> session;
    std::unique_ptr<hexecutor::broker::Client> client;
    std::vector<hexecutor::broker::Client::Stream> streams;
    std::unique_ptr<hand_tracking::Tracker> tracker;
    std::unique_ptr<mercury::ExecutorNetworks> mercury_networks;
    std::unique_ptr<mercury::Tracker> mercury_tracker;
    uint64_t mercury_detection_calls=0;
    auto close=[&] {
        tracker.reset();
        mercury_tracker.reset();
        mercury_networks.reset();
        if (session) { session->close(); session.reset(); std::cerr << "[Hands] DSP closed and restored\n"; }
        // With a broker the daemon owns the DSP and releases it when idle.
        if (client) { streams.clear(); client.reset(); std::cerr << "[Hands] broker streams closed\n"; }
    };
    try {
        FrameSpatial::DMACapture camera;
        FrameSpatial::DepthPipeline rectifier;
        uint64_t last_hash=0, sequence=0;
        bool have_hash=false;
        pid_t compositor=0;
        auto last_new=Clock::now(), last_pid_check=Clock::now();
        const auto period=std::chrono::nanoseconds(1000000000/options.fps);
        while (!shared.quit.load()) {
            const auto start=Clock::now();
            if (!shared.active.load()) {
                close();
                if (camera.isReady()) camera.release();
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
                continue;
            }
            if (camera.isReady() && start-last_pid_check>std::chrono::seconds(1)) {
                last_pid_check=start;
                if (FrameSpatial::DMACapture::findCompositorPid()!=compositor) {
                    close(); camera.release(); have_hash=false;
                }
            }
            if (!camera.isReady()) {
                compositor=FrameSpatial::DMACapture::findCompositorPid();
                if (compositor<=0 || !camera.initialize(compositor)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(250)); continue;
                }
                last_new=start; last_pid_check=start;
            }
            const uint64_t begin=nowNs();
            const bool mercury_mode=!options.mercury_dir.empty();
            const bool captured=camera.captureFrame(!mercury_mode, !mercury_mode);
            const uint64_t end=nowNs();
            const uint64_t hash=camera.leftHash() ^ (mercury_mode ? (camera.rightHash()*0x9e3779b97f4a7c15ull) : 0);
            const uint64_t captured_ns=begin+(end-begin)/2;
            if (!captured || (mercury_mode ? !camera.getRightBuffer() : !camera.getLeftUV()) ||
                (have_hash && hash==last_hash)) {
                if (start-last_new>std::chrono::milliseconds(500)) close();
                if (!captured || start-last_new>std::chrono::seconds(2)) {
                    // Preserve hash: reimporting a frozen buffer is not a new frame.
                    camera.release(); std::this_thread::sleep_for(std::chrono::milliseconds(250));
                } else std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            last_hash=hash; have_hash=true; last_new=start;
            std::vector<uint8_t> rgb;
            if (mercury_mode) rectifier.rectify(camera.getLeftBuffer(),camera.getRightBuffer());
            else rgb=rectifier.rectifyLeftRgb(camera.getLeftBuffer(),camera.getLeftUV());
            hand_tracking::Frame result;
            mercury::Result mercury_result;
            const auto inference_start=Clock::now();
            if (mercury_mode && !mercury_tracker) {
                session=std::make_unique<hexecutor::Session>(programs,options.artifacts);
                mercury_networks=std::make_unique<mercury::ExecutorNetworks>(
                    [&](size_t index,const std::vector<hexecutor::Buffer>& inputs) {
                        auto outputs=session->run(inputs,index);
                        if (index==0) ++mercury_detection_calls;
                        if (index==0 && std::getenv("FRAME_MERCURY_DEBUG") &&
                            (mercury_detection_calls<=4 || mercury_detection_calls%100==0) &&
                            !outputs.empty() && outputs[0].size()==2*sizeof(float)) {
                            float confidence[2];
                            std::memcpy(confidence,outputs[0].data(),sizeof confidence);
                            std::cerr << "[Mercury] detection=" << mercury_detection_calls
                                      << " confidence=" << confidence[0] << ',' << confidence[1] << '\n';
                        }
                        if (index==0 && mercury_detection_calls==1) {
                            if (const char* dump=std::getenv("FRAME_HAND_DUMP")) {
                                const auto path=std::string(dump)+"_detection_input.f32";
                                if (FILE* f=std::fopen(path.c_str(),"wb")) {
                                    std::fwrite(inputs[0].data,1,inputs[0].bytes,f);
                                    std::fclose(f);
                                }
                            }
                        }
                        return outputs;
                    });
                const mercury::PinholeCamera pinhole{FrameSpatial::RECT_FOCAL,FrameSpatial::RECT_FOCAL,
                    FrameSpatial::RECT_CX,FrameSpatial::RECT_CY,FrameSpatial::RECT_WIDTH,FrameSpatial::RECT_HEIGHT};
                mercury::StereoCalibration calibration;
                calibration.left=calibration.right=pinhole;
                calibration.left_in_right.position.x=-rectifier.rectifiedBaseline();
                mercury_tracker=std::make_unique<mercury::Tracker>(calibration,*mercury_networks);
            } else if (!mercury_mode && !tracker && !options.broker.empty()) {
                client=std::make_unique<hexecutor::broker::Client>(options.broker);
                for (const char* graph : {"palm","landmark"}) {
                    hexecutor::broker::StreamSpec spec;
                    spec.graph=graph;
                    spec.cls=hexecutor::broker::Class::interactive;
                    spec.deadline_ms=1000.0/options.fps*2;
                    spec.latest_only=true;
                    streams.push_back(client->open(spec));
                }
                tracker=std::make_unique<hand_tracking::Tracker>(
                    [&](size_t index,const std::vector<float>& x) {
                        return client->run(streams.at(index),{{x.data(),x.size()*4}});
                    });
            } else if (!mercury_mode && !tracker) {
                session=std::make_unique<hexecutor::Session>(programs,options.artifacts);
                tracker=std::make_unique<hand_tracking::Tracker>(*session);
            }
            if (mercury_mode) {
                mercury_result=mercury_tracker->process(
                    {rectifier.getRectifiedLeft().data(),FrameSpatial::RECT_WIDTH,FrameSpatial::RECT_HEIGHT,
                     size_t(FrameSpatial::RECT_WIDTH)},
                    {rectifier.getRectifiedRight().data(),FrameSpatial::RECT_WIDTH,FrameSpatial::RECT_HEIGHT,
                     size_t(FrameSpatial::RECT_WIDTH)},captured_ns);
            } else {
                result=tracker->process({rgb.data(),FrameSpatial::RECT_WIDTH,FrameSpatial::RECT_HEIGHT,
                    size_t(FrameSpatial::RECT_WIDTH)*3},captured_ns);
            }
            const double tracking_ms=std::chrono::duration<double,std::milli>(Clock::now()-inference_start).count();
            // Joints are fitted in the rectified camera, moved into the head
            // frame by calibration, then world-locked with the head pose at
            // capture time so head motion during inference does not drag them.
            auto scene=std::make_shared<HandScene>();
            HeadPose head;
            const bool world_locked=scene->local_space=viewer.headPoseAt(captured_ns,head);
            if (mercury_mode) {
                for (size_t h=0;h<mercury_result.hands.size();++h) {
                    const auto& hand=mercury_result.hands[h];
                    if (!hand.active) continue;
                    HandSkeleton skeleton;
                    skeleton.right=(h==1);
                    skeleton.joint_count=mercury::kJointCount;
                    for (int i=0;i<mercury::kJointCount;++i) {
                        const auto& joint=hand.joints[i];
                        const auto& c=joint.pose.position;
                        // Mercury uses +Y up, -Z forward; rectifier uses +Y down, +Z forward.
                        auto p=rectifier.rectifiedCameraPointInHead({c.x,-c.y,-c.z});
                        if (scene->local_space) {
                            p=rotate(head.orientation,p);
                            for (int k=0;k<3;++k) p[k]+=head.position[k];
                        }
                        skeleton.joints[i]=p;
                        skeleton.radii[i]=joint.radius;
                    }
                    scene->hands.push_back(skeleton);
                }
            }
            for (const auto& hand : result.hands) {
                std::array<hand_tracking::Point,21> joints;
                if (!hand_tracking::viewer::camera_joints(hand,FrameSpatial::RECT_WIDTH,FrameSpatial::RECT_HEIGHT,
                        FrameSpatial::RECT_FOCAL,FrameSpatial::RECT_CX,FrameSpatial::RECT_CY,joints)) continue;
                HandSkeleton skeleton;
                skeleton.right=hand.right_probability>=.5f;
                for (int i=0;i<21;++i) {
                    auto p=rectifier.rectifiedCameraPointInHead({joints[i].x,joints[i].y,joints[i].z});
                    if (scene->local_space) {
                        p=rotate(head.orientation,p);
                        for (int k=0;k<3;++k) p[k]+=head.position[k];
                    }
                    skeleton.joints[i]=p;
                }
                scene->hands.push_back(skeleton);
                // FRAME_HAND_DEBUG=1: wrist in image, rectified camera, head and world, every 10th frame.
                static const bool debug=std::getenv("FRAME_HAND_DEBUG")!=nullptr;
                if (debug && sequence%10==0 && scene->hands.size()==1) {
                    const auto c=joints[0];
                    const auto h=rectifier.rectifiedCameraPointInHead({c.x,c.y,c.z});
                    const auto& w=skeleton.joints[0];
                    std::cerr << "[HandDebug] seq=" << sequence << " wrist_px=" << hand.landmarks[0].x*FrameSpatial::RECT_WIDTH
                              << ',' << hand.landmarks[0].y*FrameSpatial::RECT_HEIGHT
                              << " cam=" << c.x << ',' << c.y << ',' << c.z
                              << " head=" << h[0] << ',' << h[1] << ',' << h[2]
                              << " world=" << w[0] << ',' << w[1] << ',' << w[2]
                              << " head_pos=" << head.position[0] << ',' << head.position[1] << ',' << head.position[2]
                              << " head_q=" << head.orientation[0] << ',' << head.orientation[1] << ','
                              << head.orientation[2] << ',' << head.orientation[3]
                              << " right=" << skeleton.right << '\n';
                }
            }
            const size_t visible_hands=scene->hands.size();
            if (mercury_mode && visible_hands && (sequence==199 || sequence==399 || sequence==599)) {
                if (const char* dump=std::getenv("FRAME_HAND_DUMP")) {
                    const auto prefix=std::string(dump)+"_frame"+std::to_string(sequence+1);
                    if (FILE* f=std::fopen((prefix+"_left.pgm").c_str(),"wb")) {
                        std::fprintf(f,"P5\n%d %d\n255\n",FrameSpatial::RECT_WIDTH,FrameSpatial::RECT_HEIGHT);
                        const auto& pixels=rectifier.getRectifiedLeft();
                        std::fwrite(pixels.data(),1,pixels.size(),f);
                        std::fclose(f);
                    }
                    if (FILE* f=std::fopen((prefix+"_joints.txt").c_str(),"w")) {
                        for (size_t h=0;h<mercury_result.hands.size();++h) {
                            const auto& hand=mercury_result.hands[h];
                            if (!hand.active) continue;
                            std::fprintf(f,"hand %zu reprojection_error %g\n",h,hand.reprojection_error);
                            for (int j=0;j<mercury::kJointCount;++j) {
                                const auto& p=hand.joints[j].pose.position;
                                std::fprintf(f,"%d %.8g %.8g %.8g\n",j,p.x,p.y,p.z);
                            }
                        }
                        std::fclose(f);
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lock(shared.mutex);
                shared.scene=std::move(scene); shared.published_ns=nowNs();
            }
            // FRAME_HAND_DUMP=/path/prefix: save the tracker's rectified RGB at frame 150.
            if (mercury_mode) if (const char* dump=std::getenv("FRAME_HAND_DUMP"); dump && sequence==149) {
                for (int side=0;side<2;++side) {
                    const auto& pixels=side ? rectifier.getRectifiedRight() : rectifier.getRectifiedLeft();
                    const auto path=std::string(dump)+(side ? "_right.pgm" : "_left.pgm");
                    if (FILE* f=std::fopen(path.c_str(),"wb")) {
                        std::fprintf(f,"P5\n%d %d\n255\n",FrameSpatial::RECT_WIDTH,FrameSpatial::RECT_HEIGHT);
                        std::fwrite(pixels.data(),1,pixels.size(),f);
                        std::fclose(f);
                        std::cerr << "[Mercury] saved " << path << '\n';
                    }
                }
            }
            if (!mercury_mode) if (const char* dump=std::getenv("FRAME_HAND_DUMP"); dump && sequence==149) {
                if (FILE* f=std::fopen((std::string(dump)+"_rect.ppm").c_str(),"wb")) {
                    std::fprintf(f,"P6\n%d %d\n255\n",FrameSpatial::RECT_WIDTH,FrameSpatial::RECT_HEIGHT);
                    std::fwrite(rgb.data(),1,rgb.size(),f); std::fclose(f);
                    std::cerr << "[HandDebug] saved " << dump << "_rect.ppm head_q=" << head.orientation[0] << ','
                              << head.orientation[1] << ',' << head.orientation[2] << ',' << head.orientation[3] << '\n';
                }
            }
            if (++sequence==1 || sequence%100==0)
                std::cerr << "[Hands] frame=" << sequence << " hands=" << visible_hands
                          << (mercury_mode ? " mercury=1" : " mercury=0") << " tracking_ms=" << tracking_ms
                          << " capture_to_publish_ms=" << (nowNs()-captured_ns)*1e-6
                          << " world_locked=" << world_locked
                          << (mercury_mode ? " detections=" + std::to_string(mercury_result.detections_run) +
                              " keypoints=" + std::to_string(mercury_result.keypoints_run) +
                              " tracked=" + std::to_string(int(mercury_result.hands[0].tracked)+int(mercury_result.hands[1].tracked))
                                           : "") << '\n';
            if (options.max_frames && sequence>=options.max_frames) { shared.quit.store(true); break; }
            std::this_thread::sleep_until(start+period);
        }
        close();
    } catch (const std::exception& error) {
        std::cerr << "[Hands] " << error.what() << '\n';
        try { close(); } catch (const std::exception& cleanup) { std::cerr << "[Hands] cleanup: " << cleanup.what() << '\n'; }
        shared.failed.store(true); shared.quit.store(true);
    }
}
}
int main(int argc,char** argv) {
    try {
        std::cout << std::unitbuf;
        std::signal(SIGINT,interrupt); std::signal(SIGTERM,interrupt);
        const auto options=parse(argc,argv);
        std::vector<hexecutor::Program> programs;
        if (!options.mercury_dir.empty()) {
            for (const char* name : {"mercury_detection","mercury_keypoint"}) {
                const auto prefix=options.mercury_dir/name;
                programs.push_back(hexecutor::load_qnn16_program(prefix.string()+".q16p",prefix.string()+".q16c"));
            }
        } else if (options.broker.empty()) programs=hand_tracking::load_models(options.models);
        HandViewerXr viewer;
        viewer.initialize();
        Shared shared;
        std::thread worker(track,std::cref(options),std::cref(programs),std::cref(viewer),std::ref(shared));
        struct Join { Shared& shared; std::thread& thread; ~Join() { shared.quit.store(true); if (thread.joinable()) thread.join(); } } join{shared,worker};
        const HandScene waiting;
        const auto started=Clock::now();
        while (!interrupted && !shared.quit.load() && viewer.pollEvents()) {
            if (options.seconds && Clock::now()-started>=std::chrono::seconds(options.seconds)) break;
            shared.active.store(viewer.active());
            std::shared_ptr<const HandScene> scene;
            {
                std::lock_guard<std::mutex> lock(shared.mutex);
                if (nowNs()-shared.published_ns<500000000ull) scene=shared.scene;
            }
            viewer.render(scene?*scene:waiting);
        }
        shared.quit.store(true); worker.join();
        return shared.failed.load()?1:0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
