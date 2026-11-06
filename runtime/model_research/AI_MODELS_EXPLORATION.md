# AI Models Exploration for Frame Spatial Runtime

This document outlines potential AI models and architectures that can be integrated into the Frame Spatial Runtime to replace or augment the existing pipeline (currently using 64-bit Census Transform and SGM Stereo Disparity Matching). The target hardware is the Valve Steam Frame (Deckard) with a Linux ARM64 environment, AMD Zen 4 CPU, and Qualcomm Adreno GPU.

## 1. Real-Time Stereo Depth Estimation (15-30+ fps)

To achieve real-time stereo depth estimation on edge hardware, consider these lightweight models:

*   **Fast-FoundationStereo:** An accelerated version of FoundationStereo that utilizes structured pruning, neural architecture search, and knowledge distillation. It is extremely fast and can achieve >10x speedup over the original model. Deployable via NVIDIA TAO Toolkit or ONNX.
*   **HITNET:** A highly popular architecture for real-time stereo matching. It is extremely fast and well-supported in ONNX and TFLite formats. Given its efficiency on mobile CPUs and edge devices, this is a very strong candidate to replace SGM disparity matching.
*   **MobileStereoNet:** Specifically designed for embedded devices and mobile robotics. It sacrifices a small amount of accuracy for very high throughput.

**Deployment Strategy:** Since the headset features a Qualcomm Adreno GPU, leveraging the **Qualcomm AI Hub** to convert models into **QNN DLC** or optimized **ONNX** formats will yield the highest performance (often utilizing the Hexagon NPU).

## 2. Point Cloud Generation

Point cloud generation is typically a direct mathematical back-projection from the generated depth map using the Kannala-Brandt intrinsics of the cameras. However, AI can aid in temporal consistency and noise reduction:

*   **NeuralRecon:** Instead of just back-projecting, NeuralRecon reconstructs local surfaces as sparse Truncated Signed Distance Function (TSDF) volumes from video sequences in real-time. It uses Gated Recurrent Units (GRUs) to maintain temporal coherence across frames, reducing the flickering often seen in frame-by-frame point clouds.

## 3. Real-Time Mesh Generation

Converting depth/point clouds into meshes in real-time can be computationally heavy. 

*   **Traditional Methods (Accelerated):** Fast implementations of Poisson Surface Reconstruction or Ball Pivoting are standard, but often produce noisy meshes if the input depth is imperfect.
*   **DeepMesh / Neural Approaches:** Newer AI architectures learn to generate topology directly from data (often using graph structures or tokenization). While some Generative Large Reconstruction Models (LRMs) are too heavy for edge real-time, models focusing on local surface reconstruction (like variations of NeuralRecon) can output meshes dynamically.
*   **EFM3D (Egocentric Voxel Lifting):** Another Meta open-source model designed for 3D object detection and surface reconstruction on edge/wearable recordings.

## 4. Running SceneScript

## 4. Running SceneScript

[Project Aria SceneScript](https://www.projectaria.com/scenescript/) is a novel approach that represents environments as a sequence of structured language commands to generate geometry (walls, doors, boxes, etc.).

Since this does **not need to be real-time**, you can run it as an asynchronous background worker or offline process:

*   **Availability:** The official code is open-source at [facebookresearch/scenescript](https://github.com/facebookresearch/scenescript).
*   **Models (Balanced Approach):** Meta provides pre-trained models trained on the Aria Synthetic Environments (ASE) dataset. Since latency is relaxed but we still want to avoid starving the headset's resources, using the **base or standard transformer models** (rather than the heaviest versions) is recommended. You can further **quantize these models (e.g., to INT8 or FP16)** to reduce memory footprint and prevent thermal throttling while running in the background.
*   **Input Requirements:** SceneScript expects semi-dense point clouds. The best workflow would be to have the user scan the room (building a point cloud over several seconds or minutes) and then dispatch that accumulated point cloud to the SceneScript model to generate the room layout asynchronously.

## 5. Running Segment Anything (SAM) in 3D

For 3D segmentation and object extraction, since this also **does not need to be real-time**, we can move away from aggressive mobile variants without resorting to the heaviest models.

*   **Reasonable Fidelity SAM 2 / SAM 3D:** Instead of using the heaviest ViT-Huge versions, the **ViT-Base (ViT-B) or ViT-Large (ViT-L)** variants of SAM 2 or [SAM3D](https://huggingface.co/tuandao-zenai/sam-3d-objects) offer an excellent middle ground. They provide a massive accuracy leap over MobileSAM but remain manageable.
*   **Quantization (INT8 / W8A16):** To run these efficiently as background tasks on the Valve Deckard (sharing memory between the Zen 4 CPU and Adreno GPU), applying INT8 or W8A16 quantization is highly recommended. This ensures the background segmentation doesn't monopolize the RAM or compute needed by the 80Hz VR render loop.
*   **Implementation:** You can capture keyframes of the user's environment, queue them up, and run the quantized SAM model in a low-priority background thread. Once the inference finishes, the 3D map metadata is retroactively updated.

## 6. Recommendations & Hardware Synergy

1.  **Embrace Qualcomm AI Hub:** The Valve Deckard runs on a mixed AMD/Qualcomm stack. For the Adreno GPU, compiling your ONNX models (like HITNET or MobileSAM) through the Qualcomm AI Engine Direct (QNN) SDK is critical to hit the 15-30+ fps target without melting the headset.
2.  **Monochrome Invariance:** Remember that the OG01A1B cameras are monochrome (grayscale). Any pre-trained AI model you use (especially for SAM or depth) must either be fine-tuned for grayscale images or robust enough to handle 1-channel inputs (e.g., duplicating the grayscale channel to RGB).
3.  **Unified Pipelines:** Instead of disparate models (Stereo -> Depth -> Point Cloud -> SAM), look into unified architectures like **SLAM3R** or **NeuralRecon** that handle spatial tracking and depth fusion in one optimized pass.
