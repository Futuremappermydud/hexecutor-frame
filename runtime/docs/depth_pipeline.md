# Stereo Depth Estimation Pipeline

## 1. Overview of the Pipeline

The live default converts raw, distorted $2464 \times 2464$ fisheye buffers to
metric depth through these stages:

1. **Full-extrinsic Kannala-Brandt rectification LUT**
2. **7×7 census descriptors on Vulkan**
3. **96-disparity bidirectional matching with uniqueness checks**
4. **Left/right consistency and local-support rejection**
5. **Metric conversion with `Z = fB/d`**

The native uniqueness test compares the best disparity with hypotheses outside
its immediate `d±1` basin. Neighboring samples of one subpixel minimum are not
independent competing matches. Left/right consistency and local-support
rejection remain mandatory; blank and periodically repeating texture remain
invalid. This shader still uses local cross aggregation, not full SGM.

The OpenXR occlusion worker additionally performs bounded image-guided plane
completion before temporal surface construction. This is separate from the
measured stereo depth supplied to ordinary point-cloud fusion. See
[occlusion completion](occlusion_completion.md).

The CPU Census/WTA algorithm documented below remains the fallback used when
Vulkan initialization or execution fails.

---

## 2. Precomputed Rectification Lookup Tables (LUTs)

In a traditional stereo system, epipolar lines are curved if cameras are not rectified. To enable 1D horizontal disparity search:
* We define a virtual pinhole camera with resolution $616 \times 616$, focal length $f = 380\text{ px}$, and principal point $(c_x, c_y) = (308, 308)$.
* For every pixel $(x, y)$ in the rectified virtual frame, the normalized optical ray is:
  $$r_x = \frac{x - c_x}{f}, \quad r_y = \frac{y - c_y}{f}, \quad r_z = 1.0$$

### Ray Transformation

The runtime reads the newest right-from-left translation and quaternion from
`colorPassthroughCalibrationHistory.json`. It constructs a common basis whose
+X axis follows the baseline and whose +Z axis bisects the optical axes. Each
virtual ray is transformed into both physical camera frames before KB4
projection. This includes the right sensor's near-180° parity rotation plus
the measured relative tilt; a simple sign flip is not sufficient.

Both rays are projected through the factory Kannala-Brandt distortion model at initialization, generating two lookup tables (`m_lutL` and `m_lutR`) mapping each rectified coordinate $(x, y)$ to floating-point raw sensor coordinates $(u, v)$.

During runtime, bilinear interpolation samples the raw $2464 \times 2464$ image in **~5 milliseconds** using OpenMP:
```cpp
#pragma omp parallel for collapse(2)
for (int y = 0; y < RECT_HEIGHT; y++) {
    for (int x = 0; x < RECT_WIDTH; x++) {
        int idx = y * RECT_WIDTH + x;
        m_rectL[idx] = sampleBilinear(leftRaw, m_lutL[idx].x, m_lutL[idx].y);
        m_rectR[idx] = sampleBilinear(rightRaw, m_lutR[idx].x, m_lutR[idx].y);
    }
}
```

---

## 3. 64-bit Census Transform

Stereo matching based solely on raw intensity (SAD/SSD) fails under auto-exposure differences between stereo cameras. The **Census Transform** encodes local spatial structure rather than absolute luminance.

### Window & Descriptor
* We utilize a $7 \times 7$ window centered at $(x, y)$, which contains 48 neighbors (excluding the center pixel).
* For each neighbor $(x + dx, y + dy)$, we compare its intensity to the center $I(x, y)$:
  $$\xi(p, q) = \begin{cases} 1 & \text{if } I(q) > I(p) \\ 0 & \text{otherwise} \end{cases}$$
* The 48 bits fit cleanly within a single unsigned 64-bit integer (`uint64_t`).

```cpp
uint64_t code = 0;
int bit = 0;
for (int dy = -3; dy <= 3; dy++) {
    for (int dx = -3; dx <= 3; dx++) {
        if (dx == 0 && dy == 0) continue;
        if (rectImg[(y + dy) * W + (x + dx)] > centerVal) code |= (1ULL << bit);
        bit++;
    }
}
```

---

## 4. Disparity Matching via Hamming Distance

Because the rectified stereo pair is epipolar-aligned along horizontal scanlines, for a pixel at $(x, y)$ in the Left image, the matching candidate in the Right image must lie at $(x - d, y)$, where $d$ is disparity ($1 \le d < 64$).

* **Matching Cost**: The Hamming distance between the two 64-bit descriptors:
  $$C(x, y, d) = \text{popcount}\left(\text{Census}_L(x, y) \oplus \text{Census}_R(x - d, y)\right)$$
* On modern ARM64 (NEON), `__builtin_popcountll` executes in a single clock cycle.
* **Winner-Take-All (WTA) for dense display**:
  * The disparity $d^*$ minimizing $C(x, y, d)$ is selected.
  * Valid cells with $d > 1$ and cost $< 28$ write $Z = fB / d$ into a dense map for the linear depth HUD.
  * Aggressive uniqueness/speckle rejection is **not** applied on the HUD map (it previously turned the panel into sparse colored dots). Reintroduce for 3D meshing quality later.

---

## 5. Metric 3D Reprojection

Given optimal disparity $d^*$:
1. **Metric Depth ($Z$)**:
   $$Z = \frac{f \cdot B}{d^*} = \frac{380\text{ px} \times 0.06396\text{ m}}{d^*} = \frac{24.3048}{d^*}\text{ meters}$$
2. **Rectified Cartesian Coordinates** (right, down, forward):
   $$\mathbf{P}_{rect} = ((x-c_x)Z/f, (y-c_y)Z/f, Z)^T$$
3. **Rig Transform to Head Space**:
   $$\mathbf{P}_{head} = R_{head\leftarrow left} R_{left\leftarrow rect} \mathbf{P}_{rect} + \mathbf{t}_{head\leftarrow left}$$
   `DepthPipeline::rectifiedPointInHead()` uses the same virtual basis as the
   sampling LUT and the aligned XRService transforms into the tracked head
   frame. Depth starts at the left optical center. A simple image-axis flip
   omits the optical pitch (approximately 13 degrees down on this headset).
4. **World Tracking Transform**:
   $$\mathbf{P}_{world} = \mathbf{T}_{world\_from\_hmd} \mathbf{P}_{head}$$
   The OpenXR cloud uses the head pose associated with DMA capture time.

The native Vulkan support pass requires three agreeing neighbors in a 3×3
window. The OpenXR worker additionally rejects eight-connected depth patches
smaller than 32 pixels before world fusion. It requires three distinct stereo
pairs and four nearby stable voxels before drawing a voxel. See
[OpenXR point cloud](openxr_point_cloud.md) for thresholds and regression checks.

The OpenXR worker also regularizes supported measured samples with local affine
inverse-depth fits. This reduces depth jitter on flat/sloped surfaces while
keeping holes and discontinuities. Existing world voxels are reprojected into
current depth to reject repeatedly contradicted foreground geometry. Capture
pose lookup supports `FRAME_CAMERA_LATENCY_MS` for exposure-age tuning; the
sensor timestamp is still unavailable through DMA-BUF capture.

Processed rectified luma, stereo depth, completed depth, provenance, poses, and
stage timings can be recorded by the bounded [occlusion benchmark](occlusion_benchmark.md).
Recording I/O runs outside the depth and render loops; history replay compares
legacy and experimental confidence rules on identical inputs.
