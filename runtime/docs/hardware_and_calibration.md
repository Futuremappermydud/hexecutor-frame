# Hardware & Sensor Calibration

## 1. Physical Sensor Specifications

The Valve Steam Frame (prototype / Deckard hardware) utilizes a multi-sensor array managed by the internal `XRService` computer vision daemon.

### Sensor Overview (as used by this runtime)

* **Stereo RGB pair (spatial depth + HUD)**: Dual **Sony IMX616** via `arcimx616` (`0-0010` / `0-001a` → typically `video0` / `video3`).
  * **Resolution**: **2464 × 2464** NV12 (Qualcomm UBWC in `vrcompositor` DMA-BUFs).
  * **DMA-BUF size**: Exactly **9,564,160** bytes — this is what `DMACapture` auto-discovers.
  * **Baseline**: **≈ 63.96 mm** horizontal.
  * **Mounting**: Right sensor is **physically inverted 180°** relative to left → rectification uses `r_right = (-rx, -ry, 1)`.
  * **HUD**: Detiles Y + UV and shows RGB; stereo matching uses the Y plane after KB rectification.
* **Tracking monochrome pair**: Dual **OmniVision OG01A1B** global-shutter GREY sensors (**≈ 1056 × 1024**).
  * Used by SteamVR tracking / SLAM internals.
  * **Not** the 9.5 MB UBWC slots this project binds today.

> **Doc correction:** Older notes called the 2464×2464 DMA-BUFs “OG01A1B monochrome.” On current Frame software they are the **IMX616 stereo RGB** buffers. See [`progress_imx_monitor.md`](progress_imx_monitor.md).

---

## 2. Why Stereo Matching Still Uses Luma
1. **Census / SGM** operate on spatial intensity gradients, not chroma.
2. IMX616 NV12 provides a full-resolution Y plane after UBWC detile — that is the stereo signal.
3. UV is only needed for the RGB diagnostic panels (and future textured meshes).

---

## 3. Mathematical Lens Distortion Model

The lenses are ultra-wide fisheye optics modeled using the **Kannala-Brandt (KB)** equidistant fisheye projection model.

### Kannala-Brandt Equidistant Formulation
Given a 3D point $\mathbf{P} = (X, Y, Z)$ in camera coordinates:
1. Compute the radial distance and incident angle:
   $$r_{xy} = \sqrt{X^2 + Y^2}$$
   $$\theta = \text{atan2}(r_{xy}, Z)$$
2. Apply the polynomial distortion series:
   $$r_d(\theta) = \theta \cdot \left(1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8\right)$$
3. Project to distorted pixel coordinates $(u, v)$:
   $$u = c_x + f_x \cdot \frac{X}{r_{xy}} \cdot r_d(\theta)$$
   $$v = c_y + f_y \cdot \frac{Y}{r_{xy}} \cdot r_d(\theta)$$

### Factory Calibration Values
From the device calibration (`fppae620002b6/config.json` and `sensor-rig-calibration.json`):

#### Left Camera
| Parameter | Value | Description |
| :--- | :--- | :--- |
| $f_x$ | `1477.96` | Horizontal focal length (pixels) |
| $f_y$ | `1480.84` | Vertical focal length (pixels) |
| $c_x$ | `1086.97` | Principal point horizontal center |
| $c_y$ | `1237.34` | Principal point vertical center |
| $k_1$ | `-0.030051` | 1st order radial coefficient |
| $k_2$ | `-0.161829` | 2nd order radial coefficient |
| $k_3$ | ` 0.091782` | 3rd order radial coefficient |
| $k_4$ | `-0.012317` | 4th order radial coefficient |

#### Right Camera (Mounted 180° Inverted)
| Parameter | Value | Description |
| :--- | :--- | :--- |
| $f_x$ | `1479.22` | Horizontal focal length (pixels) |
| $f_y$ | `1481.90` | Vertical focal length (pixels) |
| $c_x$ | `1387.79` | Raw-image principal point horizontal center |
| $c_y$ | `1220.44` | Principal point vertical center |
| $k_1$ | `-0.032597` | 1st order radial coefficient |
| $k_2$ | `-0.152976` | 2nd order radial coefficient |
| $k_3$ | ` 0.082236` | 3rd order radial coefficient |
| $k_4$ | `-0.008963` | 4th order radial coefficient |

---

## 4. Extrinsics & Coordinate Reference Frames

### OpenVR Tracking Coordinate System (`TrackingUniverseStanding`)
* **$+X$**: Towards the user's **Right**.
* **$+Y$**: Upwards towards the **Ceiling** (Gravity is $-Y$).
* **$-Z$**: Forwards along the user's line of sight (**Negative Z is Forward**).

### Camera-to-Head Rig Calibration
Extracted from `sensors-from-head.json`:
* Left camera optical center relative to HMD origin (midpoint between displays):
  $$T_{\text{cam\_to\_head}} = \begin{bmatrix} -0.0528\text{ m} \\ -0.0186\text{ m} \\ -0.0368\text{ m} \end{bmatrix}$$
* The aligned right-from-left transform is read from
  `colorPassthroughCalibrationHistory.json` at runtime. Its translation has a
  norm of approximately `0.06390 m`; its quaternion includes the near-180°
  sensor parity rotation and roughly 21.5° of relative optical tilt.
* Rectification rotates that measured baseline onto the virtual camera's +X
  axis before horizontal disparity search.

### OpenXR Point Cloud Rig Transform

The world point cloud composes the rectified basis with XRService's aligned
right-from-left and frontCam0-from-right transforms, then this headset's
SteamVR `cv.cad_from_cal` and `head` axes. Its virtual forward ray points about
13 degrees downward in tracked head coordinates. Backprojection starts at the
left optical center; a stereo-midpoint translation and simple image-axis flip
are insufficient. See [OpenXR point cloud](openxr_point_cloud.md).
