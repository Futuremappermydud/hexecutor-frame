#include "overlay_renderer.hpp"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <sstream>

namespace FrameSpatial {

namespace {
// OpenVR SetOverlayRaw stores packed RGBA as 0xAABBGGRR on little-endian.
constexpr uint32_t rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    return (uint32_t)r | ((uint32_t)g << 8) | ((uint32_t)b << 16) | ((uint32_t)a << 24);
}

std::vector<std::string> splitExtensions(const std::vector<char>& buffer) {
    std::vector<std::string> result;
    if (buffer.empty()) return result;
    std::istringstream stream(buffer.data());
    std::string extension;
    while (stream >> extension) result.push_back(extension);
    return result;
}
} // namespace

OverlayRenderer::OverlayRenderer()
    : m_vrSys(nullptr)
    , m_overlay(nullptr)
    , m_ovHandle(vr::k_ulOverlayHandleInvalid)
    , m_eyeWidth(640)
    , m_eyeHeight(720)
    , m_totalWidth(960)
    , m_totalHeight(400)
    , m_vkInstance(VK_NULL_HANDLE)
    , m_vkPhysicalDevice(VK_NULL_HANDLE)
    , m_vkDevice(VK_NULL_HANDLE)
    , m_vkQueue(VK_NULL_HANDLE)
    , m_vkQueueFamily(0)
    , m_vkCommandPool(VK_NULL_HANDLE)
    , m_vkCommandBuffer(VK_NULL_HANDLE)
    , m_vulkanData{}
    , m_textureIdx(0)
    , m_uploadSeq(0)
    , m_textureFails(0)
    , m_lastUpload()
    , m_fontInitialized(false)
    , m_isInitialized(false) {}

OverlayRenderer::~OverlayRenderer() {
    shutdown();
    // OpenVR requires submitted Vulkan resources to outlive VR_Shutdown(). main()
    // shuts OpenVR down before this stack object is destroyed.
    releaseVulkanTexture();
}

void OverlayRenderer::initFont() {
    if (m_fontInitialized) return;
    std::memset(m_font5x7, 0, sizeof(m_font5x7));
    auto setG = [this](char c, uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3, uint8_t b4) {
        m_font5x7[(int)c][0] = b0; m_font5x7[(int)c][1] = b1; m_font5x7[(int)c][2] = b2;
        m_font5x7[(int)c][3] = b3; m_font5x7[(int)c][4] = b4;
    };
    setG(' ', 0x00, 0x00, 0x00, 0x00, 0x00);
    setG('-', 0x08, 0x08, 0x08, 0x08, 0x08);
    setG('+', 0x08, 0x08, 0x3E, 0x08, 0x08);
    setG('.', 0x00, 0x60, 0x60, 0x00, 0x00);
    setG(',', 0x00, 0x80, 0x60, 0x00, 0x00);
    setG(':', 0x00, 0x36, 0x36, 0x00, 0x00);
    setG('/', 0x20, 0x10, 0x08, 0x04, 0x02);
    setG('=', 0x14, 0x14, 0x14, 0x14, 0x14);
    setG('%', 0x26, 0x16, 0x08, 0x64, 0x62);
    setG('[', 0x00, 0x7F, 0x41, 0x41, 0x00);
    setG(']', 0x00, 0x41, 0x41, 0x7F, 0x00);
    setG('|', 0x00, 0x00, 0x7F, 0x00, 0x00);
    setG('0', 0x3E, 0x51, 0x49, 0x45, 0x3E); setG('1', 0x00, 0x42, 0x7F, 0x40, 0x00);
    setG('2', 0x42, 0x61, 0x51, 0x49, 0x46); setG('3', 0x21, 0x41, 0x45, 0x4B, 0x31);
    setG('4', 0x18, 0x14, 0x12, 0x7F, 0x10); setG('5', 0x27, 0x45, 0x45, 0x45, 0x39);
    setG('6', 0x3C, 0x4A, 0x49, 0x49, 0x30); setG('7', 0x01, 0x71, 0x09, 0x05, 0x03);
    setG('8', 0x36, 0x49, 0x49, 0x49, 0x36); setG('9', 0x06, 0x49, 0x49, 0x29, 0x1E);
    setG('A', 0x7C, 0x12, 0x11, 0x12, 0x7C); setG('B', 0x7F, 0x49, 0x49, 0x49, 0x36);
    setG('C', 0x3E, 0x41, 0x41, 0x41, 0x22); setG('D', 0x7F, 0x41, 0x41, 0x22, 0x1C);
    setG('E', 0x7F, 0x49, 0x49, 0x49, 0x41); setG('F', 0x7F, 0x09, 0x09, 0x09, 0x01);
    setG('G', 0x3E, 0x41, 0x49, 0x49, 0x7A); setG('H', 0x7F, 0x08, 0x08, 0x08, 0x7F);
    setG('I', 0x00, 0x41, 0x7F, 0x41, 0x00); setG('J', 0x20, 0x40, 0x41, 0x3F, 0x01);
    setG('K', 0x7F, 0x08, 0x14, 0x22, 0x41); setG('L', 0x7F, 0x40, 0x40, 0x40, 0x40);
    setG('M', 0x7F, 0x02, 0x0C, 0x02, 0x7F); setG('N', 0x7F, 0x04, 0x08, 0x10, 0x7F);
    setG('O', 0x3E, 0x41, 0x41, 0x41, 0x3E); setG('P', 0x7F, 0x09, 0x09, 0x09, 0x06);
    setG('Q', 0x3E, 0x41, 0x51, 0x21, 0x5E); setG('R', 0x7F, 0x09, 0x19, 0x29, 0x46);
    setG('S', 0x46, 0x49, 0x49, 0x49, 0x31); setG('T', 0x01, 0x01, 0x7F, 0x01, 0x01);
    setG('U', 0x3F, 0x40, 0x40, 0x40, 0x3F); setG('V', 0x07, 0x38, 0x40, 0x38, 0x07);
    setG('W', 0x7F, 0x20, 0x18, 0x20, 0x7F); setG('X', 0x63, 0x14, 0x08, 0x14, 0x63);
    setG('Y', 0x07, 0x08, 0x70, 0x08, 0x07); setG('Z', 0x61, 0x51, 0x49, 0x45, 0x43);
    m_fontInitialized = true;
}

void OverlayRenderer::fillRect(int x, int y, int w, int h, uint32_t color) {
    int x1 = std::min(x + w, m_totalWidth);
    int y1 = std::min(y + h, m_totalHeight);
    x = std::max(x, 0);
    y = std::max(y, 0);
    for (int py = y; py < y1; py++) {
        uint32_t* row = m_pixels.data() + py * m_totalWidth;
        for (int px = x; px < x1; px++) row[px] = color;
    }
}

void OverlayRenderer::drawRect(int x, int y, int w, int h, uint32_t color) {
    if (w <= 0 || h <= 0) return;
    drawLine(x, y, x + w - 1, y, color);
    drawLine(x, y + h - 1, x + w - 1, y + h - 1, color);
    drawLine(x, y, x, y + h - 1, color);
    drawLine(x + w - 1, y, x + w - 1, y + h - 1, color);
}

void OverlayRenderer::drawLine(int x0, int y0, int x1, int y1, uint32_t color) {
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    while (true) {
        if (x0 >= 0 && x0 < m_totalWidth && y0 >= 0 && y0 < m_totalHeight) {
            m_pixels[y0 * m_totalWidth + x0] = color;
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void OverlayRenderer::drawText(int startX, int startY, const std::string& str, uint32_t color, int scale, int clipX0, int clipX1) {
    if (scale < 1) scale = 1;
    if (clipX1 < 0) clipX1 = m_totalWidth;
    int curX = startX;
    for (char c : str) {
        if (c >= 'a' && c <= 'z') c = (char)(c - 32);
        if (c < 0) c = ' ';
        const uint8_t* glyph = m_font5x7[(int)c];
        for (int col = 0; col < 5; col++) {
            uint8_t bits = glyph[col];
            for (int row = 0; row < 7; row++) {
                if (bits & (1 << row)) {
                    for (int sy = 0; sy < scale; sy++) {
                        int py = startY + row * scale + sy;
                        if (py < 0 || py >= m_totalHeight) continue;
                        for (int sx = 0; sx < scale; sx++) {
                            int px = curX + col * scale + sx;
                            if (px >= clipX0 && px < clipX1 && px < m_totalWidth) {
                                m_pixels[py * m_totalWidth + px] = color;
                            }
                        }
                    }
                }
            }
        }
        curX += 6 * scale;
    }
}

void OverlayRenderer::splat(int x, int y, uint32_t color, int radius) {
    for (int dy = -radius; dy <= radius; dy++) {
        int py = y + dy;
        if (py < 0 || py >= m_totalHeight) continue;
        for (int dx = -radius; dx <= radius; dx++) {
            int px = x + dx;
            if (px < 0 || px >= m_totalWidth) continue;
            m_pixels[py * m_totalWidth + px] = color;
        }
    }
}

void OverlayRenderer::blitGray(const uint8_t* src, int srcW, int srcH, int dstX, int dstY, int dstW, int dstH) {
    if (!src || srcW <= 0 || srcH <= 0) return;
    for (int y = 0; y < dstH; y++) {
        int sy = y * srcH / dstH;
        int py = dstY + y;
        if (py < 0 || py >= m_totalHeight) continue;
        for (int x = 0; x < dstW; x++) {
            int sx = x * srcW / dstW;
            int px = dstX + x;
            if (px < 0 || px >= m_totalWidth) continue;
            uint8_t g = src[sy * srcW + sx];
            m_pixels[py * m_totalWidth + px] = rgba(g, g, g, 255);
        }
    }
}

void OverlayRenderer::blitRgba(const uint32_t* src, int srcW, int srcH, int dstX, int dstY, int dstW, int dstH) {
    if (!src || srcW <= 0 || srcH <= 0) return;
    for (int y = 0; y < dstH; y++) {
        int sy = y * srcH / dstH;
        int py = dstY + y;
        if (py < 0 || py >= m_totalHeight) continue;
        for (int x = 0; x < dstW; x++) {
            int sx = x * srcW / dstW;
            int px = dstX + x;
            if (px < 0 || px >= m_totalWidth) continue;
            m_pixels[py * m_totalWidth + px] = src[sy * srcW + sx] | 0xFF000000u;
        }
    }
}

uint32_t OverlayRenderer::depthColor(float zMeters) {
    if (zMeters < 0.25f || zMeters > 5.0f) return rgba(8, 8, 12, 255);
    float t = (zMeters - 0.25f) / 4.75f;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    uint8_t v = (uint8_t)((1.0f - t) * 255.0f); // linear metric: near=white, far=black
    return rgba(v, v, v, 255);
}

void OverlayRenderer::blitDepth(const float* src, int srcW, int srcH, int dstX, int dstY, int dstW, int dstH) {
    if (!src || srcW <= 0 || srcH <= 0) return;
    for (int y = 0; y < dstH; y++) {
        int sy = y * srcH / dstH;
        int py = dstY + y;
        if (py < 0 || py >= m_totalHeight) continue;
        for (int x = 0; x < dstW; x++) {
            int sx = x * srcW / dstW;
            int px = dstX + x;
            if (px < 0 || px >= m_totalWidth) continue;
            m_pixels[py * m_totalWidth + px] = depthColor(src[sy * srcW + sx]);
        }
    }
}

bool OverlayRenderer::projectWorld(float wx, float wy, float wz,
                                  const float R[3][3], const float hmdPos[3],
                                  bool leftEye, int& sx, int& sy) {
    float dx = wx - hmdPos[0];
    float dy = wy - hmdPos[1];
    float dz = wz - hmdPos[2];
    float hx = R[0][0] * dx + R[0][1] * dy + R[0][2] * dz;
    float hy = R[1][0] * dx + R[1][1] * dy + R[1][2] * dz;
    float hz = R[2][0] * dx + R[2][1] * dy + R[2][2] * dz;
    if (hz > -0.12f) return false;
    float invZ = -1.0f / hz;
    if (leftEye) {
        float hxL = hx - m_eyeToHeadL.m[0][3];
        sx = (int)(m_projCenterXL + hxL * m_projScaleXL * invZ);
        sy = (int)(m_projCenterYL - hy * m_projScaleYL * invZ);
        return sx >= 0 && sx < m_eyeWidth && sy >= 0 && sy < m_eyeHeight;
    }
    float hxR = hx - m_eyeToHeadR.m[0][3];
    sx = (int)(m_projCenterXR + hxR * m_projScaleXR * invZ) + m_eyeWidth;
    sy = (int)(m_projCenterYR - hy * m_projScaleYR * invZ);
    return sx >= m_eyeWidth && sx < m_totalWidth && sy >= 0 && sy < m_eyeHeight;
}

void OverlayRenderer::drawWorldLine(float ax, float ay, float az,
                                   float bx, float by, float bz,
                                   const float R[3][3], const float hmdPos[3],
                                   uint32_t color) {
    int x0, y0, x1, y1;
    if (projectWorld(ax, ay, az, R, hmdPos, true, x0, y0) &&
        projectWorld(bx, by, bz, R, hmdPos, true, x1, y1)) {
        drawLine(x0, y0, x1, y1, color);
    }
    if (projectWorld(ax, ay, az, R, hmdPos, false, x0, y0) &&
        projectWorld(bx, by, bz, R, hmdPos, false, x1, y1)) {
        drawLine(x0, y0, x1, y1, color);
    }
}

bool OverlayRenderer::initialize(vr::IVRSystem* vrSys, int eyeWidth, int eyeHeight, const char* overlayKey) {
    m_vrSys = vrSys;
    m_eyeWidth = eyeWidth;
    m_eyeHeight = eyeHeight;
    // Single opaque 2D HUD (not SBS). Both eyes see the same stereo-RGB + depth card.
    m_totalWidth = eyeWidth;
    m_totalHeight = eyeHeight;
    m_pixels.assign((size_t)m_totalWidth * m_totalHeight, rgba(12, 14, 20, 255));
    m_textureIdx = 0;
    m_uploadSeq = 0;
    m_lastUpload = std::chrono::steady_clock::time_point{};

    initFont();

    m_overlay = vr::VROverlay();
    if (!m_overlay) return false;

    auto destroyKey = [&](const char* key) {
        vr::VROverlayHandle_t existing = vr::k_ulOverlayHandleInvalid;
        if (m_overlay->FindOverlay(key, &existing) == vr::VROverlayError_None) {
            m_overlay->DestroyOverlay(existing);
        }
    };
    // A separate research preview must not replace an existing runtime HUD.
    if (std::strcmp(overlayKey, "frame.spatial.imx_monitor.vulkan") == 0) {
        destroyKey("frame.spatial.room_mesh");
        destroyKey("frame.spatial.imx_monitor");
        destroyKey("frame.spatial.imx_monitor.vulkan");
    }

    vr::EVROverlayError ovErr = m_overlay->CreateOverlay(overlayKey, "IMX Stereo Monitor", &m_ovHandle);
    if (ovErr != vr::VROverlayError_None) {
        std::cerr << "[OverlayRenderer] CreateOverlay failed: " << ovErr << std::endl;
        return false;
    }

    m_overlay->SetOverlayFlag(m_ovHandle, vr::VROverlayFlags_SideBySide_Parallel, false);
    m_overlay->SetOverlayFlag(m_ovHandle, vr::VROverlayFlags_SortWithNonSceneOverlays, true);
    m_overlay->SetOverlaySortOrder(m_ovHandle, 100);

    vr::HmdMatrix34_t hmdTransform = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, -0.06f,
        0.0f, 0.0f, 1.0f, -0.72f
    };
    m_hmdTransform = hmdTransform;
    m_overlay->SetOverlayTransformTrackedDeviceRelative(m_ovHandle, vr::k_unTrackedDeviceIndex_Hmd, &m_hmdTransform);
    m_overlay->SetOverlayWidthInMeters(m_ovHandle, 0.92f);
    m_overlay->SetOverlayAlpha(m_ovHandle, 1.0f);
    m_overlay->SetOverlayColor(m_ovHandle, 1.0f, 1.0f, 1.0f);
    m_overlay->SetOverlayTextureColorSpace(m_ovHandle, vr::ColorSpace_Gamma);
    m_overlay->ShowOverlay(m_ovHandle);

    if (!initializeVulkanTexture()) {
        std::cerr << "[OverlayRenderer] Vulkan overlay texture initialization failed" << std::endl;
        shutdown();
        releaseVulkanTexture();
        return false;
    }

    m_projL = m_vrSys->GetProjectionMatrix(vr::Eye_Left, 0.1f, 30.0f);
    m_projR = m_vrSys->GetProjectionMatrix(vr::Eye_Right, 0.1f, 30.0f);

    m_projScaleXL = m_projL.m[0][0] * (m_eyeWidth / 2.0f);
    m_projScaleYL = m_projL.m[1][1] * (m_eyeHeight / 2.0f);
    m_projCenterXL = (m_eyeWidth / 2.0f) * (1.0f - m_projL.m[0][2]);
    m_projCenterYL = (m_eyeHeight / 2.0f) * (1.0f + m_projL.m[1][2]);

    m_eyeToHeadL = m_vrSys->GetEyeToHeadTransform(vr::Eye_Left);
    m_eyeToHeadR = m_vrSys->GetEyeToHeadTransform(vr::Eye_Right);

    m_isInitialized = true;
    std::cout << "[OverlayRenderer] Vulkan IMX stereo HUD " << m_totalWidth << "x" << m_totalHeight << std::endl;
    return true;
}

void OverlayRenderer::shutdown() {
    if (m_overlay && m_ovHandle != vr::k_ulOverlayHandleInvalid) {
        m_overlay->ClearOverlayTexture(m_ovHandle);
        m_overlay->HideOverlay(m_ovHandle);
        m_overlay->DestroyOverlay(m_ovHandle);
        m_ovHandle = vr::k_ulOverlayHandleInvalid;
    }
    m_isInitialized = false;
}

HmdPose OverlayRenderer::getHmdPose() {
    HmdPose pose = {0};
    if (!m_vrSys) return pose;

    vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
    m_vrSys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0.0f, poses, vr::k_unMaxTrackedDeviceCount);

    if (poses[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid) {
        pose.isValid = true;
        std::memcpy(pose.matrix, poses[vr::k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking.m, sizeof(pose.matrix));
    }
    return pose;
}

void OverlayRenderer::render(const RenderSnapshot& snap, float liveFps) {
    if (!m_isInitialized) return;

    // MNN's compute submission and this HUD texture share Turnip's graphics
    // queue. Waiting for vkQueueWaitIdle during inference would stall pose and
    // event polling. OpenVR retains the previously submitted texture, so defer
    // the next upload until the learned depth dispatch has completed.
    if (snap.gpuBusy) return;

    // Overlay pixels only need rebuilding when a texture will actually be submitted.
    // Keeping this gate ahead of the 768x444 CPU composition preserves the pose/event loop rate.
    auto now = std::chrono::steady_clock::now();
    if (now - m_lastUpload < std::chrono::milliseconds(33)) return;
    m_lastUpload = now;

    std::fill(m_pixels.begin(), m_pixels.end(), rgba(10, 12, 18, 255));

    const int margin = 8;
    const int hudH = 42;
    const int gap = 6;
    const int panel = (m_totalWidth - margin * 2 - gap * 2) / 3;
    const int panelH = m_totalHeight - hudH - margin * 2 - 14;
    const int px0 = margin;
    const int px1 = px0 + panel + gap;
    const int px2 = px1 + panel + gap;
    const int py0 = hudH + margin;
    const uint32_t cyan = rgba(0, 230, 255, 255);
    const uint32_t green = rgba(80, 255, 120, 255);
    const uint32_t yellow = rgba(255, 210, 60, 255);
    const uint32_t white = rgba(240, 240, 245, 255);

    auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    bool blink = ((nowMs / 400) % 2) == 0;

    char line1[96];
    char line2[96];
    std::snprintf(line1, sizeof(line1), "IMX616 STEREO  F%u  FPS %0.0f", snap.frameIndex, liveFps);
    std::snprintf(line2, sizeof(line2), "CAM %0.0fHZ  %s  L%d/R%d",
                  snap.captureHz, snap.status.c_str(), snap.leftSlot, snap.rightSlot);

    uint32_t statusCol = snap.camerasReady ? green : yellow;
    drawText(margin, 6, line1, cyan, 1);
    drawText(margin, 20, line2, statusCol, 1);
    if (blink && snap.camerasReady) {
        drawText(m_totalWidth - 40, 6, "LIVE", green, 1);
    } else if (!snap.camerasReady) {
        drawText(m_totalWidth - 40, 6, "WAIT", yellow, 1);
    }

    bool havePreview = snap.previewW > 0 && snap.previewH > 0 &&
                       snap.previewL.size() >= (size_t)(snap.previewW * snap.previewH) &&
                       snap.previewR.size() >= (size_t)(snap.previewW * snap.previewH);
    bool haveDepth = snap.rectW > 0 && snap.rectH > 0 &&
                     snap.depth.size() >= (size_t)(snap.rectW * snap.rectH);
    if (havePreview) {
        blitRgba(snap.previewL.data(), snap.previewW, snap.previewH, px0, py0, panel, panelH);
        blitRgba(snap.previewR.data(), snap.previewW, snap.previewH, px2, py0, panel, panelH);
    } else {
        fillRect(px0, py0, panel, panelH, rgba(22, 22, 30, 255));
        fillRect(px2, py0, panel, panelH, rgba(22, 22, 30, 255));
    }
    if (haveDepth) blitDepth(snap.depth.data(), snap.rectW, snap.rectH, px1, py0, panel, panelH);
    else fillRect(px1, py0, panel, panelH, rgba(22, 22, 30, 255));

    drawRect(px0, py0, panel, panelH, white);
    drawRect(px1, py0, panel, panelH, white);
    drawRect(px2, py0, panel, panelH, white);
    drawText(px0 + 4, py0 + 4, "L IMX616", white, 1);
    drawText(px1 + 4, py0 + 4, "DEPTH 0.25-5M", white, 1);
    drawText(px2 + 4, py0 + 4, "R IMX616", white, 1);

    int tickW = 24 + (int)(m_uploadSeq % (uint32_t)std::max(8, panel));
    fillRect(px1, py0 + panelH - 6, tickW, 4, rgba(255, 80, 40, 255));

    m_overlay->ShowOverlay(m_ovHandle);

    uploadVulkanTexture();
}

bool OverlayRenderer::renderComparison(const std::vector<uint8_t>& image,
        const std::vector<float>& baseline, const std::vector<float>& experiment,
        int width, int height, const std::string& title, const std::string& status) {
    if (!m_isInitialized || width <= 0 || height <= 0) return false;
    const size_t n = size_t(width) * height;
    if (image.size() != n || baseline.size() != n || experiment.size() != n) return false;
    const auto now = std::chrono::steady_clock::now();
    if (now - m_lastUpload < std::chrono::milliseconds(100)) return false;
    m_lastUpload = now;
    std::fill(m_pixels.begin(), m_pixels.end(), rgba(10, 12, 18));
    const uint32_t white = rgba(240, 240, 245), cyan = rgba(0, 230, 255);
    drawText(8, 8, title, cyan, 2);
    drawText(8, 30, status, rgba(255, 210, 60));
    const int panel = (m_totalWidth - 28) / 3;
    const int panelH = std::min(m_totalHeight - 110, panel);
    const int panelW = std::min(panel, panelH * width / height);
    for (int k = 0; k < 3; ++k) {
        const int x = 8 + k * (panel + 6) + (panel - panelW) / 2;
        drawText(8 + k * (panel + 6), 48, k == 0 ? "RECTIFIED LEFT" : k == 1 ? "BASELINE" : "GUARDED EXPERIMENT", white);
        if (k == 0) blitGray(image.data(), width, height, x, 64, panelW, panelH);
        else {
            const auto& depth = k == 1 ? baseline : experiment;
            for (int yy = 0; yy < panelH; ++yy) for (int xx = 0; xx < panelW; ++xx) {
                const float z = depth[size_t(yy * height / panelH) * width + xx * width / panelW];
                m_pixels[size_t(64 + yy) * m_totalWidth + x + xx] =
                    std::isfinite(z) && z >= .25f && z <= 5.f ? depthColor(z) : rgba(24, 40, 80);
            }
        }
        drawRect(x, 64, panelW, panelH, white);
    }
    drawText(8, m_totalHeight - 36, "SAME PAIR / 0.25-5M / WHITE = NEAR / BLACK = FAR / BLUE = NO DEPTH", white);
    drawText(8, m_totalHeight - 20, "RESEARCH PREVIEW / GUARDED RESULTS FAILED QUALITY GATES", rgba(255, 210, 60));
    fillRect(m_totalWidth - 48, m_totalHeight - 20, 8 + int(m_uploadSeq % 32), 5, rgba(255, 80, 40));
    m_overlay->ShowOverlay(m_ovHandle);
    return uploadVulkanTexture();
}

uint32_t OverlayRenderer::findVulkanMemoryType(uint32_t bits, VkMemoryPropertyFlags flags) const {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(m_vkPhysicalDevice, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags) return i;
    }
    return UINT32_MAX;
}

bool OverlayRenderer::initializeVulkanTexture() {
    vr::IVRCompositor* compositor = vr::VRCompositor();
    if (!compositor) return false;
    uint32_t instanceBytes = compositor->GetVulkanInstanceExtensionsRequired(nullptr, 0);
    std::vector<char> instanceBuffer(instanceBytes);
    if (instanceBytes) compositor->GetVulkanInstanceExtensionsRequired(instanceBuffer.data(), instanceBytes);
    std::vector<std::string> instanceNames = splitExtensions(instanceBuffer);
    std::vector<const char*> instanceExtensions;
    for (const auto& name : instanceNames) instanceExtensions.push_back(name.c_str());

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "frame-overlay";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = (uint32_t)instanceExtensions.size();
    ici.ppEnabledExtensionNames = instanceExtensions.data();
    if (vkCreateInstance(&ici, nullptr, &m_vkInstance) != VK_SUCCESS) return false;

    uint64_t outputDevice = 0;
    m_vrSys->GetOutputDevice(&outputDevice, vr::TextureType_Vulkan, (VkInstance_T*)m_vkInstance);
    m_vkPhysicalDevice = (VkPhysicalDevice)outputDevice;
    if (!m_vkPhysicalDevice) return false;

    uint32_t deviceBytes = compositor->GetVulkanDeviceExtensionsRequired(
        (VkPhysicalDevice_T*)m_vkPhysicalDevice, nullptr, 0);
    std::vector<char> deviceBuffer(deviceBytes);
    if (deviceBytes) compositor->GetVulkanDeviceExtensionsRequired(
        (VkPhysicalDevice_T*)m_vkPhysicalDevice, deviceBuffer.data(), deviceBytes);
    std::vector<std::string> deviceNames = splitExtensions(deviceBuffer);
    std::vector<const char*> deviceExtensions;
    for (const auto& name : deviceNames) deviceExtensions.push_back(name.c_str());

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_vkPhysicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_vkPhysicalDevice, &familyCount, families.data());
    bool foundFamily = false;
    for (uint32_t i = 0; i < familyCount; ++i) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { m_vkQueueFamily = i; foundFamily = true; break; }
    }
    if (!foundFamily) return false;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = m_vkQueueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)deviceExtensions.size();
    dci.ppEnabledExtensionNames = deviceExtensions.data();
    if (vkCreateDevice(m_vkPhysicalDevice, &dci, nullptr, &m_vkDevice) != VK_SUCCESS) return false;
    vkGetDeviceQueue(m_vkDevice, m_vkQueueFamily, 0, &m_vkQueue);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = m_vkQueueFamily;
    if (vkCreateCommandPool(m_vkDevice, &pci, nullptr, &m_vkCommandPool) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = m_vkCommandPool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(m_vkDevice, &cai, &m_vkCommandBuffer) != VK_SUCCESS) return false;

    const VkDeviceSize byteSize = m_pixels.size() * sizeof(uint32_t);
    for (auto& texture : m_textures) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = VK_FORMAT_R8G8B8A8_UNORM;
        ii.extent = {(uint32_t)m_totalWidth, (uint32_t)m_totalHeight, 1};
        ii.mipLevels = ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(m_vkDevice, &ii, nullptr, &texture.image) != VK_SUCCESS) return false;
        VkMemoryRequirements ir{};
        vkGetImageMemoryRequirements(m_vkDevice, texture.image, &ir);
        uint32_t imageType = findVulkanMemoryType(ir.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (imageType == UINT32_MAX) return false;
        VkMemoryAllocateInfo ia{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ia.allocationSize = ir.size;
        ia.memoryTypeIndex = imageType;
        if (vkAllocateMemory(m_vkDevice, &ia, nullptr, &texture.imageMemory) != VK_SUCCESS ||
            vkBindImageMemory(m_vkDevice, texture.image, texture.imageMemory, 0) != VK_SUCCESS) return false;

        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = byteSize;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(m_vkDevice, &bi, nullptr, &texture.staging) != VK_SUCCESS) return false;
        VkMemoryRequirements br{};
        vkGetBufferMemoryRequirements(m_vkDevice, texture.staging, &br);
        uint32_t stagingType = findVulkanMemoryType(br.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stagingType == UINT32_MAX) return false;
        VkMemoryAllocateInfo ba{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ba.allocationSize = br.size;
        ba.memoryTypeIndex = stagingType;
        if (vkAllocateMemory(m_vkDevice, &ba, nullptr, &texture.stagingMemory) != VK_SUCCESS ||
            vkBindBufferMemory(m_vkDevice, texture.staging, texture.stagingMemory, 0) != VK_SUCCESS ||
            vkMapMemory(m_vkDevice, texture.stagingMemory, 0, byteSize, 0, &texture.mapped) != VK_SUCCESS) return false;
    }
    return true;
}

bool OverlayRenderer::uploadVulkanTexture() {
    OverlayTexture& texture = m_textures[m_textureIdx];
    std::memcpy(texture.mapped, m_pixels.data(), m_pixels.size() * sizeof(uint32_t));
    vkResetCommandBuffer(m_vkCommandBuffer, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(m_vkCommandBuffer, &begin) != VK_SUCCESS) return false;
    VkImageMemoryBarrier toCopy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toCopy.srcAccessMask = texture.initialized ? VK_ACCESS_SHADER_READ_BIT : 0;
    toCopy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toCopy.oldLayout = texture.initialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    toCopy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toCopy.srcQueueFamilyIndex = toCopy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toCopy.image = texture.image;
    toCopy.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toCopy.subresourceRange.levelCount = toCopy.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(m_vkCommandBuffer, texture.initialized ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toCopy);
    VkBufferImageCopy copy{};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = {(uint32_t)m_totalWidth, (uint32_t)m_totalHeight, 1};
    vkCmdCopyBufferToImage(m_vkCommandBuffer, texture.staging, texture.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    VkImageMemoryBarrier readable = toCopy;
    readable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    readable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    readable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    readable.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(m_vkCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &readable);
    if (vkEndCommandBuffer(m_vkCommandBuffer) != VK_SUCCESS) return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &m_vkCommandBuffer;
    if (vkQueueSubmit(m_vkQueue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS || vkQueueWaitIdle(m_vkQueue) != VK_SUCCESS) return false;
    texture.initialized = true;

    m_vulkanData.m_nImage = (uint64_t)texture.image;
    m_vulkanData.m_pDevice = (VkDevice_T*)m_vkDevice;
    m_vulkanData.m_pPhysicalDevice = (VkPhysicalDevice_T*)m_vkPhysicalDevice;
    m_vulkanData.m_pInstance = (VkInstance_T*)m_vkInstance;
    m_vulkanData.m_pQueue = (VkQueue_T*)m_vkQueue;
    m_vulkanData.m_nQueueFamilyIndex = m_vkQueueFamily;
    m_vulkanData.m_nWidth = (uint32_t)m_totalWidth;
    m_vulkanData.m_nHeight = (uint32_t)m_totalHeight;
    m_vulkanData.m_nFormat = VK_FORMAT_R8G8B8A8_UNORM;
    m_vulkanData.m_nSampleCount = 1;
    vr::Texture_t openVrTexture{&m_vulkanData, vr::TextureType_Vulkan, vr::ColorSpace_Gamma};
    vr::EVROverlayError err = m_overlay->SetOverlayTexture(m_ovHandle, &openVrTexture);
    if (err != vr::VROverlayError_None) {
        ++m_textureFails;
        if (m_textureFails == 1 || (m_textureFails % 30) == 0)
            std::cerr << "[OverlayRenderer] SetOverlayTexture error " << (int)err << std::endl;
        return false;
    }
    m_textureFails = 0;
    if ((m_uploadSeq % 150u) == 0) std::cout << "[OverlayRenderer] texture ok seq=" << m_uploadSeq << std::endl;
    ++m_uploadSeq;
    m_textureIdx = (m_textureIdx + 1) % kTextureCount;
    return true;
}

void OverlayRenderer::releaseVulkanTexture() {
    if (m_vkDevice) vkDeviceWaitIdle(m_vkDevice);
    for (auto& texture : m_textures) {
        if (texture.mapped) vkUnmapMemory(m_vkDevice, texture.stagingMemory);
        if (texture.staging) vkDestroyBuffer(m_vkDevice, texture.staging, nullptr);
        if (texture.stagingMemory) vkFreeMemory(m_vkDevice, texture.stagingMemory, nullptr);
        if (texture.image) vkDestroyImage(m_vkDevice, texture.image, nullptr);
        if (texture.imageMemory) vkFreeMemory(m_vkDevice, texture.imageMemory, nullptr);
        texture = {};
    }
    if (m_vkDevice && m_vkCommandPool) vkDestroyCommandPool(m_vkDevice, m_vkCommandPool, nullptr);
    if (m_vkDevice) vkDestroyDevice(m_vkDevice, nullptr);
    if (m_vkInstance) vkDestroyInstance(m_vkInstance, nullptr);
    m_vkInstance = VK_NULL_HANDLE;
    m_vkPhysicalDevice = VK_NULL_HANDLE;
    m_vkDevice = VK_NULL_HANDLE;
    m_vkQueue = VK_NULL_HANDLE;
    m_vkCommandPool = VK_NULL_HANDLE;
    m_vkCommandBuffer = VK_NULL_HANDLE;
}

} // namespace FrameSpatial
