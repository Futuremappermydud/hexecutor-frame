#include "timed_stereo_capture.hpp"
#include "v4l2_capture_clock.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <vulkan/vulkan.h>

namespace FrameSpatial {
namespace {
constexpr uint64_t kQcomCompressed = 0x500000000000001ULL; // UBWC DRM modifier
constexpr int kStripRows = 16;
constexpr int64_t kMatchWindowNs = 16000000;   // write start must follow EOF by < 16 ms
constexpr int64_t kIdleNs = 6000000;           // slot complete after 6 ms without change
void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string("TimedStereoCapture: ") + what + " failed (" + std::to_string(r) + ")");
}
uint64_t hashBytes(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i + 8 <= n; i += 8) { uint64_t v; std::memcpy(&v, p + i, 8); h = (h ^ v) * 1099511628211ULL; }
    return h;
}
} // namespace

struct TimedStereoCapture::Vk {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem{};
    PFN_vkGetMemoryFdPropertiesKHR getFdProps = nullptr;
    struct Image { VkImage image = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; };
    struct Host { VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; uint8_t* mapped = nullptr; };
    std::vector<Image> images;
    Host strips, y[2], uv[2];

    uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags need) const {
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & want) == want) return i;
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & need) == need) return i;
        throw std::runtime_error("TimedStereoCapture: no memory type");
    }
    Host host(VkDeviceSize size) {
        Host b;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size; info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check(vkCreateBuffer(device, &info, nullptr, &b.buffer), "vkCreateBuffer");
        VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(device, b.buffer, &req);
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        check(vkAllocateMemory(device, &alloc, nullptr, &b.memory), "vkAllocateMemory");
        check(vkBindBufferMemory(device, b.buffer, b.memory, 0), "vkBindBufferMemory");
        check(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&b.mapped)), "vkMapMemory");
        return b;
    }
    void invalidate(const Host& b) const {
        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; r.memory = b.memory; r.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(device, 1, &r);
    }
    void destroy(Host& b) {
        if (b.mapped) vkUnmapMemory(device, b.memory);
        if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
        if (b.memory) vkFreeMemory(device, b.memory, nullptr);
        b = {};
    }
    void createDevice() {
        const char* iexts[] = {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME};
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "frame-timed-stereo-capture"; app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app; ici.enabledExtensionCount = 2; ici.ppEnabledExtensionNames = iexts;
        check(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance");
        uint32_t count = 0; vkEnumeratePhysicalDevices(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> gpus(count); vkEnumeratePhysicalDevices(instance, &count, gpus.data());
        for (auto g : gpus) { VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(g, &p); if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) { phys = g; break; } }
        if (!phys) throw std::runtime_error("TimedStereoCapture: no Vulkan GPU");
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
        std::vector<VkQueueFamilyProperties> fams(count); vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, fams.data());
        for (uint32_t i = 0; i < count; ++i) if (fams[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { family = i; break; }
        const float priority = 1.f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = family; qci.queueCount = 1; qci.pQueuePriorities = &priority;
        const char* dexts[] = {VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                               VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
                               VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, VK_KHR_BIND_MEMORY_2_EXTENSION_NAME};
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.enabledExtensionCount = 6; dci.ppEnabledExtensionNames = dexts;
        check(vkCreateDevice(phys, &dci, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, family, 0, &queue);
        vkGetPhysicalDeviceMemoryProperties(phys, &mem);
        getFdProps = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryFdPropertiesKHR"));
        if (!getFdProps) throw std::runtime_error("TimedStereoCapture: vkGetMemoryFdPropertiesKHR missing");
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pci.queueFamilyIndex = family;
        check(vkCreateCommandPool(device, &pci, nullptr, &pool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device, &cai, &cmd), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(device, &fci, nullptr, &fence), "vkCreateFence");
    }
    void import(int fd, int w, int h) {
        Image img;
        const int importFd = dup(fd);
        if (importFd < 0) throw std::runtime_error("TimedStereoCapture: dup failed");
        VkMemoryFdPropertiesKHR fdProps{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
        if (getFdProps(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, importFd, &fdProps) != VK_SUCCESS) {
            close(importFd); throw std::runtime_error("TimedStereoCapture: fd properties failed");
        }
        uint64_t modifier = kQcomCompressed;
        VkImageDrmFormatModifierListCreateInfoEXT mod{VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT};
        mod.drmFormatModifierCount = 1; mod.pDrmFormatModifiers = &modifier;
        VkExternalMemoryImageCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
        ext.pNext = &mod; ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.pNext = &ext; info.imageType = VK_IMAGE_TYPE_2D; info.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
        info.extent = {uint32_t(w), uint32_t(h), 1}; info.mipLevels = 1; info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT; info.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
        info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT; info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &info, nullptr, &img.image) != VK_SUCCESS) { close(importFd); throw std::runtime_error("TimedStereoCapture: vkCreateImage failed"); }
        VkImageMemoryRequirementsInfo2 ri{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2}; ri.image = img.image;
        VkMemoryDedicatedRequirements dr{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
        VkMemoryRequirements2 req{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &dr};
        vkGetImageMemoryRequirements2(device, &ri, &req);
        uint32_t bits = req.memoryRequirements.memoryTypeBits & fdProps.memoryTypeBits;
        if (!bits) bits = fdProps.memoryTypeBits;
        VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO}; ded.image = img.image;
        VkImportMemoryFdInfoKHR imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
        imp.pNext = &ded; imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT; imp.fd = importFd;
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.pNext = &imp; alloc.allocationSize = req.memoryRequirements.size ? req.memoryRequirements.size : 9564160;
        alloc.memoryTypeIndex = memoryType(bits, 0, 0);
        if (vkAllocateMemory(device, &alloc, nullptr, &img.memory) != VK_SUCCESS) {
            vkDestroyImage(device, img.image, nullptr); close(importFd); throw std::runtime_error("TimedStereoCapture: DMA-BUF import failed");
        }
        images.push_back(img);
        check(vkBindImageMemory(device, img.image, img.memory, 0), "vkBindImageMemory");
    }
    void barrier(VkImage image, bool toTransfer) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = toTransfer ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.newLayout = toTransfer ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = toTransfer ? VK_QUEUE_FAMILY_EXTERNAL : family;
        b.dstQueueFamilyIndex = toTransfer ? family : VK_QUEUE_FAMILY_EXTERNAL;
        b.srcAccessMask = toTransfer ? 0 : VK_ACCESS_TRANSFER_READ_BIT;
        b.dstAccessMask = toTransfer ? VK_ACCESS_TRANSFER_READ_BIT : 0;
        b.image = image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT;
        b.subresourceRange.levelCount = 1; b.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd, toTransfer ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
                             toTransfer ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    }
    void begin() {
        check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cmd, &info), "vkBeginCommandBuffer");
    }
    void submitAndWait() {
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        VkSubmitInfo s{VK_STRUCTURE_TYPE_SUBMIT_INFO}; s.commandBufferCount = 1; s.pCommandBuffers = &cmd;
        check(vkResetFences(device, 1, &fence), "vkResetFences");
        check(vkQueueSubmit(queue, 1, &s, fence), "vkQueueSubmit");
        check(vkWaitForFences(device, 1, &fence, VK_TRUE, 1000000000ULL), "vkWaitForFences");
    }
    void copy(VkImage image, VkImageAspectFlagBits plane, int y0, int w, int h, VkBuffer dst, VkDeviceSize offset) {
        VkBufferImageCopy r{};
        r.bufferOffset = offset; r.bufferRowLength = uint32_t(w); r.bufferImageHeight = uint32_t(h);
        r.imageSubresource.aspectMask = plane; r.imageSubresource.layerCount = 1;
        r.imageOffset = {0, y0, 0}; r.imageExtent = {uint32_t(w), uint32_t(h), 1};
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, 1, &r);
    }
    ~Vk() {
        if (device) {
            vkDeviceWaitIdle(device);
            for (auto& i : images) { vkDestroyImage(device, i.image, nullptr); vkFreeMemory(device, i.memory, nullptr); }
            destroy(strips); for (int i = 0; i < 2; ++i) { destroy(y[i]); destroy(uv[i]); }
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (pool) vkDestroyCommandPool(device, pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

// One eye: its compositor ring slots and its kernel clock.
struct TimedStereoCapture::Ring {
    struct Slot {
        int image = -1;              // index into Vk::images
        uint64_t top = 0, centre = 0; // latest strip hashes
        bool complete = false;        // a full camera frame is in the slot
        uint32_t sequence = 0;        // matched V4L2 sequence (0 = unmatched)
        int64_t eofNs = 0, writeStartNs = 0;
        uint64_t doneTop = 0, doneCentre = 0; // strip hashes when the write completed
    };
    std::unique_ptr<V4l2CaptureClock> clock;
    std::vector<Slot> slots;
    int pending = -1;
    int64_t pendingLastChangeNs = 0;
    bool primed = false;
    uint64_t frames = 0, unmatched = 0;
};

TimedStereoCapture::TimedStereoCapture(const std::vector<int>& leftFds, const std::vector<int>& rightFds, int width, int height)
    : width_(width), height_(height), vk_(new Vk), left_(new Ring), right_(new Ring) {
    if (leftFds.empty() || rightFds.empty()) throw std::runtime_error("TimedStereoCapture: empty ring");
    left_->clock = std::make_unique<V4l2CaptureClock>("/dev/video0");
    right_->clock = std::make_unique<V4l2CaptureClock>("/dev/video3");
    vk_->createDevice();
    for (int fd : leftFds) { left_->slots.push_back({}); left_->slots.back().image = int(vk_->images.size()); vk_->import(fd, width, height); }
    for (int fd : rightFds) { right_->slots.push_back({}); right_->slots.back().image = int(vk_->images.size()); vk_->import(fd, width, height); }
    vk_->strips = vk_->host(VkDeviceSize(width) * kStripRows * 2 * vk_->images.size());
    for (int e = 0; e < 2; ++e) {
        vk_->y[e] = vk_->host(VkDeviceSize(width) * height);
        vk_->uv[e] = vk_->host(VkDeviceSize(width) * (height / 2));
    }
    leftY_.resize(size_t(width) * height); rightY_.resize(leftY_.size());
    leftUV_.resize(size_t(width) * (height / 2)); rightUV_.resize(leftUV_.size());
    std::cout << "[TimedStereoCapture] " << leftFds.size() << "+" << rightFds.size() << " ring slots; clocks "
              << left_->clock->description() << ", " << right_->clock->description() << std::endl;
    thread_ = std::thread([this] { run(); });
}

TimedStereoCapture::~TimedStereoCapture() {
    { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void TimedStereoCapture::run() {
    const size_t stripBytes = size_t(width_) * kStripRows;
    const int centreY = height_ / 2 - kStripRows / 2;
    const int total = int(vk_->images.size());
    // Strip layout: image i -> top at (2i)*stripBytes, centre at (2i+1)*stripBytes.
    const auto recordStrips = [&](const std::vector<int>& which) {
        for (int i : which) {
            VkImage img = vk_->images[i].image;
            vk_->barrier(img, true);
            vk_->copy(img, VK_IMAGE_ASPECT_PLANE_0_BIT, 0, width_, kStripRows, vk_->strips.buffer, stripBytes * (2 * i));
            vk_->copy(img, VK_IMAGE_ASPECT_PLANE_0_BIT, centreY, width_, kStripRows, vk_->strips.buffer, stripBytes * (2 * i + 1));
            vk_->barrier(img, false);
        }
    };
    const auto stripHash = [&](int i, int which) { return hashBytes(vk_->strips.mapped + stripBytes * (2 * i + which), stripBytes); };
    std::vector<int> all(total);
    for (int i = 0; i < total; ++i) all[i] = i;
    int64_t lastProbeNs = 0;

    const auto complete = [&](Ring& ring, int s) {
        auto& slot = ring.slots[s];
        slot.complete = true; slot.doneTop = slot.top; slot.doneCentre = slot.centre;
    };
    const auto update = [&](Ring& ring, int64_t probeMid) {
        int started = -1;
        const int n = int(ring.slots.size());
        for (int s = 0; s < n; ++s) {
            auto& slot = ring.slots[s];
            const uint64_t top = stripHash(slot.image, 0), centre = stripHash(slot.image, 1);
            const bool changed = ring.primed && (top != slot.top || centre != slot.centre);
            slot.top = top; slot.centre = centre;
            if (!changed) continue;
            if (s == ring.pending) { ring.pendingLastChangeNs = probeMid; continue; }
            const int base = ring.pending >= 0 ? ring.pending : s;
            const auto ahead = [&](int x) { return (x - base + n) % n; };
            if (started < 0 || ahead(s) > ahead(started)) started = s;
        }
        if (started >= 0) {
            if (ring.pending >= 0) complete(ring, ring.pending);
            auto& slot = ring.slots[started];
            slot.complete = false;
            slot.writeStartNs = lastProbeNs ? lastProbeNs + (probeMid - lastProbeNs) / 2 : probeMid;
            const auto eof = ring.clock->atOrBefore(slot.writeStartNs);
            if (eof && slot.writeStartNs - eof->endOfReadoutNs < kMatchWindowNs) {
                slot.sequence = eof->sequence; slot.eofNs = eof->endOfReadoutNs;
                std::lock_guard<std::mutex> lock(mutex_);
                matchMsSum_ += (slot.writeStartNs - eof->endOfReadoutNs) * 1e-6; ++matches_;
            } else {
                slot.sequence = 0; slot.eofNs = 0; ++ring.unmatched;
            }
            ++ring.frames;
            ring.pending = started; ring.pendingLastChangeNs = probeMid;
        } else if (ring.pending >= 0 && probeMid - ring.pendingLastChangeNs > kIdleNs) {
            complete(ring, ring.pending); ring.pending = -1;
        }
        ring.primed = true;
    };

    try {
        while (true) {
            bool want, wantUV;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stop_) return;
                want = want_; wantUV = wantUV_;
            }
            left_->clock->poll(); right_->clock->poll();
            const int64_t t0 = V4l2CaptureClock::monotonicNs();
            vk_->begin(); recordStrips(all); vk_->submitAndWait(); vk_->invalidate(vk_->strips);
            const int64_t t1 = V4l2CaptureClock::monotonicNs();
            const int64_t probeMid = t0 + (t1 - t0) / 2;
            update(*left_, probeMid); update(*right_, probeMid);
            lastProbeNs = probeMid;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++stats_.probes; probeMsSum_ += (t1 - t0) * 1e-6;
                stats_.leftFrames = left_->frames; stats_.rightFrames = right_->frames;
                stats_.leftUnmatched = left_->unmatched; stats_.rightUnmatched = right_->unmatched;
            }
            if (want) {
                // Newest sequence completed in both rings and newer than the last delivered.
                int bl = -1, br = -1; uint32_t best = 0; uint64_t newer = 0;
                for (int a = 0; a < int(left_->slots.size()); ++a) {
                    const auto& l = left_->slots[a];
                    if (!l.complete || !l.sequence || l.sequence <= lastDelivered_) continue;
                    for (int b = 0; b < int(right_->slots.size()); ++b) {
                        const auto& r = right_->slots[b];
                        if (r.complete && r.sequence == l.sequence) {
                            ++newer;
                            if (l.sequence > best) { best = l.sequence; bl = a; br = b; }
                        }
                    }
                }
                if (bl >= 0) {
                    auto& l = left_->slots[bl]; auto& r = right_->slots[br];
                    const int64_t c0 = V4l2CaptureClock::monotonicNs();
                    vk_->begin();
                    const int imgs[2] = {l.image, r.image};
                    for (int e = 0; e < 2; ++e) {
                        VkImage img = vk_->images[imgs[e]].image;
                        vk_->barrier(img, true);
                        vk_->copy(img, VK_IMAGE_ASPECT_PLANE_0_BIT, 0, width_, height_, vk_->y[e].buffer, 0);
                        if (wantUV) vk_->copy(img, VK_IMAGE_ASPECT_PLANE_1_BIT, 0, width_ / 2, height_ / 2, vk_->uv[e].buffer, 0);
                        vk_->barrier(img, false);
                    }
                    recordStrips({l.image, r.image}); // post-copy verification
                    vk_->submitAndWait();
                    const int64_t c1 = V4l2CaptureClock::monotonicNs();
                    vk_->invalidate(vk_->strips);
                    const bool intact = stripHash(l.image, 0) == l.doneTop && stripHash(l.image, 1) == l.doneCentre &&
                                        stripHash(r.image, 0) == r.doneTop && stripHash(r.image, 1) == r.doneCentre;
                    if (!intact) {
                        std::lock_guard<std::mutex> lock(mutex_);
                        ++stats_.tornCopies;
                        lastDelivered_ = best; // never retry a slot that was being overwritten
                        continue;
                    }
                    for (int e = 0; e < 2; ++e) {
                        vk_->invalidate(vk_->y[e]);
                        if (wantUV) vk_->invalidate(vk_->uv[e]);
                    }
                    std::lock_guard<std::mutex> lock(mutex_);
                    std::memcpy(leftY_.data(), vk_->y[0].mapped, leftY_.size());
                    std::memcpy(rightY_.data(), vk_->y[1].mapped, rightY_.size());
                    if (wantUV) {
                        std::memcpy(leftUV_.data(), vk_->uv[0].mapped, leftUV_.size());
                        std::memcpy(rightUV_.data(), vk_->uv[1].mapped, rightUV_.size());
                    }
                    uvValid_ = wantUV;
                    result_ = {};
                    result_.valid = true; result_.sequence = best;
                    result_.leftSequence = l.sequence; result_.rightSequence = r.sequence;
                    result_.leftEndOfReadoutNs = l.eofNs; result_.rightEndOfReadoutNs = r.eofNs;
                    result_.leftWriteStartNs = l.writeStartNs; result_.rightWriteStartNs = r.writeStartNs;
                    result_.copyBeginNs = c0; result_.copyEndNs = c1;
                    result_.leftSlot = bl; result_.rightSlot = br;
                    result_.skippedSequences = lastDelivered_ ? best - lastDelivered_ - 1 : 0;
                    (void)newer;
                    lastDelivered_ = best;
                    ++stats_.deliveredPairs;
                    want_ = false; ready_ = true;
                    cv_.notify_all();
                    continue;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = e.what();
        cv_.notify_all();
    }
}

bool TimedStereoCapture::acquire(TimedStereoPair& out, int timeoutMs, bool includeUV) {
    std::unique_lock<std::mutex> lock(mutex_);
    want_ = true; wantUV_ = includeUV; ready_ = false;
    const bool ok = cv_.wait_for(lock, std::chrono::milliseconds(timeoutMs), [&] { return ready_ || !error_.empty() || stop_; });
    want_ = false;
    if (!ok || !ready_) return false;
    ready_ = false;
    out = result_;
    return true;
}

TimedStereoStats TimedStereoCapture::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    TimedStereoStats s = stats_;
    s.probeMsMean = s.probes ? probeMsSum_ / double(s.probes) : 0;
    s.writeStartMinusEofMsMean = matches_ ? matchMsSum_ / double(matches_) : 0;
    return s;
}

std::string TimedStereoCapture::error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

} // namespace FrameSpatial
