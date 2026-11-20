#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <vulkan/vulkan.h>
#include <openvr.h>

struct VRVulkanDevice_t {
    VkInstance m_pInstance;
    VkDevice m_pDevice;
    VkPhysicalDevice m_pPhysicalDevice;
    VkQueue m_pQueue;
    uint32_t m_uQueueFamilyIndex;
};

struct NativeDevice_t {
    uint64_t type; // 2 for Vulkan
    VRVulkanDevice_t *dev;
};

typedef uint32_t (*SetGraphicsDevice_fn)(void *self, NativeDevice_t *device);

int main() {
    printf("[PassthroughInit] Initializing OpenVR...\n");
    vr::EVRInitError vrErr = vr::VRInitError_None;
    vr::IVRSystem *sys = vr::VR_Init(&vrErr, vr::VRApplication_Background);
    if (vrErr != vr::VRInitError_None) {
        printf("VR_Init failed: %d\n", vrErr);
        return 1;
    }

    // 1. Create Vulkan Instance
    const char *instExts[] = {
        "VK_KHR_external_memory_capabilities",
        "VK_KHR_get_physical_device_properties2",
        "VK_KHR_external_fence_capabilities",
        "VK_KHR_external_semaphore_capabilities"
    };
    VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.pApplicationName = "FramePassthrough";
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &appInfo;
    ici.enabledExtensionCount = 4;
    ici.ppEnabledExtensionNames = instExts;
    VkInstance instance;
    VkResult res = vkCreateInstance(&ici, NULL, &instance);
    if (res != VK_SUCCESS) {
        printf("vkCreateInstance failed: %d\n", res);
        return 1;
    }
    printf("[PassthroughInit] Vulkan Instance created.\n");

    // 2. Select GPU
    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(instance, &gpuCount, NULL);
    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data());
    VkPhysicalDevice phys = gpus[0];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    printf("[PassthroughInit] Using GPU: %s\n", props.deviceName);

    // Find queue family with graphics/compute
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, NULL);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, qfProps.data());
    uint32_t qFam = 0;
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfProps[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
            qFam = i;
            break;
        }
    }

    // 3. Create Logical Device
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = qFam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    const char *devExts[] = {
        "VK_KHR_external_memory",
        "VK_KHR_external_memory_fd",
        "VK_KHR_external_semaphore",
        "VK_KHR_external_semaphore_fd",
        "VK_KHR_timeline_semaphore",
        "VK_KHR_dedicated_allocation",
        "VK_KHR_get_memory_requirements2",
        "VK_KHR_image_format_list"
    };

    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = sizeof(devExts) / sizeof(devExts[0]);
    dci.ppEnabledExtensionNames = devExts;

    VkDevice device;
    res = vkCreateDevice(phys, &dci, NULL, &device);
    if (res != VK_SUCCESS) {
        printf("vkCreateDevice failed: %d\n", res);
        return 1;
    }
    printf("[PassthroughInit] Vulkan Logical Device created.\n");

    VkQueue queue;
    vkGetDeviceQueue(device, qFam, 0, &queue);

    // 4. Fill device struct
    VRVulkanDevice_t vkDev;
    vkDev.m_pInstance = instance;
    vkDev.m_pDevice = device;
    vkDev.m_pPhysicalDevice = phys;
    vkDev.m_pQueue = queue;
    vkDev.m_uQueueFamilyIndex = qFam;

    NativeDevice_t nativeDev;
    nativeDev.type = 2; // Vulkan
    nativeDev.dev = &vkDev;

    // 5. Get IVRCameraPassthroughInternal_001
    vr::EVRInitError ifErr = vr::VRInitError_None;
    void *iface = vr::VR_GetGenericInterface("IVRCameraPassthroughInternal_001", &ifErr);
    if (!iface) {
        printf("Failed to get IVRCameraPassthroughInternal_001: %d\n", ifErr);
        return 1;
    }
    printf("[PassthroughInit] Acquired IVRCameraPassthroughInternal_001: %p\n", iface);

    void **vtable = *(void***)iface;
    SetGraphicsDevice_fn SetGraphicsDevice = (SetGraphicsDevice_fn)vtable[0];

    printf("[PassthroughInit] Calling SetGraphicsDevice...\n");
    uint32_t ret = SetGraphicsDevice(iface, &nativeDev);
    printf("[PassthroughInit] SetGraphicsDevice returned: %u (0 = Success)\n", ret);

    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    vr::VR_Shutdown();
    return 0;
}
