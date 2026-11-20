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
    uint64_t type;
    VRVulkanDevice_t *dev;
};

typedef uint32_t (*SetGraphicsDevice_fn)(void *self, NativeDevice_t *device);
typedef uint32_t (*Method7_fn)(void *self, uint8_t *enabled);
typedef uint32_t (*Method8_fn)(void *self, uint8_t enabled);
typedef uint32_t (*Method1_fn)(void *self, uint32_t arg1, void *arg2, void *arg3);

int main() {
    vr::EVRInitError vrErr = vr::VRInitError_None;
    vr::IVRSystem *sys = vr::VR_Init(&vrErr, vr::VRApplication_Background);
    if (vrErr != vr::VRInitError_None) return 1;

    // 1. Vulkan Init
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
    vkCreateInstance(&ici, NULL, &instance);

    uint32_t gpuCount = 1;
    VkPhysicalDevice phys;
    vkEnumeratePhysicalDevices(instance, &gpuCount, &phys);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = 0;
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
    vkCreateDevice(phys, &dci, NULL, &device);
    VkQueue queue;
    vkGetDeviceQueue(device, 0, 0, &queue);

    VRVulkanDevice_t vkDev = { instance, device, phys, queue, 0 };
    NativeDevice_t nativeDev = { 2, &vkDev };

    vr::EVRInitError ifErr = vr::VRInitError_None;
    void *iface = vr::VR_GetGenericInterface("IVRCameraPassthroughInternal_001", &ifErr);
    void **vtable = *(void***)iface;
    SetGraphicsDevice_fn SetGraphicsDevice = (SetGraphicsDevice_fn)vtable[0];
    Method1_fn Method1 = (Method1_fn)vtable[1];
    Method7_fn GetEnabled = (Method7_fn)vtable[7];
    Method8_fn SetEnabled = (Method8_fn)vtable[8];

    uint32_t ret = SetGraphicsDevice(iface, &nativeDev);
    printf("SetGraphicsDevice: %u\n", ret);

    uint8_t enabled = 0;
    GetEnabled(iface, &enabled);
    printf("GetEnabled initially: %u\n", enabled);

    printf("Calling SetEnabled(1)...\n");
    SetEnabled(iface, 1);

    GetEnabled(iface, &enabled);
    printf("GetEnabled after SetEnabled(1): %u\n", enabled);

    // Now test Method 1 again!
    uint32_t arg2[32] = {0};
    arg2[0] = 2; // e.g. 2 cameras
    uint8_t arg3[1024] = {0};
    uint32_t m1_ret = Method1(iface, 0, arg2, arg3);
    printf("Method1 returned: %u\n", m1_ret);

    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    vr::VR_Shutdown();
    return 0;
}
