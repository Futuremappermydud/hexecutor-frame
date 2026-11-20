#include <stdio.h>
#include <vector>
#include <vulkan/vulkan.h>
#include <openvr.h>

int main() {
    vr::EVRInitError err = vr::VRInitError_None;
    vr::IVRSystem *sys = vr::VR_Init(&err, vr::VRApplication_Background);
    if (err != vr::VRInitError_None) return 1;

    // Create VkInstance
    const char *instExts[] = {
        "VK_KHR_external_memory_capabilities",
        "VK_KHR_get_physical_device_properties2",
        "VK_KHR_external_fence_capabilities",
        "VK_KHR_external_semaphore_capabilities"
    };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.enabledExtensionCount = 4;
    ici.ppEnabledExtensionNames = instExts;
    VkInstance instance;
    if (vkCreateInstance(&ici, NULL, &instance) != VK_SUCCESS) {
        printf("vkCreateInstance failed\n");
        return 1;
    }

    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(instance, &gpuCount, NULL);
    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data());
    printf("Found %u GPUs\n", gpuCount);

    vr::IVRCompositor *comp = vr::VRCompositor();
    if (comp && gpuCount > 0) {
        char buf[2048] = {0};
        uint32_t len = comp->GetVulkanDeviceExtensionsRequired((VkPhysicalDevice_T*)gpus[0], buf, sizeof(buf));
        printf("Required Device Extensions (%u bytes): %s\n", len, buf);
    }

    vkDestroyInstance(instance, NULL);
    vr::VR_Shutdown();
    return 0;
}
