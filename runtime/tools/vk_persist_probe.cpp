#define _GNU_SOURCE
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <vulkan/vulkan.h>

#ifndef __NR_pidfd_open
#define __NR_pidfd_open 434
#endif
#ifndef __NR_pidfd_getfd
#define __NR_pidfd_getfd 438
#endif

static const uint64_t QCOM_COMPRESSED = 0x500000000000001ULL;
static const uint32_t CAM_W = 2464, CAM_H = 2464;
static const VkDeviceSize DMA_SIZE = 9564160;
static const VkDeviceSize YSIZE = (VkDeviceSize)CAM_W * CAM_H;

#define CHECK(e) do { VkResult r=(e); if(r){ fprintf(stderr,"FAIL %s -> %d\n", #e, (int)r); return 1;} } while(0)

static uint64_t hashY(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i += 64) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static int findCams(pid_t pid, std::vector<int>& fds) {
    char dirPath[256];
    snprintf(dirPath, sizeof(dirPath), "/proc/%d/fd", pid);
    DIR* d = opendir(dirPath);
    if (!d) return 0;
    while (dirent* e = readdir(d)) {
        if (e->d_name[0]=='.') continue;
        char linkPath[512], target[512]={0};
        snprintf(linkPath, sizeof(linkPath), "/proc/%d/fd/%s", pid, e->d_name);
        ssize_t n = readlink(linkPath, target, sizeof(target)-1);
        if (n<=0) continue;
        struct stat st;
        if (strstr(target, "dmabuf") && stat(linkPath, &st)==0 && st.st_size==(off_t)DMA_SIZE)
            fds.push_back(atoi(e->d_name));
    }
    closedir(d);
    std::sort(fds.begin(), fds.end());
    return (int)fds.size();
}

int main() {
    FILE* p = popen("pgrep -n vrcompositor","r");
    pid_t pid=0; fscanf(p,"%d",&pid); pclose(p);
    std::vector<int> cams;
    findCams(pid, cams);
    printf("pid=%d slots=%zu\n", pid, cams.size());
    int pfd = syscall(__NR_pidfd_open, pid, 0);

    const char* instExts[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME
    };
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo=&app; ici.enabledExtensionCount=2; ici.ppEnabledExtensionNames=instExts;
    VkInstance instance; CHECK(vkCreateInstance(&ici,nullptr,&instance));
    uint32_t gpuCount=0; vkEnumeratePhysicalDevices(instance,&gpuCount,nullptr);
    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(instance,&gpuCount,gpus.data());
    VkPhysicalDevice phys=gpus[0];
    float prio=1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex=0; qci.queueCount=1; qci.pQueuePriorities=&prio;
    const char* devExts[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        VK_KHR_BIND_MEMORY_2_EXTENSION_NAME
    };
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
    dci.enabledExtensionCount=6; dci.ppEnabledExtensionNames=devExts;
    VkDevice device; CHECK(vkCreateDevice(phys,&dci,nullptr,&device));
    VkQueue queue; vkGetDeviceQueue(device,0,0,&queue);
    auto vkGetMemoryFdPropertiesKHR=(PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(device,"vkGetMemoryFdPropertiesKHR");

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool; CHECK(vkCreateCommandPool(device,&cpci,nullptr,&pool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool=pool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(device,&cbai,&cmd));

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size=YSIZE; bci.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buf; CHECK(vkCreateBuffer(device,&bci,nullptr,&buf));
    VkMemoryRequirements br; vkGetBufferMemoryRequirements(device,buf,&br);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys,&mp);
    uint32_t hostType=0;
    for (uint32_t i=0;i<mp.memoryTypeCount;i++)
        if ((br.memoryTypeBits&(1u<<i)) && (mp.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) { hostType=i; break; }
    VkMemoryAllocateInfo hai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    hai.allocationSize=br.size; hai.memoryTypeIndex=hostType;
    VkDeviceMemory hostMem; CHECK(vkAllocateMemory(device,&hai,nullptr,&hostMem));
    CHECK(vkBindBufferMemory(device,buf,hostMem,0));
    void* mapped=nullptr; CHECK(vkMapMemory(device,hostMem,0,YSIZE,0,&mapped));

    auto importOne = [&](int compositorFd, VkImage& img, VkDeviceMemory& mem)->int {
        int dma = syscall(__NR_pidfd_getfd, pfd, compositorFd, 0);
        uint64_t modifier=QCOM_COMPRESSED;
        VkImageDrmFormatModifierListCreateInfoEXT modInfo{VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT};
        modInfo.drmFormatModifierCount=1; modInfo.pDrmFormatModifiers=&modifier;
        VkExternalMemoryImageCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
        ext.pNext=&modInfo; ext.handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.pNext=&ext; ii.imageType=VK_IMAGE_TYPE_2D; ii.format=VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
        ii.extent={CAM_W,CAM_H,1}; ii.mipLevels=1; ii.arrayLayers=1; ii.samples=VK_SAMPLE_COUNT_1_BIT;
        ii.tiling=VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
        ii.usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;
        ii.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device,&ii,nullptr,&img)!=VK_SUCCESS) return -1;
        VkImageMemoryRequirementsInfo2 ri{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2};
        ri.image=img;
        VkMemoryDedicatedRequirements dr{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
        VkMemoryRequirements2 r2{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,&dr};
        vkGetImageMemoryRequirements2(device,&ri,&r2);
        VkMemoryFdPropertiesKHR fp{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
        vkGetMemoryFdPropertiesKHR(device,VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,dma,&fp);
        uint32_t bits=r2.memoryRequirements.memoryTypeBits & fp.memoryTypeBits;
        uint32_t mt=0; for (uint32_t t=0;t<mp.memoryTypeCount;t++) if (bits&(1u<<t)){mt=t;break;}
        VkMemoryDedicatedAllocateInfo da{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO}; da.image=img;
        VkImportMemoryFdInfoKHR im{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
        im.pNext=&da; im.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT; im.fd=dma;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext=&im; mai.allocationSize=r2.memoryRequirements.size; mai.memoryTypeIndex=mt;
        if (vkAllocateMemory(device,&mai,nullptr,&mem)!=VK_SUCCESS) return -1;
        vkBindImageMemory(device,img,mem,0);
        return 0;
    };

    auto copyOnce = [&](VkImage img) {
        vkResetCommandBuffer(cmd,0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd,&bi);
        VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        bar.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
        bar.oldLayout=VK_IMAGE_LAYOUT_GENERAL;
        bar.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        bar.srcQueueFamilyIndex=VK_QUEUE_FAMILY_EXTERNAL;
        bar.dstQueueFamilyIndex=0;
        bar.image=img;
        bar.subresourceRange.aspectMask=VK_IMAGE_ASPECT_PLANE_0_BIT;
        bar.subresourceRange.levelCount=1; bar.subresourceRange.layerCount=1;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,0,0,0,1,&bar);
        VkBufferImageCopy region{};
        region.bufferRowLength=CAM_W; region.bufferImageHeight=CAM_H;
        region.imageSubresource.aspectMask=VK_IMAGE_ASPECT_PLANE_0_BIT;
        region.imageSubresource.layerCount=1; region.imageExtent={CAM_W,CAM_H,1};
        vkCmdCopyImageToBuffer(cmd,img,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buf,1,&region);
        VkImageMemoryBarrier rel=bar;
        rel.srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT; rel.dstAccessMask=0;
        rel.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; rel.newLayout=VK_IMAGE_LAYOUT_GENERAL;
        rel.srcQueueFamilyIndex=0; rel.dstQueueFamilyIndex=VK_QUEUE_FAMILY_EXTERNAL;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,0,0,0,0,0,1,&rel);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount=1; si.pCommandBuffers=&cmd;
        vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        VkMappedMemoryRange rng{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        rng.memory=hostMem; rng.size=VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(device,1,&rng);
        return hashY((uint8_t*)mapped, (size_t)YSIZE);
    };

    printf("=== persist copy same imported image 5x ===\n");
    VkImage img; VkDeviceMemory mem;
    importOne(cams[0], img, mem);
    for (int i=0;i<5;i++) {
        uint64_t h = copyOnce(img);
        printf("  t=%d hash=%016llx mean=%u\n", i, (unsigned long long)h, ((uint8_t*)mapped)[CAM_W*1232+1232]);
        usleep(200000);
    }
    vkDestroyImage(device,img,nullptr);
    vkFreeMemory(device,mem,nullptr);

    printf("=== all slots, two passes 300ms apart ===\n");
    std::vector<VkImage> imgs(cams.size());
    std::vector<VkDeviceMemory> mems(cams.size());
    for (size_t i=0;i<cams.size();i++) importOne(cams[i], imgs[i], mems[i]);
    for (int pass=0; pass<2; pass++) {
        printf("pass %d\n", pass);
        for (size_t i=0;i<cams.size();i++) {
            uint64_t h=copyOnce(imgs[i]);
            printf("  slot %zu fd=%d hash=%016llx\n", i, cams[i], (unsigned long long)h);
        }
        usleep(300000);
    }

    FILE* f=fopen("/tmp/persist_left.pgm","wb");
    fprintf(f,"P5\n%u %u\n255\n", CAM_W, CAM_H);
    copyOnce(imgs[4]);
    fwrite(mapped,1,(size_t)YSIZE,f); fclose(f);
    f=fopen("/tmp/persist_right.pgm","wb");
    fprintf(f,"P5\n%u %u\n255\n", CAM_W, CAM_H);
    copyOnce(imgs[0]);
    fwrite(mapped,1,(size_t)YSIZE,f); fclose(f);
    printf("wrote pgms\n");
    return 0;
}
