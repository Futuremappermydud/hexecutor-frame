// SPDX-License-Identifier: GPL-2.0
/* Large shared buffer for CDSP payloads with dynamic memory growth and shrinking.
 *
 * Valve's dsp_service only hands out a 16 MB DMA buffer. This allocates
 * `size_mb` in 2 MB physically contiguous chunks, maps them contiguously at
 * `iova` in the CDSP's SMMU domain (the domain hexagon-baremetal attached),
 * and exposes them write-combined through /dev/cdsp_mem. A payload adds its
 * own DSP TLB entries for that IOVA range.
 *
 * Dynamic memory reservation allows growing from 128 MiB baseline up to
 * 1.5 GiB (1536 MiB, the limit for 32-bit IOVA starting at 0xa0000000) and
 * shrinking back down when large workloads complete.
 * Dynamic resizing can be triggered by:
 * 1. Writing to /sys/module/cdsp_mem/parameters/size_mb
 * 2. ioctl CDSP_MEM_IOC_SET_SIZE on /dev/cdsp_mem
 */
#include <linux/device.h>
#include <linux/dma-mapping.h>
#include <linux/iommu.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/platform_device.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/capability.h>

#include "cdsp_mem_core.h"
#include "cdsp_mem_core.c"

#define CHUNK_ORDER 9 /* 2 MB with 4 KB pages */
#define CHUNK_BYTES (PAGE_SIZE << CHUNK_ORDER)

#define CDSP_MEM_IOC_MAGIC 'C'
#define CDSP_MEM_IOC_SET_SIZE _IOW(CDSP_MEM_IOC_MAGIC, 1, uint32_t)
#define CDSP_MEM_IOC_GET_SIZE _IOR(CDSP_MEM_IOC_MAGIC, 2, uint32_t)

struct cdsp_mem_info {
	uint32_t size_mb;
	uint32_t min_size_mb;
	uint32_t max_size_mb;
	uint32_t chunk_mb;
	uint64_t iova;
};
#define CDSP_MEM_IOC_GET_INFO _IOR(CDSP_MEM_IOC_MAGIC, 3, struct cdsp_mem_info)
#define CDSP_MEM_IOC_PIN      _IOW(CDSP_MEM_IOC_MAGIC, 4, uint32_t)
#define CDSP_MEM_IOC_UNPIN    _IO(CDSP_MEM_IOC_MAGIC, 5)

static unsigned int size_mb = 128;
static unsigned int iova = 0xa0000000;
module_param(iova, uint, 0444);

static bool allow_user_resize = false;
module_param(allow_user_resize, bool, 0644);
MODULE_PARM_DESC(allow_user_resize, "Allow unprivileged processes to resize CDSP memory (default: 0)");

static struct device *cdsp;
static struct iommu_domain *domain;
static struct platform_device *dma_dev;
static struct cdsp_mem_core core;
static DEFINE_MUTEX(cdsp_mem_lock);

struct cdsp_kchunk {
	struct page *page;
	dma_addr_t handle;
};

static void *cdsp_kalloc_chunk(void *ctx, uint32_t chunk_idx)
{
	struct cdsp_kchunk *kc;
	if (!dma_dev)
		return NULL;

	kc = kmalloc(sizeof(*kc), GFP_KERNEL);
	if (!kc)
		return NULL;

	kc->page = alloc_pages(GFP_KERNEL | __GFP_ZERO | __GFP_NOWARN, CHUNK_ORDER);
	if (!kc->page) {
		kfree(kc);
		return NULL;
	}

	kc->handle = dma_map_page(&dma_dev->dev, kc->page, 0, CHUNK_BYTES, DMA_BIDIRECTIONAL);
	if (dma_mapping_error(&dma_dev->dev, kc->handle)) {
		__free_pages(kc->page, CHUNK_ORDER);
		kfree(kc);
		return NULL;
	}

	return kc;
}

static void cdsp_kfree_chunk(void *ctx, void *chunk_handle, uint32_t chunk_idx)
{
	struct cdsp_kchunk *kc = (struct cdsp_kchunk *)chunk_handle;
	if (!kc)
		return;

	if (kc->handle && dma_dev)
		dma_unmap_page(&dma_dev->dev, kc->handle, CHUNK_BYTES, DMA_BIDIRECTIONAL);
	if (kc->page)
		__free_pages(kc->page, CHUNK_ORDER);
	kfree(kc);
}

static int cdsp_kiommu_map(void *ctx, uint64_t map_iova, void *chunk_handle, uint32_t chunk_idx)
{
	struct cdsp_kchunk *kc = (struct cdsp_kchunk *)chunk_handle;
	if (!domain || !kc || !kc->page)
		return -EINVAL;

	return iommu_map(domain, map_iova, page_to_phys(kc->page), CHUNK_BYTES,
			 IOMMU_READ | IOMMU_WRITE, GFP_KERNEL);
}

static void cdsp_kiommu_unmap(void *ctx, uint64_t unmap_iova, uint32_t chunk_idx)
{
	if (domain)
		iommu_unmap(domain, unmap_iova, CHUNK_BYTES);
}

static void cdsp_kflush_iotlb(void *ctx)
{
	if (domain)
		iommu_flush_iotlb_all(domain);
}

static void *cdsp_kalloc_meta(void *ctx, size_t bytes)
{
	return kzalloc(bytes, GFP_KERNEL);
}

static void cdsp_kfree_meta(void *ctx, void *ptr)
{
	kfree(ptr);
}

static const struct cdsp_mem_ops kernel_ops = {
	.alloc_chunk = cdsp_kalloc_chunk,
	.free_chunk = cdsp_kfree_chunk,
	.iommu_map = cdsp_kiommu_map,
	.iommu_unmap = cdsp_kiommu_unmap,
	.flush_iotlb = cdsp_kflush_iotlb,
	.alloc_meta = cdsp_kalloc_meta,
	.free_meta = cdsp_kfree_meta,
};

static int cdsp_mem_param_set_size_mb(const char *val, const struct kernel_param *kp)
{
	unsigned int n;
	int ret;

	if (!allow_user_resize && !capable(CAP_SYS_ADMIN))
		return -EPERM;

	ret = kstrtouint(val, 0, &n);
	if (ret)
		return ret;

	mutex_lock(&cdsp_mem_lock);
	if (!core.ops) {
		/* Before init (insmod parameter): same limits cdsp_mem_init enforces. */
		if (n < CDSP_MEM_MIN_SIZE_MB || n > CDSP_MEM_MAX_SIZE_MB || n % 2)
			ret = -EINVAL;
		else {
			size_mb = n;
			ret = 0;
		}
	} else {
		ret = cdsp_mem_core_resize(&core, n);
		if (ret == 0)
			size_mb = core.size_mb;
	}
	mutex_unlock(&cdsp_mem_lock);
	return ret;
}

static int cdsp_mem_param_get_size_mb(char *buffer, const struct kernel_param *kp)
{
	unsigned int cur;
	mutex_lock(&cdsp_mem_lock);
	cur = core.ops ? core.size_mb : size_mb;
	mutex_unlock(&cdsp_mem_lock);
	return sysfs_emit(buffer, "%u\n", cur);
}

static const struct kernel_param_ops cdsp_mem_size_mb_ops = {
	.set = cdsp_mem_param_set_size_mb,
	.get = cdsp_mem_param_get_size_mb,
};
module_param_cb(size_mb, &cdsp_mem_size_mb_ops, &size_mb, 0644);

static void cdsp_mem_vma_open(struct vm_area_struct *vma)
{
	unsigned long size = vma->vm_end - vma->vm_start;
	unsigned long off = vma->vm_pgoff << PAGE_SHIFT;

	mutex_lock(&cdsp_mem_lock);
	if (cdsp_mem_core_map_add(&core, vma, off, size) != 0) {
		core.untracked_mappings++;
	}
	mutex_unlock(&cdsp_mem_lock);
}

static void cdsp_mem_vma_close(struct vm_area_struct *vma)
{
	mutex_lock(&cdsp_mem_lock);
	cdsp_mem_core_map_close(&core, vma);
	mutex_unlock(&cdsp_mem_lock);
}

static const struct vm_operations_struct cdsp_mem_vm_ops = {
	.open = cdsp_mem_vma_open,
	.close = cdsp_mem_vma_close,
};

static int cdsp_mem_mmap(struct file *file, struct vm_area_struct *vma)
{
	unsigned long size = vma->vm_end - vma->vm_start, off = vma->vm_pgoff << PAGE_SHIFT;
	unsigned long done;
	int ret = 0;

	mutex_lock(&cdsp_mem_lock);
	if (off + size > (unsigned long)core.size_mb << 20 || off % CHUNK_BYTES) {
		mutex_unlock(&cdsp_mem_lock);
		return -EINVAL;
	}

	ret = cdsp_mem_core_map_add(&core, vma, off, size);
	if (ret) {
		mutex_unlock(&cdsp_mem_lock);
		return ret;
	}

	vma->vm_ops = &cdsp_mem_vm_ops;
	vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
	for (done = 0; done < size; done += CHUNK_BYTES) {
		unsigned long n = min_t(unsigned long, CHUNK_BYTES, size - done);
		struct cdsp_kchunk *kc = (struct cdsp_kchunk *)cdsp_mem_core_get_chunk(&core, (off + done) / CHUNK_BYTES);
		if (!kc || !kc->page) {
			ret = -EFAULT;
			break;
		}
		ret = remap_pfn_range(vma, vma->vm_start + done,
				      page_to_pfn(kc->page), n,
				      vma->vm_page_prot);
		if (ret)
			break;
	}

	if (ret != 0) {
		cdsp_mem_core_map_remove(&core, vma);
	}
	mutex_unlock(&cdsp_mem_lock);
	return ret;
}

static int cdsp_mem_release(struct inode *inode, struct file *file)
{
	mutex_lock(&cdsp_mem_lock);
	cdsp_mem_core_unpin(&core, file);
	mutex_unlock(&cdsp_mem_lock);
	return 0;
}

static long cdsp_mem_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	uint32_t val;
	int ret = 0;

	switch (cmd) {
	case CDSP_MEM_IOC_SET_SIZE:
		if (!allow_user_resize && !capable(CAP_SYS_ADMIN))
			return -EPERM;
		if (copy_from_user(&val, (void __user *)arg, sizeof(val)))
			return -EFAULT;
		mutex_lock(&cdsp_mem_lock);
		ret = cdsp_mem_core_resize(&core, val);
		if (ret == 0)
			size_mb = core.size_mb;
		mutex_unlock(&cdsp_mem_lock);
		return ret;
	case CDSP_MEM_IOC_GET_SIZE:
		mutex_lock(&cdsp_mem_lock);
		val = core.size_mb;
		mutex_unlock(&cdsp_mem_lock);
		if (copy_to_user((void __user *)arg, &val, sizeof(val)))
			return -EFAULT;
		return 0;
	case CDSP_MEM_IOC_GET_INFO: {
		struct cdsp_mem_info info;
		mutex_lock(&cdsp_mem_lock);
		info.size_mb = core.size_mb;
		info.min_size_mb = CDSP_MEM_MIN_SIZE_MB;
		info.max_size_mb = CDSP_MEM_MAX_SIZE_MB;
		info.chunk_mb = CDSP_MEM_CHUNK_MB;
		info.iova = core.iova;
		mutex_unlock(&cdsp_mem_lock);
		if (copy_to_user((void __user *)arg, &info, sizeof(info)))
			return -EFAULT;
		return 0;
	}
	case CDSP_MEM_IOC_PIN:
		if (copy_from_user(&val, (void __user *)arg, sizeof(val)))
			return -EFAULT;
		mutex_lock(&cdsp_mem_lock);
		ret = cdsp_mem_core_pin(&core, file, val);
		mutex_unlock(&cdsp_mem_lock);
		return ret;
	case CDSP_MEM_IOC_UNPIN:
		mutex_lock(&cdsp_mem_lock);
		ret = cdsp_mem_core_unpin(&core, file);
		mutex_unlock(&cdsp_mem_lock);
		return ret;
	default:
		return -ENOTTY;
	}
}

static const struct file_operations cdsp_mem_fops = {
	.owner = THIS_MODULE,
	.mmap = cdsp_mem_mmap,
	.unlocked_ioctl = cdsp_mem_ioctl,
	.release = cdsp_mem_release,
};

static struct miscdevice cdsp_mem_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "cdsp_mem",
	.fops = &cdsp_mem_fops,
	.mode = 0666,
};

static void cdsp_mem_free(void)
{
	mutex_lock(&cdsp_mem_lock);
	cdsp_mem_core_destroy(&core);
	if (dma_dev) {
		platform_device_unregister(dma_dev);
		dma_dev = NULL;
	}
	if (cdsp) {
		put_device(cdsp);
		cdsp = NULL;
	}
	domain = NULL;
	mutex_unlock(&cdsp_mem_lock);
}

static int __init cdsp_mem_init(void)
{
	unsigned int i;
	int ret;

	if (size_mb < CDSP_MEM_MIN_SIZE_MB)
		size_mb = CDSP_MEM_MIN_SIZE_MB;
	if (size_mb > CDSP_MEM_MAX_SIZE_MB)
		size_mb = CDSP_MEM_MAX_SIZE_MB;
	if (size_mb % 2 || iova % CHUNK_BYTES ||
	    ((u64)size_mb << 20) > (1ULL << 32) - iova)
		return -EINVAL;

	cdsp = bus_find_device_by_name(&platform_bus_type, NULL, "32300000.cdsp");
	if (!cdsp)
		return -ENODEV;
	domain = iommu_get_domain_for_dev(cdsp);
	if (!domain) {
		ret = -ENODEV;
		goto fail;
	}
	for (i = 0; i < size_mb / 2; i++) {
		if (iommu_iova_to_phys(domain, iova + (u64)i * CHUNK_BYTES)) {
			pr_err("cdsp_mem: IOVA %#llx is already mapped\n", iova + (u64)i * CHUNK_BYTES);
			ret = -EEXIST;
			goto fail;
		}
	}

	/* A plain non-coherent device gives the cache maintenance we need:
	 * the zeroed chunks must not leave dirty lines in the linear map. */
	dma_dev = platform_device_register_simple("cdsp_mem_dma", -1, NULL, 0);
	if (IS_ERR(dma_dev)) {
		ret = PTR_ERR(dma_dev);
		dma_dev = NULL;
		goto fail;
	}
	ret = dma_coerce_mask_and_coherent(&dma_dev->dev, DMA_BIT_MASK(64));
	if (ret)
		goto fail;

	ret = cdsp_mem_core_init(&core, size_mb, iova, &kernel_ops, NULL);
	if (ret)
		goto fail;

	ret = misc_register(&cdsp_mem_misc);
	if (ret)
		goto fail;

	pr_info("cdsp_mem: %u MB at CDSP IOVA %#x (dynamic growth up to %u MB enabled)\n",
		core.size_mb, iova, CDSP_MEM_MAX_SIZE_MB);
	return 0;

fail:
	cdsp_mem_free();
	return ret;
}

static void __exit cdsp_mem_exit(void)
{
	misc_deregister(&cdsp_mem_misc);
	cdsp_mem_free();
	pr_info("cdsp_mem: released\n");
}

module_init(cdsp_mem_init);
module_exit(cdsp_mem_exit);
MODULE_LICENSE("GPL");
