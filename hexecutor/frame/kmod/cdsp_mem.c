// SPDX-License-Identifier: GPL-2.0
/* Large shared buffer for CDSP payloads.
 *
 * Valve's dsp_service only hands out a 16 MB DMA buffer. This allocates
 * `size_mb` in 2 MB physically contiguous chunks, maps them contiguously at
 * `iova` in the CDSP's SMMU domain (the domain hexagon-baremetal attached),
 * and exposes them write-combined through /dev/cdsp_mem. A payload adds its
 * own DSP TLB entries for that IOVA range. Unloading unmaps and frees it, so
 * only unload while no DSP payload is running.
 */
#include <linux/device.h>
#include <linux/dma-mapping.h>
#include <linux/iommu.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/slab.h>

#define CHUNK_ORDER 9 /* 2 MB with 4 KB pages */
#define CHUNK_BYTES (PAGE_SIZE << CHUNK_ORDER)

static unsigned int size_mb = 512;
static unsigned int iova = 0xa0000000;
module_param(size_mb, uint, 0444);
module_param(iova, uint, 0444);

static struct device *cdsp;
static struct iommu_domain *domain;
static struct platform_device *dma_dev;
static struct page **chunks;
static dma_addr_t *handles;
static unsigned int nchunks, mapped;

static int cdsp_mem_mmap(struct file *file, struct vm_area_struct *vma)
{
	unsigned long size = vma->vm_end - vma->vm_start, off = vma->vm_pgoff << PAGE_SHIFT;
	unsigned long done;

	if (off + size > (unsigned long)nchunks * CHUNK_BYTES || off % CHUNK_BYTES)
		return -EINVAL;
	vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
	for (done = 0; done < size; done += CHUNK_BYTES) {
		unsigned long n = min_t(unsigned long, CHUNK_BYTES, size - done);
		int ret = remap_pfn_range(vma, vma->vm_start + done,
					  page_to_pfn(chunks[(off + done) / CHUNK_BYTES]), n,
					  vma->vm_page_prot);
		if (ret)
			return ret;
	}
	return 0;
}

static const struct file_operations cdsp_mem_fops = {
	.owner = THIS_MODULE,
	.mmap = cdsp_mem_mmap,
};

static struct miscdevice cdsp_mem_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "cdsp_mem",
	.fops = &cdsp_mem_fops,
	.mode = 0666,
};

static void cdsp_mem_free(void)
{
	unsigned int i;

	if (mapped) {
		iommu_unmap(domain, iova, (size_t)mapped * CHUNK_BYTES);
		/* By-VA invalidation alone left stale CDSP translations after
		 * reloads (chunks resolving to old pages); flush the context. */
		iommu_flush_iotlb_all(domain);
	}
	for (i = 0; i < nchunks; i++) {
		if (handles && handles[i])
			dma_unmap_page(&dma_dev->dev, handles[i], CHUNK_BYTES, DMA_BIDIRECTIONAL);
		if (chunks && chunks[i])
			__free_pages(chunks[i], CHUNK_ORDER);
	}
	kfree(handles);
	kfree(chunks);
	if (dma_dev)
		platform_device_unregister(dma_dev);
	if (cdsp)
		put_device(cdsp);
}

static int __init cdsp_mem_init(void)
{
	unsigned int i;
	int ret;

	if (!size_mb || size_mb % 2 || iova % CHUNK_BYTES ||
	    ((u64)size_mb << 20) > (1ULL << 32) - iova)
		return -EINVAL;
	nchunks = size_mb / 2;
	cdsp = bus_find_device_by_name(&platform_bus_type, NULL, "32300000.cdsp");
	if (!cdsp)
		return -ENODEV;
	domain = iommu_get_domain_for_dev(cdsp);
	if (!domain) {
		ret = -ENODEV;
		goto fail;
	}
	for (i = 0; i < nchunks; i++)
		if (iommu_iova_to_phys(domain, iova + (u64)i * CHUNK_BYTES)) {
			pr_err("cdsp_mem: IOVA %#llx is already mapped\n", iova + (u64)i * CHUNK_BYTES);
			ret = -EEXIST;
			goto fail;
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
	chunks = kcalloc(nchunks, sizeof(*chunks), GFP_KERNEL);
	handles = kcalloc(nchunks, sizeof(*handles), GFP_KERNEL);
	if (!chunks || !handles) {
		ret = -ENOMEM;
		goto fail;
	}
	for (i = 0; i < nchunks; i++) {
		chunks[i] = alloc_pages(GFP_KERNEL | __GFP_ZERO | __GFP_NOWARN, CHUNK_ORDER);
		if (!chunks[i]) {
			ret = -ENOMEM;
			goto fail;
		}
		handles[i] = dma_map_page(&dma_dev->dev, chunks[i], 0, CHUNK_BYTES, DMA_BIDIRECTIONAL);
		if (dma_mapping_error(&dma_dev->dev, handles[i])) {
			handles[i] = 0;
			ret = -EIO;
			goto fail;
		}
		ret = iommu_map(domain, iova + (u64)i * CHUNK_BYTES, page_to_phys(chunks[i]),
				CHUNK_BYTES, IOMMU_READ | IOMMU_WRITE, GFP_KERNEL);
		if (ret)
			goto fail;
		mapped = i + 1;
	}
	iommu_flush_iotlb_all(domain);
	ret = misc_register(&cdsp_mem_misc);
	if (ret)
		goto fail;
	pr_info("cdsp_mem: %u MB at CDSP IOVA %#x\n", size_mb, iova);
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
