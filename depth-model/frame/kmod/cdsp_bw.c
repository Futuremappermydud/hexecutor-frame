// SPDX-License-Identifier: GPL-2.0
/* Extra interconnect vote on the CDSP's "mem" path (NSP -> DDR).
 * Valve's hexagon-baremetal driver votes only 1 GB/s peak for the CDSP;
 * votes aggregate, so this raises the NoC/DDR rate the DSP sees.
 * Unloading the module drops the vote again. */
#include <linux/device.h>
#include <linux/interconnect.h>
#include <linux/module.h>
#include <linux/platform_device.h>

static unsigned int avg_mbps = 20000, peak_mbps = 40000;
module_param(avg_mbps, uint, 0444);
module_param(peak_mbps, uint, 0444);

static struct icc_path *path;
static struct device *cdsp;

static int __init cdsp_bw_init(void)
{
	int ret;

	cdsp = bus_find_device_by_name(&platform_bus_type, NULL, "32300000.cdsp");
	if (!cdsp)
		return -ENODEV;
	path = of_icc_get(cdsp, "mem");
	if (IS_ERR_OR_NULL(path)) {
		put_device(cdsp);
		return path ? PTR_ERR(path) : -ENODEV;
	}
	ret = icc_set_bw(path, MBps_to_icc(avg_mbps), MBps_to_icc(peak_mbps));
	if (ret) {
		icc_put(path);
		put_device(cdsp);
		return ret;
	}
	pr_info("cdsp_bw: voted avg %u MB/s, peak %u MB/s\n", avg_mbps, peak_mbps);
	return 0;
}

static void __exit cdsp_bw_exit(void)
{
	icc_put(path);
	put_device(cdsp);
	pr_info("cdsp_bw: vote released\n");
}

module_init(cdsp_bw_init);
module_exit(cdsp_bw_exit);
MODULE_LICENSE("GPL");
