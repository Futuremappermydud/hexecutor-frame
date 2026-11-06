// SPDX-License-Identifier: GPL-2.0
/* Raise the compute-NoC (NSP) bus clock for the CDSP.
 *
 * The CDSP's path to DDR runs through BCM "CO0". Linux's SM8650 interconnect
 * driver treats CO0 as enable-only, so it only ever votes the lowest level;
 * QuRT would vote it up from the DSP, but Valve's firmware does not. The DSP
 * then sustains only ~1.25 GB/s from DDR. This votes CO0 directly through the
 * apps RSC (same command format as bcm-voter) and restores the enable-only
 * vote on unload. The vote is in CO0 units (cmd-db aux data: unit 4170000,
 * width 64); 14 bits, larger values select the top level.
 */
#include <linux/device.h>
#include <linux/module.h>
#include <linux/platform_device.h>
#include <soc/qcom/cmd-db.h>
#include <soc/qcom/rpmh.h>
#include <soc/qcom/tcs.h>

static unsigned int vote = 9592;   /* ~20 GB/s through qnm_nsp */
module_param(vote, uint, 0444);
static char *bcm = "CO0";
module_param(bcm, charp, 0444);

static struct device *voter;
static u32 addr;

static int co0_vote(u32 x, u32 y)
{
	struct tcs_cmd cmd = {
		.addr = addr,
		.data = BCM_TCS_CMD(true, true, x, y),
		.wait = true,
	};
	int ret = rpmh_write(voter, RPMH_ACTIVE_ONLY_STATE, &cmd, 1);

	if (ret)
		return ret;
	cmd.wait = false;
	return rpmh_write(voter, RPMH_WAKE_ONLY_STATE, &cmd, 1);
}

static int __init cdsp_noc_init(void)
{
	int ret;

	if (vote > BCM_TCS_CMD_VOTE_MASK)
		vote = BCM_TCS_CMD_VOTE_MASK;
	addr = cmd_db_read_addr(bcm);
	if (!addr)
		return -ENODEV;
	voter = bus_find_device_by_name(&platform_bus_type, NULL, "17a00000.rsc:bcm-voter");
	if (!voter)
		return -ENODEV;
	ret = co0_vote(vote, vote);
	if (ret) {
		put_device(voter);
		return ret;
	}
	pr_info("cdsp_noc: %s (%#x) voted %u\n", bcm, addr, vote);
	return 0;
}

static void __exit cdsp_noc_exit(void)
{
	co0_vote(0, 1);
	put_device(voter);
	pr_info("cdsp_noc: %s back to enable-only\n", bcm);
}

module_init(cdsp_noc_init);
module_exit(cdsp_noc_exit);
MODULE_LICENSE("GPL");
