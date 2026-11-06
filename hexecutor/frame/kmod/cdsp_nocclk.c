// SPDX-License-Identifier: GPL-2.0
/* Inspect or raise the NSP NoC clock feeding the CDSP's path to DDR.
 *
 * QuRT's clock driver (CDSP.HT.3.0, turing_cc_nspnoc_clk) programs a GCC
 * RCG at GCC+0x45018 that Linux's gcc-sm8650 driver does not model. Its
 * mux: 0 XO, 1 GPLL0 (600 MHz), 6 GPLL0_OUT_EVEN (300 MHz), 7 400 MHz;
 * QuRT runs it at 300 MHz (LOW_SVS) or 400 MHz (SVS). Valve's firmware
 * never sets it, and at 19.2 MHz x 64 B the DSP tops out at ~1.25 GB/s.
 *
 * mhz=0 (default) only reports the RCG. mhz=300 selects GPLL0_OUT_EVEN / 1
 * after checking that GPLL0 and its even output are running. Unloading
 * restores the original configuration if it was changed.
 */
#include <linux/bitfield.h>
#include <linux/delay.h>
#include <linux/io.h>
#include <linux/module.h>

#define GCC_BASE	0x00100000
#define GPLL0_MODE	0x0
#define GPLL0_USER_CTL	0x18
#define TURING_Q6_AXI_CBCR	0x45004
#define TURING_CFG_AHB_CBCR	0x45008
#define NSPNOC_RCG	0x45018	/* CMD; CFG follows at +4 */

#define PLL_LOCK_DET	BIT(31)
#define PLL_OUT_EVEN	BIT(1)
#define CMD_UPDATE	BIT(0)
#define CMD_ROOT_OFF	BIT(31)
#define CFG_SRC_DIV	GENMASK(4, 0)
#define CFG_SRC_SEL	GENMASK(10, 8)
#define CFG_MODE	GENMASK(13, 12)
#define CFG_HW_CTRL	BIT(20)

static unsigned int mhz;
module_param(mhz, uint, 0444);

static void __iomem *gcc;
static u32 saved_cfg;
static bool changed;

static void report(const char *when)
{
	u32 cmd = readl(gcc + NSPNOC_RCG), cfg = readl(gcc + NSPNOC_RCG + 4);
	static const char * const src[8] = { "XO", "GPLL0", "?", "?", "?", "?",
					     "GPLL0_EVEN", "src7" };

	pr_info("cdsp_nocclk: %s CMD %#x%s CFG %#x src %s div %u.%u%s\n", when,
		cmd, cmd & CMD_ROOT_OFF ? " (root off)" : "", cfg,
		src[FIELD_GET(CFG_SRC_SEL, cfg)],
		(FIELD_GET(CFG_SRC_DIV, cfg) + 1) / 2,
		(FIELD_GET(CFG_SRC_DIV, cfg) + 1) % 2 ? 5 : 0,
		cfg & CFG_HW_CTRL ? " hw-ctrl" : "");
	pr_info("cdsp_nocclk: %s GPLL0 MODE %#x USER_CTL %#x, turing_q6_axi CBCR %#x, turing_cfg_ahb CBCR %#x\n",
		when, readl(gcc + GPLL0_MODE), readl(gcc + GPLL0_USER_CTL),
		readl(gcc + TURING_Q6_AXI_CBCR), readl(gcc + TURING_CFG_AHB_CBCR));
}

static int rcg_update(u32 cfg)
{
	int i;

	writel(cfg, gcc + NSPNOC_RCG + 4);
	writel(readl(gcc + NSPNOC_RCG) | CMD_UPDATE, gcc + NSPNOC_RCG);
	for (i = 0; i < 500; i++) {
		if (!(readl(gcc + NSPNOC_RCG) & CMD_UPDATE))
			return 0;
		udelay(1);
	}
	return -ETIMEDOUT;
}

static int __init cdsp_nocclk_init(void)
{
	u32 cfg;
	int ret;

	if (mhz && mhz != 300)
		return -EINVAL;
	gcc = ioremap(GCC_BASE, NSPNOC_RCG + 8);
	if (!gcc)
		return -ENOMEM;
	report("before");
	saved_cfg = readl(gcc + NSPNOC_RCG + 4);
	if (!mhz)
		return 0;

	if (!(readl(gcc + GPLL0_MODE) & PLL_LOCK_DET) ||
	    !(readl(gcc + GPLL0_USER_CTL) & PLL_OUT_EVEN)) {
		pr_err("cdsp_nocclk: GPLL0 even output not running, not switching\n");
		iounmap(gcc);
		return -ENODEV;
	}
	cfg = saved_cfg & ~(CFG_SRC_DIV | CFG_SRC_SEL | CFG_MODE);
	cfg |= FIELD_PREP(CFG_SRC_SEL, 6) | FIELD_PREP(CFG_SRC_DIV, 1);
	ret = rcg_update(cfg);
	changed = true;
	report(ret ? "update timed out" : "after");
	return 0;
}

static void __exit cdsp_nocclk_exit(void)
{
	if (changed && rcg_update(saved_cfg))
		pr_err("cdsp_nocclk: restore timed out\n");
	if (changed)
		report("restored");
	iounmap(gcc);
}

module_init(cdsp_nocclk_init);
module_exit(cdsp_nocclk_exit);
MODULE_LICENSE("GPL");
