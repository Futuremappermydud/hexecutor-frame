#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Boot Qualcomm's stock QuRT CDSP image through Valve's /dev/cdsp (hexagon-baremetal) driver.

No TrustZone involvement: the image is placed in dmabufs mapped at its physical load
addresses in the CDSP's SMMU domain, then the core is started at the ELF entry point,
exactly like dsp_service does for Valve's own firmware.

Requires Valve's original DTB, dsp_service stopped, and root.
Liveness signal: QuRT raising IPCC signal 2 (smp2p) or 0 (glink) towards the apps CPU.
"""
import argparse, fcntl, mmap, os, select, struct, subprocess, sys, time

FW = "/usr/lib/firmware/qcom/sm8650/cdsp.mbn"
DTB = "/usr/lib/firmware/qcom/sm8650/cdsp_dtb.mbn"

DSP_IOCTL_IRQ_STATUS = 0x4300
DSP_IOCTL_BOOT = 0x4302
DSP_IOCTL_SHUTDOWN = 0x4303
DSP_IOCTL_MAP_DMABUF = 0x4304
DSP_IOCTL_MAP_DIRECT = 0x400C4305  # _IOW('C', 5, struct map_direct{u32 va, pa, size})
UDMABUF_CREATE = 0x40187542        # _IOW('u', 0x42, struct udmabuf_create)
F_ADD_SEALS, F_SEAL_SHRINK = 1033, 0x2

# Identity mappings: Valve's CDSP set (IPCC, AOSS/RPMh, CDSP subsystem, cmd-db) plus SMEM.
DIRECT = [
    (0x00405000, 0x1000, "ipcc"),
    (0x0b000000, 0x1000000, "aoss"),
    (0x32000000, 0x1000000, "cdsp-ss"),
    (0x81c60000, 0x20000, "cmd-db"),
    (0x81d00000, 0x200000, "smem"),
]
# Fresh zeroed buffers at addresses the stock image expects (not identity: the DTB we
# boot with doesn't reserve these, so they're Linux RAM physically).
SCRATCH = [
    (0x82600000, 0x100000, "global-sync"),
]

PAGE = 0x1000
LOGF = None


def log(msg):
    """print + append to a file with fsync, so the last step survives a hard reset"""
    print(msg, flush=True)
    if LOGF:
        LOGF.write(msg + "\n")
        LOGF.flush()
        os.fsync(LOGF.fileno())


def dma_sync(fd, flags):
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    a = ctypes.create_string_buffer(struct.pack("<Q", flags))
    libc.ioctl(fd, 0x40086200, a)


def align(x, a=PAGE):
    return (x + a - 1) & ~(a - 1)


def load_segments(path):
    d = open(path, "rb").read()
    entry, phoff = struct.unpack_from("<II", d, 24)
    n, = struct.unpack_from("<H", d, 44)
    segs = []
    for i in range(n):
        t, off, va, pa, fs, ms, fl, al = struct.unpack_from("<8I", d, phoff + i * 32)
        if t != 1 or ms == 0 or ((fl >> 24) & 7) in (2, 7):  # skip hash / ELF-header segments
            continue
        segs.append((pa, d[off:off + fs], ms))
    return entry, segs


def make_dmabuf(size):
    size = align(size)
    mfd = os.memfd_create("qurt", os.MFD_ALLOW_SEALING)
    os.ftruncate(mfd, size)
    fcntl.fcntl(mfd, F_ADD_SEALS, F_SEAL_SHRINK)
    # UDMABUF_CREATE returns the new dmabuf fd as the ioctl's return value
    with open("/dev/udmabuf", "rb") as u:
        dfd = dsp_ioctl(u.fileno(), UDMABUF_CREATE, struct.pack("<IIQQ", mfd, 1, 0, size))
    m = mmap.mmap(mfd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    return mfd, dfd, m


def dsp_ioctl(fd, cmd, arg):
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if isinstance(arg, bytes):
        a = ctypes.create_string_buffer(arg)
        r = libc.ioctl(fd, cmd, a)
    else:
        r = libc.ioctl(fd, ctypes.c_ulong(cmd), ctypes.c_ulong(arg))
    if r < 0:
        raise OSError(ctypes.get_errno(), "ioctl 0x%x" % cmd)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15, help="how long to keep the DSP running")
    ap.add_argument("--no-dtb", action="store_true", help="don't load cdsp_dtb.mbn")
    ap.add_argument("--no-boot", action="store_true", help="do all mappings but don't start the core")
    ap.add_argument("--skip", default="", help="comma-separated DIRECT/SCRATCH names to leave unmapped")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds to wait between steps")
    ap.add_argument("--log", default="/home/steamos/frame-dsp/steps.log")
    ap.add_argument("--hold", action="store_true")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--fw", default=FW, help="firmware ELF to boot (e.g. Valve's cdsp_sm8650.mbn as a control)")
    args = ap.parse_args()
    global LOGF
    LOGF = open(args.log, "a")
    skip = set(filter(None, args.skip.split(",")))
    log("---- %s args=%s" % (time.ctime(), vars(args)))

    if subprocess.run(["pidof", "dsp_service"], capture_output=True).returncode == 0:
        sys.exit("dsp_service is running; stop it first (it owns /dev/cdsp)")

    dumps = []
    keep = []  # keep fds/mmaps alive while the DSP runs
    fd = os.open("/dev/cdsp", os.O_RDWR)
    log("opened /dev/cdsp"); time.sleep(args.pause)

    images = [(os.path.basename(args.fw),) + load_segments(args.fw)]
    if not args.no_dtb:
        images.append(("cdsp_dtb.mbn",) + load_segments(DTB))
    entry = images[0][1]

    for name, _, segs in images:
        lo = min(pa for pa, _, _ in segs) & ~(PAGE - 1)
        hi = align(max(pa + ms for pa, _, ms in segs))
        mfd, dfd, m = make_dmabuf(hi - lo)
        for pa, data, ms in segs:
            m[pa - lo:pa - lo + len(data)] = data
        m.flush()
        mapped = dsp_ioctl(fd, DSP_IOCTL_MAP_DMABUF, (lo << 32) | dfd)
        log("%-13s %08x-%08x mapped (%d segments, 0x%x bytes)" % (name, lo, hi, len(segs), mapped))
        keep += [mfd, dfd, m]
        dumps.append((name, lo, m, dfd))
        time.sleep(args.pause)

    for va, size, name in SCRATCH:
        if name in skip:
            continue
        mfd, dfd, m = make_dmabuf(size)
        dsp_ioctl(fd, DSP_IOCTL_MAP_DMABUF, (va << 32) | dfd)
        log("%-13s %08x-%08x scratch" % (name, va, va + size))
        keep += [mfd, dfd, m]
        time.sleep(args.pause)

    for pa, size, name in DIRECT:
        if name in skip:
            continue
        dsp_ioctl(fd, DSP_IOCTL_MAP_DIRECT, struct.pack("<III", pa, pa, size))
        log("%-13s %08x-%08x identity" % (name, pa, pa + size))
        time.sleep(args.pause)

    if args.no_boot:
        log("--no-boot: all mappings done, closing")
        os.close(fd)
        log("closed /dev/cdsp")
        return

    dsp_ioctl(fd, DSP_IOCTL_IRQ_STATUS, 0)  # clear stale signals
    log("about to BOOT at 0x%08x" % entry)
    status = dsp_ioctl(fd, DSP_IOCTL_BOOT, entry)
    log("BOOT at 0x%08x -> QDSP6SS_BOOT_STATUS=0x%x" % (entry, status))

    seen = 0
    wm = dumps[0][2]
    wdfd = dumps[0][3]
    last = None
    t0w = time.time()
    while time.time() - t0w < args.seconds:
        dma_sync(wdfd, 1)
        cur = bytes(wm[0x1a000:0x1a048]) + bytes(wm[0x1300000:0x1300048])
        if cur != last:
            log("t=%.4f dump %s" % (time.time() - t0w, cur.hex()))
            last = cur
        dma_sync(wdfd, 5)
        time.sleep(0.002)
    t0 = time.time()
    p = select.poll()
    p.register(fd, select.POLLIN)
    while time.time() - t0 < args.seconds:
        if p.poll(250):
            sig = dsp_ioctl(fd, DSP_IOCTL_IRQ_STATUS, 0)
            if sig:
                seen |= sig
                log("t=%.3fs IPCC signals from CDSP: 0x%x%s%s" % (
                    time.time() - t0, sig,
                    " [glink]" if sig & 1 else "", " [smp2p]" if sig & 4 else ""))
    if args.dump:
        for name, lo_, m_, dfd_ in dumps:
            dma_sync(dfd_, 1)
            open("/home/steamos/frame-dsp/dump_%s.bin" % name, "wb").write(m_[:])
        log("dumped")
    if args.hold:
        log("holding (no close); polling forever")
        while True:
            time.sleep(1)
    log("done; signals seen: 0x%x. Closing /dev/cdsp shuts the DSP down." % seen)
    os.close(fd)
    log("closed /dev/cdsp (DSP shut down)")


if __name__ == "__main__":
    main()
