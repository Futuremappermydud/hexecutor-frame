"""Dynamic CDSP memory reservation manager for Frame workloads."""

import fcntl
import os
from pathlib import Path
from contextlib import contextmanager

SYSFS_SIZE = Path("/sys/module/cdsp_mem/parameters/size_mb")
SYSFS_IOVA = Path("/sys/module/cdsp_mem/parameters/iova")
DEV_PATH = Path("/dev/cdsp_mem")

CDSP_MEM_MIN_SIZE_MB = 128
CDSP_MEM_MAX_SIZE_MB = 1536  # 1.5 GiB
CDSP_MEM_CHUNK_MB = 2

# _IOW('C', 1, uint32_t) -> 0x40044301
CDSP_MEM_IOC_SET_SIZE = 0x40044301
# _IOR('C', 2, uint32_t) -> 0x80044302
CDSP_MEM_IOC_GET_SIZE = 0x80044302


def _sysfs_exists() -> bool:
    return SYSFS_SIZE.exists()


def _dev_exists() -> bool:
    return DEV_PATH.exists()


def get_cdsp_mem_mb() -> int:
    """Return current cdsp_mem reservation in MiB."""
    if _sysfs_exists():
        try:
            return int(SYSFS_SIZE.read_text().strip())
        except Exception:
            pass
    if _dev_exists():
        import struct
        fd = os.open(DEV_PATH, os.O_RDWR)
        try:
            buf = bytearray(4)
            fcntl.ioctl(fd, CDSP_MEM_IOC_GET_SIZE, buf)
            return struct.unpack('I', buf)[0]
        except Exception:
            pass
        finally:
            os.close(fd)
    return CDSP_MEM_MIN_SIZE_MB


def set_cdsp_mem_mb(target_mb: int) -> int:
    """Dynamically resize cdsp_mem reservation between 128 MiB and 1536 MiB."""
    if target_mb < CDSP_MEM_MIN_SIZE_MB or target_mb > CDSP_MEM_MAX_SIZE_MB:
        raise ValueError(
            f"size {target_mb} MiB outside allowed range [{CDSP_MEM_MIN_SIZE_MB}, {CDSP_MEM_MAX_SIZE_MB}]"
        )
    if target_mb % 2 != 0:
        raise ValueError(
            f"size {target_mb} MiB must be multiple of {CDSP_MEM_CHUNK_MB} MiB chunk"
        )

    if _dev_exists():
        import struct
        fd = os.open(DEV_PATH, os.O_RDWR)
        try:
            buf = struct.pack('I', target_mb)
            fcntl.ioctl(fd, CDSP_MEM_IOC_SET_SIZE, buf)
            return target_mb
        except Exception:
            pass
        finally:
            os.close(fd)

    if _sysfs_exists():
        SYSFS_SIZE.write_text(f"{target_mb}\n")
        return int(SYSFS_SIZE.read_text().strip())

    return target_mb


@contextmanager
def cdsp_mem_lease(required_mb: int, auto_shrink: bool = True):
    """Context manager for acquiring a dynamic CDSP memory lease."""
    prev = get_cdsp_mem_mb()
    if prev < required_mb:
        set_cdsp_mem_mb(required_mb)
    try:
        yield get_cdsp_mem_mb()
    finally:
        if auto_shrink and get_cdsp_mem_mb() > prev:
            set_cdsp_mem_mb(prev)
