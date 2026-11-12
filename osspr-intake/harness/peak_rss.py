"""Sample process RSS while a call runs.

Large NumPy and Torch buffers are often mmap'd and released before the call
returns, so a before/after reading misses them. The sampler records the
high-water RSS during the call and reports how far it rose above the RSS
just before the call (inputs already allocated).
"""

import gc
import os
import threading
from typing import Callable


def _rss_bytes() -> int:
    with open("/proc/self/statm") as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def peak_extra_bytes(fn: Callable[[], object]) -> int:
    gc.collect()
    stop = threading.Event()
    base = _rss_bytes()
    peak = base

    def sample() -> None:
        nonlocal peak
        while not stop.is_set():
            current = _rss_bytes()
            if current > peak:
                peak = current

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        fn()
    finally:
        stop.set()
        worker.join()
    return peak - base
