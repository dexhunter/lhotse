"""Peak Python-allocator memory of one call, including buffers freed before return."""

import gc
import tracemalloc
from typing import Callable


def traced_peak_bytes(fn: Callable[[], object]) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
