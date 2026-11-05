"""Baseline timing for next_fast_len.

Workload: FFT sizes for 880 distinct utterance lengths around 2-3 seconds at
16 kHz, the sizes convolve1d asks for while reverberating a batch. The
module cache is cleared before each timed block so every length takes the
factor-search path, matching a batch of lengths that have not been seen yet.
"""
import gc
import os
import statistics
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from lhotse.augmentation.utils import _NEXT_FAST_LEN, next_fast_len

REPEATS = 21
WARMUP = 5
# Padded convolution sizes for about 2.0s to 2.9s of 16 kHz audio.
SIZES = list(range(32000, 32880))


def run():
    for size in SIZES:
        next_fast_len(size)


def main():
    for _ in range(WARMUP):
        _NEXT_FAST_LEN.clear()
        run()
    samples = []
    for _ in range(REPEATS):
        gc.collect()
        _NEXT_FAST_LEN.clear()
        started = time.perf_counter()
        run()
        samples.append((time.perf_counter() - started) * 1000.0)
    print(f"{statistics.median(samples):.4f}")


if __name__ == "__main__":
    main()
