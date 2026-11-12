"""Peak RSS of _apply_sinc_resample_kernel on 8-channel 48 kHz audio.

Thirty seconds down to 16 kHz is one recording resample before feature
extraction. The sinc kernel is built first; the measured call is the apply
used by Resample.forward.
"""

import numpy as np
import torch
from peak_rss import peak_extra_bytes

from lhotse.augmentation.resample import Resample, _apply_sinc_resample_kernel

CHANNELS = 8
SOURCE_RATE = 48000
TARGET_RATE = 16000
SECONDS = 30


def main() -> None:
    samples = (
        np.random.default_rng(0)
        .standard_normal((CHANNELS, SOURCE_RATE * SECONDS))
        .astype(np.float32)
    )
    waveform = torch.from_numpy(samples)
    resampler = Resample(SOURCE_RATE, TARGET_RATE)
    expected = (CHANNELS, TARGET_RATE * SECONDS)

    def run() -> None:
        resampled = _apply_sinc_resample_kernel(
            waveform,
            resampler.orig_freq,
            resampler.new_freq,
            resampler.gcd,
            resampler.kernel,
            resampler.width,
        )
        if tuple(resampled.shape) != expected or resampled.dtype != waveform.dtype:
            raise SystemExit(
                f"unexpected resample {tuple(resampled.shape)} {resampled.dtype}"
            )

    print(peak_extra_bytes(run))


if __name__ == "__main__":
    main()
