"""Peak RSS of Clipping.__call__ on one 8-channel minute of 16 kHz audio.

Soft clipping with a +6 dB pre-gain is the augmentation attached to a
recording and applied inside load_audio.
"""

import numpy as np
from peak_rss import peak_extra_bytes

from lhotse.augmentation.clipping import Clipping

CHANNELS = 8
SAMPLE_RATE = 16000
SECONDS = 60


def main() -> None:
    audio = (
        np.random.default_rng(0)
        .standard_normal((CHANNELS, SAMPLE_RATE * SECONDS))
        .astype(np.float32)
    )
    clipping = Clipping(hard=False, gain_db=6.0, normalize=True)

    def run() -> None:
        clipped = clipping(audio, SAMPLE_RATE)
        if clipped.shape != audio.shape or clipped.dtype != audio.dtype:
            raise SystemExit(
                f"unexpected clip {clipped.shape} {clipped.dtype}, expected {audio.shape}"
            )

    print(peak_extra_bytes(run))


if __name__ == "__main__":
    main()
