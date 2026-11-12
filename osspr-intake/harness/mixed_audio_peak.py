"""Peak RSS of AudioMixer.mixed_audio for mono noise mixed into an 8-channel minute.

The mixer is built first. The measured call is only the mix, which is what
MixedCut.load_audio does for every augmented array recording.
"""

import numpy as np
from peak_rss import peak_extra_bytes

from lhotse.audio.mixer import AudioMixer

CHANNELS = 8
SAMPLE_RATE = 16000
SECONDS = 60


def main() -> None:
    base = (
        np.random.default_rng(0)
        .standard_normal((CHANNELS, SAMPLE_RATE * SECONDS))
        .astype(np.float32)
    )
    mono_noise = np.ascontiguousarray(base[:1])
    mixer = AudioMixer(base_audio=base, sampling_rate=SAMPLE_RATE)
    mixer.add_to_mix(mono_noise, snr=10.0, offset=0.0)

    def run() -> None:
        mixed = mixer.mixed_audio
        if mixed.shape != base.shape or mixed.dtype != base.dtype:
            raise SystemExit(
                f"unexpected mix {mixed.shape} {mixed.dtype}, expected {base.shape}"
            )

    print(peak_extra_bytes(run))


if __name__ == "__main__":
    main()
