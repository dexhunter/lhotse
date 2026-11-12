"""Peak RSS of audio_energy on one 8-channel minute of 16 kHz audio.

That shape is a single array-recording chunk (CHiME / AMI style). SNR mixing
calls audio_energy on the reference track and on every added track.
"""

import numpy as np
from peak_rss import peak_extra_bytes

from lhotse.audio.mixer import audio_energy

CHANNELS = 8
SAMPLE_RATE = 16000
SECONDS = 60


def main() -> None:
    audio = (
        np.random.default_rng(0)
        .standard_normal((CHANNELS, SAMPLE_RATE * SECONDS))
        .astype(np.float32)
    )

    def run() -> None:
        energy = audio_energy(audio)
        if not isinstance(energy, float) or energy <= 0.0:
            raise SystemExit(f"unexpected energy {energy!r}")

    print(peak_extra_bytes(run))


if __name__ == "__main__":
    main()
