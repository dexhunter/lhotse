"""Time RecordingSet.__getitem__ for 48 distinct ids at the tail of a large set.

An eager manifest of 450,000 recordings is the size of a combined corpus
(LibriSpeech plus a few public sets). Each lookup walks that list. Forty-eight
tail ids is one batch of cuts resolving recordings stored late in the
manifest. The printed number is milliseconds for those forty-eight calls.
"""

import gc
import time

from lhotse.audio.recording import AudioSource, Recording
from lhotse.audio.recording_set import RecordingSet

N_RECORDINGS = 450_000
N_LOOKUPS = 48


def main() -> None:
    recordings = [
        Recording(
            id=f"r{i}",
            sources=[AudioSource(type="command", channels=[0], source=":")],
            sampling_rate=16000,
            num_samples=1600,
            duration=0.1,
        )
        for i in range(N_RECORDINGS)
    ]
    recording_set = RecordingSet.from_recordings(recordings)
    ids = [f"r{i}" for i in range(N_RECORDINGS - N_LOOKUPS, N_RECORDINGS)]

    def resolve() -> None:
        for recording_id in ids:
            found = recording_set[recording_id]
            if found.id != recording_id:
                raise SystemExit(f"lookup returned {found.id}, expected {recording_id}")

    resolve()
    gc.collect()
    started = time.perf_counter()
    resolve()
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"{elapsed_ms:.6f}")


if __name__ == "__main__":
    main()
