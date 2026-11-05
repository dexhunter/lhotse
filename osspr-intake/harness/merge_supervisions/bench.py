"""Baseline timing for MonoCut.merge_supervisions.

Workload: one session-level cut whose turns do not overlap. Each turn carries
a word alignment. Merging concatenates those alignment lists with repeated
list addition, which copies the growing result on every turn.
"""
import gc
import os
import statistics
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from lhotse.audio import AudioSource, Recording
from lhotse.cut import MonoCut
from lhotse.supervision import AlignmentItem, SupervisionSegment

REPEATS = 21
WARMUP = 5
N_SUPERVISIONS = 520
N_WORDS = 32


def build():
    supervisions = []
    for index in range(N_SUPERVISIONS):
        start = index * 0.5
        words = [
            AlignmentItem(
                symbol=f"w{index}_{word_index}",
                start=start + word_index * 0.01,
                duration=0.01,
            )
            for word_index in range(N_WORDS)
        ]
        supervisions.append(
            SupervisionSegment(
                id=f"seg-{index}",
                recording_id="rec",
                start=start,
                duration=0.4,
                channel=0,
                text=f"token{index}",
                speaker=f"spk{index % 3}",
                language="en",
                gender="M",
                custom={"utt": f"id{index}"},
                alignment={"word": words},
            )
        )
    duration = N_SUPERVISIONS * 0.5
    recording = Recording(
        id="rec",
        sources=[AudioSource(type="command", channels=[0], source=":")],
        sampling_rate=16000,
        num_samples=int(duration * 16000),
        duration=duration,
    )
    return MonoCut(
        id="meeting",
        start=0.0,
        duration=duration,
        channel=0,
        recording=recording,
        supervisions=supervisions,
    )


def main():
    cut = build()

    def run():
        cut.merge_supervisions(merge_policy="delimiter")

    for _ in range(WARMUP):
        run()
    samples = []
    for _ in range(REPEATS):
        gc.collect()
        started = time.perf_counter()
        run()
        samples.append((time.perf_counter() - started) * 1000.0)
    print(f"{statistics.median(samples):.4f}")


if __name__ == "__main__":
    main()
