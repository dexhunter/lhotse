"""Baseline timing for find_segments_with_speaker_count.

Workload: one pass over 300 session cuts, 40 overlapping speaker turns each.
That is the sweep describe() repeats while counting speakers. Each call rebuilds
and sorts every boundary.
"""
import gc
import os
import statistics
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from lhotse.cut import CutSet, MonoCut
from lhotse.cut.describe import find_segments_with_speaker_count
from lhotse.supervision import SupervisionSegment

REPEATS = 21
WARMUP = 5
N_CUTS = 300
N_SUPERVISIONS = 40


def build():
    cuts = []
    for cut_index in range(N_CUTS):
        supervisions = [
            SupervisionSegment(
                id=f"c{cut_index}-s{segment_index}",
                recording_id=f"rec{cut_index}",
                start=(segment_index % 8) * 0.7,
                duration=1.2,
                channel=0,
                speaker=f"spk{segment_index % 5}",
                text="hello there",
            )
            for segment_index in range(N_SUPERVISIONS)
        ]
        cuts.append(
            MonoCut(
                id=f"c{cut_index}",
                start=0.0,
                duration=12.0,
                channel=0,
                supervisions=supervisions,
            )
        )
    return CutSet.from_cuts(cuts)


def main():
    cuts = build()

    def run():
        for cut in cuts:
            find_segments_with_speaker_count(cut, min_speakers=1)

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
