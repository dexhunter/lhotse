"""Baseline timing for SupervisionSet.with_alignment_from_ctm.

Workload: four long recordings, 80 supervision turns each, and 500 CTM words
per recording. Each turn scans every word on that recording.
"""
import gc
import os
import statistics
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from lhotse.supervision import SupervisionSegment, SupervisionSet

REPEATS = 21
WARMUP = 5


def build():
    segments = []
    lines = []
    for recording_index in range(4):
        recording_id = f"reco{recording_index}"
        for segment_index in range(80):
            segments.append(
                SupervisionSegment(
                    id=f"{recording_id}-seg-{segment_index}",
                    recording_id=recording_id,
                    start=segment_index * 1.5,
                    duration=2.0,
                    channel=0,
                    text="the quick brown fox",
                    speaker=f"spk{segment_index % 6}",
                )
            )
        cursor = 0.0
        for word_index in range(500):
            lines.append(f"{recording_id} 0 {cursor:.3f} 0.250 w{word_index}\n")
            cursor += 0.25
    handle = tempfile.NamedTemporaryFile("w", suffix=".ctm", delete=False)
    handle.writelines(lines)
    handle.close()
    return SupervisionSet.from_segments(segments), handle.name


def main():
    supervisions, ctm_path = build()

    def run():
        supervisions.with_alignment_from_ctm(ctm_path, type="word")

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
