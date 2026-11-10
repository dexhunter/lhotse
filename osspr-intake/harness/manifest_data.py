"""Deterministic CutSet used by the manifest load and serialize benches."""

from lhotse import AudioSource, CutSet, MonoCut, Recording
from lhotse.supervision import SupervisionSegment


def build_cutset(n: int = 1500) -> CutSet:
    cuts = []
    for i in range(n):
        cuts.append(
            MonoCut(
                id=f"cut-{i:05d}",
                start=0.0,
                duration=1.5,
                channel=0,
                recording=Recording(
                    id=f"rec-{i:05d}",
                    sources=[
                        AudioSource(
                            type="file", channels=[0], source=f"/data/{i}.wav"
                        )
                    ],
                    sampling_rate=16000,
                    num_samples=24000,
                    duration=1.5,
                ),
                supervisions=[
                    SupervisionSegment(
                        id=f"s-{i:05d}",
                        recording_id=f"rec-{i:05d}",
                        start=0.1,
                        duration=1.2,
                        channel=0,
                        text=f"transcript {i} of a short utterance",
                        speaker="spk",
                        language="en",
                    )
                ],
            )
        )
    return CutSet.from_cuts(cuts)
