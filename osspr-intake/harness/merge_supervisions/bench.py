"""Object allocations of MonoCut.merge_supervisions on one long session cut."""

import gc
import json
import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from count_allocs import count_call
from lhotse import AudioSource, MonoCut, Recording
from lhotse.supervision import AlignmentItem, SupervisionSegment

TURNS = 400
WORDS = 48
WARMUP = 3
REPEATS = 5


def build_cut() -> MonoCut:
    supervisions = []
    for turn in range(TURNS):
        alignment = [
            AlignmentItem(
                symbol=f"w{turn}-{word}",
                start=turn + word * 0.01,
                duration=0.01,
                score=0.5,
            )
            for word in range(WORDS)
        ]
        supervisions.append(
            SupervisionSegment(
                id=f"seg-{turn:04d}",
                recording_id="rec-1",
                start=float(turn),
                duration=0.5,
                channel=0,
                text=f"turn number {turn} says hello",
                speaker=f"spk{turn % 7}",
                language="en",
                gender="f" if turn % 2 == 0 else "m",
                custom={"field": f"value-{turn}"},
                alignment={"word": alignment},
            )
        )
    recording = Recording(
        id="rec-1",
        sources=[AudioSource(type="file", channels=[0], source="dummy.wav")],
        sampling_rate=16000,
        num_samples=16000 * (TURNS + 1),
        duration=float(TURNS + 1),
    )
    return MonoCut(
        id="session",
        start=0.0,
        duration=float(TURNS),
        channel=0,
        recording=recording,
        supervisions=supervisions,
    )


def measure(cut: MonoCut) -> list:
    def run():
        cut.merge_supervisions()

    for _ in range(WARMUP):
        count_call(run)
    counts = []
    for _ in range(REPEATS):
        gc.collect()
        _, n = count_call(run)
        counts.append(int(n))
    return counts


def main() -> None:
    cut = build_cut()
    original = cut.to_dict()
    first = cut.merge_supervisions()
    second = cut.merge_supervisions()
    if first != second:
        raise SystemExit("merge_supervisions results differ across calls")
    if cut.to_dict() != original:
        raise SystemExit("merge_supervisions mutated the input cut")
    if len(first.supervisions) != 1:
        raise SystemExit("expected a single merged supervision")
    words = first.supervisions[0].alignment["word"]
    if len(words) != TURNS * WORDS:
        raise SystemExit("merged alignment lost items")
    if any(words is source.alignment["word"] for source in cut.supervisions):
        raise SystemExit("merged word alignment aliases a source list")
    counts = measure(cut)
    med = sorted(counts)[len(counts) // 2]
    noise = (max(counts) - min(counts)) / med * 100 if med else 0.0
    print(json.dumps({"median": med, "noise_pct": noise, "values": counts}))


if __name__ == "__main__":
    main()
