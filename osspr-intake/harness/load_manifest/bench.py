"""Peak traced bytes of load_manifest on an untyped JSONL CutSet."""

import gc
import json
import os
import sys
import tempfile
import tracemalloc
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manifest_data import build_cutset
from lhotse import load_manifest

WARMUP = 2
REPEATS = 5


def measure(path: Path) -> list:
    def run():
        load_manifest(path)

    tracemalloc.start(1)
    try:
        for _ in range(WARMUP):
            run()
        peaks = []
        for _ in range(REPEATS):
            gc.collect()
            tracemalloc.clear_traces()
            tracemalloc.reset_peak()
            run()
            peaks.append(tracemalloc.get_traced_memory()[1])
        return peaks
    finally:
        tracemalloc.stop()


def main() -> None:
    cuts = build_cutset()
    with tempfile.TemporaryDirectory(prefix="lhotse-load-bench-") as tmp:
        path = Path(tmp) / "cuts.jsonl"
        cuts.to_jsonl(path)
        loaded = load_manifest(path)
        if loaded != cuts:
            raise SystemExit("load_manifest result is not equal to the written CutSet")
        peaks = measure(path)
    med = sorted(peaks)[len(peaks) // 2]
    noise = (max(peaks) - min(peaks)) / med * 100 if med else 0.0
    print(json.dumps({"median": med, "noise_pct": noise, "values": peaks}))


if __name__ == "__main__":
    main()
