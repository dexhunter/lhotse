"""Object allocations of JsonMixin.to_json for an in-memory CutSet."""

import gc
import json
import os
import sys
import tempfile
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from count_allocs import count_call
from manifest_data import build_cutset

WARMUP = 3
REPEATS = 5


def measure(cuts, path: Path) -> list:
    def run():
        cuts.to_json(path)

    for _ in range(WARMUP):
        count_call(run)
    counts = []
    for _ in range(REPEATS):
        gc.collect()
        _, n = count_call(run)
        counts.append(int(n))
    return counts


def main() -> None:
    cuts = build_cutset()
    with tempfile.TemporaryDirectory(prefix="lhotse-tojson-bench-") as tmp:
        first = Path(tmp) / "a.json"
        second = Path(tmp) / "b.json"
        cuts.to_json(first)
        cuts.to_json(second)
        if first.read_bytes() != second.read_bytes():
            raise SystemExit("to_json output bytes are not stable")
        counts = measure(cuts, Path(tmp) / "out.json")
    med = sorted(counts)[len(counts) // 2]
    noise = (max(counts) - min(counts)) / med * 100 if med else 0.0
    print(json.dumps({"median": med, "noise_pct": noise, "values": counts}))


if __name__ == "__main__":
    main()
