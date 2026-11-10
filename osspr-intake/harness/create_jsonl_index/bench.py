"""Peak traced bytes of create_jsonl_index on a fixed JSONL manifest."""

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

from lhotse.indexing import create_jsonl_index, read_index

N_LINES = 200_000
WARMUP = 2
REPEATS = 5


def write_manifest(path: Path) -> None:
    with path.open("w") as fh:
        for i in range(N_LINES):
            fh.write(f'{{"id":"c{i}","duration":1.25,"text":"hello world {i}"}}\n')


def reference_offsets(path: Path) -> list:
    offsets = []
    pos = 0
    with path.open("rb") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            offsets.append(pos)
            pos += len(line)
    offsets.append(pos)
    return offsets


def measure(jsonl: Path) -> list:
    def run():
        create_jsonl_index(jsonl)

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
    with tempfile.TemporaryDirectory(prefix="lhotse-index-bench-") as tmp:
        jsonl = Path(tmp) / "cuts.jsonl"
        write_manifest(jsonl)
        create_jsonl_index(jsonl)
        got = [int(v) for v in read_index(jsonl.with_name(jsonl.name + ".idx"))]
        if got != reference_offsets(jsonl):
            raise SystemExit("create_jsonl_index offsets diverged from a linear scan")
        peaks = measure(jsonl)
    med = sorted(peaks)[len(peaks) // 2]
    noise = (max(peaks) - min(peaks)) / med * 100 if med else 0.0
    print(json.dumps({"median": med, "noise_pct": noise, "values": peaks}))


if __name__ == "__main__":
    main()
