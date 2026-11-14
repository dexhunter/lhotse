"""Peak traced memory of LilcomChunkyReader.read on a 10-minute fbank.

60,000 frames by 80 dims is 10 minutes of 16 kHz filterbanks at a 10 ms shift,
stored with the chunked lilcom writer (500 frames per chunk). The returned
matrix has to match that shape and stay within lilcom's tick tolerance.
"""

import tempfile
from pathlib import Path

import numpy as np
from trace_peak import traced_peak_bytes

from lhotse.features.io import LilcomChunkyReader, LilcomChunkyWriter

FRAMES = 60_000
FEATURES = 80
# tick_power=-5 quantizes to multiples of 2**-5.
TOLERANCE = 2**-5 + 1e-6


def main() -> None:
    features = (
        np.random.default_rng(0).standard_normal((FRAMES, FEATURES)).astype(np.float32)
    )
    with tempfile.TemporaryDirectory() as directory:
        with LilcomChunkyWriter(Path(directory) / "feats") as writer:
            key = writer.write("meeting", features)
            storage_path = writer.storage_path
        reader = LilcomChunkyReader(storage_path)

        decoded = {}

        def read_features() -> None:
            decoded["value"] = reader.read(key)

        peak = traced_peak_bytes(read_features)

    restored = decoded["value"]
    if restored.shape != features.shape or restored.dtype != features.dtype:
        raise SystemExit(
            f"unexpected read {restored.shape} {restored.dtype}, "
            f"expected {features.shape} {features.dtype}"
        )
    max_abs = float(np.max(np.abs(restored - features)))
    if max_abs > TOLERANCE:
        raise SystemExit(f"lilcom round trip drifted by {max_abs}")

    print(peak)


if __name__ == "__main__":
    main()
