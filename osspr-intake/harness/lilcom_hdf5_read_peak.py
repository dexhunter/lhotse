"""Peak traced memory of ChunkedLilcomHdf5Reader.read on a 10-minute fbank.

Same 60,000 by 80 matrix as the lilcom-chunky harness, stored with the HDF5
chunked lilcom writer (500 frames per chunk). The returned matrix has to
match that shape and stay within lilcom's tick tolerance.
"""

import tempfile
from pathlib import Path

import numpy as np
from trace_peak import traced_peak_bytes

from lhotse.features.io import ChunkedLilcomHdf5Reader, ChunkedLilcomHdf5Writer

FRAMES = 60_000
FEATURES = 80
TOLERANCE = 2**-5 + 1e-6


def main() -> None:
    features = (
        np.random.default_rng(0).standard_normal((FRAMES, FEATURES)).astype(np.float32)
    )
    with tempfile.TemporaryDirectory() as directory:
        with ChunkedLilcomHdf5Writer(
            Path(directory) / "feats.h5", chunk_size=500
        ) as writer:
            key = writer.write("meeting", features)
            storage_path = writer.storage_path
        reader = ChunkedLilcomHdf5Reader(storage_path)

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
