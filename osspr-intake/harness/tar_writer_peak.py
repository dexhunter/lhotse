"""Peak traced memory of TarWriter.write for one large shard member.

AudioTarWriter and ArrayTarWriter hand the encoded bytes to TarWriter.write.
One minute of 8-channel 16 kHz float32 PCM is 30,720,000 bytes, which is the
member measured here. The call must put those bytes in the tar unchanged.
"""

import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

from trace_peak import traced_peak_bytes

from lhotse.shar.writers.tar import TarWriter

PAYLOAD_BYTES = 8 * 16000 * 60 * 4  # 8 ch * 16 kHz * 60 s * float32


def main() -> None:
    payload = bytes(PAYLOAD_BYTES)
    blob = BytesIO()
    blob.write(payload)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "shard.tar"
        with TarWriter(str(path), shard_size=None) as writer:
            writer.write("open.bin", BytesIO(b"\x00"))

            def write_member() -> None:
                blob.seek(0)
                writer.write("audio.f32", blob)

            peak = traced_peak_bytes(write_member)

        with tarfile.open(path) as archive:
            stored = archive.extractfile(archive.getmember("audio.f32")).read()
        if stored != payload:
            raise SystemExit(
                f"tar member changed: {len(stored)} bytes, expected {len(payload)}"
            )

    print(peak)


if __name__ == "__main__":
    main()
