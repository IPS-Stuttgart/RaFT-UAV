from __future__ import annotations

import struct
from pathlib import Path
import zlib

import numpy as np

from raft_uav.mmuad.image_evidence import build_image_evidence


def _write_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(pixels, dtype=np.uint8)
    height, width = pixels.shape[:2]

    def chunk(name: bytes, data: bytes) -> bytes:
        payload = name + data
        checksum = zlib.crc32(payload) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", checksum)

    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_image_evidence_counts_image_derived_targets_without_truth(tmp_path: Path) -> None:
    image_dir = tmp_path / "mmuad" / "seq001" / "Image"
    pixels = np.full((4, 4, 3), 64, dtype=np.uint8)
    _write_png(image_dir / "0.0.png", pixels)
    _write_png(image_dir / "1.0.png", pixels)

    result = build_image_evidence(
        tmp_path / "mmuad",
        max_frames_per_sequence=8,
        max_image_time_delta_s=0.1,
    )

    summary = result.sequence_features.iloc[0]
    assert summary["image_sampled_frame_count"] == 2
    assert summary["image_target_count"] == 2
    assert summary["image_matched_target_fraction"] == 1.0
