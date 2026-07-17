from __future__ import annotations

import gzip

import numpy as np
import pytest

from raft_uav.mmuad.calibration import RigidTransform
from raft_uav.mmuad.camera import (
    CameraIntrinsics,
    CameraModel,
    load_camera_detections_csv_as_candidates,
)


def _write_png_size_header(path, *, width: int, height: int) -> None:
    raw = bytearray(24)
    raw[:8] = b"\x89PNG\r\n\x1a\n"
    raw[16:20] = int(width).to_bytes(4, "big")
    raw[20:24] = int(height).to_bytes(4, "big")
    path.write_bytes(raw)


def test_gzipped_yolo_labels_use_same_stem_image_and_decompressed_text(tmp_path) -> None:
    image_path = tmp_path / "frame_12.5.png"
    _write_png_size_header(image_path, width=200, height=100)
    label_path = tmp_path / "frame_12.5.txt.gz"
    with gzip.open(label_path, "wt", encoding="utf-8") as handle:
        handle.write("2 0.5 0.25 0.2 0.4 0.8\n")

    model = CameraModel(
        source="cam0",
        intrinsics=CameraIntrinsics(fx=200.0, fy=100.0, cx=100.0, cy=50.0),
        transform_camera_to_world=RigidTransform(
            rotation=np.eye(3),
            translation_m=np.zeros(3),
        ),
    )

    candidates = load_camera_detections_csv_as_candidates(
        label_path,
        camera_models={"cam0": model},
        default_source="cam0",
        fixed_depth_m=10.0,
    ).rows

    assert len(candidates) == 1
    row = candidates.iloc[0]
    assert float(row["time_s"]) == pytest.approx(12.5)
    assert float(row["x_m"]) == pytest.approx(0.0)
    assert float(row["y_m"]) == pytest.approx(-2.5)
    assert float(row["z_m"]) == pytest.approx(10.0)
    assert float(row["confidence"]) == pytest.approx(0.8)
    assert row["class_name"] == "2"
    assert row["track_id"] == "frame_12.5:0"
