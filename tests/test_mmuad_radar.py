from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.radar import load_radar_polar_csv_as_candidates, radar_polar_frame_to_candidates


def test_radar_polar_frame_honors_explicit_radian_angle_columns() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0],
            "range_m": [10.0],
            "azimuth_rad": [np.pi / 2.0],
            "elevation_rad": [0.0],
        }
    )

    candidates = radar_polar_frame_to_candidates(frame)

    np.testing.assert_allclose(
        candidates.rows[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
        [[10.0, 0.0, 0.0]],
        atol=1.0e-9,
    )


def test_radar_polar_frame_honors_explicit_degree_angle_columns() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0],
            "range_m": [10.0],
            "bearing_deg": [90.0],
            "pitch_deg": [0.0],
        }
    )

    candidates = radar_polar_frame_to_candidates(frame, angle_unit="rad")

    np.testing.assert_allclose(
        candidates.rows[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
        [[10.0, 0.0, 0.0]],
        atol=1.0e-9,
    )


def test_radar_polar_csv_preserves_zero_padded_sequence_and_track_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "radar.csv"
    pd.DataFrame(
        {
            "Sequence": ["001"],
            "time_s": [0.0],
            "range_m": [10.0],
            "azimuth_deg": [90.0],
            "track": ["0007"],
        }
    ).to_csv(path, index=False)

    frame = load_radar_polar_csv_as_candidates(path)

    assert frame.rows["sequence_id"].tolist() == ["001"]
    assert frame.rows["track_id"].tolist() == ["0007"]
    np.testing.assert_allclose(
        frame.rows[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
        [[10.0, 0.0, 0.0]],
        atol=1.0e-9,
    )


def test_nested_radar_json_detections_inherit_parent_metadata(tmp_path: Path) -> None:
    path = tmp_path / "radar.json"
    path.write_text(
        json.dumps(
            {
                "sequence_id": "seq_radar",
                "timestamp_s": 2.5,
                "radar_detections": [
                    {"range_m": 10.0, "azimuth_deg": 90.0, "confidence": 0.8},
                    {"range_m": 20.0, "azimuth_deg": 0.0, "track_id": "north"},
                ],
            }
        ),
        encoding="utf-8",
    )

    frame = load_radar_polar_csv_as_candidates(path)

    assert len(frame.rows) == 2
    assert frame.rows["sequence_id"].tolist() == ["seq_radar", "seq_radar"]
    assert frame.rows["time_s"].tolist() == [2.5, 2.5]
    xyz = frame.rows[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    assert np.any(
        np.all(np.isclose(xyz, [10.0, 0.0, 0.0], atol=1.0e-9), axis=1)
    )
    assert np.any(
        np.all(np.isclose(xyz, [0.0, 20.0, 0.0], atol=1.0e-9), axis=1)
    )


def test_nested_radar_json_detections_inherit_parent_microsecond_timestamp(
    tmp_path: Path,
) -> None:
    path = tmp_path / "radar_us.json"
    path.write_text(
        json.dumps(
            {
                "sequence_id": "seq_radar_us",
                "timestamp_us": 1_250_000,
                "radar_detections": [
                    {"range_m": 10.0, "azimuth_deg": 0.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    frame = load_radar_polar_csv_as_candidates(path)

    assert len(frame.rows) == 1
    assert frame.rows["sequence_id"].tolist() == ["seq_radar_us"]
    assert frame.rows["time_s"].tolist() == [1.25]


def test_radar_polar_frame_fills_missing_sequence_ids_from_call_default() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": [None, ""],
            "time_s": [0.0, 1.0],
            "range_m": [10.0, 20.0],
            "azimuth_deg": [0.0, 90.0],
        }
    )

    candidates = radar_polar_frame_to_candidates(
        frame,
        default_sequence_id="seq-from-folder",
    )

    assert candidates.rows["sequence_id"].tolist() == [
        "seq-from-folder",
        "seq-from-folder",
    ]
