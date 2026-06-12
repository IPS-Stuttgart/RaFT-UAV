from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from raft_uav.mmuad.radar import load_radar_polar_csv_as_candidates


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
