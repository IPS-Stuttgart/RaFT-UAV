from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.radar import (
    load_radar_polar_csv_as_candidates,
    radar_polar_frame_to_candidates,
)


def test_radar_polar_frame_drops_negative_ranges_but_keeps_zero() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "range_m": [10.0, -5.0, 0.0],
            "azimuth_deg": [90.0, 90.0, 45.0],
            "elevation_deg": [0.0, 0.0, 0.0],
        }
    )

    candidates = radar_polar_frame_to_candidates(frame)

    assert candidates.rows["time_s"].tolist() == [0.0, 2.0]
    np.testing.assert_allclose(
        candidates.rows[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
        [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        atol=1.0e-9,
    )


def test_radar_polar_csv_drops_serialized_negative_ranges(tmp_path: Path) -> None:
    path = tmp_path / "radar.csv"
    pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "range_m": ["-12.5", "8.0"],
            "azimuth_deg": [0.0, 0.0],
        }
    ).to_csv(path, index=False)

    candidates = load_radar_polar_csv_as_candidates(path)

    assert candidates.rows["time_s"].tolist() == [1.0]
    np.testing.assert_allclose(
        candidates.rows[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
        [[0.0, 8.0, 0.0]],
        atol=1.0e-9,
    )
