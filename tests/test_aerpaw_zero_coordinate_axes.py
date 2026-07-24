from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.coordinates import LocalENUProjector
from raft_uav.io.aerpaw import normalize_rf


def test_normalize_rf_preserves_single_zero_coordinate_axes() -> None:
    rf = pd.DataFrame(
        {
            "Time": [
                "2026-01-01 00:00:01",
                "2026-01-01 00:00:02",
                "2026-01-01 00:00:03",
            ],
            "Latitude": [0.0, 10.0, 0.0],
            "Longitude": [20.0, 0.0, 0.0],
            "Elevation": [0.0, 0.0, 0.0],
            "CEP": [5.0, 5.0, 5.0],
        }
    )
    projector = LocalENUProjector(0.0, 0.0, 0.0)

    normalized = normalize_rf(
        rf,
        projector,
        pd.Timestamp("2026-01-01"),
        clock_offset_s=0.0,
    )

    assert normalized["time_s"].tolist() == [1.0, 2.0]
    assert normalized["Latitude"].tolist() == [0.0, 10.0]
    assert normalized["Longitude"].tolist() == [20.0, 0.0]
    assert np.isfinite(
        normalized[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    ).all()


def test_normalize_rf_still_drops_zero_coordinate_pair() -> None:
    rf = pd.DataFrame(
        {
            "Time": ["2026-01-01 00:00:01"],
            "Latitude": [0.0],
            "Longitude": [0.0],
            "Elevation": [0.0],
            "CEP": [5.0],
        }
    )
    projector = LocalENUProjector(0.0, 0.0, 0.0)

    normalized = normalize_rf(
        rf,
        projector,
        pd.Timestamp("2026-01-01"),
        clock_offset_s=0.0,
    )

    assert normalized.empty
