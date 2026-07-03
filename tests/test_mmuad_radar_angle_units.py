from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.radar import radar_polar_frame_to_candidates


def _first_xyz(frame: pd.DataFrame, *, angle_unit: str) -> np.ndarray:
    candidates = radar_polar_frame_to_candidates(frame, angle_unit=angle_unit)
    return candidates.rows.iloc[0][["x_m", "y_m", "z_m"]].to_numpy(dtype=float)


def test_radar_polar_frame_respects_explicit_radian_angle_columns() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0],
            "range_m": [10.0],
            "azimuth_rad": [np.pi / 2.0],
            "elevation_rad": [0.0],
        }
    )

    xyz = _first_xyz(frame, angle_unit="deg")

    assert np.allclose(xyz, [10.0, 0.0, 0.0], atol=1.0e-9)


def test_radar_polar_frame_respects_explicit_degree_aliases_with_radian_default() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0],
            "range_m": [10.0],
            "bearing_deg": [90.0],
            "pitch_deg": [0.0],
        }
    )

    xyz = _first_xyz(frame, angle_unit="rad")

    assert np.allclose(xyz, [10.0, 0.0, 0.0], atol=1.0e-9)
