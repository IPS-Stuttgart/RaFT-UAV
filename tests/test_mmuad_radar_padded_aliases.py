from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.radar import radar_polar_frame_to_candidates


def test_radar_polar_frame_strips_padded_column_aliases() -> None:
    frame = pd.DataFrame(
        {
            " Time_S ": [0.0],
            " Range_M ": [10.0],
            " Azimuth_Deg ": [90.0],
            " Elevation_Deg ": [0.0],
            " Confidence ": [0.5],
        }
    )

    candidates = radar_polar_frame_to_candidates(frame)

    assert candidates.rows["time_s"].tolist() == [0.0]
    assert candidates.rows["confidence"].tolist() == [0.5]
    np.testing.assert_allclose(
        candidates.rows[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
        [[10.0, 0.0, 0.0]],
        atol=1.0e-9,
    )
