from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.radar import radar_polar_frame_to_candidates


def test_radar_polar_frame_accepts_plain_distance_alias() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0],
            "distance": [10.0],
            "azimuth_deg": [90.0],
            "elevation_deg": [0.0],
        }
    )

    candidates = radar_polar_frame_to_candidates(frame)

    np.testing.assert_allclose(
        candidates.rows[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
        [[10.0, 0.0, 0.0]],
        atol=1.0e-9,
    )
