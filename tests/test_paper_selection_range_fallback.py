from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.paper_selection import range_gated_radar_candidates


def test_non_strict_range_gate_fills_only_missing_native_ranges() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "frame_index": [0, 1, 2],
            "track_id": [1, 1, 1],
            # The first row must use its native range even though its ENU norm
            # exceeds the gate. Missing native ranges may use ENU fallback.
            "east_m": [1000.0, 100.0, 900.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "range_m": [100.0, np.nan, np.nan],
        }
    )

    selected = range_gated_radar_candidates(
        radar,
        range_gate_m=800.0,
        require_range_m=False,
    )

    assert selected["frame_index"].tolist() == [0, 1]
    assert selected["association_range_source"].unique().tolist() == [
        "range_m_with_enu_fallback"
    ]
