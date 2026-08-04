from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.paper_selection import (
    select_paper_compatible_radar_track,
    select_paper_strict_raw_radar_track,
)


def _radar_frame(times: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": times,
            "track_id": [7] * len(times),
            "track_index": [0] * len(times),
            "range_m": [100.0] * len(times),
            "east_m": [10.0] * len(times),
            "north_m": [20.0] * len(times),
            "up_m": [30.0] * len(times),
            "cat_prob_uav": [0.9] * len(times),
        }
    )


def test_paper_track_selection_sorts_numeric_string_times_numerically() -> None:
    numeric = _radar_frame([10.0, 1.0, 2.0, 3.0])
    strings = numeric.assign(time_s=numeric["time_s"].astype(str))

    expected_strict = select_paper_strict_raw_radar_track(numeric)
    actual_strict = select_paper_strict_raw_radar_track(strings)
    expected_compatible = select_paper_compatible_radar_track(
        numeric,
        range_gate_m=800.0,
        catprob_threshold=0.5,
    )
    actual_compatible = select_paper_compatible_radar_track(
        strings,
        range_gate_m=800.0,
        catprob_threshold=0.5,
    )

    for expected, actual in (
        (expected_strict, actual_strict),
        (expected_compatible, actual_compatible),
    ):
        np.testing.assert_allclose(
            pd.to_numeric(actual["time_s"]).to_numpy(dtype=float),
            expected["time_s"].to_numpy(dtype=float),
        )
        assert pd.to_numeric(actual["time_s"]).tolist() == [1.0, 2.0, 3.0]
        assert actual["track_id"].tolist() == [7, 7, 7]
