from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_scorecard import (
    build_candidate_regret_summary,
    build_pose_by_sequence_table,
)


def test_track5_scorecard_treats_float_encoded_boolean_flags_as_true() -> None:
    public_rows = pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "matched": [1.0, 0.0],
            "error_3d_m": [2.0, 99.0],
            "squared_error_3d_m2": [4.0, 9801.0],
        }
    )

    pose = build_pose_by_sequence_table(public_rows)

    assert pose["sequence_id"].tolist() == ["seq001"]
    assert int(pose.loc[0, "count"]) == 1
    assert float(pose.loc[0, "mse"]) == pytest.approx(4.0)

    candidate_gap = pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "sensor": ["lidar_360", "lidar_360"],
            "nearest_candidate_found": [1.0, 0.0],
            "selected_candidate_found": [1.0, 1.0],
            "selected_source_matches_sensor": [1.0, 0.0],
            "candidate_count_at_nearest_time": [2.0, 0.0],
            "selected_minus_truth_error_m": [1.0, 2.0],
            "nearest_minus_truth_error_m": [1.0, None],
            "candidate_regret_m": [0.0, 1.0],
            "nearest_candidate_time_delta_s": [0.0, 0.1],
        }
    )

    regret = build_candidate_regret_summary(candidate_gap)
    lidar = regret.loc[regret["sequence_id"] == "seq001"].iloc[0]

    assert float(lidar["nearest_found_fraction"]) == pytest.approx(0.5)
    assert float(lidar["selected_found_fraction"]) == pytest.approx(1.0)
    assert float(lidar["selected_source_match_fraction"]) == pytest.approx(0.5)
