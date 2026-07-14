from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.mot import _greedy_truth_matches, compute_multi_object_metrics


@pytest.mark.parametrize(
    "invalid_distance",
    [float("nan"), float("inf"), float("-inf"), -1.0, True, False, [1.0]],
)
def test_multi_object_metrics_rejects_invalid_match_distance(invalid_distance) -> None:
    with pytest.raises(ValueError, match="match_distance_m must be finite and nonnegative"):
        compute_multi_object_metrics(pd.DataFrame(), None, match_distance_m=invalid_distance)


def test_greedy_truth_matches_rejects_invalid_distance_directly() -> None:
    with pytest.raises(ValueError, match="match_distance_m must be finite and nonnegative"):
        _greedy_truth_matches(pd.DataFrame(), pd.DataFrame(), max_distance_m=np.nan)


def test_multi_object_metrics_accepts_zero_match_distance() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq0"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
            "output_track_id": ["estimate_1"],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq0"],
            "time_s": [0.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
            "track_id": ["truth_1"],
        }
    )

    metrics = compute_multi_object_metrics(estimates, truth, match_distance_m=0.0)

    assert metrics["matches"] == 1
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0
    assert metrics["motp_3d_m"] == pytest.approx(0.0)
