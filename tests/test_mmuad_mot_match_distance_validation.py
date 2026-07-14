from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.mot import compute_multi_object_metrics


def _estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0"],
            "time_s": [0.0],
            "state_x_m": [1000.0],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
            "output_track_id": ["mot_1"],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
            "track_id": ["uav_1"],
        }
    )


@pytest.mark.parametrize(
    "match_distance_m",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        -1.0,
        True,
        False,
        np.bool_(True),
        np.array([25.0]),
    ],
)
def test_mot_metrics_reject_invalid_match_distances(match_distance_m: object) -> None:
    with pytest.raises(ValueError, match="match_distance_m.*finite non-negative"):
        compute_multi_object_metrics(
            _estimates(),
            _truth(),
            match_distance_m=match_distance_m,
        )


def test_mot_metrics_accept_finite_numpy_scalar_match_distance() -> None:
    metrics = compute_multi_object_metrics(
        _estimates(),
        _truth(),
        match_distance_m=np.float64(999.0),
    )

    assert metrics["matches"] == 0
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
