from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_branch_consensus import (
    attach_candidate_branch_consensus,
)


NUMERIC_CONTROL_FIELDS = (
    "time_window_s",
    "time_scale_s",
    "distance_gate_m",
    "distance_scale_m",
    "base_score_weight",
    "consensus_weight",
    "pair_advantage_weight",
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [0.0, 0.0],
            "source": ["lidar_360", "livox_avia"],
            "track_id": ["lidar", "livox"],
            "x_m": [0.0, 0.5],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "ranker_score": [0.8, 0.7],
            "confidence": [0.8, 0.7],
        }
    )


@pytest.mark.parametrize("field", NUMERIC_CONTROL_FIELDS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_branch_consensus_rejects_nonfinite_numeric_controls(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        attach_candidate_branch_consensus(
            _candidate_rows(),
            **{field: value},
        )


def test_branch_consensus_validates_controls_before_empty_input_return() -> None:
    with pytest.raises(ValueError, match="consensus_weight"):
        attach_candidate_branch_consensus(
            pd.DataFrame(),
            consensus_weight=np.nan,
        )


@pytest.mark.parametrize("value", [True, np.bool_(False), np.array([0.1])])
def test_branch_consensus_rejects_boolean_and_nonscalar_controls(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="time_window_s"):
        attach_candidate_branch_consensus(
            _candidate_rows(),
            time_window_s=value,
        )


def test_branch_consensus_accepts_finite_numpy_scalar_controls() -> None:
    augmented = attach_candidate_branch_consensus(
        _candidate_rows(),
        time_window_s=np.array(0.01),
        time_scale_s=np.float64(0.005),
        distance_gate_m=np.float32(2.0),
        distance_scale_m=np.array(2.0),
        base_score_weight=np.float64(1.0),
        consensus_weight=np.float32(2.0),
        pair_advantage_weight=np.array(0.25),
    ).rows

    assert np.isfinite(augmented["branch_consensus_rank_score"]).all()
