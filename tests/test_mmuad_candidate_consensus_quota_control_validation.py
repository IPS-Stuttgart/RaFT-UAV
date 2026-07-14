from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_consensus_quota import (
    build_consensus_quota_reservoir,
)


INTEGER_CONTROLS = (
    "consensus_top_n",
    "min_neighbor_count",
    "min_unique_source_count",
    "max_per_origin",
    "max_per_source",
)

WEIGHT_CONTROLS = (
    "base_score_weight",
    "consensus_weight",
    "pair_advantage_weight",
)


@pytest.mark.parametrize("field", INTEGER_CONTROLS)
@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        np.nan,
        np.inf,
        np.array([1.0]),
        None,
        "not-an-integer",
    ],
)
def test_consensus_quota_rejects_malformed_integer_controls(
    field: str,
    value: object,
) -> None:
    expected = "positive integer" if field == "max_per_origin" else "non-negative integer"
    with pytest.raises(ValueError, match=rf"{field} must be a {expected}"):
        build_consensus_quota_reservoir(
            pd.DataFrame(),
            **{field: value},
        )


@pytest.mark.parametrize("field", WEIGHT_CONTROLS)
@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        np.nan,
        np.inf,
        -np.inf,
        np.array([1.0]),
        None,
        "not-a-weight",
    ],
)
def test_consensus_quota_rejects_nonfinite_weights_before_empty_return(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be a finite scalar"):
        build_consensus_quota_reservoir(
            pd.DataFrame(),
            **{field: value},
        )


def test_consensus_quota_accepts_zero_dimensional_numeric_scalars() -> None:
    reservoir = build_consensus_quota_reservoir(
        pd.DataFrame(),
        consensus_top_n=np.array(1),
        min_neighbor_count=np.int64(1),
        min_unique_source_count=np.float64(1.0),
        max_per_origin=np.array(1),
        max_per_source=np.float32(0.0),
        time_window_s=np.array(0.05),
        time_scale_s=np.float64(0.05),
        distance_gate_m=np.float32(5.0),
        distance_scale_m=np.array(5.0),
        max_nearest_distance_m=np.float64(5.0),
        base_score_weight=np.array(-1.0),
        consensus_weight=np.float32(1.0),
        pair_advantage_weight=np.int64(0),
    )

    assert reservoir.rows.empty
