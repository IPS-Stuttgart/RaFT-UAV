from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.cluster_ranker import build_cluster_feature_table
from raft_uav.mmuad.cluster_ranker import label_cluster_features_against_truth


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )


@pytest.mark.parametrize("field", ["good_threshold_m", "max_truth_time_delta_s"])
@pytest.mark.parametrize(
    "gate",
    [
        -0.1,
        np.nan,
        np.inf,
        -np.inf,
        True,
        1.0 + 0.0j,
        np.array([0.5]),
        np.ma.masked,
    ],
)
def test_cluster_ranker_rejects_invalid_truth_labeling_gates(
    field: str,
    gate: object,
) -> None:
    arguments = {
        "good_threshold_m": 1.0,
        "max_truth_time_delta_s": 0.5,
        field: gate,
    }

    with pytest.raises(ValueError, match=field):
        label_cluster_features_against_truth(
            _features(),
            _truth(),
            **arguments,
        )


def test_cluster_ranker_accepts_zero_dimensional_zero_gates() -> None:
    labeled = label_cluster_features_against_truth(
        _features(),
        _truth(),
        good_threshold_m=np.array(0.0),
        max_truth_time_delta_s=np.array(0.0),
    )

    assert labeled.loc[0, "truth_matched"]
    assert labeled.loc[0, "good_cluster"]


def test_feature_builder_uses_truth_gate_validation() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [10.0],
            "source": ["lidar_360"],
            "track_id": ["candidate"],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
            "confidence": [1.0],
        }
    )

    with pytest.raises(ValueError, match="max_truth_time_delta_s"):
        build_cluster_feature_table(
            candidates,
            truth=_truth(),
            max_truth_time_delta_s=np.nan,
        )
