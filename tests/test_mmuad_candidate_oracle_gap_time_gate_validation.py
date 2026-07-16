from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_oracle_gap import build_candidate_oracle_gap


def _truth_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )


def _candidate_frame(time_s: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001"],
            "time_s": [time_s],
            "source": ["lidar_360"],
            "track_id": ["candidate"],
            "x_m": [1.0],
            "y_m": [0.0],
            "z_m": [0.0],
            "confidence": [0.8],
        }
    )


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        np.bool_(False),
        1 + 0j,
        np.array([0.5]),
        "invalid",
    ],
)
def test_candidate_oracle_gap_rejects_malformed_time_gates(value: object) -> None:
    with pytest.raises(ValueError, match="max_time_delta_s"):
        build_candidate_oracle_gap(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            max_time_delta_s=value,
        )


def test_candidate_oracle_gap_none_time_gate_allows_distant_candidates() -> None:
    candidates = _candidate_frame(5.0)

    rows = build_candidate_oracle_gap(
        candidates,
        candidates,
        _truth_frame(),
        max_time_delta_s=None,
    )

    assert len(rows) == 1
    assert bool(rows.loc[0, "nearest_candidate_found"])
    assert bool(rows.loc[0, "selected_candidate_found"])
    assert float(rows.loc[0, "nearest_candidate_time_delta_s"]) == pytest.approx(5.0)
