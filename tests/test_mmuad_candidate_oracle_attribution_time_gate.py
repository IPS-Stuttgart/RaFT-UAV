from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_oracle_attribution import (
    build_candidate_oracle_attribution_tables,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "source": ["radar"],
            "track_id": ["candidate"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
            "candidate_reservoir_score": [1.0],
        }
    )


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    )


@pytest.mark.parametrize(
    "invalid_gate",
    (
        -0.1,
        np.nan,
        np.inf,
        -np.inf,
        True,
        1.0 + 0.0j,
        [0.5],
        np.asarray([0.5]),
    ),
)
def test_candidate_oracle_attribution_rejects_invalid_time_gates(
    invalid_gate: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_truth_time_delta_s must be a non-negative finite scalar",
    ):
        build_candidate_oracle_attribution_tables(
            _candidate_rows(),
            _truth_rows(),
            max_truth_time_delta_s=invalid_gate,
        )


def test_candidate_oracle_attribution_accepts_zero_time_gate() -> None:
    frame_rows, pooled, _by_branch, _by_source = (
        build_candidate_oracle_attribution_tables(
            _candidate_rows(),
            _truth_rows(),
            max_truth_time_delta_s=0.0,
        )
    )

    assert len(frame_rows) == 1
    assert pooled.loc[0, "frame_count"] == 1
