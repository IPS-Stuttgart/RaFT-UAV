from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_oracle_targets import CandidateOracleTargetConfig
from raft_uav.mmuad.candidate_oracle_targets import build_candidate_oracle_targets


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "source": ["radar"],
            "track_id": ["candidate"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )


def _truth_rows(*, time_s: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [time_s],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )


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
    ],
)
def test_candidate_oracle_targets_rejects_invalid_truth_time_gate(gate: object) -> None:
    with pytest.raises(ValueError, match="max_truth_time_delta_s"):
        build_candidate_oracle_targets(
            _candidate_rows(),
            _truth_rows(time_s=100.0),
            config=CandidateOracleTargetConfig(max_truth_time_delta_s=gate),
        )


def test_candidate_oracle_targets_accepts_zero_dimensional_zero_gate() -> None:
    target_rows, frame_summary, summary = build_candidate_oracle_targets(
        _candidate_rows(),
        _truth_rows(),
        config=CandidateOracleTargetConfig(
            max_truth_time_delta_s=np.array(0.0),
        ),
    )

    assert len(target_rows) == 1
    assert len(frame_summary) == 1
    assert summary["config"]["max_truth_time_delta_s"] == 0.0
