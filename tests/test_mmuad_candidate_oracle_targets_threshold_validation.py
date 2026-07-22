from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_oracle_targets import CandidateOracleTargetConfig
from raft_uav.mmuad.candidate_oracle_targets import build_candidate_oracle_targets


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
            "source": ["radar", "radar"],
            "track_id": ["near", "far"],
            "x_m": [0.0, 5.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )


@pytest.mark.parametrize(
    "tau",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
        True,
        1.0 + 0.0j,
        np.array([1.0]),
        np.ma.masked,
        None,
    ],
)
def test_candidate_oracle_targets_rejects_invalid_soft_tau(tau: object) -> None:
    with pytest.raises(ValueError, match="soft_tau_m"):
        build_candidate_oracle_targets(
            _candidate_rows(),
            _truth_rows(),
            config=CandidateOracleTargetConfig(soft_tau_m=(tau,)),
        )


@pytest.mark.parametrize(
    "threshold",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
        True,
        1.0 + 0.0j,
        np.array([1.0]),
        np.ma.masked,
        None,
    ],
)
def test_candidate_oracle_targets_rejects_invalid_good_threshold(
    threshold: object,
) -> None:
    with pytest.raises(ValueError, match="good_thresholds_m"):
        build_candidate_oracle_targets(
            _candidate_rows(),
            _truth_rows(),
            config=CandidateOracleTargetConfig(good_thresholds_m=(threshold,)),
        )


@pytest.mark.parametrize("field", ["soft_tau_m", "good_thresholds_m"])
def test_candidate_oracle_targets_rejects_scalar_threshold_collections(
    field: str,
) -> None:
    config = CandidateOracleTargetConfig(**{field: "1.0"})

    with pytest.raises(ValueError, match=field):
        build_candidate_oracle_targets(_candidate_rows(), _truth_rows(), config=config)


def test_candidate_oracle_targets_normalizes_valid_threshold_scalars() -> None:
    target_rows, frame_summary, summary = build_candidate_oracle_targets(
        _candidate_rows(),
        _truth_rows(),
        config=CandidateOracleTargetConfig(
            soft_tau_m=("2.5", np.array(3.0)),
            good_thresholds_m=("0", np.array(1.5)),
        ),
    )

    assert len(target_rows) == 2
    assert len(frame_summary) == 1
    assert summary["config"]["soft_tau_m"] == [2.5, 3.0]
    assert summary["config"]["good_thresholds_m"] == [0.0, 1.5]
    assert "soft_oracle_weight_tau_2p5_m" in target_rows.columns
    assert "candidate_good_le_0_m" in target_rows.columns
