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


@pytest.mark.parametrize("config", [False, 0, "", {}, []])
def test_candidate_oracle_targets_rejects_non_config_objects(config: object) -> None:
    with pytest.raises(TypeError, match="CandidateOracleTargetConfig"):
        build_candidate_oracle_targets(
            _candidate_rows(),
            _truth_rows(),
            config=config,
        )


@pytest.mark.parametrize("score_column", [None, "", "   ", True, 1])
def test_candidate_oracle_targets_rejects_invalid_score_column(
    score_column: object,
) -> None:
    config = CandidateOracleTargetConfig(score_column=score_column)

    with pytest.raises(ValueError, match="score_column"):
        build_candidate_oracle_targets(_candidate_rows(), _truth_rows(), config=config)


@pytest.mark.parametrize(
    "fallback_score_columns",
    [
        None,
        "ranker_score",
        ("ranker_score", ""),
        ("ranker_score", True),
    ],
)
def test_candidate_oracle_targets_rejects_invalid_fallback_score_columns(
    fallback_score_columns: object,
) -> None:
    config = CandidateOracleTargetConfig(
        fallback_score_columns=fallback_score_columns,
    )

    with pytest.raises(ValueError, match="fallback_score_columns"):
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


def test_candidate_oracle_targets_normalizes_score_column_names() -> None:
    candidates = _candidate_rows().assign(
        ranker_score=[0.1, 0.9],
        confidence=[0.8, 0.2],
    )

    target_rows, _, summary = build_candidate_oracle_targets(
        candidates,
        _truth_rows(),
        config=CandidateOracleTargetConfig(
            score_column=" ranker_score ",
            fallback_score_columns=(" confidence ",),
        ),
    )

    config_summary = summary["config"]
    assert config_summary["score_column"] == "ranker_score"
    assert config_summary["fallback_score_columns"] == ["confidence"]
    near_rank = target_rows.loc[target_rows["track_id"] == "near", "candidate_score_rank"]
    assert int(near_rank.iloc[0]) == 2
