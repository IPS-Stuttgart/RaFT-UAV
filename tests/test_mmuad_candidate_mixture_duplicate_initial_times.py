from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureMapConfig,
    _normalize_initial_estimates,
    run_candidate_mixture_map,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "candidate-0",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 1.0,
                "source": "lidar_360",
                "track_id": "candidate-1",
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
        ]
    )


def _initial_row(sequence_id: str, *, time_s: float, x_m: float) -> dict[str, object]:
    return {
        "sequence_id": sequence_id,
        "time_s": time_s,
        "state_x_m": x_m,
        "state_y_m": 0.0,
        "state_z_m": 0.0,
    }


def test_candidate_mixture_rejects_duplicate_initial_estimate_times() -> None:
    initial = pd.DataFrame(
        [
            _initial_row("seqA", time_s=0.0, x_m=0.0),
            _initial_row("seqA", time_s=0.0, x_m=10.0),
            _initial_row("seqA", time_s=1.0, x_m=1.0),
        ]
    )

    with pytest.raises(
        ValueError,
        match="at most one row per sequence_id/time_s",
    ):
        run_candidate_mixture_map(
            _candidate_rows(),
            initial_estimates=initial,
            config=CandidateMixtureMapConfig(
                top_k=1,
                score_column="ranker_score",
                sigma_column="predicted_sigma_m",
                smoothness_weight=0.0,
                anchor_weight=1.0,
                iterations=1,
            ),
        )


def test_initial_estimate_timestamp_uniqueness_is_sequence_scoped() -> None:
    initial = pd.DataFrame(
        [
            _initial_row("seqA", time_s=0.0, x_m=0.0),
            _initial_row("seqB", time_s=0.0, x_m=10.0),
        ]
    )

    normalized = _normalize_initial_estimates(initial)

    assert normalized is not None
    assert normalized[["sequence_id", "time_s"]].to_records(index=False).tolist() == [
        ("seqA", 0.0),
        ("seqB", 0.0),
    ]
