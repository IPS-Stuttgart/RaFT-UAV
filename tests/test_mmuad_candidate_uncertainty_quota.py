from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_reservoir import ReservoirConfig
from raft_uav.mmuad.candidate_uncertainty_quota import build_uncertainty_quota_reservoir


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [1.0] * 4,
            "source": ["livox"] * 4,
            "candidate_branch": ["raw"] * 4,
            "track_id": ["score", "sigma", "other", "bad_sigma"],
            "x_m": [0.0, 10.0, 20.0, 30.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "ranker_score": [0.99, 0.30, 0.20, 0.10],
            "confidence": [0.99, 0.30, 0.20, 0.10],
            "predicted_sigma_m": [20.0, 1.0, 5.0, -1.0],
        }
    )


def test_uncertainty_quota_preserves_low_sigma_candidate() -> None:
    reservoir = build_uncertainty_quota_reservoir(
        _rows(),
        reservoir_config=ReservoirConfig(
            global_top_n=1,
            per_source_top_n=0,
            per_branch_top_n=0,
            max_candidates_per_frame=2,
        ),
        per_source_branch_top_n=1,
        uncertainty_top_n=1,
    ).rows

    assert set(reservoir["track_id"]) == {"score", "sigma"}
    sigma_row = reservoir.loc[reservoir["track_id"] == "sigma"].iloc[0]
    assert bool(sigma_row["candidate_uncertainty_quota_selected"])
    assert "source_branch_uncertainty:livox|raw" in sigma_row[
        "candidate_reservoir_reason"
    ]


def test_uncertainty_quota_disabled_matches_score_driven_cell_choice() -> None:
    reservoir = build_uncertainty_quota_reservoir(
        _rows(),
        reservoir_config=ReservoirConfig(
            global_top_n=0,
            per_source_top_n=0,
            per_branch_top_n=0,
            max_candidates_per_frame=1,
        ),
        per_source_branch_top_n=1,
        uncertainty_top_n=0,
    ).rows

    assert reservoir["track_id"].tolist() == ["score"]


def test_uncertainty_quota_ignores_nonpositive_sigma() -> None:
    rows = _rows().copy()
    rows.loc[rows["track_id"] == "sigma", "predicted_sigma_m"] = 0.0

    reservoir = build_uncertainty_quota_reservoir(
        rows,
        reservoir_config=ReservoirConfig(max_candidates_per_frame=4),
        per_source_branch_top_n=0,
        uncertainty_top_n=1,
    ).rows

    selected = reservoir.loc[reservoir["candidate_uncertainty_quota_selected"].fillna(False)]
    assert selected["track_id"].tolist() == ["other"]


def test_uncertainty_quota_rejects_negative_budget() -> None:
    try:
        build_uncertainty_quota_reservoir(_rows(), uncertainty_top_n=-1)
    except ValueError as exc:
        assert "uncertainty_top_n" in str(exc)
    else:
        raise AssertionError("expected negative uncertainty quota to fail")
