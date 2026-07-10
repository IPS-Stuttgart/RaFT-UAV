from __future__ import annotations

import json

import pandas as pd

from raft_uav.mmuad.candidate_reservoir import ReservoirConfig
from raft_uav.mmuad.candidate_risk_reservoir import (
    attach_candidate_risk_score,
    build_risk_adjusted_reservoir,
    main as risk_reservoir_main,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqA"],
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "source": ["lidar_360", "livox_avia", "lidar_360", "livox_avia"],
            "track_id": ["low-risk-0", "high-risk-0", "low-risk-1", "high-risk-1"],
            "candidate_branch": ["raw", "translated", "raw", "translated"],
            "x_m": [0.0, 20.0, 1.0, 18.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "candidate_class_calibrated_score": [0.80, 0.90, 0.80, 0.90],
            "ranker_score": [0.80, 0.90, 0.80, 0.90],
            "predicted_sigma_m": [1.0, 20.0, 1.0, 20.0],
        }
    )


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
        }
    )


def test_risk_score_prefers_lower_uncertainty_candidate() -> None:
    scored = attach_candidate_risk_score(_candidate_rows(), uncertainty_weight=1.0)
    first = scored.rows.loc[scored.rows["time_s"] == 0.0].sort_values(
        "candidate_risk_adjusted_score",
        ascending=False,
    )

    assert first.iloc[0]["track_id"] == "low-risk-0"
    assert first.iloc[0]["candidate_risk_sigma_m"] == 1.0
    assert first.iloc[1]["candidate_risk_log_sigma_penalty"] > 0.0


def test_risk_adjusted_reservoir_changes_global_top1() -> None:
    scored, reservoir = build_risk_adjusted_reservoir(
        _candidate_rows(),
        uncertainty_weight=1.0,
        reservoir_config=ReservoirConfig(
            global_top_n=1,
            per_source_top_n=0,
            per_branch_top_n=0,
            max_candidates_per_frame=1,
            score_column="candidate_risk_adjusted_score",
        ),
    )

    assert len(scored.rows) == 4
    assert set(reservoir.rows["track_id"]) == {"low-risk-0", "low-risk-1"}


def test_risk_reservoir_cli_writes_oracle_outputs(tmp_path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    scored_csv = tmp_path / "scored.csv"
    reservoir_csv = tmp_path / "reservoir.csv"
    summary_json = tmp_path / "summary.json"
    oracle_csv = tmp_path / "oracle.csv"
    _candidate_rows().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = risk_reservoir_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--output-candidates-csv",
            str(scored_csv),
            "--output-reservoir-csv",
            str(reservoir_csv),
            "--summary-json",
            str(summary_json),
            "--truth-csv",
            str(truth_csv),
            "--oracle-summary-csv",
            str(oracle_csv),
            "--global-top-n",
            "1",
            "--per-source-top-n",
            "0",
            "--per-branch-top-n",
            "0",
            "--max-candidates-per-frame",
            "1",
            "--top-k",
            "1",
            "--max-truth-time-delta-s",
            "0.1",
        ]
    )

    assert status == 0
    assert scored_csv.exists()
    assert reservoir_csv.exists()
    assert oracle_csv.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["mode"] == "logit-minus-log-sigma"
    assert summary["reservoir_candidate_rows"] == 2
    oracle = pd.read_csv(oracle_csv)
    assert oracle.loc[0, "oracle_top1_3d_m_mse"] == 0.0


def test_probability_over_sigma_mode_is_finite_with_missing_sigma() -> None:
    rows = _candidate_rows().drop(columns=["predicted_sigma_m"])
    scored = attach_candidate_risk_score(rows, mode="probability-over-sigma")

    assert scored.rows["candidate_risk_adjusted_score"].notna().all()
    assert (scored.rows["candidate_risk_sigma_m"] == 1.0).all()
