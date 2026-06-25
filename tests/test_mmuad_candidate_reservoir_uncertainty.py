from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_reservoir_uncertainty import (
    main as uncertainty_main,
    uncertainty_aware_cap_reservoir,
    uncertainty_cap_summary,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 6,
            "time_s": [0.0] * 6,
            "source": [
                "lidar_360",
                "lidar_360",
                "lidar_360",
                "lidar_360",
                "livox_avia",
                "livox_avia",
            ],
            "track_id": ["score1", "score2", "score3", "low_sigma", "livox1", "livox2"],
            "candidate_branch": [
                "translated",
                "translated",
                "translated",
                "raw",
                "raw",
                "dynamic",
            ],
            "x_m": [10.0, 11.0, 12.0, 0.0, 20.0, 21.0],
            "y_m": [0.0] * 6,
            "z_m": [1.0] * 6,
            "candidate_reservoir_score": [0.99, 0.98, 0.97, 0.05, 0.80, 0.70],
            "confidence": [0.99, 0.98, 0.97, 0.05, 0.80, 0.70],
            "predicted_sigma_m_hgb": [20.0, 18.0, 16.0, 0.5, 12.0, 10.0],
        }
    )


def test_uncertainty_cap_preserves_low_sigma_candidate() -> None:
    capped = uncertainty_aware_cap_reservoir(
        _rows(),
        max_candidates_per_frame=3,
        min_per_source=0,
        min_per_branch=0,
        min_low_uncertainty=1,
        uncertainty_weight=0.0,
    )

    assert len(capped) == 3
    assert "low_sigma" in set(capped["track_id"])
    low_sigma = capped.loc[capped["track_id"] == "low_sigma"].iloc[0]
    assert "low_uncertainty" in low_sigma["candidate_uncertainty_cap_reason"]
    assert low_sigma["candidate_uncertainty_value_m"] == 0.5
    assert low_sigma["candidate_uncertainty_column"] == "predicted_sigma_m_hgb"


def test_uncertainty_weight_changes_score_fill_without_hard_replacement() -> None:
    rows = _rows().iloc[:3].copy()
    rows.loc[:, "track_id"] = ["high_score_high_sigma", "mid", "low_score_low_sigma"]
    rows.loc[:, "candidate_reservoir_score"] = [1.0, 0.8, 0.7]
    rows.loc[:, "predicted_sigma_m_hgb"] = [30.0, 20.0, 1.0]

    score_only = uncertainty_aware_cap_reservoir(
        rows,
        max_candidates_per_frame=2,
        min_per_source=0,
        min_per_branch=0,
        min_low_uncertainty=0,
        uncertainty_weight=0.0,
    )
    uncertainty_weighted = uncertainty_aware_cap_reservoir(
        rows,
        max_candidates_per_frame=2,
        min_per_source=0,
        min_per_branch=0,
        min_low_uncertainty=0,
        uncertainty_weight=0.75,
    )

    assert set(score_only["track_id"]) == {"high_score_high_sigma", "mid"}
    assert "low_score_low_sigma" in set(uncertainty_weighted["track_id"])
    assert len(uncertainty_weighted) == 2


def test_uncertainty_cap_falls_back_to_score_when_sigma_missing() -> None:
    rows = _rows().drop(columns=["predicted_sigma_m_hgb"])
    capped = uncertainty_aware_cap_reservoir(
        rows,
        max_candidates_per_frame=2,
        min_per_source=0,
        min_per_branch=0,
        min_low_uncertainty=2,
        uncertainty_weight=1.0,
    )

    assert set(capped["track_id"]) == {"score1", "score2"}
    assert capped["candidate_uncertainty_value_m"].isna().all()
    assert set(capped["candidate_uncertainty_column"]) == {""}


def test_uncertainty_cap_summary_counts_reasons() -> None:
    capped = uncertainty_aware_cap_reservoir(
        _rows(),
        max_candidates_per_frame=3,
        min_per_source=0,
        min_per_branch=0,
        min_low_uncertainty=1,
    )
    summary = uncertainty_cap_summary(_rows(), capped)

    assert summary["input_rows"] == 6
    assert summary["output_rows"] == 3
    assert summary["uncertainty_cap_reason_counts"]["low_uncertainty"] == 1
    assert summary["uncertainty_column_counts"]["predicted_sigma_m_hgb"] == 3


def test_uncertainty_cap_cli_writes_outputs_and_oracle(tmp_path: Path) -> None:
    input_csv = tmp_path / "reservoir.csv"
    output_csv = tmp_path / "capped.csv"
    summary_json = tmp_path / "summary.json"
    truth_csv = tmp_path / "truth.csv"
    oracle_csv = tmp_path / "oracle.csv"
    _rows().to_csv(input_csv, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    ).to_csv(truth_csv, index=False)

    status = uncertainty_main(
        [
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--truth-csv",
            str(truth_csv),
            "--oracle-summary-csv",
            str(oracle_csv),
            "--max-candidates-per-frame",
            "3",
            "--min-per-source",
            "0",
            "--min-per-branch",
            "0",
            "--min-low-uncertainty",
            "1",
            "--uncertainty-column",
            "predicted_sigma_m_hgb",
            "--max-truth-time-delta-s",
            "0.1",
        ]
    )

    assert status == 0
    assert output_csv.exists()
    assert summary_json.exists()
    assert oracle_csv.exists()
    assert "low_sigma" in set(pd.read_csv(output_csv)["track_id"])
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["output_rows"] == 3
    oracle = pd.read_csv(oracle_csv)
    assert oracle.loc[0, "oracle_all_3d_m_mse"] == 0.0
