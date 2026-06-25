from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_reservoir_diversity import (
    diversity_cap_reservoir,
    diversity_cap_summary,
    main as diversity_main,
)


def _reservoir_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 6,
            "time_s": [0.0] * 6,
            "source": ["lidar_360", "lidar_360", "lidar_360", "lidar_360", "lidar_360", "livox_avia"],
            "track_id": ["translated1", "translated2", "translated3", "translated4", "translated5", "raw_candidate"],
            "candidate_branch": ["translated", "translated", "translated", "translated", "translated", "raw"],
            "x_m": [10.0, 11.0, 12.0, 13.0, 14.0, 0.0],
            "y_m": [0.0] * 6,
            "z_m": [1.0] * 6,
            "candidate_reservoir_score": [0.99, 0.98, 0.97, 0.96, 0.95, 0.01],
            "confidence": [0.99, 0.98, 0.97, 0.96, 0.95, 0.01],
        }
    )


def test_diversity_cap_preserves_lower_score_branch_and_source() -> None:
    capped = diversity_cap_reservoir(
        _reservoir_rows(),
        max_candidates_per_frame=3,
        min_per_source=1,
        min_per_branch=1,
    )

    assert len(capped) == 3
    assert "raw_candidate" in set(capped["track_id"])
    raw = capped.loc[capped["track_id"] == "raw_candidate"].iloc[0]
    assert "branch:raw" in raw["candidate_diversity_cap_reason"]
    assert "source:livox_avia" in raw["candidate_diversity_cap_reason"]
    assert set(capped["candidate_branch"]) == {"translated", "raw"}


def test_diversity_cap_summary_counts_reasons() -> None:
    capped = diversity_cap_reservoir(
        _reservoir_rows(),
        max_candidates_per_frame=3,
        min_per_source=1,
        min_per_branch=1,
    )

    summary = diversity_cap_summary(_reservoir_rows(), capped)

    assert summary["input_rows"] == 6
    assert summary["output_rows"] == 3
    assert summary["branch_counts"]["raw"] == 1
    assert any(key.startswith("branch:") for key in summary["diversity_cap_reason_counts"])


def test_diversity_cap_cli_writes_capped_reservoir_and_oracle(tmp_path: Path) -> None:
    input_csv = tmp_path / "reservoir.csv"
    output_csv = tmp_path / "capped.csv"
    summary_json = tmp_path / "summary.json"
    truth_csv = tmp_path / "truth.csv"
    oracle_summary_csv = tmp_path / "oracle_summary.csv"
    _reservoir_rows().to_csv(input_csv, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    ).to_csv(truth_csv, index=False)

    status = diversity_main(
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
            str(oracle_summary_csv),
            "--max-candidates-per-frame",
            "3",
            "--min-per-source",
            "1",
            "--min-per-branch",
            "1",
            "--max-truth-time-delta-s",
            "0.1",
        ]
    )

    assert status == 0
    capped = pd.read_csv(output_csv)
    assert "raw_candidate" in set(capped["track_id"])
    assert json.loads(summary_json.read_text(encoding="utf-8"))["output_rows"] == 3
    oracle = pd.read_csv(oracle_summary_csv)
    assert oracle.loc[0, "oracle_all_3d_m_mse"] == 0.0
