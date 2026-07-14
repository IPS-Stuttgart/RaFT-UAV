from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir_diversity import diversity_cap_reservoir
from raft_uav.mmuad.candidate_reservoir_diversity import diversity_cap_summary
from raft_uav.mmuad.candidate_reservoir_diversity import main


def _custom_branch_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001"],
            "time_s": [0.0, 0.0, 0.0],
            "source": ["radar", "radar", "radar"],
            "track_id": ["a-high", "a-low", "b"],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "confidence": [0.9, 0.8, 0.7],
            "candidate_reservoir_score": [0.9, 0.8, 0.7],
            "track_family": ["family_a", "family_a", "family_b"],
        }
    )


def test_summary_uses_configured_branch_column() -> None:
    rows = _custom_branch_rows()
    capped = diversity_cap_reservoir(
        rows,
        max_candidates_per_frame=1,
        min_per_source=0,
        min_per_branch=1,
        branch_column="track_family",
    )

    summary = diversity_cap_summary(
        rows,
        capped,
        branch_column="track_family",
    )

    assert summary["branch_column"] == "track_family"
    assert sum(summary["branch_counts"].values()) == 1
    assert set(summary["branch_counts"]) <= {"family_a", "family_b"}
    assert summary["mean_branch_coverage_fraction"] == pytest.approx(0.5)
    assert summary["frames_all_branches_preserved_fraction"] == 0.0


def test_cli_forwards_custom_branch_column_to_summary(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    summary_json = tmp_path / "summary.json"
    _custom_branch_rows().to_csv(input_csv, index=False)

    result = main(
        [
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--branch-column",
            "track_family",
            "--max-candidates-per-frame",
            "1",
            "--min-per-source",
            "0",
            "--min-per-branch",
            "1",
        ]
    )

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert result == 0
    assert output_csv.exists()
    assert summary["branch_column"] == "track_family"
    assert sum(summary["branch_counts"].values()) == 1
    assert summary["mean_branch_coverage_fraction"] == pytest.approx(0.5)
