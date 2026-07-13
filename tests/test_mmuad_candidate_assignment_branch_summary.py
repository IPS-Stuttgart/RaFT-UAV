from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_assignment_branch_summary import (
    build_candidate_assignment_branch_summary,
)
from raft_uav.mmuad.candidate_assignment_branch_summary import main as branch_summary_main


def _frame_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqB"],
            "time_s": [0.0, 1.0, 2.0, 0.0],
            "assignment_failure_mode": [
                "covered",
                "good_candidate_buried",
                "wrong_dominant_assignment",
                "covered",
            ],
            "oracle_candidate_branch": ["raw", "raw", "dynamic", "translated"],
            "dominant_candidate_branch": ["raw", "translated", "raw", "translated"],
            "oracle_source": ["lidar", "lidar", "livox", "lidar"],
            "dominant_source": ["lidar", "radar", "lidar", "lidar"],
            "state_error_3d_m": [1.0, 10.0, 7.0, 2.0],
            "oracle_error_3d_m": [0.5, 0.4, 0.6, 1.0],
            "dominant_error_3d_m": [0.5, 9.0, 6.0, 1.2],
            "state_regret_m": [0.5, 9.6, 6.4, 1.0],
            "dominant_regret_m": [0.0, 8.6, 5.4, 0.2],
            "oracle_mixture_weight": [0.8, 0.02, 0.15, 0.7],
            "oracle_weight_rank": [1, 5, 3, 1],
            "candidate_count": [4, 8, 6, 5],
            "dominant_is_oracle": [True, False, False, True],
            "oracle_in_topk_by_weight": [True, False, True, True],
        }
    )


def test_branch_summary_groups_by_oracle_and_dominant_branch() -> None:
    summary = build_candidate_assignment_branch_summary(_frame_rows())

    pooled = summary.loc[
        (summary["sequence_id"] == "__pooled__")
        & (summary["group_label"] == "__all__")
    ].iloc[0]
    assert pooled["frame_count"] == 4
    assert pooled["dominant_matches_oracle_rate"] == 0.5
    assert pooled["state_vs_oracle_mse_gap"] > 0.0
    assert pooled["dominant_vs_oracle_mse_gap"] > 0.0
    assert pooled["oracle_weight_deficit_mean"] == 1.0 - pooled["oracle_mixture_weight_mean"]
    assert pooled["assignment_priority_score"] > 0.0
    buried = summary.loc[
        (summary["sequence_id"] == "__pooled__")
        & (summary["assignment_failure_mode"] == "good_candidate_buried")
        & (summary["oracle_candidate_branch"] == "raw")
        & (summary["dominant_candidate_branch"] == "translated")
    ].iloc[0]
    assert buried["frame_count"] == 1
    assert buried["oracle_mixture_weight_mean"] == 0.02
    assert buried["oracle_weight_deficit_mean"] == 0.98
    assert buried["state_vs_oracle_mse_gap"] > 90.0
    assert buried["assignment_priority_score"] > 0.0
    assert buried["dominant_source"] == "radar"


def test_branch_summary_cli_writes_artifacts(tmp_path: Path) -> None:
    frame_csv = tmp_path / "frames.csv"
    output_dir = tmp_path / "out"
    _frame_rows().to_csv(frame_csv, index=False)

    status = branch_summary_main(
        [
            "--frame-csv",
            str(frame_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    summary_csv = output_dir / "mmuad_candidate_assignment_branch_summary.csv"
    summary_json = output_dir / "mmuad_candidate_assignment_branch_summary.json"
    assert summary_csv.exists()
    assert summary_json.exists()
    written = pd.read_csv(summary_csv)
    assert "assignment_priority_score" in written.columns
    assert "state_vs_oracle_mse_gap" in written.columns
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "raft-uav-mmuad-candidate-assignment-branch-summary-v1"
    assert payload["row_count"] == len(written)


def test_branch_summary_parses_explicit_boolean_representations() -> None:
    rows = _frame_rows()
    rows["dominant_is_oracle"] = ["False", "TRUE", 0, 1]
    rows["oracle_in_topk_by_weight"] = ["0", "1.0", "no", "yes"]

    summary = build_candidate_assignment_branch_summary(rows)
    pooled = summary.loc[
        (summary["sequence_id"] == "__pooled__")
        & (summary["group_label"] == "__all__")
    ].iloc[0]

    assert pooled["dominant_matches_oracle_rate"] == 0.5
    assert pooled["oracle_in_topk_by_weight_rate"] == 0.5


@pytest.mark.parametrize("invalid", ["maybe", 2, -1, np.inf])
@pytest.mark.parametrize("column", ["dominant_is_oracle", "oracle_in_topk_by_weight"])
def test_branch_summary_rejects_ambiguous_boolean_diagnostics(
    column: str,
    invalid: object,
) -> None:
    rows = _frame_rows()
    rows[column] = rows[column].astype(object)
    rows.loc[1, column] = invalid

    with pytest.raises(ValueError, match=rf"{column}.*1"):
        build_candidate_assignment_branch_summary(rows)
