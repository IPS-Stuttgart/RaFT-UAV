from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_reservoir_bottleneck import BottleneckConfig
from raft_uav.mmuad.candidate_reservoir_bottleneck import annotate_gap_table
from raft_uav.mmuad.candidate_reservoir_bottleneck import classify_gap_row
from raft_uav.mmuad.candidate_reservoir_bottleneck import main as bottleneck_main


def test_classify_gap_row_marks_assignment_limited_when_oracle_is_good() -> None:
    row = {
        "mixture_mse_3d_m2": 100.0,
        "reservoir_oracle_all_mse_3d_m2": 20.0,
        "best_reservoir_oracle_topk_mse_3d_m2": 22.0,
    }

    result = classify_gap_row(row, config=BottleneckConfig(target_mse_3d_m2=24.51))

    assert result["primary_bottleneck"] == "assignment_limited"
    assert result["recommended_action"] == "improve_mixture_weighting_sigma_or_assignment"
    assert result["reservoir_oracle_all_beats_target"] is True


def test_classify_gap_row_marks_topk_recall_when_deep_pool_is_good() -> None:
    row = {
        "mixture_mse_3d_m2": 100.0,
        "reservoir_oracle_all_mse_3d_m2": 20.0,
        "best_reservoir_oracle_topk_mse_3d_m2": 80.0,
    }

    result = classify_gap_row(row, config=BottleneckConfig(target_mse_3d_m2=24.51))

    assert result["primary_bottleneck"] == "topk_recall_limited"
    assert result["recommended_action"] == "increase_reservoir_quota_or_rebalance_branch_scores"


def test_annotate_gap_table_and_cli_write_actionable_outputs(tmp_path: Path) -> None:
    gap_csv = tmp_path / "gap.csv"
    output_csv = tmp_path / "annotated.csv"
    summary_json = tmp_path / "summary.json"
    pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "mixture_mse_3d_m2": 100.0,
                "reservoir_oracle_all_mse_3d_m2": 20.0,
                "best_reservoir_oracle_topk_mse_3d_m2": 22.0,
            },
            {
                "sequence_id": "seqB",
                "mixture_mse_3d_m2": 60.0,
                "reservoir_oracle_all_mse_3d_m2": 58.0,
                "best_reservoir_oracle_topk_mse_3d_m2": 59.0,
            },
        ]
    ).to_csv(gap_csv, index=False)

    annotated = annotate_gap_table(pd.read_csv(gap_csv), config=BottleneckConfig())
    assert annotated["primary_bottleneck"].tolist() == [
        "assignment_limited",
        "reservoir_ceiling_limited",
    ]

    status = bottleneck_main(
        [
            "--gap-csv",
            str(gap_csv),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--target-mse-3d-m2",
            "24.51",
        ]
    )

    assert status == 0
    written = pd.read_csv(output_csv)
    assert "primary_bottleneck" in written.columns
    assert "recommended_action" in written.columns
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["row_count"] == 2
    assert payload["bottleneck_counts"]["assignment_limited"] == 1
    assert payload["bottleneck_counts"]["reservoir_ceiling_limited"] == 1
    assert payload["worst_assignment_gap"]["sequence_id"] == "seqA"
