from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_reservoir_bottleneck import BottleneckConfig
from raft_uav.mmuad.candidate_reservoir_bottleneck import annotate_gap_table
from raft_uav.mmuad.candidate_reservoir_bottleneck import build_bottleneck_summary


def test_annotate_gap_table_preserves_nondefault_index_alignment() -> None:
    rows = pd.DataFrame(
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
        ],
        index=["first", "second"],
    )

    annotated = annotate_gap_table(rows, config=BottleneckConfig())

    assert annotated.index.tolist() == ["first", "second"]
    assert annotated["primary_bottleneck"].tolist() == [
        "assignment_limited",
        "reservoir_ceiling_limited",
    ]


def test_build_bottleneck_summary_selects_by_position_with_duplicate_labels() -> None:
    rows = pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "primary_bottleneck": "assignment_limited",
                "recommended_action": "improve_mixture_weighting_sigma_or_assignment",
                "assignment_gap_mse_3d_m2": 80.0,
                "topk_recall_gap_mse_3d_m2": 2.0,
                "reservoir_oracle_all_mse_3d_m2": 20.0,
            },
            {
                "sequence_id": "seqB",
                "primary_bottleneck": "reservoir_ceiling_limited",
                "recommended_action": "add_or_repair_candidate_branches",
                "assignment_gap_mse_3d_m2": 2.0,
                "topk_recall_gap_mse_3d_m2": 1.0,
                "reservoir_oracle_all_mse_3d_m2": 58.0,
            },
        ],
        index=["result", "result"],
    )

    summary = build_bottleneck_summary(rows, config=BottleneckConfig())

    assert summary["worst_assignment_gap"]["sequence_id"] == "seqA"
    assert summary["worst_reservoir_ceiling"]["sequence_id"] == "seqB"
