from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_assignment_branch_summary import build_candidate_assignment_branch_summary


def test_candidate_assignment_branch_summary_accepts_float_boolean_flags() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "dominant_is_oracle": [1.0, 0.0],
            "oracle_in_topk_by_weight": [1.0, 0.0],
        }
    )

    summary = build_candidate_assignment_branch_summary(rows)
    pooled = summary.loc[
        (summary["sequence_id"] == "__pooled__")
        & (summary["group_label"] == "__all__")
    ].iloc[0]

    assert pooled["dominant_matches_oracle_rate"] == 0.5
    assert pooled["oracle_in_topk_by_weight_rate"] == 0.5
