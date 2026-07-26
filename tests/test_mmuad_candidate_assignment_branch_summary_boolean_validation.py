from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_assignment_branch_summary import (
    build_candidate_assignment_branch_summary,
)


def _pooled_row(summary: pd.DataFrame) -> pd.Series:
    return summary.loc[
        (summary["sequence_id"] == "__pooled__")
        & (summary["group_label"] == "__all__")
    ].iloc[0]


def test_assignment_branch_summary_parses_serialized_boolean_flags() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq"] * 4,
            "dominant_is_oracle": [" TRUE ", "False", "1.0", "0"],
            "oracle_in_topk_by_weight": ["yes", "no", 1.0, 0.0],
        }
    )

    pooled = _pooled_row(build_candidate_assignment_branch_summary(rows))

    assert pooled["dominant_matches_oracle_rate"] == 0.5
    assert pooled["oracle_in_topk_by_weight_rate"] == 0.5


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("dominant_is_oracle", 2.0),
        ("dominant_is_oracle", -1.0),
        ("oracle_in_topk_by_weight", 0.5),
        ("oracle_in_topk_by_weight", "maybe"),
        ("oracle_in_topk_by_weight", np.inf),
    ],
)
def test_assignment_branch_summary_rejects_invalid_boolean_flags(
    column: str,
    value: object,
) -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "dominant_is_oracle": [value if column == "dominant_is_oracle" else False],
            "oracle_in_topk_by_weight": [
                value if column == "oracle_in_topk_by_weight" else False
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match=rf"{column} contains invalid Boolean values at rows \[0\]",
    ):
        build_candidate_assignment_branch_summary(rows)
