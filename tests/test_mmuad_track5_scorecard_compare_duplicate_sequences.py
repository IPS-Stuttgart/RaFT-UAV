from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_scorecard_compare import compare_pose_by_sequence_tables


def _pose_rows(*sequence_ids: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": list(sequence_ids),
            "count": [1] * len(sequence_ids),
            "mse": [1.0] * len(sequence_ids),
        }
    )


@pytest.mark.parametrize(
    ("duplicate_side", "expected_label"),
    [("baseline", "reference"), ("candidate", "trial")],
)
def test_pose_comparison_rejects_duplicate_sequence_rows(
    duplicate_side: str,
    expected_label: str,
) -> None:
    baseline = _pose_rows("seq001", "seq002")
    candidate = _pose_rows("seq001", "seq002")
    duplicate = _pose_rows("seq001", "seq001", "seq002")
    if duplicate_side == "baseline":
        baseline = duplicate
    else:
        candidate = duplicate

    with pytest.raises(
        ValueError,
        match=rf"{expected_label}.*duplicate sequence_id.*seq001",
    ):
        compare_pose_by_sequence_tables(
            baseline,
            candidate,
            baseline_label="reference",
            candidate_label="trial",
        )


def test_pose_comparison_still_accepts_unique_sequence_rows() -> None:
    baseline = _pose_rows("seq001", "seq002")
    candidate = _pose_rows("seq001", "seq002")

    delta, summary = compare_pose_by_sequence_tables(baseline, candidate)

    assert delta["sequence_id"].tolist() == ["seq001", "seq002"]
    assert summary["common_sequence_count"] == 2
