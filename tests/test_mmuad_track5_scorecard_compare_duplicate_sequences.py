from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_scorecard_compare import compare_pose_by_sequence_tables


def _pose_rows(sequence_column: str, sequence_ids: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            sequence_column: sequence_ids,
            "count": [10] * len(sequence_ids),
            "mse": [float(index + 1) for index in range(len(sequence_ids))],
        }
    )


@pytest.mark.parametrize(
    ("duplicate_side", "sequence_column"),
    [
        ("baseline", "sequence_id"),
        ("candidate", "sequence"),
    ],
)
def test_pose_comparison_rejects_duplicate_sequence_rows(
    duplicate_side: str,
    sequence_column: str,
) -> None:
    duplicate = _pose_rows(sequence_column, ["seq001", "seq001"])
    unique = _pose_rows("sequence_id", ["seq001"])
    baseline = duplicate if duplicate_side == "baseline" else unique
    candidate = duplicate if duplicate_side == "candidate" else unique

    with pytest.raises(
        ValueError,
        match=rf"{duplicate_side} pose table contains duplicate sequence_id rows",
    ):
        compare_pose_by_sequence_tables(baseline, candidate)


@pytest.mark.parametrize(
    ("missing_side", "sequence_column", "missing_value"),
    [
        ("baseline", "sequence_id", None),
        ("candidate", "sequence", np.nan),
        ("baseline", "sequence_id", "   "),
    ],
)
def test_pose_comparison_rejects_missing_sequence_rows(
    missing_side: str,
    sequence_column: str,
    missing_value: object,
) -> None:
    malformed = _pose_rows(sequence_column, [missing_value])
    valid = _pose_rows("sequence_id", ["seq001"])
    baseline = malformed if missing_side == "baseline" else valid
    candidate = malformed if missing_side == "candidate" else valid

    with pytest.raises(
        ValueError,
        match=rf"{missing_side} pose table contains missing sequence_id rows",
    ):
        compare_pose_by_sequence_tables(baseline, candidate)


def test_pose_comparison_preserves_unique_sequence_alias_rows() -> None:
    baseline = _pose_rows("sequence", ["001", "010"])
    candidate = _pose_rows("sequence_id", ["001", "010"])

    delta, summary = compare_pose_by_sequence_tables(baseline, candidate)

    assert delta["sequence_id"].tolist() == ["001", "010"]
    assert summary["common_sequence_count"] == 2
