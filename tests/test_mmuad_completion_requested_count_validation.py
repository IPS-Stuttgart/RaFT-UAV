from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.completion import CompletionResult, completion_summary
from raft_uav.mmuad.submission import UG2_RESULT_COLUMNS


def _empty_result() -> CompletionResult:
    return CompletionResult(
        rows=pd.DataFrame(columns=UG2_RESULT_COLUMNS),
        diagnostics=pd.DataFrame(),
    )


@pytest.mark.parametrize(
    "requested_count",
    [
        True,
        np.bool_(False),
        -1,
        1.5,
        float("nan"),
        np.ma.masked,
        np.asarray([3]),
    ],
)
def test_completion_summary_rejects_invalid_requested_counts(
    requested_count: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="requested_count must be a non-negative integer",
    ):
        completion_summary(_empty_result(), requested_count=requested_count)


def test_completion_summary_accepts_exact_scalar_like_requested_count() -> None:
    summary = completion_summary(
        _empty_result(),
        requested_count=np.asarray(3),
    )

    assert summary["requested_count"] == 3
    assert summary["dropped_count"] == 3


def test_completion_summary_rejects_count_below_completed_rows() -> None:
    completed = CompletionResult(
        rows=pd.DataFrame([{"sequence_id": "seq1"}]),
        diagnostics=pd.DataFrame(),
    )

    with pytest.raises(
        ValueError,
        match=r"requested_count \(0\) cannot be smaller than completed_count \(1\)",
    ):
        completion_summary(completed, requested_count=0)
