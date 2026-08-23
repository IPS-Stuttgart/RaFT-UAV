from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pull import refine_official_results_with_candidate_pull


def _results(timestamp: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [timestamp],
            "Position": ["(1,2,3)"],
            "Classification": ["UAV"],
        },
        index=[17],
    )


@pytest.mark.parametrize(
    "timestamp",
    [None, "not-a-time", np.nan, np.inf, -np.inf, True, np.bool_(False), 1.0 + 2.0j],
)
def test_candidate_pull_rejects_malformed_result_timestamps(timestamp: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"official result Timestamp must be a finite real scalar at row 17",
    ):
        refine_official_results_with_candidate_pull(_results(timestamp), pd.DataFrame())


def test_candidate_pull_keeps_numeric_text_result_timestamps_supported() -> None:
    result = refine_official_results_with_candidate_pull(_results("1.25"), pd.DataFrame())

    assert result.rows["Timestamp"].tolist() == [pytest.approx(1.25)]
    assert result.rows["Position"].tolist() == ["(1,2,3)"]
