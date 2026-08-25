from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.completion import complete_results_to_truth_timestamps
from raft_uav.mmuad.submission import UG2_RESULT_COLUMNS


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=UG2_RESULT_COLUMNS)


def _template() -> pd.DataFrame:
    return pd.DataFrame({"sequence_id": ["seq1"], "time_s": [0.0]})


@pytest.mark.parametrize(
    "default_score",
    [
        True,
        np.bool_(False),
        float("nan"),
        float("inf"),
        float("-inf"),
        1.0 + 0.0j,
        [0.5],
        np.asarray([0.5]),
        np.ma.masked,
        None,
        object(),
    ],
)
def test_completion_rejects_invalid_default_scores_before_empty_result_path(
    default_score: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="default_score must be a finite real scalar",
    ):
        complete_results_to_truth_timestamps(
            _empty_results(),
            _template(),
            default_score=default_score,
        )


def test_completion_accepts_scalar_like_finite_default_score() -> None:
    completed = complete_results_to_truth_timestamps(
        _empty_results(),
        _template(),
        default_score=np.asarray("0.25"),
    )

    assert completed.rows.empty
    assert completed.diagnostics["completion_method"].tolist() == [
        "missing_sequence_prediction"
    ]
