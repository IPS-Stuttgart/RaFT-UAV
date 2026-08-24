from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.evaluation.oracle_coverage import _bucket_summary, _coverage_summary


def _nested_complex_scalar(value: complex) -> np.ndarray:
    inner = np.asarray(value)
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def _coverage_frame(*, column: str, value: object) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "oracle_available": [True],
            "oracle_retained": [True],
            "oracle_drop_reason": ["retained"],
            "retained_candidate_count": [1],
            "frame_candidate_count": [2],
            "oracle_truth_error_m": [3.0],
            "oracle_range_m": [100.0],
            "oracle_cat_prob_uav": [0.8],
            "oracle_miss_streak_before": [0],
        },
        index=[73],
    )
    frame[column] = pd.Series([value], index=frame.index, dtype=object)
    return frame


def _summary(frame: pd.DataFrame) -> dict[str, object]:
    return _coverage_summary(
        frame,
        candidate_catprob_threshold=0.5,
        config=TrackletViterbiAssociationConfig(max_candidates_per_frame=8),
        truth_time_gate_s=1.0,
    )


@pytest.mark.parametrize(
    "value",
    [
        1.0 + 0.0j,
        0.0 + 0.0j,
        np.complex128(1.0 + 0.0j),
        np.asarray(0.0 + 0.0j),
        _nested_complex_scalar(1.0 + 0.0j),
    ],
)
@pytest.mark.parametrize("column", ["oracle_available", "oracle_retained"])
@pytest.mark.parametrize(
    "consumer",
    [
        pytest.param(_summary, id="coverage-summary"),
        pytest.param(_bucket_summary, id="bucket-summary"),
    ],
)
def test_oracle_coverage_rejects_complex_boolean_diagnostics(
    value: object,
    column: str,
    consumer: Callable[[pd.DataFrame], Any],
) -> None:
    frame = _coverage_frame(column=column, value=value)

    with pytest.raises(
        ValueError,
        match=rf"{column} contains invalid Boolean values at rows \[73\]",
    ):
        consumer(frame)
