from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.evaluation.oracle_coverage import _bucket_summary, _coverage_summary


def _coverage_frame(
    *,
    oracle_available: object,
    oracle_retained: object,
) -> pd.DataFrame:
    available = pd.Series(oracle_available)
    retained = pd.Series(oracle_retained)
    row_count = len(available)
    return pd.DataFrame(
        {
            "oracle_available": available,
            "oracle_retained": retained,
            "oracle_drop_reason": ["retained"] * row_count,
            "retained_candidate_count": [1] * row_count,
            "frame_candidate_count": [2] * row_count,
            "oracle_truth_error_m": [3.0] * row_count,
            "oracle_range_m": [100.0] * row_count,
            "oracle_cat_prob_uav": [0.8] * row_count,
            "oracle_miss_streak_before": [0] * row_count,
        }
    )


def _summary(frame: pd.DataFrame) -> dict[str, object]:
    return _coverage_summary(
        frame,
        candidate_catprob_threshold=0.5,
        config=TrackletViterbiAssociationConfig(max_candidates_per_frame=8),
        truth_time_gate_s=1.0,
    )


def test_oracle_coverage_summary_parses_serialized_boolean_diagnostics() -> None:
    frame = _coverage_frame(
        oracle_available=[" TRUE ", "False", "1.0", "0", None],
        oracle_retained=["false", "true", 1, 0, "yes"],
    )

    summary = _summary(frame)
    buckets = _bucket_summary(frame)
    range_buckets = buckets.loc[buckets["bucket_type"] == "range_m"]

    assert summary["oracle_available_frames"] == 2
    assert summary["oracle_retained_frames"] == 1
    assert summary["oracle_retention_rate"] == 0.5
    assert int(range_buckets["frame_count"].sum()) == 2
    assert int(range_buckets["retained_count"].sum()) == 1


def test_oracle_coverage_summary_handles_nullable_numeric_booleans() -> None:
    frame = _coverage_frame(
        oracle_available=pd.Series([1.0, 0.0, pd.NA], dtype="Float64"),
        oracle_retained=pd.Series([1.0, 0.0, pd.NA], dtype="Float64"),
    )

    summary = _summary(frame)

    assert summary["oracle_available_frames"] == 1
    assert summary["oracle_retained_frames"] == 1


def test_oracle_coverage_summary_rejects_ambiguous_boolean_diagnostics() -> None:
    frame = _coverage_frame(
        oracle_available=["true", "maybe"],
        oracle_retained=["true", "false"],
    )

    with pytest.raises(
        ValueError,
        match=r"oracle_available contains invalid Boolean values at rows \[1\]",
    ):
        _summary(frame)
