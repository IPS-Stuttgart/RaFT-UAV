from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.evaluate import match_submission_to_truth, metrics_from_matches


def _submission(*, track_ids: list[str] | None = None) -> pd.DataFrame:
    rows: dict[str, list[object]] = {
        "sequence_id": ["seq"] if track_ids is None else ["seq", "seq"],
        "time_s": [0.0] if track_ids is None else [0.0, 0.0],
        "x_m": [0.0] if track_ids is None else [0.0, 10.0],
        "y_m": [0.0] if track_ids is None else [0.0, 0.0],
        "z_m": [0.0] if track_ids is None else [0.0, 0.0],
    }
    if track_ids is not None:
        rows["track_id"] = track_ids
    return pd.DataFrame(rows)


def test_submission_evaluator_uses_final_same_time_truth_row() -> None:
    submission = _submission()
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": ["0", 0.0],
            "x_m": [100.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )

    matches = match_submission_to_truth(
        submission,
        truth,
        max_time_delta_s=0.0,
    )
    summary = metrics_from_matches(
        matches,
        submission=submission,
        truth=truth,
    )

    assert matches["matched"].tolist() == [True]
    assert matches["error_3d_m"].tolist() == pytest.approx([0.0])
    assert summary["pooled"]["truth_count"] == 1
    assert summary["pooled"]["matched_count"] == 1
    assert summary["pooled"]["covered_truth_count"] == 1
    assert summary["pooled"]["truth_coverage_fraction"] == pytest.approx(1.0)


def test_duplicate_truth_collapse_preserves_distinct_track_ids() -> None:
    submission = _submission(track_ids=["uav0", "uav1"])
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq", "seq"],
            "time_s": [0.0, 0.0, "0"],
            "track_id": ["uav0", "uav1", "uav0"],
            "x_m": [100.0, 10.0, 0.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
        }
    )

    matches = match_submission_to_truth(
        submission,
        truth,
        max_time_delta_s=0.0,
    )
    summary = metrics_from_matches(
        matches,
        submission=submission,
        truth=truth,
    )

    assert matches["matched"].tolist() == [True, True]
    assert matches["error_3d_m"].tolist() == pytest.approx([0.0, 0.0])
    assert summary["pooled"]["truth_count"] == 2
    assert summary["pooled"]["matched_count"] == 2
