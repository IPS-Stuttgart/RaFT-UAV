from pathlib import Path

import pandas as pd

from raft_uav.mmuad.evaluate import (
    load_submission_csv,
    match_submission_to_truth,
    metrics_from_matches,
)


def test_load_submission_csv_accepts_case_insensitive_alias_columns(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "Sequence_ID": ["seqA"],
            "Time_S": [1.25],
            "Track": ["track7"],
            "X": [10.0],
            "Y": [20.0],
            "Z": [30.0],
            "Score": [0.8],
        }
    ).to_csv(path, index=False)

    frame = load_submission_csv(path)

    assert frame.loc[0, "sequence_id"] == "seqA"
    assert frame.loc[0, "time_s"] == 1.25
    assert frame.loc[0, "track_id"] == "track7"
    assert frame.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [10.0, 20.0, 30.0]
    assert frame.loc[0, "score"] == 0.8


def test_truth_coverage_counts_unique_truth_rows_not_duplicate_predictions() -> None:
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [2.0, 2.0],
        }
    )
    submission = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 0.0, 1.0],
            "track_id": ["dup_a", "dup_b", "last"],
            "x_m": [0.0, 0.0, 1.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [2.0, 2.0, 2.0],
        }
    )

    matches = match_submission_to_truth(submission, truth, max_time_delta_s=0.1)
    metrics = metrics_from_matches(matches, submission=submission, truth=truth)

    assert metrics["pooled"]["matched_count"] == 3
    assert metrics["pooled"]["truth_count"] == 2
    assert metrics["pooled"]["covered_truth_count"] == 2
    assert metrics["pooled"]["truth_coverage_fraction"] == 1.0


def test_metrics_include_truth_only_sequences_when_no_predictions() -> None:
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [2.0, 2.0],
        }
    )
    submission = pd.DataFrame(
        columns=["sequence_id", "time_s", "track_id", "x_m", "y_m", "z_m"]
    )

    matches = match_submission_to_truth(submission, truth, max_time_delta_s=0.1)
    metrics = metrics_from_matches(matches, submission=submission, truth=truth)

    assert metrics["pooled"]["truth_count"] == 2
    assert metrics["pooled"]["prediction_count"] == 0
    assert metrics["pooled"]["covered_truth_count"] == 0
    assert metrics["pooled"]["truth_coverage_fraction"] == 0.0
    assert set(metrics["sequences"]) == {"seqA"}
    assert metrics["sequences"]["seqA"]["truth_count"] == 2
    assert metrics["sequences"]["seqA"]["prediction_count"] == 0
    assert metrics["sequences"]["seqA"]["matched_count"] == 0
    assert metrics["sequences"]["seqA"]["covered_truth_count"] == 0
    assert metrics["sequences"]["seqA"]["truth_coverage_fraction"] == 0.0
