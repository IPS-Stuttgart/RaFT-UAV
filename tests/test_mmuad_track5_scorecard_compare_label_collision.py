from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_scorecard_compare import (
    compare_pose_by_sequence_tables,
    main as compare_main,
)


def _pose_table(mse: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_id": "seq001",
                "count": 1,
                "mse": mse,
                "rmse": mse**0.5,
                "mean_3d": mse**0.5,
                "median_3d": mse**0.5,
                "p95_3d": mse**0.5,
                "max_3d": mse**0.5,
            }
        ]
    )


def test_pose_comparison_rejects_equal_labels_before_merge() -> None:
    with pytest.raises(
        ValueError,
        match="baseline_label and candidate_label must be distinct",
    ):
        compare_pose_by_sequence_tables(
            _pose_table(4.0),
            _pose_table(9.0),
            baseline_label="same",
            candidate_label="same",
        )


def test_pose_comparison_rejects_labels_with_equal_string_suffixes() -> None:
    with pytest.raises(
        ValueError,
        match="baseline_label and candidate_label must be distinct",
    ):
        compare_pose_by_sequence_tables(
            _pose_table(4.0),
            _pose_table(9.0),
            baseline_label=1,
            candidate_label="1",
        )


def test_scorecard_compare_cli_rejects_equal_labels(tmp_path: Path) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    candidate_csv = tmp_path / "candidate.csv"
    _pose_table(4.0).to_csv(baseline_csv, index=False)
    _pose_table(9.0).to_csv(candidate_csv, index=False)

    with pytest.raises(
        ValueError,
        match="baseline_label and candidate_label must be distinct",
    ):
        compare_main(
            [
                "--baseline-pose-by-sequence-csv",
                str(baseline_csv),
                "--candidate-pose-by-sequence-csv",
                str(candidate_csv),
                "--pose-delta-csv",
                str(tmp_path / "delta.csv"),
                "--baseline-label",
                "same",
                "--candidate-label",
                "same",
            ]
        )
