from __future__ import annotations

import numpy as np
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


@pytest.mark.parametrize(
    "invalid_tolerance",
    [
        -1.0,
        float("nan"),
        float("inf"),
        True,
        1.0 + 0.0j,
        np.array([-1.0]),
    ],
)
def test_pose_comparison_rejects_invalid_regression_tolerance(
    invalid_tolerance: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="regression_tolerance_mse must be a finite non-negative real scalar",
    ):
        compare_pose_by_sequence_tables(
            _pose_table(1.0),
            _pose_table(1.0),
            regression_tolerance_mse=invalid_tolerance,
        )


def test_pose_comparison_accepts_boxed_nonnegative_regression_tolerance() -> None:
    _, summary = compare_pose_by_sequence_tables(
        _pose_table(1.0),
        _pose_table(1.5),
        regression_tolerance_mse=np.array(np.array(0.25, dtype=object), dtype=object),
    )

    assert summary["regressed_sequence_count"] == 1
    assert summary["unchanged_sequence_count"] == 0
    assert summary["regression_tolerance_mse"] == 0.25


def test_scorecard_compare_cli_rejects_negative_regression_tolerance(
    tmp_path,
) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    candidate_csv = tmp_path / "candidate.csv"
    _pose_table(1.0).to_csv(baseline_csv, index=False)
    _pose_table(1.0).to_csv(candidate_csv, index=False)

    with pytest.raises(
        ValueError,
        match="regression_tolerance_mse must be a finite non-negative real scalar",
    ):
        compare_main(
            [
                "--baseline-pose-by-sequence-csv",
                str(baseline_csv),
                "--candidate-pose-by-sequence-csv",
                str(candidate_csv),
                "--pose-delta-csv",
                str(tmp_path / "delta.csv"),
                "--regression-tolerance-mse",
                "-1",
            ]
        )
