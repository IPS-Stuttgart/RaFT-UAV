from __future__ import annotations

import pandas as pd

from raft_uav.evaluation.radar_oracle_diagnostics import best_time_offset


def test_best_time_offset_maximizes_coverage_and_count() -> None:
    sweep = pd.DataFrame(
        {
            "time_offset_s": [-1.0, 0.0, 1.0],
            "coverage": [0.25, 1.0, 0.5],
            "count": [1.0, 4.0, 2.0],
        }
    )

    assert best_time_offset(sweep, metric="coverage") == 0.0
    assert best_time_offset(sweep, metric="count") == 0.0


def test_best_time_offset_keeps_minimizing_error_metrics() -> None:
    sweep = pd.DataFrame(
        {
            "time_offset_s": [-1.0, 0.0, 1.0],
            "mean_3d_error_m": [2.0, 1.0, 3.0],
        }
    )

    assert best_time_offset(sweep) == 0.0


def test_best_time_offset_resolves_symmetric_ties_independently_of_row_order() -> None:
    sweep = pd.DataFrame(
        {
            "time_offset_s": [-1.0, 1.0],
            "mean_3d_error_m": [2.0, 2.0],
        }
    )

    assert best_time_offset(sweep) == -1.0
    assert best_time_offset(sweep.iloc[::-1].reset_index(drop=True)) == -1.0


def test_best_time_offset_returns_none_without_offset_column() -> None:
    sweep = pd.DataFrame({"mean_3d_error_m": [1.0]})

    assert best_time_offset(sweep) is None
