from __future__ import annotations

from pathlib import Path

import pandas as pd

import raft_uav.mmuad.track5_estimate_ensemble_grid as grid
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput


def test_best_weights_follow_sorted_summary_for_nan_and_inf(monkeypatch) -> None:
    """Keep weights and policy/config metadata on the same winning grid row."""

    monkeypatch.setattr(grid, "read_estimate_csv", lambda _path: pd.DataFrame())
    monkeypatch.setattr(
        grid,
        "build_track5_estimate_ensemble",
        lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        grid,
        "_ensemble_results_frame",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(grid, "evaluate_mmaud_results", lambda *_args, **_kwargs: {})

    def fake_grid_row(
        weights: tuple[float, ...],
        _evaluation: dict,
        *,
        aggregation_policy: str,
        trim_fraction: float,
    ) -> grid.EnsembleGridRow:
        metric = float("nan") if weights == (1.0, 0.0) else float("inf")
        return grid.EnsembleGridRow(
            weights=weights,
            aggregation_policy=aggregation_policy,
            trim_fraction=trim_fraction,
            pose_mse=metric,
            rmse_m=metric,
            mean_error_m=metric,
            p95_error_m=metric,
            max_error_m=metric,
            class_accuracy=None,
            matched_count=0,
        )

    monkeypatch.setattr(grid, "_grid_row", fake_grid_row)

    inputs = (
        EstimateInput(label="first", path=Path("first.csv")),
        EstimateInput(label="second", path=Path("second.csv")),
    )
    summary, by_sequence, best_weights = grid.evaluate_estimate_ensemble_weight_grid(
        inputs,
        template=pd.DataFrame(),
        truth=pd.DataFrame(),
        weight_grid=((1.0, 0.0), (0.0, 1.0)),
    )

    assert by_sequence.empty
    assert summary.iloc[0]["weights"] == "0.0;1.0"
    assert best_weights == (0.0, 1.0)
