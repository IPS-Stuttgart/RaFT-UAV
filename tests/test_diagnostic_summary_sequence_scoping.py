from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.evaluation.diagnostics import _position_error_frame
from raft_uav.evaluation.diagnostics import build_diagnostic_summary


def _pooled_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )


def _pooled_estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB", "seqC"],
            "time_s": [0.0, 0.0, 0.0],
            "source": ["radar", "radar", "radar"],
            "track_id": [10, 20, 30],
            "east_m": [0.0, 100.0, 50.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "residual_norm_m": [0.0, 0.0, 0.0],
            "covariance_scale": [1.0, 1.0, 1.0],
        }
    )


def test_position_errors_match_truth_within_sequence() -> None:
    errors = _position_error_frame(
        estimate_frame=_pooled_estimates(),
        truth=_pooled_truth(),
        max_eval_time_delta_s=0.1,
    )

    assert errors["sequence_id"].tolist() == ["seqA", "seqB"]
    assert errors["error_3d_m"].tolist() == pytest.approx([0.0, 0.0])


def test_pooled_summary_does_not_count_sequence_boundary_switches() -> None:
    estimates = _pooled_estimates()
    selected_radar = estimates[["sequence_id", "time_s", "track_id"]]

    summary = build_diagnostic_summary(
        estimate_frame=estimates,
        selected_radar=selected_radar,
        truth=_pooled_truth(),
        max_eval_time_delta_s=0.1,
        top_n=10,
        window_s=10.0,
    )

    assert summary["track_switches"]["posterior_radar"]["count"] == 0
    assert summary["track_switches"]["selected_radar"]["count"] == 0
    assert summary["worst_time_windows"][0]["count"] == 2
    assert summary["worst_time_windows"][0]["rmse_3d_m"] == pytest.approx(0.0)
    assert summary["worst_time_windows"][0]["track_switch_count"] == 0
