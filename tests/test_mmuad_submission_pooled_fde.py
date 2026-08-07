from __future__ import annotations

import importlib

import pandas as pd

from raft_uav.mmuad import submission


def _multi_sequence_estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["short", "short", "long", "long"],
            "time_s": [0.0, 1.0, 0.0, 2.0],
            "error_3d_m": [1.0, 100.0, 2.0, 20.0],
            "error_2d_m": [1.0, 60.0, 2.0, 12.0],
        }
    )


def test_exported_pooled_fde_averages_each_sequence_endpoint() -> None:
    metrics = submission.compute_trajectory_metrics(_multi_sequence_estimates())

    assert metrics["pooled"]["fde_3d_m"] == 60.0
    assert metrics["pooled"]["fde_2d_m"] == 36.0
    assert metrics["sequences"]["short"]["fde_3d_m"] == 100.0
    assert metrics["sequences"]["long"]["fde_3d_m"] == 20.0


def test_exported_sequence_fde_averages_each_track_endpoint() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1", "seq1"],
            "track_id": ["a", "a", "b", "b"],
            "time_s": [0.0, 2.0, 0.0, 1.0],
            "error_3d_m": [1.0, 20.0, 2.0, 100.0],
            "error_2d_m": [1.0, 12.0, 2.0, 60.0],
        }
    )

    metrics = submission.compute_trajectory_metrics(estimates)

    assert metrics["pooled"]["fde_3d_m"] == 60.0
    assert metrics["pooled"]["fde_2d_m"] == 36.0
    assert metrics["sequences"]["seq1"]["fde_3d_m"] == 60.0
    assert metrics["sequences"]["seq1"]["fde_2d_m"] == 36.0


def test_exported_pooled_fde_survives_submission_wrapper_reload() -> None:
    reloaded = importlib.reload(submission)
    reloaded = importlib.reload(reloaded)

    metrics = reloaded.compute_trajectory_metrics(_multi_sequence_estimates())

    assert metrics["pooled"]["fde_3d_m"] == 60.0
    assert metrics["pooled"]["fde_2d_m"] == 36.0
