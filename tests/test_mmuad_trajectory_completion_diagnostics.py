from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.trajectory_completion import (
    TrajectoryCompletionConfig,
    _speed_gate_summary_row,
    _trajectory_roughness,
)


def test_pooled_kinematic_diagnostics_do_not_bridge_independent_tracks() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1"] * 6,
            "output_track_id": ["a"] * 3 + ["b"] * 3,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "state_x_m": [0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0],
            "state_y_m": [0.0] * 6,
            "state_z_m": [5.0] * 6,
            "trajectory_speed_gate_outlier": [False] * 6,
            "trajectory_outlier_replaced": [False] * 6,
        }
    )

    summary = _speed_gate_summary_row(
        estimates,
        sequence_id="seq1",
        trajectory_id="__all__",
        config=TrajectoryCompletionConfig(),
    )

    assert summary["segment_count"] == 4
    assert summary["max_segment_speed_mps"] == pytest.approx(0.0)
    assert summary["p95_segment_speed_mps"] == pytest.approx(0.0)
    assert _trajectory_roughness(estimates) == pytest.approx(0.0)
