from __future__ import annotations

import json

import pandas as pd

from raft_uav.research.factor_graph import FactorGraphSmoothingResult
from scripts import run_factor_graph_smoother


def test_radar_mode_reports_diagnostics_for_written_estimates(
    tmp_path,
    monkeypatch,
) -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [10.0, 20.0],
            "north_m": [0.0, 0.0],
            "up_m": [2.0, 3.0],
        }
    )
    rf = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [11.0, 21.0],
            "north_m": [1.0, 1.0],
            "up_m": [2.5, 3.5],
        }
    )
    initial_estimates = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [10.5, 20.5],
            "north_m": [0.5, 0.5],
            "up_m": [2.25, 3.25],
        }
    )
    final_estimates = initial_estimates.copy()
    final_estimates["east_m"] += 0.25

    radar_path = tmp_path / "radar.csv"
    rf_path = tmp_path / "rf.csv"
    output_dir = tmp_path / "output"
    radar.to_csv(radar_path, index=False)
    rf.to_csv(rf_path, index=False)

    def fake_coordinate_descent(
        radar_frame,
        rf_frame,
        *,
        iterations,
        config,
    ):
        pd.testing.assert_frame_equal(radar_frame, radar)
        pd.testing.assert_frame_equal(rf_frame, rf)
        assert iterations == 2
        assert config.measurement_std_m == 25.0
        return initial_estimates, radar

    smoothing_calls = []

    def fake_smooth(measurements, *, initial=None, config=None):
        smoothing_calls.append((measurements.copy(), initial.copy(), config))
        return FactorGraphSmoothingResult(
            estimates=final_estimates,
            cost=12.5,
            optimality=0.25,
            iterations=7,
            success=True,
            message="converged",
        )

    monkeypatch.setattr(
        run_factor_graph_smoother,
        "coordinate_descent_association_and_smoothing",
        fake_coordinate_descent,
    )
    monkeypatch.setattr(
        run_factor_graph_smoother,
        "smooth_position_trajectory",
        fake_smooth,
    )

    exit_code = run_factor_graph_smoother.main(
        [
            "--radar",
            str(radar_path),
            "--rf",
            str(rf_path),
            "--iterations",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert len(smoothing_calls) == 1
    measurements, initial, _ = smoothing_calls[0]
    assert measurements["source"].tolist() == ["rf", "rf", "radar", "radar"]
    pd.testing.assert_frame_equal(initial, initial_estimates)
    pd.testing.assert_frame_equal(
        pd.read_csv(output_dir / "factor_graph_estimates.csv"),
        final_estimates,
    )
    summary = json.loads(
        (output_dir / "factor_graph_summary.json").read_text(encoding="utf-8")
    )
    assert summary["cost"] == 12.5
    assert summary["optimality"] == 0.25
    assert summary["iterations"] == 7
    assert summary["message"] == "converged"
