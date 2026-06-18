from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.cli import main as mmuad_cli_main
from raft_uav.mmuad.trajectory_completion import (
    TrajectoryCompletionConfig,
    complete_and_smooth_estimates,
)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [5.0, 5.0, 5.0],
        }
    )


def test_fixed_lag_smoothing_prefers_selected_positions() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1"],
            "time_s": [0.0, 1.0, 2.0],
            "source": ["lidar", "soft", "lidar"],
            "track_id": ["a", "b", "a"],
            "class_name": ["uav", "uav", "uav"],
            "update_action": ["selected_update", "soft_anchor", "selected_update"],
            "selected_path_update": [True, False, True],
            "state_x_m": [0.0, 10.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [5.0, 5.0, 5.0],
            "v_x_mps": [0.0, 0.0, 0.0],
            "v_y_mps": [0.0, 0.0, 0.0],
            "v_z_mps": [0.0, 0.0, 0.0],
        }
    )

    result = complete_and_smooth_estimates(
        estimates,
        _truth_rows(),
        config=TrajectoryCompletionConfig(mode="fixed-lag", fixed_lag_s=2.0),
    )

    middle = result.estimates.loc[result.estimates["time_s"] == 1.0].iloc[0]
    assert abs(float(middle["state_x_m"]) - 1.0) < 1.0e-6
    assert middle["trajectory_completion_method"] == "fixed-lag_smoothed"
    assert set(result.smoothing_ablation["trajectory_completion_mode"]) == {
        "raw",
        "gap-interpolation",
        "fixed-lag",
        "constant-velocity",
        "constant-acceleration",
    }
    summary = result.sequence_error_summary.iloc[0]
    assert float(summary["final_rmse_3d_m"]) < float(summary["raw_rmse_3d_m"])


def test_gap_interpolation_fills_truth_timestamp_and_reports_gap() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 2.0],
            "source": ["lidar", "lidar"],
            "track_id": ["a", "a"],
            "class_name": ["uav", "uav"],
            "update_action": ["selected_update", "selected_update"],
            "selected_path_update": [True, True],
            "state_x_m": [0.0, 2.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [5.0, 5.0],
            "v_x_mps": [0.0, 0.0],
            "v_y_mps": [0.0, 0.0],
            "v_z_mps": [0.0, 0.0],
        }
    )

    result = complete_and_smooth_estimates(
        estimates,
        _truth_rows(),
        config=TrajectoryCompletionConfig(
            mode="gap-interpolation",
            max_gap_s=3.0,
        ),
    )

    assert result.estimates["time_s"].tolist() == [0.0, 1.0, 2.0]
    filled = result.estimates.loc[result.estimates["time_s"] == 1.0].iloc[0]
    assert filled["trajectory_completion_filled"]
    assert filled["trajectory_completion_method"] == "interpolated_short_gap"
    assert result.gap_summary.loc[0, "filled_count"] == 1


def test_mmuad_cli_writes_trajectory_completion_diagnostics(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 2.0],
            "source": ["lidar", "lidar"],
            "track_id": ["a", "a"],
            "x_m": [0.0, 2.0],
            "y_m": [0.0, 0.0],
            "z_m": [5.0, 5.0],
            "confidence": [1.0, 1.0],
        }
    ).to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = mmuad_cli_main(
        [
            "--candidate-csv",
            str(candidates_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--trajectory-completion-mode",
            "gap-interpolation",
            "--trajectory-completion-max-gap-s",
            "3.0",
        ]
    )

    assert status == 0
    estimates = pd.read_csv(output_dir / "mmuad_estimates.csv")
    gap_summary = pd.read_csv(output_dir / "mmuad_gap_summary.csv")
    ablation = pd.read_csv(output_dir / "mmuad_smoothing_ablation.csv")
    sequence_summary = pd.read_csv(output_dir / "mmuad_sequence_error_summary.csv")
    assert estimates["time_s"].tolist() == [0.0, 1.0, 2.0]
    assert int(gap_summary.loc[0, "filled_count"]) == 1
    assert "__pooled__" in set(ablation["sequence_id"])
    assert sequence_summary.loc[0, "trajectory_completion_mode"] == "gap-interpolation"
