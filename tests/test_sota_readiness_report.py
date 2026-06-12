import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_sota_readiness_report.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("run_sota_readiness_report", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_error_columns_preserve_counts_and_paper_metrics():
    report = _load_report_module()

    columns = report.paper_error_columns(
        {
            "count": 12.0,
            "eval_sample_count": 20,
            "coverage": 0.6,
            "mean_3d_error_m": 11.2345,
            "std_3d_error_m": 2.3456,
            "rmse_3d_error_m": 12.3456,
            "p95_3d_error_m": 17.6543,
            "max_3d_error_m": 23.4567,
        }
    )

    assert columns["matched_count"] == 12
    assert columns["eval_sample_count"] == 20
    assert columns["coverage"] == 0.6
    assert columns["mean_3d_error_m"] == 11.234
    assert columns["std_3d_error_m"] == 2.346
    assert columns["max_3d_error_m"] == 23.457
    assert columns["rmse_3d_error_m"] == 12.346


def test_truth_grid_coverage_uses_truth_sample_denominator():
    report = _load_report_module()
    estimates = pd.DataFrame(
        {
            "time_s": [0.0, 10.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            "east_m": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            "north_m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    columns = report.error_summary_from_estimates(estimates, truth, max_time_delta_s=20.0)

    assert columns["paper_sample_eval_sample_count"] == 2
    assert columns["truth_grid_matched_count"] == 6
    assert columns["truth_grid_eval_sample_count"] == 6
    assert columns["truth_grid_coverage"] == 1.0


def test_leaderboard_ranks_by_mean_3d_error_with_rmse_secondary_only():
    report = _load_report_module()
    rows = [
        {
            "flight": "Opt1",
            "method": "low-rmse-high-mean",
            "row_type": "tracking",
            "mean_3d_error_m": 15.0,
            "rmse_3d_error_m": 15.1,
        },
        {
            "flight": "Opt1",
            "method": "high-rmse-low-mean",
            "row_type": "tracking",
            "mean_3d_error_m": 10.0,
            "rmse_3d_error_m": 30.0,
        },
        {
            "flight": "Opt1",
            "method": "oracle",
            "row_type": "oracle",
            "mean_3d_error_m": 1.0,
            "rmse_3d_error_m": 1.1,
        },
        {
            "flight": "Opt2",
            "method": "missing-metric",
            "row_type": "tracking",
            "mean_3d_error_m": "",
            "rmse_3d_error_m": 0.1,
        },
    ]

    leaderboard = report.build_leaderboard_rows(rows)
    opt1_tracking = [
        row for row in leaderboard if row["flight"] == "Opt1" and row["row_type"] == "tracking"
    ]
    opt1_oracle = [
        row for row in leaderboard if row["flight"] == "Opt1" and row["row_type"] == "oracle"
    ]
    opt2_tracking = [
        row for row in leaderboard if row["flight"] == "Opt2" and row["row_type"] == "tracking"
    ]

    assert [row["method"] for row in opt1_tracking] == [
        "high-rmse-low-mean",
        "low-rmse-high-mean",
    ]
    assert [row["rank"] for row in opt1_tracking] == [1, 2]
    assert opt1_oracle[0]["rank"] == 1
    assert opt2_tracking[0]["rank"] == ""
    assert all(row["paper_primary_metric"] == report.PAPER_PRIMARY_METRIC for row in leaderboard)


def test_tracklet_viterbi_command_uses_canonical_wrapper(tmp_path: Path):
    report = _load_report_module()
    args = SimpleNamespace(
        dataset_root=tmp_path / "dataset",
        radar_catprob_threshold=0.4,
        smoother="fixed-lag",
        smoother_lag_s=20.0,
    )

    command = report._tracklet_viterbi_command(args, "Opt1", tmp_path / "out")

    assert command[:4] == [
        sys.executable,
        "-m",
        "raft_uav.tracklet_viterbi_cli",
        "run-baseline",
    ]
    assert "run_tracklet_viterbi_baseline.py" not in " ".join(command)
    assert command[command.index("--radar-association") + 1] == "tracklet-viterbi"
    assert command[command.index("--tracklet-variant") + 1] == "range-covariance"
    assert command[command.index("--tracklet-replay-tracker") + 1] == "imm"
    assert command[command.index("--robust-update") + 1] == "nis-inflate"
    assert command[command.index("--rf-inflation-alpha") + 1] == "0.5"
    assert command[command.index("--radar-inflation-alpha") + 1] == "0.5"
