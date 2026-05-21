from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_leave_flight_out_sota as lfo  # noqa: E402


def test_truth_coverage_counts_truth_timestamps_with_nearby_estimate() -> None:
    truth_times = np.array([0.0, 1.0, 2.0, 3.0])
    estimate_times = np.array([0.1, 2.2, 10.0])

    coverage = lfo.truth_coverage(truth_times, estimate_times, max_time_delta_s=0.25)

    assert coverage["truth_rows"] == 4
    assert coverage["covered_truth_rows"] == 2
    assert coverage["truth_coverage_rate"] == 0.5


def test_summarize_scalar_errors_includes_tail_metrics() -> None:
    summary = lfo.summarize_scalar_errors(np.array([1.0, 2.0, 3.0, 4.0]))

    assert summary["count"] == 4.0
    assert np.isclose(summary["rmse_m"], np.sqrt((1.0 + 4.0 + 9.0 + 16.0) / 4.0))
    assert np.isclose(summary["p50_m"], 2.5)
    assert np.isclose(summary["p90_m"], 3.7)
    assert np.isclose(summary["p95_m"], 3.85)
    assert np.isclose(summary["p99_m"], 3.97)


def test_aggregate_method_rows_pools_errors_and_ranks_methods() -> None:
    methods = [
        lfo.MethodSpec("method_a", "baseline", "a"),
        lfo.MethodSpec("method_b", "baseline", "b"),
    ]
    evaluations = {
        "method_a": [
            lfo.RunEvaluation(
                row={"posterior_records": 2, "selected_radar_rows": 1},
                errors_2d_m=np.array([1.0, 2.0]),
                errors_3d_m=np.array([2.0, 3.0]),
                covered_truth_rows=8,
                truth_rows=10,
            )
        ],
        "method_b": [
            lfo.RunEvaluation(
                row={"posterior_records": 2, "selected_radar_rows": 1},
                errors_2d_m=np.array([0.5, 1.0]),
                errors_3d_m=np.array([1.0, 1.5]),
                covered_truth_rows=10,
                truth_rows=10,
            )
        ],
    }

    rows = lfo._aggregate_method_rows(methods, evaluations)
    by_method = {row["method"]: row for row in rows}

    assert by_method["method_a"]["truth_coverage_rate"] == 0.8
    assert by_method["method_b"]["truth_coverage_rate"] == 1.0
    assert by_method["method_b"]["rank_rmse_3d"] == 1
    assert by_method["method_a"]["rank_rmse_3d"] == 2


def test_sota_protocol_exposes_online_fixed_lag_and_rts_imm_rows() -> None:
    assert lfo.METHODS["imm_catprob"].runner == "imm"
    assert lfo.METHODS["imm_catprob_fixed_lag"].fixed_lag is True
    assert lfo.METHODS["imm_catprob_rts"].rts is True
    assert lfo.METHODS["imm_tracklet_viterbi_fixed_lag"].fixed_lag is True


def test_calibrated_heteroscedastic_method_is_available() -> None:
    method = lfo.METHODS["hetero_cv_lofo_nis_fixed_lag"]

    assert method.runner == "hetero"
    assert method.fixed_lag
    assert method.nis_calibrated


def test_nis_calibration_summary_csv_flattens_groups(tmp_path: Path) -> None:
    payload = {
        "groups": {
            "radar:3": {
                "source": "radar",
                "measurement_dim": 3,
                "count": 42,
                "enabled": True,
                "applied_scale": 1.5,
            }
        }
    }
    path = tmp_path / "summary.csv"

    lfo._write_nis_calibration_summary_csv(payload, path)

    text = path.read_text(encoding="utf-8")
    assert "radar:3" in text
    assert "applied_scale" in text


def test_imm_runner_forwards_smoother_options(monkeypatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        lfo, "_run", lambda command, **_kwargs: captured.append(list(command))
    )
    args = SimpleNamespace(
        dataset_root=tmp_path / "dataset",
        acceleration_std=4.0,
        candidate_threshold=0.4,
        fixed_lag_s=12.0,
        rf_gate_prob=0.99,
        radar_gate_prob=0.99,
        rf_inflation_alpha=0.5,
        radar_inflation_alpha=0.5,
    )

    lfo._run_method(
        args,
        lfo.METHODS["imm_catprob_fixed_lag"],
        "Opt1",
        tmp_path / "run",
        tmp_path / "model.json",
    )

    command = captured[0]
    assert command[:3] == [sys.executable, "-m", "raft_uav.imm_cli"]
    assert "--smoother" in command
    assert command[command.index("--smoother") + 1] == "fixed-lag"
    assert "--smoother-lag-s" in command
    assert command[command.index("--smoother-lag-s") + 1] == "12.0"
    assert "--acceleration-std" in command
    assert command[command.index("--acceleration-std") + 1] == "4.0"
