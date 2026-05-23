from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_lofo_radar_covariance_tuning.py"
spec = importlib.util.spec_from_file_location("run_lofo_radar_covariance_tuning", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _command_args(**overrides):
    values = {
        "dataset_root": Path("data/raw/AADM2025Dryad"),
        "baseline_runner": "canonical-tracklet",
        "baseline_script": None,
        "baseline_arg": [],
        "radar_catprob_threshold": 0.4,
        "acceleration_std": 4.0,
        "fixed_lag_s": 20.0,
        "rf_gate_prob": 0.99,
        "radar_gate_prob": 0.99,
        "rf_safety_gate_prob": 0.9999999,
        "radar_safety_gate_prob": 0.9999999,
        "rf_max_residual_m": 750.0,
        "radar_max_residual_m": 0.0,
        "robust_update": "nis-inflate",
        "rf_inflation_alpha": 0.5,
        "radar_inflation_alpha": 0.5,
    }
    values.update(overrides)
    return module.argparse.Namespace(**values)


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_parse_float_list_rejects_nonpositive_values():
    assert module._parse_float_list("1,2.5") == [1.0, 2.5]
    try:
        module._parse_float_list("1,0")
    except ValueError as exc:
        assert "finite and positive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_candidate_environment_uses_raft_uav_radar_variables():
    candidate = module.RadarCovarianceCandidate(
        candidate_id="cov0000",
        range_std_m=5.0,
        azimuth_std_deg=2.0,
        elevation_std_deg=3.0,
        min_std_m=4.0,
        max_std_m=250.0,
    )

    env = candidate.environment()

    assert env["RAFT_UAV_RADAR_COVARIANCE_MODE"] == "range-angle"
    assert env["RAFT_UAV_RADAR_RANGE_STD_M"] == "5"
    assert env["RAFT_UAV_RADAR_AZIMUTH_STD_DEG"] == "2"
    assert env["RAFT_UAV_RADAR_ELEVATION_STD_DEG"] == "3"
    assert env["RAFT_UAV_RADAR_COVARIANCE_MIN_STD_M"] == "4"
    assert env["RAFT_UAV_RADAR_COVARIANCE_MAX_STD_M"] == "250"


def test_select_candidate_aggregates_training_metric():
    sweep = pd.DataFrame(
        [
            {"candidate_id": "bad", "metric_value": 10.0},
            {"candidate_id": "bad", "metric_value": 12.0},
            {"candidate_id": "good", "metric_value": 8.0},
            {"candidate_id": "good", "metric_value": 9.0},
        ]
    )

    selected = module._select_candidate(sweep, metric_column="metric_value", aggregate="mean")

    assert selected["candidate_id"] == "good"
    assert selected["aggregate_metric_value"] == 8.5
    assert selected["finite_train_flights"] == 2


def test_default_baseline_command_uses_canonical_sota_tracklet_wrapper():
    args = _command_args(baseline_arg=["--max-eval-time-delta-s", "1.5"])

    command = module._baseline_command(args, flight="Opt2", output_dir=Path("out"))

    assert command[:4] == [
        module.sys.executable,
        "-m",
        "raft_uav.tracklet_viterbi_cli",
        "run-baseline",
    ]
    assert _value_after(command, "--radar-association") == "tracklet-viterbi"
    assert _value_after(command, "--tracklet-variant") == "range-covariance"
    assert _value_after(command, "--tracklet-replay-tracker") == "imm"
    assert _value_after(command, "--radar-catprob-threshold") == "0.4"
    assert _value_after(command, "--acceleration-std") == "4"
    assert _value_after(command, "--rf-gate-prob") == "0.99"
    assert _value_after(command, "--radar-gate-prob") == "0.99"
    assert _value_after(command, "--rf-safety-gate-prob") == "0.9999999"
    assert _value_after(command, "--radar-safety-gate-prob") == "0.9999999"
    assert _value_after(command, "--rf-max-residual-m") == "750"
    assert _value_after(command, "--radar-max-residual-m") == "0"
    assert _value_after(command, "--robust-update") == "nis-inflate"
    assert _value_after(command, "--rf-inflation-alpha") == "0.5"
    assert _value_after(command, "--radar-inflation-alpha") == "0.5"
    assert _value_after(command, "--smoother") == "fixed-lag"
    assert _value_after(command, "--smoother-lag-s") == "20"
    assert command[-2:] == ["--max-eval-time-delta-s", "1.5"]


def test_baseline_script_option_preserves_legacy_command_path():
    args = _command_args(
        baseline_script=Path("scripts/custom_tracklet.py"),
        baseline_arg=["--sentinel"],
    )

    command = module._baseline_command(args, flight="Opt1", output_dir=Path("out"))

    assert command[:2] == [module.sys.executable, "scripts/custom_tracklet.py"]
    assert command[2:7] == [
        "data/raw/AADM2025Dryad",
        "--flight",
        "Opt1",
        "--output-dir",
        "out",
    ]
    assert command[-1] == "--sentinel"
