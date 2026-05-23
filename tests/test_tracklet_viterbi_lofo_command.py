from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_tracklet_viterbi_lofo.py"
spec = importlib.util.spec_from_file_location("run_tracklet_viterbi_lofo", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _args(tmp_path: Path, *, disable_rf_anchor: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root=tmp_path / "dataset",
        acceleration_std=4.0,
        candidate_threshold=0.4,
        max_candidates_per_frame=8,
        missed_detection_cost=7.0,
        track_switch_cost=8.0,
        catprob_weight=2.5,
        anchor_nis_weight=0.35,
        transition_nis_weight=1.0,
        velocity_nis_weight=0.15,
        max_speed_mps=55.0,
        range_gate_m=850.0,
        fixed_lag_s=20.0,
        max_eval_time_delta_s=2.0,
        rf_gate_prob=0.99,
        radar_gate_prob=0.99,
        rf_safety_gate_prob=0.9999999,
        radar_safety_gate_prob=0.9999999,
        rf_max_residual_m=750.0,
        radar_max_residual_m=0.0,
        robust_update="nis-inflate",
        rf_inflation_alpha=0.5,
        radar_inflation_alpha=0.5,
        disable_rf_anchor=disable_rf_anchor,
    )


def test_tracklet_lofo_command_uses_canonical_wrapper(tmp_path: Path):
    command = module._tracklet_viterbi_command(_args(tmp_path), "Opt1", tmp_path / "run")
    command_text = [str(item) for item in command]

    assert command_text[:4] == [
        sys.executable,
        "-m",
        "raft_uav.tracklet_viterbi_cli",
        "run-baseline",
    ]
    assert "scripts/run_tracklet_viterbi_baseline.py" not in command_text
    assert command_text[command_text.index("--radar-association") + 1] == "tracklet-viterbi"
    assert command_text[command_text.index("--tracklet-variant") + 1] == "range-covariance"
    assert command_text[command_text.index("--tracklet-replay-tracker") + 1] == "imm"
    assert command_text[command_text.index("--tracklet-max-candidates") + 1] == "8"
    assert command_text[command_text.index("--tracklet-missed-detection-cost") + 1] == "7.0"
    assert command_text[command_text.index("--tracklet-track-switch-cost") + 1] == "8.0"
    assert "--max-candidates-per-frame" not in command_text
    assert "--missed-detection-cost" not in command_text


def test_tracklet_lofo_disable_rf_anchor_uses_runtime_flag(tmp_path: Path):
    command = module._tracklet_viterbi_command(
        _args(tmp_path, disable_rf_anchor=True),
        "Opt3",
        tmp_path / "run",
    )
    command_text = [str(item) for item in command]

    assert "--disable-tracklet-rf-anchor" in command_text
    assert "--disable-rf-anchor" not in command_text
