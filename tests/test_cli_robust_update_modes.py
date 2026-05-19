from pathlib import Path
from types import SimpleNamespace
import inspect

import numpy as np
import pandas as pd
import pytest

from raft_uav import cli
from raft_uav.baselines.update_logic import DEFAULT_HUBER_THRESHOLD, DEFAULT_STUDENT_T_DOF


@pytest.mark.parametrize("mode", ["nis-inflate", "student-t", "huber"])
def test_run_baseline_cli_accepts_all_backend_robust_update_modes(monkeypatch, mode):
    captured = {}
    signature = inspect.signature(cli._run_baseline)

    def fake_run_baseline(*args):
        captured.update(signature.bind(*args).arguments)
        return 0

    monkeypatch.setattr(cli, "_run_baseline", fake_run_baseline)

    assert cli.main(["run-baseline", "dataset-root", "--flight", "flight-1", "--robust-update", mode]) == 0
    assert captured["dataset_root"] == Path("dataset-root")
    assert captured["flight_name"] == "flight-1"
    assert captured["robust_update"] == mode


def test_run_baseline_cli_rejects_unknown_robust_update_mode(monkeypatch):
    monkeypatch.setattr(cli, "_run_baseline", lambda *args: 0)

    with pytest.raises(SystemExit):
        cli.main(
            [
                "run-baseline",
                "dataset-root",
                "--flight",
                "flight-1",
                "--robust-update",
                "typo-mode",
            ]
        )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "nis-inflate",
            {
                "rf_gate_probability": 0.91,
                "radar_gate_probability": 0.92,
                "student_t_degrees_of_freedom": None,
                "huber_threshold": None,
                "rf_inflation_alpha": 1.3,
                "radar_inflation_alpha": 1.4,
            },
        ),
        (
            "student-t",
            {
                "rf_gate_probability": None,
                "radar_gate_probability": None,
                "student_t_degrees_of_freedom": DEFAULT_STUDENT_T_DOF,
                "huber_threshold": None,
                "rf_inflation_alpha": None,
                "radar_inflation_alpha": None,
            },
        ),
        (
            "huber",
            {
                "rf_gate_probability": None,
                "radar_gate_probability": None,
                "student_t_degrees_of_freedom": None,
                "huber_threshold": DEFAULT_HUBER_THRESHOLD,
                "rf_inflation_alpha": None,
                "radar_inflation_alpha": None,
            },
        ),
    ],
)
def test_baseline_metrics_documents_mode_specific_robust_update_parameters(mode, expected):
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    estimate_frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "source": ["rf", "radar"],
            "east_m": [0.0, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "accepted": [True, True],
            "update_action": ["student_t" if mode == "student-t" else "updated", "huberized" if mode == "huber" else "inflated"],
            "nis": [1.0, 2.0],
            "covariance_scale": [1.0, 2.5],
        }
    )
    metrics = cli._baseline_metrics(
        flight_name="flight-1",
        flight=SimpleNamespace(truth_txt=None, rf_csv=None, radar_json=None),
        truth=truth,
        rf=pd.DataFrame(),
        radar=pd.DataFrame(),
        selected_radar=pd.DataFrame(),
        estimate_frame=estimate_frame,
        acceleration_std=4.0,
        radar_association="catprob",
        radar_catprob_threshold=0.5,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
        track_switch_nis_ratio=0.5,
        geometry_velocity_std=12.0,
        geometry_velocity_weight=0.25,
        geometry_switch_penalty=4.0,
        geometry_catprob_weight=2.0,
        rf_anchor_weight=0.35,
        rf_anchor_time_gate_s=2.0,
        rf_anchor_nis_cap=25.0,
        rf_anchor_gate_nis=25.0,
        pda_nis_temperature=1.0,
        pda_catprob_exponent=1.0,
        track_bank_max_hypotheses=16,
        track_bank_max_assignments=16,
        track_bank_max_candidates=16,
        track_bank_gate_prob=0.9999999,
        track_bank_detection_prob=0.999,
        track_bank_clutter_intensity=1.0e-12,
        track_bank_prune_delta=80.0,
        stable_segment_min_frames=100,
        stable_segment_max_transition_speed_mps=65.0,
        stable_segment_range_gate_m=800.0,
        stable_segment_interpolation_max_gap_s=5.0,
        stable_segment_interpolation_max_speed_mps=65.0,
        stable_segment_interpolation_std_scale=2.0,
        stable_segment_interpolation_gap_std_mps=12.0,
        stable_segment_rf_score_weight=1.0,
        stable_segment_rf_time_gate_s=2.0,
        stable_segment_rf_nis_cap=25.0,
        smoother="none",
        smoother_lag_s=20.0,
        max_eval_time_delta_s=2.0,
        enable_gating=False,
        robust_update=mode,
        rf_gate_prob=0.91,
        radar_gate_prob=0.92,
        enable_association_safety_gate=True,
        rf_safety_gate_prob=0.9999999,
        radar_safety_gate_prob=0.9999999,
        rf_max_residual_m=750.0,
        radar_max_residual_m=0.0,
        rf_inflation_alpha=1.3,
        radar_inflation_alpha=1.4,
    )

    robust = metrics["robust_update"]
    assert robust["method"] == mode
    for key, value in expected.items():
        assert robust[key] == value
    assert metrics["reweighted_measurements"] == 1
    assert metrics["reweighted_by_source"] == {"radar": 1}
