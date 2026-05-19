from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from raft_uav import cli
from raft_uav.baselines.kalman import TrackingMeasurement


def test_run_baseline_passes_sensor_measurements_to_map_smoother(monkeypatch, tmp_path):
    """The CLI should pass sensor measurements to robust MAP smoothing."""

    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    rf_frame = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [1.0],
            "north_m": [2.0],
            "up_m": [0.0],
        }
    )
    rf_measurement = TrackingMeasurement(
        time_s=0.0,
        vector=np.array([1.0, 2.0]),
        covariance=np.eye(2),
        source="rf",
    )
    baseline_records = [
        {
            "time_s": 0.0,
            "source": "rf",
            "state": np.zeros(6),
            "covariance": np.eye(6),
            "accepted": True,
            "measurement_dim": 2,
            "update_action": "updated",
        }
    ]
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "select_flight",
        lambda _dataset_root, _flight_name: SimpleNamespace(
            name="synthetic-flight",
            root=tmp_path,
            rf_csv=Path("rf.csv"),
            radar_json=None,
            truth_txt=Path("truth.txt"),
        ),
    )
    monkeypatch.setattr(cli, "read_truth", lambda _path: object())
    monkeypatch.setattr(cli, "normalize_truth", lambda _raw: (truth, object(), 0.0))
    monkeypatch.setattr(cli, "read_rf_csv", lambda _path: object())
    monkeypatch.setattr(
        cli, "normalize_rf", lambda _raw, _projector, _origin_time: rf_frame
    )
    monkeypatch.setattr(cli, "rf_measurements_to_enu", lambda _frame: [rf_measurement])
    monkeypatch.setattr(
        cli, "select_radar_measurement_rows", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(cli, "radar_measurements_to_enu", lambda _frame: [])
    monkeypatch.setattr(
        cli,
        "run_async_cv_baseline",
        lambda measurements, **_kwargs: baseline_records,
    )

    def fake_smooth_tracking_records(records, **kwargs):
        captured["records"] = records
        captured["kwargs"] = kwargs
        return records

    monkeypatch.setattr(cli, "smooth_tracking_records", fake_smooth_tracking_records)
    monkeypatch.setattr(cli, "_write_trajectory_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "build_diagnostic_summary", lambda **kwargs: {})
    monkeypatch.setattr(
        cli,
        "_baseline_metrics",
        lambda **_kwargs: {
            "accepted_measurements": 1,
            "rejected_measurements": 0,
            "reweighted_measurements": 0,
            "selected_radar_track_ids": [],
            "position_error_2d": {"rmse_m": 0.0},
            "position_error_3d": {"rmse_m": 0.0},
        },
    )

    result = cli._run_baseline(
        dataset_root=tmp_path,
        flight_name="synthetic-flight",
        output_dir=tmp_path / "out",
        acceleration_std=4.0,
        radar_association="catprob",
        legacy_radar_selection="none",
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
        smoother="robust-map",
        smoother_lag_s=20.0,
        max_eval_time_delta_s=2.0,
        enable_gating=False,
        robust_update="none",
        rf_gate_prob=0.99,
        radar_gate_prob=0.99,
        enable_association_safety_gate=True,
        rf_safety_gate_prob=0.9999999,
        radar_safety_gate_prob=0.9999999,
        rf_max_residual_m=750.0,
        radar_max_residual_m=0.0,
        rf_inflation_alpha=1.0,
        radar_inflation_alpha=1.0,
    )

    assert result == 0
    assert captured["kwargs"]["method"] == "robust-map"
    passed_measurements = captured["kwargs"]["measurements"]
    assert len(passed_measurements) == 1
    assert passed_measurements[0] is rf_measurement
    assert captured["records"] is baseline_records
