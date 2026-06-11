from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from raft_uav import cli as base_cli
from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.robust_map import RobustMapSmootherConfig
from raft_uav.baselines.smoothing import smooth_tracking_records


def _synthetic_records_and_measurements():
    times = np.arange(8, dtype=float)
    truth = np.column_stack(
        [
            10.0 * times,
            0.5 * times,
            np.zeros_like(times),
            np.full_like(times, 10.0),
            np.full_like(times, 0.5),
            np.zeros_like(times),
        ]
    )
    measurements = []
    records = []
    covariance = np.diag([2.0**2, 2.0**2, 5.0**2])
    record_covariance = np.diag([4.0**2, 4.0**2, 8.0**2, 3.0**2, 3.0**2, 3.0**2])
    for idx, time_s in enumerate(times):
        vector = truth[idx, :3].copy()
        if idx == 3:
            vector[0] += 150.0
            vector[1] -= 120.0
        measurements.append(
            TrackingMeasurement(time_s=time_s, vector=vector, covariance=covariance, source="radar")
        )
        state = truth[idx].copy()
        state[:3] = vector
        records.append(
            {
                "time_s": float(time_s),
                "source": "radar",
                "state": state,
                "covariance": record_covariance.copy(),
                "accepted": True,
                "measurement_dim": 3,
            }
        )
    return truth, records, measurements


def test_robust_map_downweights_single_position_outlier():
    truth, records, measurements = _synthetic_records_and_measurements()
    smoothed = smooth_tracking_records(
        records,
        method="robust-map",
        acceleration_std_mps2=0.5,
        measurements=measurements,
        robust_map_config=RobustMapSmootherConfig(
            loss="cauchy",
            loss_scale=2.0,
            max_iterations=100,
            process_position_floor_m=0.5,
            process_velocity_floor_mps=0.2,
        ),
    )

    before = np.linalg.norm(records[3]["state"][:2] - truth[3, :2])
    after = np.linalg.norm(smoothed[3]["state"][:2] - truth[3, :2])
    assert after < 0.5 * before
    assert smoothed[0]["smoother_method"] == "robust-map"
    assert smoothed[0]["map_matched_measurements"] == len(records)
    assert smoothed[0]["map_final_cost"] <= smoothed[0]["map_initial_cost"]


def test_fixed_lag_map_emits_one_record_per_input():
    _truth, records, measurements = _synthetic_records_and_measurements()
    smoothed = smooth_tracking_records(
        records,
        method="fixed-lag-map",
        acceleration_std_mps2=0.5,
        lag_s=3.0,
        measurements=measurements,
        robust_map_config=RobustMapSmootherConfig(max_iterations=50),
    )

    assert len(smoothed) == len(records)
    assert smoothed[0]["smoother_method"] == "fixed-lag-map"
    assert smoothed[0]["smoother_lag_s"] == 3.0
    assert "map_success" in smoothed[0]


def test_robust_map_falls_back_to_posterior_pseudo_measurements():
    _truth, records, _measurements = _synthetic_records_and_measurements()
    smoothed = smooth_tracking_records(
        records,
        method="robust-map",
        acceleration_std_mps2=0.5,
        robust_map_config=RobustMapSmootherConfig(max_iterations=20),
    )

    assert len(smoothed) == len(records)
    assert smoothed[0]["smoother_method"] == "robust-map"
    assert smoothed[0]["map_matched_measurements"] == len(records)


def test_baseline_cli_passes_original_measurements_to_map_smoother(monkeypatch, tmp_path):
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    rf_frame = pd.DataFrame({"time_s": [0.0]})
    rf_measurement = TrackingMeasurement(
        time_s=0.0,
        vector=np.array([0.0, 0.0]),
        covariance=np.eye(2),
        source="rf",
    )
    record = {
        "time_s": 0.0,
        "source": "rf",
        "state": np.zeros(6),
        "covariance": np.eye(6),
        "accepted": True,
        "measurement_dim": 2,
        "update_action": "initialized",
        "nis": 0.0,
        "gate_threshold": None,
        "safety_gate_threshold": None,
        "residual_gate_threshold_m": None,
        "covariance_scale": 1.0,
        "inflation_alpha": None,
        "residual_norm_m": 0.0,
    }
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        base_cli,
        "select_flight",
        lambda *_args, **_kwargs: SimpleNamespace(
            name="OptX",
            truth_txt=Path("truth.txt"),
            rf_csv=Path("rf.csv"),
            radar_json=None,
        ),
    )
    monkeypatch.setattr(base_cli, "read_truth", lambda _path: object())
    monkeypatch.setattr(
        base_cli,
        "normalize_truth",
        lambda _raw: (truth.copy(), object(), 0.0),
    )
    monkeypatch.setattr(base_cli, "read_rf_csv", lambda _path: pd.DataFrame())
    monkeypatch.setattr(
        base_cli,
        "normalize_rf",
        lambda *_args, **_kwargs: rf_frame.copy(),
    )
    monkeypatch.setattr(base_cli, "rf_measurements_to_enu", lambda _rf: [rf_measurement])
    monkeypatch.setattr(
        base_cli,
        "select_radar_measurement_rows",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(base_cli, "radar_measurements_to_enu", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(base_cli, "run_async_cv_baseline", lambda *_args, **_kwargs: [record])

    def fake_smooth(records, **kwargs):
        seen["method"] = kwargs["method"]
        seen["measurement_ids"] = [id(item) for item in kwargs["measurements"]]
        return records

    monkeypatch.setattr(base_cli, "smooth_tracking_records", fake_smooth)
    monkeypatch.setattr(base_cli, "_write_trajectory_plot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(base_cli, "build_diagnostic_summary", lambda **_kwargs: {})
    monkeypatch.setattr(base_cli, "_baseline_metrics", lambda **_kwargs: _minimal_metrics())

    assert (
        base_cli._run_baseline(
            dataset_root=tmp_path,
            flight_name="OptX",
            output_dir=tmp_path / "out",
            acceleration_std=4.0,
            radar_association="catprob",
            legacy_radar_selection=None,
            rf_clock_offset_s=0.0,
            radar_clock_offset_s=0.0,
            rf_time_offset_correction_s=0.0,
            radar_time_offset_correction_s=0.0,
            calibration_bundle_path=None,
            radar_catprob_threshold=0.5,
            paper_compatible_catprob_threshold=None,
            paper_compatible_bootstrap_source="radar",
            radar_covariance_model="cartesian",
            radar_range_std_m=12.0,
            radar_range_std_fraction=0.005,
            radar_crossrange_angle_std_deg=1.5,
            radar_crossrange_min_std_m=5.0,
            radar_crossrange_max_std_m=80.0,
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
            enable_radar_velocity_update=False,
            radar_velocity_std_mps=12.0,
        )
        == 0
    )
    assert seen == {"method": "robust-map", "measurement_ids": [id(rf_measurement)]}


def _minimal_metrics() -> dict[str, object]:
    error = {"mean_m": 0.0, "std_m": 0.0, "rmse_m": 0.0, "max_m": 0.0}
    paper_error = {"mean_m": None, "std_m": None, "max_m": None}
    return {
        "accepted_measurements": 1,
        "rejected_measurements": 0,
        "reweighted_measurements": 0,
        "selected_radar_track_ids": [],
        "position_error_2d": error,
        "position_error_3d": error,
        "paper_position_error_2d": paper_error,
        "paper_position_error_3d": paper_error,
    }
