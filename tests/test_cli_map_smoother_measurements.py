from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from raft_uav import cli
from raft_uav.baselines.kalman import TrackingMeasurement


def test_run_baseline_passes_original_measurements_to_map_smoother(monkeypatch, tmp_path):
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    rf_frame = pd.DataFrame({"time_s": [0.0]})
    measurement = TrackingMeasurement(
        time_s=0.0,
        vector=np.array([12.0, -3.0]),
        covariance=np.diag([4.0, 9.0]),
        source="rf",
    )
    records = [
        {
            "time_s": 0.0,
            "source": "rf",
            "state": np.zeros(6),
            "covariance": np.eye(6),
            "accepted": True,
            "measurement_dim": 2,
        }
    ]
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "select_flight",
        lambda _root, name: SimpleNamespace(
            name=name,
            truth_txt=Path("truth.txt"),
            rf_csv=Path("rf.csv"),
            radar_json=None,
        ),
    )
    monkeypatch.setattr(cli, "read_truth", lambda _path: object())
    monkeypatch.setattr(cli, "normalize_truth", lambda _raw: (truth, object(), 0.0))
    monkeypatch.setattr(cli, "read_rf_csv", lambda _path: rf_frame)
    monkeypatch.setattr(cli, "normalize_rf", lambda _raw, _projector, _origin_time: rf_frame)
    monkeypatch.setattr(cli, "rf_measurements_to_enu", lambda _rf: [measurement])
    monkeypatch.setattr(cli, "_inside_truth_window", lambda frame, _truth: frame)
    monkeypatch.setattr(cli, "run_async_cv_baseline", lambda measurements, **_kwargs: records)

    def fake_smooth_tracking_records(records_arg, **kwargs):
        seen["records"] = records_arg
        seen["measurements"] = kwargs.get("measurements")
        seen["method"] = kwargs.get("method")
        return records_arg

    monkeypatch.setattr(cli, "smooth_tracking_records", fake_smooth_tracking_records)
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
    monkeypatch.setattr(cli, "build_diagnostic_summary", lambda **_kwargs: {})
    monkeypatch.setattr(cli, "_write_trajectory_plot", lambda *_args, **_kwargs: None)

    assert (
        cli.main(
            [
                "run-baseline",
                str(tmp_path),
                "--flight",
                "synthetic-flight",
                "--output-dir",
                str(tmp_path / "out"),
                "--smoother",
                "robust-map",
                "--radar-association",
                "catprob",
            ]
        )
        == 0
    )

    passed_measurements = seen["measurements"]
    assert seen["method"] == "robust-map"
    assert isinstance(passed_measurements, list)
    assert len(passed_measurements) == 1
    assert passed_measurements[0] is measurement
