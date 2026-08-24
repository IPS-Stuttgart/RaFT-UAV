from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.bias_cli import train_bias_model


def test_train_bias_model_rejects_duplicate_requested_flights(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="requested_flights must not contain duplicate flight names: flight-a",
    ):
        train_bias_model(
            dataset_root=tmp_path,
            requested_flights=["flight-a", "flight-b", "flight-a"],
            output_path=tmp_path / "bias_model.json",
            max_time_delta_s=2.0,
            max_position_error_m=250.0,
            ridge_alpha=1.0,
            min_samples=5,
        )


def test_train_bias_model_rejects_selectors_for_same_flight(tmp_path: Path) -> None:
    flight_dir = tmp_path / "RF Sensor and Radar" / "flight-a"
    flight_dir.mkdir(parents=True)
    (flight_dir / "rf.csv").write_text(
        "Time,Latitude,Longitude\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "bias_model.json"

    with pytest.raises(
        ValueError,
        match="requested_flights must not resolve to duplicate flights: flight-a",
    ):
        train_bias_model(
            dataset_root=tmp_path,
            requested_flights=["flight-a", "a"],
            output_path=output_path,
            max_time_delta_s=2.0,
            max_position_error_m=250.0,
            ridge_alpha=1.0,
            min_samples=5,
        )

    assert not output_path.exists()
