from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import raft_uav.diagnostics.time_offset as time_offset_module
from raft_uav.diagnostics.time_offset import (
    best_offset_row,
    offset_grid,
    radar_frame_groups,
    sweep_positions_against_truth,
    truth_positions_at_times,
)


def test_offset_grid_is_inclusive():
    grid = offset_grid(-1.0, 1.0, 0.5)
    assert np.allclose(grid, np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))


def test_truth_positions_at_times_interpolates_and_masks_outside_window():
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 10.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 10.0],
        }
    )

    positions, mask = truth_positions_at_times(
        truth,
        np.array([5.0, 12.0]),
        max_delta_s=10.0,
    )

    assert mask.tolist() == [True, False]
    assert positions[0].tolist() == [50.0, 0.0, 5.0]


def test_truth_positions_at_times_handles_unsorted_truth_times():
    truth = pd.DataFrame(
        {
            "time_s": [10.0, 0.0],
            "east_m": [100.0, 0.0],
            "north_m": [0.0, 0.0],
            "up_m": [10.0, 0.0],
        }
    )

    positions, mask = truth_positions_at_times(
        truth,
        np.array([5.0]),
        max_delta_s=10.0,
    )

    assert mask.tolist() == [True]
    assert positions[0].tolist() == [50.0, 0.0, 5.0]


def test_position_offset_sweep_recovers_known_shift():
    truth = pd.DataFrame(
        {
            "time_s": np.arange(0.0, 11.0),
            "east_m": np.arange(0.0, 11.0),
            "north_m": np.zeros(11),
            "up_m": np.zeros(11),
        }
    )
    measurement_times = np.array([0.0, 1.0, 2.0, 3.0])
    measurement_positions = np.zeros((4, 3))
    measurement_positions[:, 0] = measurement_times + 2.0

    sweep = sweep_positions_against_truth(
        measurement_times_s=measurement_times,
        measurement_positions_m=measurement_positions,
        truth=truth,
        taus_s=offset_grid(-3.0, 3.0, 1.0),
        dimensions=2,
        max_truth_time_delta_s=1.0,
    )
    best = best_offset_row(sweep, objective="mean")

    assert float(best["tau_s"]) == 2.0
    assert float(best["mean_error_m"]) == 0.0


def test_run_time_offset_diagnostic_keeps_rows_shiftable_into_truth_window(
    monkeypatch,
):
    measurements = pd.DataFrame(
        {"time_s": [-2.0, -2.01, 0.0, 10.0, 12.0, 12.01]}
    )
    truth = pd.DataFrame({"time_s": [0.0, 10.0]})

    def fake_run(**_kwargs):
        filtered = time_offset_module._legacy._inside_truth_window(
            measurements,
            truth,
        )
        return {"time_s": filtered["time_s"].tolist()}

    monkeypatch.setattr(
        time_offset_module,
        "_original_run_time_offset_diagnostic",
        fake_run,
    )

    result = time_offset_module.run_time_offset_diagnostic(
        dataset_root=Path("."),
        flight_name="dummy",
        source="rf",
        tau_min_s=-2.0,
        tau_max_s=2.0,
        tau_step_s=1.0,
        write_plot=False,
    )

    assert result["time_s"] == [-2.0, 0.0, 10.0, 12.0]
    strict = time_offset_module._legacy._inside_truth_window(measurements, truth)
    assert strict["time_s"].tolist() == [0.0, 10.0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tau_min_s", np.nan),
        ("tau_max_s", np.inf),
        ("tau_step_s", True),
        ("tau_step_s", np.array([0.5])),
        ("max_truth_time_delta_s", np.nan),
        ("radar_catprob_threshold", np.nan),
    ],
)
def test_run_time_offset_diagnostic_rejects_malformed_numeric_controls_before_io(
    monkeypatch,
    field,
    value,
):
    def unexpected_run(**_kwargs):
        raise AssertionError("legacy diagnostic should not run")

    monkeypatch.setattr(
        time_offset_module,
        "_original_run_time_offset_diagnostic",
        unexpected_run,
    )
    kwargs = {
        "dataset_root": Path("."),
        "flight_name": "dummy",
        "source": "radar",
        "tau_min_s": -1.0,
        "tau_max_s": 1.0,
        "tau_step_s": 0.5,
        "radar_selection": "catprob-oracle-nearest",
        "radar_catprob_threshold": 0.4,
        "max_truth_time_delta_s": 2.0,
        "write_plot": False,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        time_offset_module.run_time_offset_diagnostic(**kwargs)


def test_run_time_offset_diagnostic_normalizes_valid_scalar_controls(monkeypatch):
    def capture_run(**kwargs):
        return kwargs

    monkeypatch.setattr(
        time_offset_module,
        "_original_run_time_offset_diagnostic",
        capture_run,
    )

    result = time_offset_module.run_time_offset_diagnostic(
        dataset_root=Path("."),
        flight_name="dummy",
        source="radar",
        tau_min_s=np.array(-1.0),
        tau_max_s=np.float64(1.0),
        tau_step_s="0.5",
        radar_catprob_threshold=np.array(0.4),
        max_truth_time_delta_s=np.float64(2.0),
        write_plot=False,
    )

    for field in (
        "tau_min_s",
        "tau_max_s",
        "tau_step_s",
        "radar_catprob_threshold",
        "max_truth_time_delta_s",
    ):
        assert isinstance(result[field], float)
    assert result["tau_min_s"] == -1.0
    assert result["tau_max_s"] == 1.0
    assert result["tau_step_s"] == 0.5
    assert result["radar_catprob_threshold"] == 0.4
    assert result["max_truth_time_delta_s"] == 2.0


@pytest.mark.parametrize("threshold", [np.nan, True, np.array([0.4])])
def test_catprob_candidate_pool_rejects_malformed_thresholds(threshold):
    candidates = pd.DataFrame({"cat_prob_uav": [0.2, 0.8]})

    with pytest.raises(ValueError, match="threshold must be a finite real scalar"):
        time_offset_module.catprob_candidate_pool(candidates, threshold)


def test_radar_frame_groups_preserves_rows_when_frame_index_is_incomplete():
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "frame_index": [10.0, 10.0, np.nan, np.nan],
            "track_id": [1, 2, 1, 2],
        }
    )

    groups = radar_frame_groups(radar)

    assert [group["time_s"].iloc[0] for group in groups] == [0.0, 1.0]
    assert sum(len(group) for group in groups) == len(radar)
    assert groups[1]["frame_index"].isna().all()


def test_radar_frame_groups_keeps_complete_frame_indices_distinct():
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "frame_index": [10, 11],
            "track_id": [1, 2],
        }
    )

    groups = radar_frame_groups(radar)

    assert [int(group["frame_index"].iloc[0]) for group in groups] == [10, 11]
    assert [len(group) for group in groups] == [1, 1]
