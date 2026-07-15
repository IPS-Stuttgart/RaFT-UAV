import numpy as np
import pandas as pd

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
