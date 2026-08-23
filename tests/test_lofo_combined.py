import numpy as np
import pandas as pd

from raft_uav.calibration.lofo_combined import (
    aggregate_offset_sweeps,
    apply_time_offsets,
    shift_time,
)


def test_shift_time_preserves_original_time_column():
    frame = pd.DataFrame({"time_s": [1.0, 2.0], "east_m": [0.0, 1.0]})

    shifted = shift_time(frame, 0.5, source="radar")

    assert shifted["time_s"].tolist() == [1.5, 2.5]
    assert shifted["time_s_uncorrected"].tolist() == [1.0, 2.0]
    assert shifted["radar_time_offset_s"].tolist() == [0.5, 0.5]


def test_shift_time_rebases_reapplied_offsets_on_original_timestamps():
    frame = pd.DataFrame({"time_s": [1.0, 2.0], "east_m": [0.0, 1.0]})

    first = shift_time(frame, 0.5, source="radar")
    reapplied = shift_time(first, -0.25, source="radar")

    assert reapplied["time_s"].tolist() == [0.75, 1.75]
    assert reapplied["time_s_uncorrected"].tolist() == [1.0, 2.0]
    assert reapplied["radar_time_offset_s"].tolist() == [-0.25, -0.25]


def test_apply_time_offsets_shifts_rf_and_radar_only():
    item = {
        "truth": pd.DataFrame({"time_s": [10.0]}),
        "rf": pd.DataFrame({"time_s": [1.0]}),
        "radar": pd.DataFrame({"time_s": [2.0]}),
    }

    shifted = apply_time_offsets(item, rf_tau_s=-0.25, radar_tau_s=0.75)

    assert shifted["truth"]["time_s"].tolist() == [10.0]
    assert shifted["rf"]["time_s"].tolist() == [0.75]
    assert shifted["radar"]["time_s"].tolist() == [2.75]


def test_aggregate_offset_sweeps_pools_rmse_from_squared_error():
    sweep = pd.DataFrame(
        {
            "tau_s": [0.0, 0.0, 1.0, 1.0],
            "flight": ["a", "b", "a", "b"],
            "candidate_count": [1, 9, 1, 9],
            "selected_count": [1, 9, 1, 9],
            "matched_count": [1, 9, 1, 9],
            "rmse_error_m": [0.0, 7.0, 10.0, 6.0],
        }
    )

    aggregate = aggregate_offset_sweeps(sweep, "rmse")

    expected = [np.sqrt(441.0 / 10.0), np.sqrt(424.0 / 10.0)]
    assert np.allclose(aggregate["rmse_error_m"].to_numpy(float), expected)
    assert aggregate.loc[aggregate["rmse_error_m"].idxmin(), "tau_s"] == 1.0


def test_aggregate_offset_sweeps_uses_global_maximum_across_flights():
    sweep = pd.DataFrame(
        {
            "tau_s": [0.0, 0.0, 1.0, 1.0],
            "flight": ["a", "b", "a", "b"],
            "candidate_count": [1, 9, 1, 9],
            "selected_count": [1, 9, 1, 9],
            "matched_count": [1, 9, 1, 9],
            "max_error_m": [20.0, 0.0, 3.0, 3.0],
        }
    )

    aggregate = aggregate_offset_sweeps(sweep, "max")

    assert aggregate["max_error_m"].tolist() == [20.0, 3.0]
    assert aggregate.loc[aggregate["max_error_m"].idxmin(), "tau_s"] == 1.0
