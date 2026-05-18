import pandas as pd

from raft_uav.calibration.lofo_combined import apply_time_offsets, shift_time


def test_shift_time_preserves_original_time_column():
    frame = pd.DataFrame({"time_s": [1.0, 2.0], "east_m": [0.0, 1.0]})

    shifted = shift_time(frame, 0.5, source="radar")

    assert shifted["time_s"].tolist() == [1.5, 2.5]
    assert shifted["time_s_uncorrected"].tolist() == [1.0, 2.0]
    assert shifted["radar_time_offset_s"].tolist() == [0.5, 0.5]


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
