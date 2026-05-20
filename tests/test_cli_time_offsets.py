import pandas as pd

from raft_uav.cli import _apply_time_offset_correction


def test_apply_time_offset_correction_shifts_normalized_time():
    frame = pd.DataFrame({"time_s": [1.0, 2.5], "east_m": [0.0, 1.0]})

    corrected = _apply_time_offset_correction(frame, 0.75)

    assert corrected["time_s"].tolist() == [1.75, 3.25]
    assert corrected["time_s_uncorrected"].tolist() == [1.0, 2.5]
    assert corrected["time_offset_correction_s"].tolist() == [0.75, 0.75]
    assert frame["time_s"].tolist() == [1.0, 2.5]


def test_zero_time_offset_correction_is_noop():
    frame = pd.DataFrame({"time_s": [1.0]})

    corrected = _apply_time_offset_correction(frame, 0.0)

    assert corrected is frame
    assert corrected["time_s"].tolist() == [1.0]
