from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.time_offset import apply_time_offset


@pytest.mark.parametrize(
    ("time_value", "expected_message"),
    [
        (True, "Boolean timestamp values"),
        (np.bool_(False), "Boolean timestamp values"),
        (1.0 + 2.0j, "nonzero imaginary components"),
        (np.complex64(1.0 + 2.0j), "nonzero imaginary components"),
        (np.array(1.0 + 2.0j), "nonzero imaginary components"),
    ],
)
def test_apply_time_offset_rejects_invalid_timestamp_cells(
    time_value: object,
    expected_message: str,
) -> None:
    frame = pd.DataFrame({"time_s": pd.Series([time_value], dtype=object)})

    with pytest.raises(ValueError, match=expected_message):
        apply_time_offset(frame, 0.25)


def test_apply_time_offset_validates_reused_uncorrected_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [1.25],
            "time_s_uncorrected": pd.Series([1.0 + 2.0j], dtype=object),
        }
    )

    with pytest.raises(ValueError, match="time_s_uncorrected.*nonzero imaginary"):
        apply_time_offset(frame, 0.5)


def test_apply_time_offset_preserves_real_like_and_missing_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "time_s": pd.Series(
                [np.complex64(1.0 + 0.0j), "2.5", np.ma.masked],
                dtype=object,
            )
        }
    )

    shifted = apply_time_offset(frame, 0.25)

    np.testing.assert_allclose(
        shifted["time_s"],
        [1.25, 2.75, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        shifted["time_s_uncorrected"],
        [1.0, 2.5, np.nan],
        equal_nan=True,
    )
