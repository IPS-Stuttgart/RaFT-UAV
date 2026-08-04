import numpy as np
import pandas as pd
import pytest

import raft_uav.io.aerpaw as aerpaw
from raft_uav.coordinates import LocalENUProjector


def _empty_rf_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": pd.Series(dtype=object),
            "Latitude": pd.Series(dtype=float),
            "Longitude": pd.Series(dtype=float),
            "Elevation": pd.Series(dtype=float),
            "CEP": pd.Series(dtype=float),
        }
    )


def _call_normalize_rf(clock_offset_s: object) -> pd.DataFrame:
    return aerpaw.normalize_rf(
        _empty_rf_frame(),
        object(),
        pd.Timestamp("2026-01-01"),
        clock_offset_s=clock_offset_s,
    )


def _call_normalize_radar(clock_offset_s: object) -> pd.DataFrame:
    return aerpaw.normalize_radar(
        pd.DataFrame(),
        object(),
        pd.Timestamp("2026-01-01"),
        clock_offset_s=clock_offset_s,
    )


@pytest.mark.parametrize(
    "normalizer",
    [_call_normalize_rf, _call_normalize_radar],
    ids=["rf", "radar"],
)
@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.nan,
        np.inf,
        -np.inf,
        1.0 + 0.0j,
        np.array([1.0]),
        np.ma.masked,
        None,
    ],
)
def test_aerpaw_normalizers_reject_invalid_clock_offsets(normalizer, value):
    with pytest.raises(
        ValueError,
        match="clock_offset_s must be a finite real scalar",
    ):
        normalizer(value)


def test_aerpaw_normalizers_accept_scalar_like_real_clock_offsets():
    origin_time = pd.Timestamp("1970-01-01")
    projector = LocalENUProjector(1.0, 1.0, 0.0)
    offset = np.array("1.5")

    rf = aerpaw.normalize_rf(
        pd.DataFrame(
            {
                "Time": ["1970-01-01 00:00:00"],
                "Latitude": [1.0],
                "Longitude": [1.0],
                "Elevation": [0.0],
                "CEP": [10.0],
            }
        ),
        projector,
        origin_time,
        clock_offset_s=offset,
    )
    radar = aerpaw.normalize_radar(
        pd.DataFrame(
            {
                "global_time_raw_s": [0.0],
                "latitude": [1.0],
                "longitude": [1.0],
                "altitude_m": [0.0],
            }
        ),
        projector,
        origin_time,
        clock_offset_s=offset,
    )

    np.testing.assert_allclose(rf["time_s"].to_numpy(), np.array([1.5]))
    np.testing.assert_allclose(radar["time_s"].to_numpy(), np.array([1.5]))
