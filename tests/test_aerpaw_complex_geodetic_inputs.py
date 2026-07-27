import numpy as np
import pandas as pd
import pytest

from raft_uav.io.aerpaw import (
    normalize_radar,
    normalize_rf,
    normalize_truth,
    projector_from_lla,
)


def _truth_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_raw": ["2024-01-01 00:00:00.000000"],
            "latitude": [35.0],
            "longitude": [-78.0],
            "altitude_m": [100.0],
        }
    )


@pytest.mark.parametrize(
    ("latitude", "longitude", "altitude", "field"),
    [
        (35.0 + 1.0j, -78.0, 100.0, "origin_latitude_deg"),
        (35.0, -78.0 + 1.0j, 100.0, "origin_longitude_deg"),
        (35.0, -78.0, 100.0 + 1.0j, "origin_altitude_m"),
    ],
)
def test_projector_from_lla_rejects_complex_values(
    latitude,
    longitude,
    altitude,
    field,
):
    with pytest.raises(ValueError, match=field):
        projector_from_lla(latitude, longitude, altitude)


def test_normalize_truth_rejects_complex_explicit_origin():
    with pytest.raises(ValueError, match="origin_latitude_deg"):
        normalize_truth(
            _truth_frame(),
            enu_origin_lla=(np.complex128(35.0 + 1.0j), -78.0, 100.0),
        )


def test_normalize_truth_rejects_complex_coordinate_column():
    truth = _truth_frame().astype({"latitude": complex})

    with pytest.raises(ValueError, match="truth latitude must contain real values"):
        normalize_truth(truth)


@pytest.mark.parametrize(
    "wrapped_complex",
    [
        np.array(np.complex64(35.0 + 1.0j), dtype=object),
        np.ma.array(np.complex64(35.0 + 1.0j), dtype=object, mask=False),
    ],
)
def test_normalize_truth_rejects_object_wrapped_complex_coordinate(
    wrapped_complex,
):
    truth = _truth_frame()
    truth["latitude"] = pd.Series([wrapped_complex], dtype=object)

    with pytest.raises(ValueError, match="truth latitude must contain real values"):
        normalize_truth(truth)


def test_normalize_rf_rejects_complex_coordinate_column():
    projector = projector_from_lla(35.0, -78.0, 100.0)
    rf = pd.DataFrame(
        {
            "Time": ["2024-01-01 00:00:00"],
            "Latitude": np.array([35.0 + 1.0j]),
            "Longitude": [-78.0],
            "CEP": [25.0],
        }
    )

    with pytest.raises(ValueError, match="RF Latitude must contain real values"):
        normalize_rf(rf, projector, pd.Timestamp("2024-01-01 00:00:00"))


def test_normalize_radar_rejects_complex_coordinate_column():
    projector = projector_from_lla(35.0, -78.0, 100.0)
    radar = pd.DataFrame(
        {
            "global_time_raw_s": [1_704_067_200.0],
            "latitude": [35.0],
            "longitude": np.array([-78.0 + 1.0j]),
            "altitude_m": [100.0],
        }
    )

    with pytest.raises(ValueError, match="radar longitude must contain real values"):
        normalize_radar(radar, projector, pd.Timestamp("2024-01-01 00:00:00"))


def test_valid_object_wrapped_real_origin_is_preserved():
    projector = projector_from_lla(
        np.array(35.0, dtype=object),
        np.array(-78.0, dtype=object),
        np.array(100.0, dtype=object),
    )

    assert projector.origin_latitude_deg == pytest.approx(35.0)
    assert projector.origin_longitude_deg == pytest.approx(-78.0)
    assert projector.origin_altitude_m == pytest.approx(100.0)
