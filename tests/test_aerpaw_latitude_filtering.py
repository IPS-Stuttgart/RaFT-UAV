import pandas as pd

from raft_uav.coordinates import LocalENUProjector
from raft_uav.io.aerpaw import normalize_radar, normalize_rf, normalize_truth


def test_normalize_truth_skips_out_of_range_latitude_rows():
    truth = pd.DataFrame(
        {
            "timestamp_raw": [
                "1970-01-01 00:00:01",
                "1970-01-01 00:00:02",
            ],
            "latitude": [95.0, 35.0],
            "longitude": [-78.0, -78.0],
            "altitude_m": [10.0, 10.0],
        }
    )

    normalized, _projector, origin_time = normalize_truth(truth)

    assert origin_time == pd.Timestamp("1970-01-01 00:00:02")
    assert normalized["time_s"].tolist() == [0.0]
    assert normalized["latitude"].tolist() == [35.0]


def test_normalize_rf_skips_out_of_range_latitude_rows():
    rf = pd.DataFrame(
        {
            "Time": [
                "1970-01-01 00:00:01",
                "1970-01-01 00:00:02",
            ],
            "Latitude": [95.0, 35.0],
            "Longitude": [-78.0, -78.0],
            "Elevation": [10.0, 10.0],
            "CEP": [25.0, 25.0],
        }
    )
    projector = LocalENUProjector(35.0, -78.0, 10.0)

    normalized = normalize_rf(
        rf,
        projector,
        pd.Timestamp("1970-01-01"),
        clock_offset_s=0.0,
    )

    assert normalized["time_s"].tolist() == [2.0]
    assert normalized["Latitude"].tolist() == [35.0]


def test_normalize_radar_skips_out_of_range_latitude_rows():
    radar = pd.DataFrame(
        {
            "global_time_raw_s": [1.0, 2.0],
            "latitude": [95.0, 35.0],
            "longitude": [-78.0, -78.0],
            "altitude_m": [10.0, 10.0],
        }
    )
    projector = LocalENUProjector(35.0, -78.0, 10.0)

    normalized = normalize_radar(
        radar,
        projector,
        pd.Timestamp("1970-01-01"),
        clock_offset_s=0.0,
    )

    assert normalized["time_s"].tolist() == [2.0]
    assert normalized["latitude"].tolist() == [35.0]
