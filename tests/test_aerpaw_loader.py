import json

import numpy as np

from raft_uav.coordinates import LocalENUProjector
from raft_uav.io.aerpaw import (
    normalize_radar,
    normalize_rf,
    normalize_truth,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    select_radar_measurement_rows,
)


def test_enu_origin_maps_to_near_zero():
    projector = LocalENUProjector(35.7274895, -78.696216, 2.717)

    enu = projector.transform(35.7274895, -78.696216, 2.717)

    np.testing.assert_allclose(enu, np.zeros(3), atol=1e-6)


def test_timestamp_normalization_is_monotonic_and_truth_relative(tmp_path):
    truth_path = tmp_path / "vehicleOut.txt"
    truth_path.write_text(
        "\n".join(
            [
                '1,-78.696216,35.7274895,2.717,"(0,0,0)","(0,0,0)",49.6,2025-10-07 15:42:19.000000,4,27',
                '2,-78.6962159,35.7274895,2.727,"(0,0,0)","(0,0,0)",49.6,2025-10-07 15:42:20.000000,4,27',
            ]
        ),
        encoding="utf-8",
    )
    rf_path = tmp_path / "rf.csv"
    rf_path.write_text(
        "\n".join(
            [
                "Meas ID,Algorithm,Center Frequency,Sample Rate,Num Samples,Latitude,Longitude,Elevation,RHO,CEP,Total Sensors,Valid Sensors,Overloaded Sensors,Sensor Names,Time,Comment",
                "1,tdoa,1,1,1,35.72749,-78.69621,0,0,75,4,4,0,s,2025-10-07 19:42:20.000,",
            ]
        ),
        encoding="utf-8",
    )

    truth, projector, origin_time = normalize_truth(read_truth(truth_path))
    rf = normalize_rf(read_rf_csv(rf_path), projector, origin_time)

    assert truth["time_s"].tolist() == [0.0, 1.0]
    assert truth["time_s"].is_monotonic_increasing
    np.testing.assert_allclose(rf["time_s"].to_numpy(), np.array([1.0]), atol=1e-6)


def test_radar_jsonl_reader_and_catprob_selection(tmp_path):
    radar_path = tmp_path / "radar.json"
    frames = [
        {"error": {"code": 200, "message": ""}},
        {
            "params": {"globalTime": 1759866140.0},
            "trackData": [
                {
                    "id": 1,
                    "lla": [35.72749, -78.69621, 30.0],
                    "globalTime": 1759866140.0,
                    "catProb": [0.6, 0.1],
                },
                {
                    "id": 2,
                    "lla": [35.727, -78.696, 30.0],
                    "globalTime": 1759866140.0,
                    "catProb": [0.1, 0.1],
                },
            ],
        },
    ]
    radar_path.write_text("\n".join(json.dumps(frame) for frame in frames), encoding="utf-8")
    truth_path = tmp_path / "vehicleOut.txt"
    truth_path.write_text(
        '1,-78.696216,35.7274895,2.717,"(0,0,0)","(0,0,0)",49.6,2025-10-07 15:42:20.000000,4,27',
        encoding="utf-8",
    )

    truth, projector, origin_time = normalize_truth(read_truth(truth_path))
    radar = normalize_radar(read_radar_tracks_json(radar_path), projector, origin_time)
    selected = select_radar_measurement_rows(radar, selection="catprob", catprob_threshold=0.5)

    assert len(radar) == 2
    assert selected["track_id"].tolist() == [1]
    assert len(truth) == 1
